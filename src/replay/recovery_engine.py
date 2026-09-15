"""Replay: what happens when something gets in a step's way.

Two lists in the signed contract decide it, and nothing else does:
- a known outcome is an answer for the caller ("no such member"): replay stops and returns
  its code;
- a known interruption is an obstacle replay clears by itself (a popup, an expired
  session), with the one recovery a person approved: click a stated element, start over,
  or wait.
Anything on neither list is a failure. Nothing is guessed, and nothing unapproved is
clicked.
"""
import asyncio
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlsplit

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Locator as PageLocator
from playwright.async_api import Page

from src.config.settings import settings
from src.locating.checks import element_wording, find_phrase
from src.locating.resolver import UnfillableLocator, resolve
from src.observability.logger import RunLogger
from src.replay.checks import CheckValues
from src.safety.allowlist import AllowlistViolation, enforce_safety
from src.safety.classifier import classify
from src.types.artifact_schema import (
    InterruptionSignal,
    KnownInterruption,
    KnownOutcome,
    OutcomeSignal,
    RecoveryAction,
)
from src.types.result_schema import RecoveryAttemptLog, RecoveryTier
from src.types.routes import route_allowed
from src.types.step_schema import ActionType, SafetyTier, Step

_POLL_S = 0.2
# What a clearing click is classified as: only its element's own wording and the page count.
_CLEARING_DESCRIPTION = "Clear a known interruption"


# --- known outcomes: answers for the caller ---

async def outcome_showing(page: Page, outcomes: Sequence[KnownOutcome]) -> Optional[KnownOutcome]:
    """The first declared outcome whose text the page shows: the whole phrase, visible."""
    for outcome in outcomes:
        if outcome.signal == OutcomeSignal.PAGE_TEXT and await _shows(page, outcome.text or ""):
            return outcome
    return None


async def missing_option(
    step: Step, dropdown: PageLocator, outcomes: Sequence[KnownOutcome], values: CheckValues
) -> Optional[KnownOutcome]:
    """A select step whose dropdown doesn't offer this run's value, when a no_such_option
    outcome names that input (a payee the member doesn't have). Checked before selecting."""
    if step.action != ActionType.SELECT or step.input_value is None:
        return None
    for outcome in outcomes:
        if outcome.signal != OutcomeSignal.NO_SUCH_OPTION or step.input_value.strip() != f"{{{outcome.input_key}}}":
            continue
        wanted = values.text.get(outcome.input_key or "")
        labels = await dropdown.evaluate("(select) => Array.from(select.options).map((option) => option.label)")
        if wanted is not None and _flat(wanted) not in {_flat(label) for label in labels}:
            return outcome
    return None


# --- known interruptions: obstacles replay clears by itself ---

async def interruption_showing(
    page: Page, interruptions: Sequence[KnownInterruption], values: CheckValues
) -> Optional[KnownInterruption]:
    """The first declared interruption the page shows, by its declared signal."""
    for interruption in interruptions:
        if await _signal_showing(page, interruption, values):
            return interruption
    return None


class Recoveries:
    """The limits for one run: the same interruption is recovered at most a set number of
    times (it may genuinely come back, but not forever); starting over happens at most once,
    and never once an irreversible step has run, since that could repeat it."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._started_over = False
        self.total = 0

    def refusal(self, interruption: KnownInterruption, irreversible_done: bool) -> Optional[str]:
        """Why this recovery isn't allowed now; None if it is."""
        done = self._counts[interruption.code]
        if done >= settings.tier1_max_dismiss_attempts:
            return f"{interruption.code} came back after {done} recoveries"
        if interruption.recovery == RecoveryAction.START_OVER:
            if irreversible_done:
                return "starting over after the irreversible step ran could repeat it"
            if self._started_over:
                return "the run already started over once"
        return None

    def note(self, interruption: KnownInterruption) -> None:
        self._counts[interruption.code] += 1
        self._started_over = self._started_over or interruption.recovery == RecoveryAction.START_OVER


@dataclass(frozen=True)
class Recovery:
    """What was done about an interruption, for the step's trace; start_over asks the
    executor to run the artifact again from its first step."""

    log: RecoveryAttemptLog
    start_over: bool = False


async def recover(
    page: Page,
    interruption: KnownInterruption,
    values: CheckValues,
    allowed_paths: Sequence[str],
    logger: RunLogger,
    recoveries: Recoveries,
    *,
    irreversible_done: bool,
) -> Recovery:
    """Run the interruption's one declared recovery, within the run's limits.

    A screenshot is taken first, as evidence of what was in the way. A click passes the
    same gate as any step (the page is allowed, the click is SAFE, its target is exactly
    one element) and counts only if the interruption is gone afterwards.
    """
    recoveries.total += 1
    screenshot = await _screenshot(page, logger, recoveries.total, interruption.code)
    refused = recoveries.refusal(interruption, irreversible_done)
    start_over = False
    if refused is not None:
        resolved, details = False, refused
    else:
        recoveries.note(interruption)
        if interruption.recovery == RecoveryAction.CLICK:
            resolved, details = await _clear_by_click(page, interruption, values, allowed_paths)
        elif interruption.recovery == RecoveryAction.WAIT:
            resolved = await _gone(page, interruption, values)
            details = "it went away" if resolved else "it was still showing when the time was up"
        else:
            resolved, details, start_over = True, "starting over from the first step", True
    logger.recovery_event(RecoveryTier.TIER_1_RULE.value, screenshot, interruption_code=interruption.code,
                          recovery=interruption.recovery.value, resolved=resolved, details=details)
    log = RecoveryAttemptLog(timestamp=datetime.now(timezone.utc), tier=RecoveryTier.TIER_1_RULE,
                             interruption_code=interruption.code, resolved=resolved,
                             screenshot_path=screenshot, details=details)
    return Recovery(log, start_over)


async def _clear_by_click(
    page: Page, interruption: KnownInterruption, values: CheckValues, allowed_paths: Sequence[str]
) -> tuple[bool, str]:
    try:
        target = resolve(page, interruption.target, values.text)
        matches = await target.count()
    except UnfillableLocator:
        return False, "its target can't be filled with this run's values"
    if matches != 1:
        return False, f"its target matched {matches} elements, not one"
    # The same gate as any step: a page this capability may act on, and a SAFE click.
    try:
        enforce_safety(page.url, ActionType.CLICK, allowed_paths=allowed_paths)
    except AllowlistViolation:
        return False, "the page isn't one this capability may act on"
    clearing = Step(sequence_index=0, action=ActionType.CLICK, description=_CLEARING_DESCRIPTION,
                    locators=[interruption.target])
    tier = classify(clearing, page.url, element_wording=await element_wording(target))
    if tier != SafetyTier.SAFE:
        return False, f"its target is classified {tier.value}; only a SAFE click clears an interruption"
    await target.click(timeout=settings.replay_checkpoint_timeout_ms)
    if await _gone(page, interruption, values):
        return True, "clicked its target; it went away"
    return False, "clicked its target, but it was still showing"


async def _signal_showing(page: Page, interruption: KnownInterruption, values: CheckValues) -> bool:
    try:
        if interruption.signal == InterruptionSignal.PAGE_TEXT:
            return await _shows(page, interruption.text or "")
        if interruption.signal == InterruptionSignal.PAGE_PATH:
            return route_allowed(urlsplit(page.url).path or "/", [interruption.text or ""])
        element = resolve(page, interruption.locator, values.text)
        return await element.count() > 0 and await element.first.is_visible()
    except (UnfillableLocator, PlaywrightError):
        return False


async def _gone(page: Page, interruption: KnownInterruption, values: CheckValues) -> bool:
    deadline = time.monotonic() + settings.replay_checkpoint_timeout_ms / 1000
    while await _signal_showing(page, interruption, values):
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_POLL_S)
    return True


async def _shows(page: Page, phrase: str) -> bool:
    matches = await find_phrase(page, phrase)
    for handle in matches:
        await handle.dispose()
    return bool(matches)


async def _screenshot(page: Page, logger: RunLogger, number: int, code: str) -> Optional[str]:
    folder = settings.evidence_dir / "replay" / "screenshots"
    path = folder / f"{logger.trace_id}_recovery{number:02d}_{code}.png"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(path))
    except (PlaywrightError, OSError):
        return None
    return str(path)


def _flat(text: str) -> str:
    return " ".join(text.split())
