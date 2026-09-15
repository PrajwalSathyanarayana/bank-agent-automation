"""Replay: after a step's action, its checks must hold before the next step runs.

Each check gets its own time (the checkpoint's timeout, 10 s by default), because the page
may still be loading: it is looked at again every short while until it holds or the time
is up. The first check that doesn't hold stops the step and says what was expected and
what was seen, ready for the result's failure block.
"""
import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
from urllib.parse import urlsplit

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Locator as PageLocator
from playwright.async_api import Page

from src.locating.checks import find_phrase, shows_pattern, text_pattern, visible_text
from src.locating.resolver import UnfillableLocator, resolve
from src.types.artifact_schema import KnownOutcome, OutcomeSignal
from src.types.placeholders import fill_text
from src.types.step_schema import CheckpointType, Step, StepCheckpoint

_POLL_S = 0.2
_SHOWN_MAX = 80


@dataclass(frozen=True)
class CheckValues:
    """This run's inputs for the checks: text fills addresses, titles and locators; numbers
    are matched in any common form in checked text."""

    text: Mapping[str, str]
    numbers: Mapping[str, float]

    def as_text(self) -> dict[str, str]:
        # Numbers written plainly, for a check that compares a whole value (50.0 → "50").
        return {**self.text, **{key: format(Decimal(str(value)).normalize(), "f") for key, value in self.numbers.items()}}


@dataclass(frozen=True)
class CheckFailed:
    """What the check expected and what replay saw instead, each in a few words. next_step
    marks the one check about the next step's element rather than this step's own result."""

    expected: str
    observed: str
    next_step: bool = False


async def verify_step_checks(
    page: Page, step: Step, next_step: Optional[Step], values: CheckValues,
    known_outcomes: Sequence[KnownOutcome] = ()
) -> Optional[CheckFailed]:
    """The first of the step's checks that doesn't hold within its time; None when all hold.
    A declared outcome appearing on the page ends the wait early — the checkpoint will never
    hold once the bank has already answered."""
    for checkpoint in step.checkpoints:
        failed = await _verify(page, checkpoint, next_step, values, known_outcomes)
        if failed is not None:
            return failed
    return None


async def verify_shown_text(
    element: PageLocator, phrase: str, values: CheckValues, timeout_ms: int,
    known_outcomes: Sequence[KnownOutcome] = ()
) -> Optional[CheckFailed]:
    """A checking step (assert_text): the element its locators found is visible and shows
    the phrase, with this run's values filled in."""
    pattern = text_pattern(phrase, values.text, values.numbers)
    expected = f'"{_cap(fill_text(phrase, values.as_text()))}" shown'
    if pattern is not None and await _within(timeout_ms, lambda: shows_pattern(element, pattern),
                                             element.page, known_outcomes):
        return None
    return CheckFailed(expected, f'shown: "{_cap(await _safely(visible_text(element), ""))}"')


async def _verify(
    page: Page, checkpoint: StepCheckpoint, next_step: Optional[Step], values: CheckValues,
    known_outcomes: Sequence[KnownOutcome]
) -> Optional[CheckFailed]:
    kind = checkpoint.type
    timeout_ms = checkpoint.timeout_ms
    if kind == CheckpointType.NEXT_STEP_TARGET:
        return await _next_step_present(page, next_step, values, timeout_ms, known_outcomes)
    if kind in (CheckpointType.PAGE_PATH, CheckpointType.PAGE_TITLE, CheckpointType.URL_CONTAINS):
        return await _page_check(page, kind, fill_text(checkpoint.expected_value or "", values.text), timeout_ms,
                                 known_outcomes)
    return await _element_check(page, checkpoint, values, timeout_ms, known_outcomes)


async def _page_check(page: Page, kind: CheckpointType, wanted: str, timeout_ms: int,
                      known_outcomes: Sequence[KnownOutcome]) -> Optional[CheckFailed]:
    async def current() -> str:
        if kind == CheckpointType.PAGE_TITLE:
            return (await page.title()).strip()
        return urlsplit(page.url).path or "/" if kind == CheckpointType.PAGE_PATH else page.url

    async def holds() -> bool:
        seen = await current()
        return wanted in seen if kind == CheckpointType.URL_CONTAINS else seen == wanted

    if await _within(timeout_ms, holds, page, known_outcomes):
        return None
    seen = await _safely(current(), "(the page was still changing)")
    if kind == CheckpointType.PAGE_TITLE:
        return CheckFailed(f'page title "{_cap(wanted)}"', f'page title "{_cap(seen)}"')
    if kind == CheckpointType.PAGE_PATH:
        return CheckFailed(f"page path {_cap(wanted)}", f"page path {_cap(seen)}")
    return CheckFailed(f"an address containing {_cap(wanted)}", f"address {_cap(seen)}")


async def _element_check(
    page: Page, checkpoint: StepCheckpoint, values: CheckValues, timeout_ms: int,
    known_outcomes: Sequence[KnownOutcome]
) -> Optional[CheckFailed]:
    kind = checkpoint.type
    target = checkpoint.target_locator
    described = f"the element {_cap(target.value)}" if target else "its element"
    try:
        element = resolve(page, target, values.text)
    except UnfillableLocator:
        return CheckFailed(f"{described} on the page", "its locator can't be filled with this run's values")

    if kind == CheckpointType.ELEMENT_VISIBLE:
        expected = f"{described} visible"

        async def holds() -> bool:
            return await element.count() == 1 and await element.is_visible()
    elif kind == CheckpointType.TEXT_MATCH:
        pattern = text_pattern(checkpoint.expected_value or "", values.text, values.numbers)
        expected = f'{described} showing "{_cap(fill_text(checkpoint.expected_value or "", values.as_text()))}"'

        async def holds() -> bool:
            return pattern is not None and await element.count() == 1 and await shows_pattern(element, pattern)
    else:  # VALUE_EQUALS: a field holding exactly this value
        wanted = fill_text(checkpoint.expected_value or "", values.as_text())
        expected = f'{described} holding "{_cap(wanted)}"'

        async def holds() -> bool:
            return await element.count() == 1 and await element.input_value() == wanted

    if await _within(timeout_ms, holds, page, known_outcomes):
        return None
    count = await _safely(element.count(), 0)
    if count != 1:
        return CheckFailed(expected, f"{count} matching elements")
    if kind == CheckpointType.VALUE_EQUALS:
        return CheckFailed(expected, f'holding "{_cap(await _safely(element.input_value(), ""))}"')
    return CheckFailed(expected, f'shown: "{_cap(await _safely(visible_text(element), ""))}"')


async def _next_step_present(
    page: Page, next_step: Optional[Step], values: CheckValues, timeout_ms: int,
    known_outcomes: Sequence[KnownOutcome]
) -> Optional[CheckFailed]:
    # Answered with the next step's own locators: any one of them matching exactly one element.
    if next_step is None or not next_step.locators:
        return None
    elements = []
    for locator in next_step.locators:
        try:
            elements.append(resolve(page, locator, values.text))
        except UnfillableLocator:
            continue

    async def holds() -> bool:
        for element in elements:
            if await element.count() == 1:
                return True
        return False

    if await _within(timeout_ms, holds, page, known_outcomes):
        return None
    return CheckFailed(f"the element for step {next_step.sequence_index} on the page", "none of its locators found it",
                       next_step=True)


async def _within(
    timeout_ms: int, holds: Callable[[], Awaitable[bool]], page: Page, known_outcomes: Sequence[KnownOutcome]
) -> bool:
    # Look again every short while until it holds, a declared outcome answers first (the
    # checkpoint will never hold once the bank has already answered), or the time is up.
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        try:
            if await holds():
                return True
            if known_outcomes and await _outcome_showing(page, known_outcomes):
                return False
        except PlaywrightError:
            pass  # the page was between documents; look again
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_POLL_S)


async def _outcome_showing(page: Page, outcomes: Sequence[KnownOutcome]) -> bool:
    for outcome in outcomes:
        if outcome.signal == OutcomeSignal.PAGE_TEXT and await _shows(page, outcome.text or ""):
            return True
    return False


async def _shows(page: Page, phrase: str) -> bool:
    matches = await find_phrase(page, phrase)
    for handle in matches:
        await handle.dispose()
    return bool(matches)


async def _safely(reading: Awaitable, fallback):
    # A reading for the failure message; a page mid-change mustn't turn a failure into a crash.
    try:
        return await reading
    except PlaywrightError:
        return fallback


def _cap(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= _SHOWN_MAX else text[:_SHOWN_MAX - 1] + "…"
