"""What a person does during a discovery handoff, recorded as steps of the artifact,
continuing the agent's own recording.

A person's click is recorded exactly as the agent's would be: its element read by the same
collector, its locators derived and proven on the live page before the click goes through,
its safety tier from the classifier, the same allowlist. Typing and choosing are stored as
the task's own placeholders ({member_id}, {credential:bank_password}): the value is only
compared, in memory, and never stored or logged. What can't be recorded safely is either
refused before it happens (a click) or kept as a problem, so the run never saves a
recording with a gap (a typed value that matches none of the task's inputs).

A click that clears one of the contract's declared interruptions (the promotion's Close)
goes through but isn't recorded, as the agent's closing of an overlay isn't: it may not
appear on the next run, and replay clears it by itself whenever it does.
"""
from typing import Optional, Sequence

from playwright.async_api import ElementHandle, Frame
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from src.config.settings import settings
from src.discovery.locators import NoProvenLocator, RunValues, derive_locators
from src.discovery.perception import element_for
from src.discovery.recorder import Action, Recorder, TypingRefused
from src.handoff.control_bar import let_through
from src.locating.checks import element_wording
from src.locating.resolver import UnfillableLocator, resolve
from src.locating.values import number_pattern
from src.observability.logger import RunLogger
from src.safety.allowlist import AllowlistViolation, check_domain, check_route
from src.surface.browser import ActionFailed, BrowserSession, click
from src.types.artifact_schema import KnownInterruption, RecoveryAction
from src.types.placeholders import CREDENTIAL_PREFIX
from src.types.step_schema import ActionType, SafetyTier

BY_A_PERSON = "(done by a person)"
NOT_RECORDABLE = "That can't be recorded as a step, so it wasn't done; use a button, link or field on the page."
OFF_LIMITS = "That leads outside the pages this task may use, so it wasn't followed."


class PersonStepRecorder:
    """Discovery's PersonSteps: each of the person's actions becomes the next step."""

    def __init__(self, recorder: Recorder, session: BrowserSession, run: RunValues, allowed_paths: Sequence[str],
                 *, username_key: str, known_interruptions: Sequence[KnownInterruption] = (),
                 logger: Optional[RunLogger] = None) -> None:
        self._recorder = recorder
        self._session = session
        self._run = run
        self._allowed_paths = list(allowed_paths)
        self._username_key = username_key
        self._interruptions = list(known_interruptions)
        self._logger = logger
        # What was last stored for each field (by its first locator), so saving it unchanged
        # adds no second step.
        self._stored: dict[str, str] = {}
        self.recorded = 0
        self.irreversible_done = False
        # What couldn't be recorded: the recording has a gap and mustn't be saved.
        self.problems: list[str] = []

    async def clicked(self, element: ElementHandle) -> Optional[str]:
        page = self._session.page
        cleared = await self._clears_interruption(element)
        if cleared is not None:
            try:
                await let_through(element)
                await click(element, timeout_ms=settings.discovery_page_action_timeout_ms)
            except ActionFailed as failure:
                return f"That click didn't go through ({failure}); try again."
            if self._logger is not None:
                self._logger.overlay_dismissed(f"a person cleared {cleared}, which replay clears by itself")
            return None
        picked = await element_for(page, element)
        if picked is None:
            return NOT_RECORDABLE
        try:
            derived = await derive_locators(page, picked, self._run)
        except NoProvenLocator:
            return NOT_RECORDABLE
        wording = await element_wording(element)
        action = Action(ActionType.CLICK, f'Clicked "{(wording or [picked.description])[0]}" {BY_A_PERSON}')
        step = await self._recorder.draft_step(action, element, derived, page.url, run=self._run)
        if not self._allowed(page.url) or await self._leads_off(element):
            return OFF_LIMITS

        boxes_before = self._session.dialogs_for_person
        navigated: list[str] = []

        def on_navigated(frame: Frame) -> None:
            if frame.parent_frame is None:
                navigated.append(frame.url)

        page.on("framenavigated", on_navigated)
        try:
            await let_through(element)
            # A confirm box the bank opens waits for the person, and the click waits with it.
            await click(element, timeout_ms=settings.operator_timeout_ms)
            await page.wait_for_load_state("load", timeout=settings.discovery_page_action_timeout_ms)
        except (ActionFailed, PlaywrightTimeoutError) as failure:
            return f"That click didn't go through ({str(failure).splitlines()[0]}); try again."
        finally:
            page.remove_listener("framenavigated", on_navigated)
        if self._session.dialogs_for_person > boxes_before and not navigated:
            # The person cancelled the bank's box: nothing happened, so nothing is recorded.
            return None
        if not self._allowed(page.url):
            await page.go_back()
            return OFF_LIMITS
        await self._recorder.commit(step, page, self._run, derived=derived)
        self.recorded += 1
        if step.safety_tier == SafetyTier.IRREVERSIBLE:
            self.irreversible_done = True
        return None

    async def changed(self, element: ElementHandle) -> Optional[str]:
        page = self._session.page
        picked = await element_for(page, element)
        if picked is None:
            return self._problem("a field the page doesn't offer was changed")
        label = picked.facts.label or picked.description
        choosing = picked.facts.tag == "select"
        shown = await (_chosen_label(element) if choosing else element.input_value())
        stored = self._placeholder_for(shown, into_password_box=picked.facts.input_type == "password")
        if stored is None and choosing and shown:
            stored = shown  # a fixed choice is kept as it reads, as the agent's would be
        if stored is None:
            return self._problem(f'the value entered in "{label}" is none of the task\'s inputs')
        try:
            derived = await derive_locators(page, picked, self._run)
        except NoProvenLocator:
            return self._problem(f'"{label}" can\'t be found again on a later run')
        field = derived.locators[0].value
        if self._stored.get(field) == stored:
            return None
        kind = ActionType.SELECT if choosing else ActionType.TYPE
        action = Action(kind, f'{"Chose" if choosing else "Entered"} "{label}" {BY_A_PERSON}', value=stored)
        try:
            step = await self._recorder.draft_step(action, element, derived, page.url, run=self._run)
        except TypingRefused as refused:
            return self._problem(str(refused))
        if not self._allowed(page.url):
            return self._problem(f'"{label}" is on a page outside this task\'s pages')
        await self._recorder.commit(step, page, self._run, derived=derived)
        self._stored[field] = stored
        self.recorded += 1
        return None

    def _placeholder_for(self, shown: str, *, into_password_box: bool) -> Optional[str]:
        """The placeholder a value stands for: one secret, the username, or one input. None
        when it matches nothing or several. Compared here only; never kept."""
        text = shown.strip()
        if not text:
            return None
        matches = [f"{{{CREDENTIAL_PREFIX}:{name}}}" for name, secret in self._run.secrets.items()
                   if shown == secret.get_secret_value()]
        if into_password_box:
            return matches[0] if len(matches) == 1 else None
        if self._username_key and self._run.username and text.lower() == self._run.username.lower():
            matches.append(f"{{{CREDENTIAL_PREFIX}:{self._username_key}}}")
        matches += [f"{{{key}}}" for key, value in self._run.text_inputs.items() if text.lower() == value.strip().lower()]
        matches += [f"{{{key}}}" for key, number in self._run.number_inputs.items()
                    if number_pattern(number).fullmatch(text)]
        return matches[0] if len(matches) == 1 else None

    async def _clears_interruption(self, element: ElementHandle) -> Optional[str]:
        """The code of the declared interruption this element is the clearing target of, if any."""
        for interruption in self._interruptions:
            if interruption.recovery != RecoveryAction.CLICK or interruption.target is None:
                continue
            try:
                target = resolve(self._session.page, interruption.target, self._run.text_inputs)
                if await target.count() == 1 and await target.evaluate("(el, other) => el === other", element):
                    return interruption.code
            except (UnfillableLocator, PlaywrightError):
                continue
        return None

    def _problem(self, what: str) -> str:
        self.problems.append(what)
        return f"That can't be saved as a step ({what}). You can carry on, but this run won't save what it learned."

    def _allowed(self, url: str) -> bool:
        try:
            check_domain(url)
            check_route(url, self._allowed_paths)
        except AllowlistViolation:
            return False
        return True

    async def _leads_off(self, element: ElementHandle) -> bool:
        # A link's destination is known before the click, so it is refused there, as the agent's is.
        try:
            destination = await element.evaluate("element => element.tagName === 'A' ? element.href : ''")
        except PlaywrightError:
            return False
        return destination.startswith(("http://", "https://")) and not self._allowed(destination)


async def _chosen_label(select: ElementHandle) -> str:
    return await select.evaluate("select => select.selectedOptions.length ? select.selectedOptions[0].label : ''")
