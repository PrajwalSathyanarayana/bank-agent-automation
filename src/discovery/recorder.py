"""Records what discovery did as the artifact's steps, with automatic checkpoints.

A step is drafted before its action runs, while the element still exists and its
locators can be proven, and committed once the action has run.
"""
import dataclasses
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

from playwright.async_api import ElementHandle, Page

from src.config.settings import settings
from src.discovery.locators import (
    MAX_LOCATORS,
    Candidate,
    DerivedLocators,
    NoProvenLocator,
    RunValues,
    Verdict,
    derive_locators,
    label_cell_value_xpath,
    parameterize_address,
    prove,
    scan,
)
from src.discovery.perception import Box, ElementFacts, PageElement
from src.locating.checks import element_wording, find_phrase, is_password_box
from src.observability.logger import RunLogger
from src.safety.classifier import classify
from src.safety.secret_typing import typing_refusal
from src.types.placeholders import find_placeholders
from src.types.step_schema import ActionType, CheckpointType, Locator, LocatorType, Step, StepCheckpoint

START_DESCRIPTION = "Open the start page"

# What the model's tools are recorded as: click, type_text, select_option,
# extract_text, and assert_visible (a checking step of its own).
RECORDED_ACTIONS = {
    ActionType.CLICK,
    ActionType.TYPE,
    ActionType.SELECT,
    ActionType.EXTRACT_TEXT,
    ActionType.ASSERT_TEXT,
}
# Actions whose input_value holds the model's text: typed on TYPE, the option label
# on SELECT, the text expected on ASSERT_TEXT.
_VALUE_ACTIONS = {ActionType.TYPE, ActionType.SELECT, ActionType.ASSERT_TEXT}


class RecordingError(RuntimeError):
    """Raised when the loop asks for a recording the artifact can't hold."""


class AssertionRefused(RecordingError):
    """The model's assertion can't be recorded; the message says why, worded for the model."""


class TypingRefused(RecordingError):
    """The value can't be typed into this field; nothing was typed. Worded for the model."""


class ExtractionRefused(RecordingError):
    """The value can't be read by that label; the message says why, worded for the model."""


# The cell after the one holding the label, in the same row: where legacy pages show a value.
_VALUE_CELL = """(element) => {
  const cell = element.closest("td, th");
  if (!cell) return null;
  let next = cell.nextElementSibling;
  while (next && !["TD", "TH"].includes(next.tagName)) next = next.nextElementSibling;
  return next;
}"""
# The label cell's whole text, the way the XPath locator will compare it.
_LABEL_CELL_TEXT = """(element) => {
  const cell = element.closest("td, th");
  return cell ? (cell.textContent || "").replace(/\\s+/g, " ").trim() : "";
}"""


# The basic facts locators.py needs about an element found by its text rather than
# picked from the numbered list.
_ELEMENT_BASICS = """(element) => {
  const box = element.getBoundingClientRect();
  const role = (element.getAttribute("role") || "").trim().toLowerCase().split(/\\s+/)[0];
  return {
    tag: element.tagName.toLowerCase(),
    input_type: element.tagName === "INPUT" ? element.type : "",
    role: role,
    box: { x: box.x, y: box.y, width: box.width, height: box.height },
  };
}"""


@dataclass(frozen=True)
class DraftedAssertion:
    """A checking step, with the locator details the run log reports."""

    step: Step
    derived: DerivedLocators


@dataclass(frozen=True)
class DraftedExtraction:
    """A reading step, its locator details for the run log, and the value read now."""

    step: Step
    derived: DerivedLocators
    value: str


@dataclass(frozen=True)
class Committed:
    """A step as it entered the recording."""

    step: Step
    # Names of secrets found in the page address or title; those checks were left out.
    secrets_found: list[str]


@dataclass(frozen=True)
class Action:
    """One of the model's actions, as the loop hands it to the recorder."""

    kind: ActionType
    # The model's reason, recorded as the step's description.
    reason: str
    # The text typed, the option label chosen, or the text expected, exactly as the
    # model wrote it, placeholders included ({member_id}, {credential:bank_password}).
    value: Optional[str] = None
    # For extract_text: the declared output this reading fills.
    output_key: Optional[str] = None


class Recorder:
    """Builds the artifact's steps in order: the start step first, then each action.

    Every committed step is written to the run log as it is recorded.
    """

    def __init__(self, logger: RunLogger) -> None:
        self._logger = logger
        self._steps: list[Step] = []

    @property
    def steps(self) -> list[Step]:
        return list(self._steps)

    def draft_start(self, start_url: str) -> Step:
        """Step 0: open the artifact's start URL. The URL itself lives in the metadata."""
        if self._steps:
            raise RecordingError("the start step can only be the first step")
        step = Step(sequence_index=0, action=ActionType.NAVIGATE, description=START_DESCRIPTION)
        return _with_tier(step, start_url, wording=[])

    async def draft_step(
        self,
        action: Action,
        element: ElementHandle,
        derived: DerivedLocators,
        current_url: str,
        *,
        run: RunValues,
    ) -> Step:
        """The step for an action about to run on `element`, with its proven locators.

        The model's text is stored as written. Its placeholders are the intended ones;
        a literal value that slipped through (a member ID typed out instead of
        {member_id}) is caught by the save-time backstop scan, not here.

        Typing is the exception, checked here because a step is drafted before its action
        runs: a password box takes exactly one secret placeholder and a secret placeholder
        goes only into a password box. Anything else raises TypingRefused before a key is
        pressed, so a wrong password never reaches the bank and a secret never shows on
        screen. run names this run's secrets.
        """
        if action.kind not in RECORDED_ACTIONS:
            raise RecordingError(f"{action.kind.value} is not one of the model's recorded actions")
        if not self._steps:
            raise RecordingError("the start step must be recorded before any action")
        if action.kind == ActionType.TYPE:
            refusal = typing_refusal(
                action.value or "",
                into_password_box=await is_password_box(element),
                secret_names=run.secrets.keys(),
            )
            if refusal:
                raise TypingRefused(refusal)

        option_value = None
        if action.kind == ActionType.SELECT and action.value and not find_placeholders(action.value):
            option_value = await _fixed_option_value(element, action.value)

        step = Step(
            sequence_index=len(self._steps),
            action=action.kind,
            description=_description(action),
            locators=derived.locators,
            input_value=action.value if action.kind in _VALUE_ACTIONS else None,
            option_value=option_value,
            output_key=action.output_key if action.kind == ActionType.EXTRACT_TEXT else None,
        )
        return _with_tier(step, current_url, wording=await element_wording(element))

    async def draft_assertion(self, phrase: str, reason: str, page: Page, run: RunValues) -> DraftedAssertion:
        """A checking step for the model's assert_visible: the one element showing the phrase.

        The element is visible and shows the phrase as whole words, ignoring case, so the
        assertion passes now by construction. It is refused (AssertionRefused, worded for
        the model) when the phrase is empty, not shown, or shown by more than one element:
        a checking step needs one element whose locators replay can find again.
        """
        if not phrase.strip():
            raise AssertionRefused("the assertion needs the text you expect to see")
        matches = await find_phrase(page, phrase)
        try:
            if not matches:
                raise AssertionRefused(
                    f'no visible element shows "{phrase}"; quote the text as it appears on the screen'
                )
            if len(matches) > 1:
                raise AssertionRefused(
                    f'"{phrase}" is shown by {len(matches)} elements; quote a longer phrase that appears once'
                )
            element = matches[0]
            basics = await element.evaluate(_ELEMENT_BASICS)
            facts = ElementFacts(
                tag=basics["tag"], box=Box(**basics["box"]), input_type=basics["input_type"], role=basics["role"]
            )
            target = PageElement(number=0, facts=facts, description="", in_viewport=True, handle=element)
            derived = await derive_locators(page, target, run)
            step = await self.draft_step(
                Action(ActionType.ASSERT_TEXT, reason, value=phrase), element, derived, page.url, run=run
            )
            return DraftedAssertion(step, derived)
        finally:
            for handle in matches:
                await handle.dispose()

    async def draft_extraction(
        self, label: str, output_key: str, reason: str, page: Page, run: RunValues
    ) -> DraftedExtraction:
        """A step that reads the value shown right after a label, for a declared output.

        The model quotes the label, not the value: the label is the same for every record,
        so the first locator is anchored on it, and the value's own text never becomes a
        locator. The label must appear on one visible element, inside a table cell that
        has a cell after it. Refused otherwise (ExtractionRefused, worded for the model).
        """
        if not label.strip():
            raise ExtractionRefused("quote the label shown next to the value")
        matches = await find_phrase(page, label)
        try:
            if not matches:
                raise ExtractionRefused(
                    f'no visible element shows "{label}"; quote the label as it appears on the screen'
                )
            if len(matches) > 1:
                raise ExtractionRefused(
                    f'"{label}" is shown by {len(matches)} elements; quote a longer label that appears once'
                )
            value_cell = (await matches[0].evaluate_handle(_VALUE_CELL)).as_element()
            if value_cell is None:
                raise ExtractionRefused(
                    f'no table cell follows "{label}"; quote the label shown right before the value'
                )
            try:
                value = " ".join((await value_cell.inner_text()).split())
                if not value:
                    raise ExtractionRefused(f'the cell after "{label}" is empty')
                label_text = await matches[0].evaluate(_LABEL_CELL_TEXT)
                derived = await self._value_locators(page, value_cell, label_text, output_key, value, run)
                step = await self.draft_step(
                    Action(ActionType.EXTRACT_TEXT, reason, output_key=output_key), value_cell, derived, page.url,
                    run=run,
                )
                return DraftedExtraction(step, derived, value)
            finally:
                await value_cell.dispose()
        finally:
            for handle in matches:
                await handle.dispose()

    async def _value_locators(
        self, page: Page, value_cell: ElementHandle, label_text: str, output_key: str, value: str, run: RunValues
    ) -> DerivedLocators:
        # The value is this run's data: kept out of every locator, the way an input's
        # value is. The label-anchored XPath goes first; generated fallbacks follow it.
        guarded = dataclasses.replace(run, text_inputs={**run.text_inputs, output_key: value})
        anchored: list[Locator] = []
        xpath = label_cell_value_xpath(label_text)
        if xpath is not None:
            outcome = scan(Candidate("label", LocatorType.XPATH, xpath, data=(label_text,)), guarded)
            if outcome.stored_value is not None:
                locator = Locator(type=LocatorType.XPATH, value=outcome.stored_value, priority=0)
                if await prove(page, value_cell, locator, run.text_inputs) is Verdict.PROVEN:
                    anchored.append(locator)

        basics = await value_cell.evaluate(_ELEMENT_BASICS)
        facts = ElementFacts(
            tag=basics["tag"], box=Box(**basics["box"]), input_type=basics["input_type"], role=basics["role"]
        )
        target = PageElement(number=0, facts=facts, description="", in_viewport=True, handle=value_cell)
        try:
            fallback = await derive_locators(page, target, guarded)
        except NoProvenLocator:
            fallback = DerivedLocators([], [], False, [], [])
        if not anchored and not fallback.locators:
            raise ExtractionRefused("that value can't be found again reliably on a later run; quote another label")

        kept = [*anchored, *fallback.locators][:MAX_LOCATORS]
        kinds = [*(["label"] if anchored else []), *fallback.kinds][: len(kept)]
        return DerivedLocators(
            locators=[Locator(type=locator.type, value=locator.value, priority=number) for number, locator in enumerate(kept)],
            kinds=kinds,
            weak=not anchored and fallback.weak,
            rejected=fallback.rejected,
            secrets_found=fallback.secrets_found,
        )

    async def commit(
        self,
        step: Step,
        page: Page,
        run: RunValues,
        *,
        derived: Optional[DerivedLocators],
        acted: bool = True,
    ) -> Committed:
        """Add a drafted step to the recording, with automatic checks of where the page landed.

        derived is what locators.py found for the step's element (None only for the start
        step, which has no element); it is required so the run log always gets the
        locator kinds, rejections and weak flag.

        After an action (or opening the start page) the step gets a page-path and a
        page-title check, each cleared of this run's data or left out if it can't be.
        The previous step gets a check that this step's element is present, so a failed
        form submission is caught at the step that caused it; the last step never gets
        one, since nothing follows it.

        A checking step gets no path or title check: it doesn't move the page. acted=False
        is for the irreversible step, recorded but never clicked: the page after it was
        never seen, so there is nothing to check.
        """
        if step.sequence_index != len(self._steps):
            raise RecordingError(
                f"step {step.sequence_index} is out of order; the next step is {len(self._steps)}"
            )
        checkpoints: list[StepCheckpoint] = []
        landing_secrets: list[str] = []
        if acted and step.action != ActionType.ASSERT_TEXT:
            checkpoints, landing_secrets = await _landing_checks(page, run)
        step = step.model_copy(update={"checkpoints": [*step.checkpoints, *checkpoints]})

        next_check_added_to = None
        if self._steps:
            previous = self._steps[-1]
            self._steps[-1] = _with_next_step_check(previous)
            if self._steps[-1] is not previous:
                next_check_added_to = previous.sequence_index
        self._steps.append(step)

        self._log_step(step, derived, acted, next_check_added_to)
        locator_secrets = derived.secrets_found if derived is not None else []
        if locator_secrets:
            self._logger.secret_on_page(step.sequence_index, locator_secrets, "locator candidates")
        if landing_secrets:
            self._logger.secret_on_page(step.sequence_index, landing_secrets, "page address or title")
        return Committed(step, sorted({*locator_secrets, *landing_secrets}))

    def _log_step(
        self, step: Step, derived: Optional[DerivedLocators], acted: bool, next_check_added_to: Optional[int]
    ) -> None:
        kinds = derived.kinds if derived is not None else [None] * len(step.locators)
        self._logger.step_recorded(
            index=step.sequence_index,
            action=step.action.value,
            description=step.description,
            safety_tier=step.safety_tier.value,
            acted=acted,
            input_value=step.input_value,
            locators=[
                {"priority": locator.priority, "kind": kind, "type": locator.type.value, "value": locator.value}
                for locator, kind in zip(step.locators, kinds)
            ],
            weak=derived.weak if derived is not None else False,
            rejected=[
                {"kind": rejection.kind, "reason": rejection.reason}
                for rejection in (derived.rejected if derived is not None else [])
            ],
            checkpoints=[
                {"type": checkpoint.type.value, "expected_value": checkpoint.expected_value}
                for checkpoint in step.checkpoints
            ],
            is_assertion=step.action == ActionType.ASSERT_TEXT,
            next_step_check_added_to=next_check_added_to,
        )


def _with_tier(step: Step, current_url: str, wording: list[str]) -> Step:
    # The element's own wording decides first; the description and locators can only
    # raise the tier further.
    tier = classify(step, current_url, element_wording=wording)
    return step.model_copy(update={"safety_tier": tier})


async def _landing_checks(page: Page, run: RunValues) -> tuple[list[StepCheckpoint], list[str]]:
    """Page-path and page-title checks for where an action landed, cleared of run data.

    The path keeps no host (each bank has its own) and no query or fragment; a segment
    equal to a text input becomes a placeholder (/member/{member_id}/accounts). Anything
    else carrying this run's data leaves that check out, with the same scan as locators.
    """
    checkpoints: list[StepCheckpoint] = []
    secrets_found: list[str] = []

    address = urlsplit(page.url)
    # Only web pages have a path to check; a blank or data page (tests) has none.
    if address.scheme in ("http", "https"):
        path = address.path or "/"
        outcome = scan(Candidate("address", LocatorType.CSS, "", address=path), run)
        stored_path = parameterize_address(path, run.text_inputs) if outcome.stored_value is not None else None
        if stored_path is not None:
            checkpoints.append(_checkpoint(CheckpointType.PAGE_PATH, stored_path))
        elif outcome.secret:
            secrets_found.append(outcome.secret)

    title = (await page.title()).strip()
    if title:
        outcome = scan(Candidate("text", LocatorType.TEXT_CONTENT, title, data=(title,)), run)
        if outcome.stored_value is not None:
            checkpoints.append(_checkpoint(CheckpointType.PAGE_TITLE, outcome.stored_value))
        elif outcome.secret:
            secrets_found.append(outcome.secret)
    return checkpoints, secrets_found


def _with_next_step_check(step: Step) -> Step:
    # Answered at replay with the next step's own locators and fallbacks.
    if any(checkpoint.type == CheckpointType.NEXT_STEP_TARGET for checkpoint in step.checkpoints):
        return step
    check = StepCheckpoint(type=CheckpointType.NEXT_STEP_TARGET, timeout_ms=settings.replay_checkpoint_timeout_ms)
    return step.model_copy(update={"checkpoints": [*step.checkpoints, check]})


def _checkpoint(kind: CheckpointType, expected_value: str) -> StepCheckpoint:
    return StepCheckpoint(type=kind, expected_value=expected_value, timeout_ms=settings.replay_checkpoint_timeout_ms)


def _description(action: Action) -> str:
    # The tool requires a reason; an empty one is recorded as such, never invented.
    return action.reason.strip() or f"{action.kind.value} (the model gave no reason)"


async def _fixed_option_value(select: ElementHandle, label: str) -> Optional[str]:
    """The hidden value of the option with this label, for replay's fallback.

    Only for a fixed choice: when the label is a placeholder such as {payee_name}, a
    value frozen at discovery would select the wrong option on fallback, so none is kept.
    """
    value = await select.evaluate(
        "(select, label) => {"
        " const option = Array.from(select.options).find((candidate) => candidate.label === label);"
        " return option ? option.value : null; }",
        label,
    )
    return value or None
