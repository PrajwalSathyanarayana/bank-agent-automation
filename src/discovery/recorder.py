"""Records what discovery did as the artifact's steps, with automatic checkpoints.

A step is drafted before its action runs, while the element still exists and its
locators can be proven, and committed once the action has run.
"""
from dataclasses import dataclass
from typing import Optional

from playwright.async_api import ElementHandle

from src.discovery.locators import DerivedLocators
from src.locating.checks import element_wording
from src.safety.classifier import classify
from src.types.placeholders import find_placeholders
from src.types.step_schema import ActionType, Step

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
    """Builds the artifact's steps in order: the start step first, then each action."""

    def __init__(self) -> None:
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
        self, action: Action, element: ElementHandle, derived: DerivedLocators, current_url: str
    ) -> Step:
        """The step for an action about to run on `element`, with its proven locators.

        The model's text is stored as written. Its placeholders are the intended ones;
        a literal value that slipped through (a member ID typed out instead of
        {member_id}) is caught by the save-time backstop scan, not here.
        """
        if action.kind not in RECORDED_ACTIONS:
            raise RecordingError(f"{action.kind.value} is not one of the model's recorded actions")
        if not self._steps:
            raise RecordingError("the start step must be recorded before any action")

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


def _with_tier(step: Step, current_url: str, wording: list[str]) -> Step:
    # The element's own wording decides first; the description and locators can only
    # raise the tier further.
    tier = classify(step, current_url, element_wording=wording)
    return step.model_copy(update={"safety_tier": tier})


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
