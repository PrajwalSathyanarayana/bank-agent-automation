from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator
import uuid
from .placeholders import find_placeholders


class SafetyTier(str, Enum):
    SAFE = "SAFE"
    RISKY = "RISKY"
    IRREVERSIBLE = "IRREVERSIBLE"


class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    HOVER = "hover"
    NAVIGATE = "navigate"
    WAIT_FOR_ELEMENT = "wait_for_element"
    ASSERT_VISIBLE = "assert_visible"
    ASSERT_TEXT = "assert_text"
    EXTRACT_TEXT = "extract_text"


class LocatorType(str, Enum):
    CSS = "css"
    XPATH = "xpath"
    TEXT_CONTENT = "text_content"
    ARIA_LABEL = "aria_label"


class Locator(BaseModel):
    type: LocatorType
    value: str = Field(min_length=1)
    priority: int = Field(
        ge=0,
        description="0 = primary, 1 = first fallback, 2 = second fallback"
    )


class CheckpointType(str, Enum):
    ELEMENT_VISIBLE = "element_visible"
    TEXT_MATCH = "text_match"
    URL_CONTAINS = "url_contains"
    VALUE_EQUALS = "value_equals"
    PAGE_TITLE = "page_title"
    # Answered at replay with the next step's own locators, so none are stored here.
    NEXT_STEP_TARGET = "next_step_target"


_NEEDS_LOCATOR = {
    CheckpointType.ELEMENT_VISIBLE,
    CheckpointType.TEXT_MATCH,
    CheckpointType.VALUE_EQUALS,
}
_NEEDS_EXPECTED_VALUE = {
    CheckpointType.TEXT_MATCH,
    CheckpointType.VALUE_EQUALS,
    CheckpointType.URL_CONTAINS,
    CheckpointType.PAGE_TITLE,
}


class StepCheckpoint(BaseModel):
    checkpoint_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    type: CheckpointType
    target_locator: Optional[Locator] = None
    expected_value: Optional[str] = Field(default=None, min_length=1)
    timeout_ms: int = Field(
        gt=0,
        description="No default: the recorder fills it from settings, the single source for the number",
    )

    @model_validator(mode="after")
    def validate_fields_for_type(self) -> "StepCheckpoint":
        needs_locator = self.type in _NEEDS_LOCATOR
        if needs_locator and self.target_locator is None:
            raise ValueError(f"{self.type.value} checkpoint requires target_locator")
        if not needs_locator and self.target_locator is not None:
            raise ValueError(f"{self.type.value} checkpoint must not have target_locator")

        needs_value = self.type in _NEEDS_EXPECTED_VALUE
        if needs_value and self.expected_value is None:
            raise ValueError(f"{self.type.value} checkpoint requires expected_value")
        if not needs_value and self.expected_value is not None:
            raise ValueError(f"{self.type.value} checkpoint must not have expected_value")
        return self


class RetryBudget(BaseModel):
    max_attempts: int = Field(default=3, gt=0)
    poll_interval_ms: int = Field(default=500, gt=0)


class Step(BaseModel):
    step_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    sequence_index: int = Field(ge=0)
    action: ActionType
    description: str = Field(min_length=1)
    # At least one for every action except NAVIGATE, which locates nothing (validator below).
    locators: list[Locator] = Field(default_factory=list)
    safety_tier: SafetyTier = SafetyTier.SAFE
    input_value: Optional[str] = None
    checkpoints: list[StepCheckpoint] = Field(default_factory=list)
    retry_budget: RetryBudget = Field(default_factory=RetryBudget)
    output_key: Optional[str] = Field(
        default=None,
        description="Matches a key in the parent Artifact's output_definitions. "
        "Required if and only if action is EXTRACT_TEXT.",
    )

    option_value: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Hidden value of a fixed dropdown option. "
        "Forbidden when the selection comes from an input.",
    )

    @model_validator(mode="after")
    def validate_navigate_locates_nothing(self) -> "Step":
        # NAVIGATE opens the artifact's start URL (or the tenant's override), which lives
        # only in the metadata; every other action works on an element.
        if self.action == ActionType.NAVIGATE:
            if self.locators:
                raise ValueError("a navigate step opens the start URL and has no locators")
            if self.input_value is not None:
                raise ValueError("a navigate step has no input_value; the start URL is in the metadata")
        elif not self.locators:
            raise ValueError(f"a {self.action.value} step needs at least one locator")
        return self

    @model_validator(mode="after")
    def validate_output_key_matches_extract_text(self) -> "Step":
        if self.action == ActionType.EXTRACT_TEXT and not self.output_key:
            raise ValueError("output_key is required when action is EXTRACT_TEXT")
        if self.action != ActionType.EXTRACT_TEXT and self.output_key:
            raise ValueError("output_key is only valid when action is EXTRACT_TEXT")
        return self

    @model_validator(mode="after")
    def validate_option_value(self) -> "Step":
        if self.option_value is None:
            return self
        if self.action != ActionType.SELECT:
            raise ValueError("option_value is only valid on SELECT steps")
        if not self.input_value:
            raise ValueError("option_value requires the option's visible label in input_value")
        if find_placeholders(self.input_value):
            # A discovery-time hidden value would select the wrong option on fallback.
            raise ValueError("option_value is not allowed when the selection comes from an input")
        return self