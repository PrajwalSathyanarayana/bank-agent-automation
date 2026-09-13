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


class StepParameter(BaseModel):
    param_key: str = Field(min_length=1)
    is_sensitive: bool = False
    description: Optional[str] = None


class CheckpointType(str, Enum):
    ELEMENT_VISIBLE = "element_visible"
    TEXT_MATCH = "text_match"
    URL_CONTAINS = "url_contains"
    VALUE_EQUALS = "value_equals"


class StepCheckpoint(BaseModel):
    checkpoint_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    type: CheckpointType
    target_locator: Locator
    expected_value: Optional[str] = None
    timeout_ms: int = Field(default=5000, gt=0)


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
    locators: list[Locator] = Field(min_length=1)
    safety_tier: SafetyTier = SafetyTier.SAFE
    input_parameter: Optional[StepParameter] = None
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