import json
from enum import Enum
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
import uuid
from .step_schema import SafetyTier

# What a failure says about a step, kept short enough to read at a glance.
FAILURE_TEXT_MAX = 200


class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    # A legitimate answer the caller needs ("no such member"), not a crash.
    BUSINESS_OUTCOME = "BUSINESS_OUTCOME"
    TECHNICAL_FAIL = "TECHNICAL_FAIL"
    HUMAN_ESCALATED = "HUMAN_ESCALATED"
    HARD_ABORT = "HARD_ABORT"


class RecoveryTier(str, Enum):
    # A known interruption the contract declares, cleared by its declared recovery.
    TIER_1_RULE = "TIER_1_RULE"
    # A person takes over the live session.
    TIER_3_HANDOFF = "TIER_3_HANDOFF"


class StepStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    RECOVERED = "RECOVERED"
    SKIPPED = "SKIPPED"


class HandoffResolution(str, Enum):
    RESUMED = "RESUMED"
    MANUAL_COMPLETED = "MANUAL_COMPLETED"
    OPERATOR_TIMED_OUT = "OPERATOR_TIMED_OUT"
    ABORTED = "ABORTED"


class RecoveryAttemptLog(BaseModel):
    timestamp: datetime
    tier: RecoveryTier
    # The declared interruption this recovered from, e.g. PROMO_POPUP; None for a handoff.
    interruption_code: Optional[str] = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    resolved: bool
    screenshot_path: Optional[str] = None
    details: Optional[str] = None


class StepExecutionTrace(BaseModel):
    step_id: str = Field(min_length=1)
    sequence_index: int = Field(ge=0)
    # The artifact's own words for this step (e.g. "Open Bill Pay for this member"), so a
    # reviewer reads what happened without cross-referencing the artifact separately.
    # Optional: a result saved before this field existed has none.
    description: Optional[str] = None
    status: StepStatus
    safety_tier: SafetyTier
    attempt_count: int = Field(gt=0)
    duration_ms: int = Field(ge=0)
    # Which of the step's locators found the element: 0 is the primary; a fallback here
    # means the page has changed since discovery.
    locator_priority: Optional[int] = Field(default=None, ge=0)
    recovery_logs: list[RecoveryAttemptLog] = Field(
        default_factory=list
    )
    failure_screenshot_path: Optional[str] = None
    error_message: Optional[str] = None


class HandoffTelemetry(BaseModel):
    triggered_timestamp: datetime
    resolved_timestamp: Optional[datetime] = None
    duration_ms: Optional[int] = Field(default=None, ge=0)
    trigger_reason: str = Field(min_length=1)
    # The step the run paused at; None when it paused before any step.
    step_index: Optional[int] = Field(default=None, ge=0)
    operator_id: Optional[str] = None
    resolution: Optional[HandoffResolution] = None
    session_lock_token: Optional[str] = None
    # How many actions the person took (clicks, fields changed, pages visited); each is a
    # line in the run log. None when no person took control.
    person_actions: Optional[int] = Field(default=None, ge=0)


class ErrorDetail(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    stack: Optional[str] = None


class BusinessOutcome(BaseModel):
    """The known answer the run ended with, as the contract declares it."""

    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    description: str = Field(min_length=1)
    # The page showing this answer, when replay could still reach one to screenshot.
    screenshot_path: Optional[str] = None


class FailureDetail(BaseModel):
    """Where a run broke and how: the step, what was expected there and what was seen.
    Short and factual, redacted like the run log, never a typed value."""

    step_index: int = Field(ge=0)
    step_description: str = Field(min_length=1)
    expected: str = Field(min_length=1, max_length=FAILURE_TEXT_MAX)
    observed: str = Field(min_length=1, max_length=FAILURE_TEXT_MAX)
    screenshot_path: Optional[str] = None


class EvidencePaths(BaseModel):
    log_file: str = Field(min_length=1)
    screenshots_dir: str = Field(min_length=1)
    playwright_trace_zip: Optional[str] = None


class ExecutionResult(BaseModel):
    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    capability: str = Field(min_length=1)
    artifact_version: Optional[str] = None
    mode: Literal["DISCOVERY", "REPLAY"]
    status: ExecutionStatus
    # What was asked and what happened, in plain English, for anyone reading the result.
    summary: Optional[str] = None
    # Whether the irreversible step (for bill pay, the payment) happened: "unknown" only when
    # the run stopped after clicking it and couldn't confirm what followed. None when the
    # capability has no irreversible step, or discovery never met one.
    irreversible_step: Optional[Literal["not_reached", "completed", "unknown"]] = None
    start_time: datetime
    end_time: datetime
    duration_ms: int = Field(ge=0)
    # True only when replay checked the artifact's signature before running it.
    integrity_verified: bool = False
    step_traces: list[StepExecutionTrace] = Field(
        default_factory=list
    )
    handoff_events: list[HandoffTelemetry] = Field(
        default_factory=list
    )
    evidence_paths: EvidencePaths
    # Money arrives as exact decimal text ("2450.32"); a float is only ever a plain number.
    terminal_outputs: Optional[dict[str, str | int | float | bool]] = Field(
        default=None,
        description="Extracted data returned to the calling agent"
    )
    outcome: Optional[BusinessOutcome] = None
    failure: Optional[FailureDetail] = None
    error: Optional[ErrorDetail] = None

    @model_validator(mode="after")
    def validate_what_each_status_carries(self) -> "ExecutionResult":
        status = self.status
        if status == ExecutionStatus.SUCCESS and (self.error or self.failure or self.outcome):
            raise ValueError("a SUCCESS result carries no error, failure or outcome")
        if status == ExecutionStatus.BUSINESS_OUTCOME and (self.outcome is None or self.error or self.failure):
            raise ValueError("a BUSINESS_OUTCOME result carries its outcome, and no error or failure")
        if status in (ExecutionStatus.TECHNICAL_FAIL, ExecutionStatus.HARD_ABORT) and (
            self.error is None or self.outcome
        ):
            raise ValueError(f"a {status.value} result carries an error, and no outcome")
        if status == ExecutionStatus.HUMAN_ESCALATED and (not self.handoff_events or self.outcome):
            raise ValueError("a HUMAN_ESCALATED result carries a handoff record, and no outcome")
        if self.integrity_verified and self.mode != "REPLAY":
            raise ValueError("only replay verifies an artifact's signature")
        return self

    def printable(self) -> dict[str, Any]:
        """The result as a caller or reviewer reads it: empty fields are left out."""
        return _without_empty(self.model_dump(mode="json"))

    def to_json(self) -> str:
        return json.dumps(self.printable(), indent=2, ensure_ascii=False)


def _without_empty(node: Any) -> Any:
    # None, [] and {} say nothing; False, 0 and "" are values and stay.
    if isinstance(node, dict):
        cleaned = {key: _without_empty(value) for key, value in node.items()}
        return {key: value for key, value in cleaned.items() if not _is_empty(value)}
    if isinstance(node, list):
        return [_without_empty(value) for value in node]
    return node


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, (list, dict)) and not value)
