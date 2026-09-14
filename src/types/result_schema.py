from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
from .step_schema import SafetyTier


class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    BUSINESS_OUTCOME_FAIL = "BUSINESS_OUTCOME_FAIL"
    TECHNICAL_FAIL = "TECHNICAL_FAIL"
    HUMAN_ESCALATED = "HUMAN_ESCALATED"
    HARD_ABORT = "HARD_ABORT"


class RecoveryTier(str, Enum):
    TIER_1_RULE = "TIER_1_RULE"
    TIER_2_LLM = "TIER_2_LLM"
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
    strategy_name: str = Field(min_length=1)
    resolved: bool
    screenshot_path: Optional[str] = None
    details: Optional[str] = None


class StepExecutionTrace(BaseModel):
    step_id: str = Field(min_length=1)
    sequence_index: int = Field(ge=0)
    status: StepStatus
    safety_tier: SafetyTier
    attempt_count: int = Field(gt=0)
    duration_ms: int = Field(ge=0)
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
    operator_id: Optional[str] = None
    resolution: Optional[HandoffResolution] = None
    session_lock_token: Optional[str] = None


class ErrorDetail(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    stack: Optional[str] = None


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
    mode: str = Field(description="DISCOVERY | REPLAY")
    status: ExecutionStatus
    start_time: datetime
    end_time: datetime
    duration_ms: int = Field(ge=0)
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
    error: Optional[ErrorDetail] = None