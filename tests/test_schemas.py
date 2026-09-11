from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.types.step_schema import (
    ActionType,
    CheckpointType,
    Locator,
    LocatorType,
    SafetyTier,
    Step,
    StepCheckpoint,
)
from src.types.artifact_schema import (
    Artifact,
    ArtifactMetadata,
    GlobalAssertion,
    GlobalAssertionType,
    InputParamDefinition,
    ParamType,
)
from src.types.result_schema import (
    EvidencePaths,
    ExecutionResult,
    ExecutionStatus,
    StepExecutionTrace,
    StepStatus,
)


def _valid_locator() -> Locator:
    return Locator(type=LocatorType.CSS, value="#member-id", priority=0)


def _valid_step(sequence_index: int = 0) -> Step:
    return Step(
        sequence_index=sequence_index,
        action=ActionType.CLICK,
        description="Click the search button",
        locators=[_valid_locator()],
    )


def _valid_metadata() -> ArtifactMetadata:
    now = datetime.now(timezone.utc)
    return ArtifactMetadata(
        capability="member-lookup",
        version="1.0.0",
        integrity_hash="a" * 64,
        target_url="http://localhost:5000/search",
        created_timestamp=now,
        last_updated_timestamp=now,
    )


# --- Step schema ---

def test_step_constructs_with_valid_data():
    step = _valid_step()
    assert step.action == ActionType.CLICK
    assert step.safety_tier == SafetyTier.SAFE
    assert step.locators[0].priority == 0


def test_step_requires_at_least_one_locator():
    with pytest.raises(ValidationError):
        Step(
            sequence_index=0,
            action=ActionType.CLICK,
            description="Click something",
            locators=[],
        )


def test_step_rejects_negative_sequence_index():
    with pytest.raises(ValidationError):
        Step(
            sequence_index=-1,
            action=ActionType.CLICK,
            description="Click something",
            locators=[_valid_locator()],
        )


def test_step_checkpoint_requires_target_locator():
    checkpoint = StepCheckpoint(
        type=CheckpointType.ELEMENT_VISIBLE,
        target_locator=_valid_locator(),
    )
    assert checkpoint.timeout_ms == 5000


# --- Artifact schema ---

def test_artifact_constructs_with_valid_data():
    artifact = Artifact(metadata=_valid_metadata(), steps=[_valid_step()])
    assert artifact.metadata.version == "1.0.0"
    assert len(artifact.steps) == 1
    assert artifact.global_assertions == []


def test_artifact_requires_at_least_one_step():
    with pytest.raises(ValidationError):
        Artifact(metadata=_valid_metadata(), steps=[])


def test_artifact_metadata_rejects_bad_semver():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        ArtifactMetadata(
            capability="member-lookup",
            version="v1",
            integrity_hash="a" * 64,
            target_url="http://localhost:5000/search",
            created_timestamp=now,
            last_updated_timestamp=now,
        )


def test_artifact_metadata_rejects_bad_integrity_hash():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        ArtifactMetadata(
            capability="member-lookup",
            version="1.0.0",
            integrity_hash="not-a-hash",
            target_url="http://localhost:5000/search",
            created_timestamp=now,
            last_updated_timestamp=now,
        )


def test_artifact_with_input_params_and_global_assertions():
    artifact = Artifact(
        metadata=_valid_metadata(),
        input_parameters=[
            InputParamDefinition(
                key="member_id",
                type=ParamType.STRING,
                description="The member ID to search for",
            )
        ],
        steps=[_valid_step()],
        global_assertions=[
            GlobalAssertion(
                type=GlobalAssertionType.FINAL_URL_MATCH,
                value="/member/",
            )
        ],
    )
    assert artifact.input_parameters[0].key == "member_id"
    assert artifact.global_assertions[0].type == GlobalAssertionType.FINAL_URL_MATCH


# --- ExecutionResult schema ---

def test_execution_result_constructs_with_valid_data():
    now = datetime.now(timezone.utc)
    result = ExecutionResult(
        capability="member-lookup",
        mode="REPLAY",
        status=ExecutionStatus.SUCCESS,
        start_time=now,
        end_time=now,
        duration_ms=1200,
        evidence_paths=EvidencePaths(
            log_file="evidence/replay/run_log.json",
            screenshots_dir="evidence/replay/screenshots",
        ),
    )
    assert result.status == ExecutionStatus.SUCCESS
    assert result.step_traces == []


def test_step_execution_trace_requires_positive_attempt_count():
    with pytest.raises(ValidationError):
        StepExecutionTrace(
            step_id="step-1",
            sequence_index=0,
            status=StepStatus.PASSED,
            safety_tier=SafetyTier.SAFE,
            attempt_count=0,
            duration_ms=100,
        )
