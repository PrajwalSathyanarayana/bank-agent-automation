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
    OutputParamDefinition,
    ParamType,
)
from src.types.placeholders import find_placeholders
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


# --- output contract (EXTRACT_TEXT + output_key + output_definitions) ---

def _extract_step(output_key: str = "balance", sequence_index: int = 1) -> Step:
    return Step(
        sequence_index=sequence_index,
        action=ActionType.EXTRACT_TEXT,
        description="Extract the savings balance",
        locators=[_valid_locator()],
        output_key=output_key,
    )


def test_extract_text_step_with_output_key_constructs():
    step = _extract_step()
    assert step.action == ActionType.EXTRACT_TEXT
    assert step.output_key == "balance"


def test_extract_text_step_without_output_key_rejected():
    with pytest.raises(ValidationError):
        Step(
            sequence_index=1,
            action=ActionType.EXTRACT_TEXT,
            description="Extract the savings balance",
            locators=[_valid_locator()],
        )


def test_non_extract_step_with_output_key_rejected():
    with pytest.raises(ValidationError):
        Step(
            sequence_index=0,
            action=ActionType.CLICK,
            description="Click the search button",
            locators=[_valid_locator()],
            output_key="balance",
        )


def test_artifact_with_matching_output_definition_and_extract_step_constructs():
    artifact = Artifact(
        metadata=_valid_metadata(),
        output_definitions=[
            OutputParamDefinition(key="balance", type=ParamType.NUMBER, description="Savings balance")
        ],
        steps=[_valid_step(), _extract_step(output_key="balance")],
    )
    assert artifact.output_definitions[0].key == "balance"


def test_artifact_rejects_orphan_output_definition():
    with pytest.raises(ValidationError):
        Artifact(
            metadata=_valid_metadata(),
            output_definitions=[
                OutputParamDefinition(key="balance", type=ParamType.NUMBER, description="Savings balance")
            ],
            steps=[_valid_step()],  # no EXTRACT_TEXT step produces "balance"
        )


def test_artifact_rejects_orphan_extract_step():
    with pytest.raises(ValidationError):
        Artifact(
            metadata=_valid_metadata(),
            output_definitions=[],  # "balance" never declared
            steps=[_valid_step(), _extract_step(output_key="balance")],
        )


def test_artifact_rejects_duplicate_output_definition_keys():
    with pytest.raises(ValidationError):
        Artifact(
            metadata=_valid_metadata(),
            output_definitions=[
                OutputParamDefinition(key="balance", type=ParamType.NUMBER, description="First"),
                OutputParamDefinition(key="balance", type=ParamType.NUMBER, description="Duplicate"),
            ],
            steps=[_extract_step(output_key="balance")],
        )


def test_artifact_rejects_two_steps_producing_same_output_key():
    with pytest.raises(ValidationError):
        Artifact(
            metadata=_valid_metadata(),
            output_definitions=[
                OutputParamDefinition(key="balance", type=ParamType.NUMBER, description="Savings balance")
            ],
            steps=[
                _extract_step(output_key="balance", sequence_index=1),
                _extract_step(output_key="balance", sequence_index=2),
            ],
        )


# --- option_value on dropdown steps ---

def _select_step(input_value, option_value=None, action=ActionType.SELECT) -> Step:
    return Step(
        sequence_index=0,
        action=action,
        description="Choose an option",
        locators=[_valid_locator()],
        input_value=input_value,
        option_value=option_value,
    )


def test_fixed_dropdown_with_label_and_hidden_value_constructs():
    step = _select_step("Member Number", option_value="MEMNUM")
    assert step.option_value == "MEMNUM"


def test_input_driven_dropdown_without_hidden_value_constructs():
    step = _select_step("{payee_name}")
    assert step.option_value is None


def test_input_driven_dropdown_with_hidden_value_rejected():
    # The wrong-payee case: discovery's P001 frozen next to a payee input.
    with pytest.raises(ValidationError):
        _select_step("{payee_name}", option_value="P001")


def test_hidden_value_on_non_select_step_rejected():
    with pytest.raises(ValidationError):
        _select_step("Log In", option_value="X", action=ActionType.CLICK)


def test_hidden_value_without_label_rejected():
    with pytest.raises(ValidationError):
        _select_step(None, option_value="MEMNUM")


def test_doubled_braces_are_literal_text_not_a_placeholder():
    step = _select_step("Plan {{A}}", option_value="PLAN_A")
    assert step.option_value == "PLAN_A"


def test_find_placeholders():
    assert find_placeholders("{payee_name}") == ["payee_name"]
    assert find_placeholders("For member {member_id}, pay {amount}") == ["member_id", "amount"]
    assert find_placeholders("{credential:bank_password}") == ["credential:bank_password"]
    assert find_placeholders("Plan {{A}}") == []
    assert find_placeholders("no placeholders here") == []
