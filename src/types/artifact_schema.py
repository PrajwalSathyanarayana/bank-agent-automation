from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import uuid
import re
from .placeholders import find_placeholders, iter_placeholders
from .step_schema import ActionType, CheckpointType, Step

_SIMPLE_NAME = r"^[a-z][a-z0-9_]*$"
_CREDENTIAL_PREFIX = "credential"
_ADDRESS_SEGMENT_END = set("/\"'?#&")


class ParamType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SECRET = "secret"


class InputParamDefinition(BaseModel):
    key: str = Field(pattern=_SIMPLE_NAME)
    type: ParamType
    required: bool = True
    description: str = Field(min_length=1)
    example_value: Optional[str] = None


class OutputParamDefinition(BaseModel):
    key: str = Field(min_length=1)
    type: ParamType
    description: str = Field(min_length=1)


class CredentialKind(str, Enum):
    CONFIG = "config"
    SECRET = "secret"


# Supplied by our system from configuration, never by the caller; stored by name only.
class CredentialDefinition(BaseModel):
    key: str = Field(pattern=_SIMPLE_NAME)
    kind: CredentialKind
    description: str = Field(min_length=1)


class GlobalAssertionType(str, Enum):
    FINAL_URL_MATCH = "final_url_match"
    SUCCESS_BANNER_TEXT = "success_banner_text"
    TERMINAL_DOM_STATE = "terminal_dom_state"


class GlobalAssertion(BaseModel):
    assertion_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    type: GlobalAssertionType
    value: str = Field(min_length=1)


class ArtifactMetadata(BaseModel):
    artifact_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    capability: str = Field(min_length=1)
    description: str = Field(
        min_length=1,
        description="What the capability does, written as a goal template, "
        "e.g. 'For member {member_id}, pay {amount} to {payee_name}.'"
    )
    version: str = Field(
        description="SemVer format e.g. 1.0.0"
    )
    integrity_hash: Optional[str] = Field(
        default=None,
        description="Keyed HMAC-SHA256 of the artifact's content; None until signed"
    )
    author: str = "DiscoveryEngine"
    target_url: str = Field(
        min_length=1,
        description="Entry point URL this artifact was recorded against"
    )
    tenant_override_url: Optional[str] = Field(
        default=None,
        description="Per-tenant URL override for multi-tenant reuse"
    )
    created_timestamp: datetime
    last_updated_timestamp: datetime

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        if not re.match(r"^\d+\.\d+\.\d+$", v):
            raise ValueError(
                "Version must follow SemVer format e.g. 1.0.0"
            )
        return v

    @field_validator("integrity_hash")
    @classmethod
    def validate_hex(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not re.fullmatch(r"[a-f0-9]{64}", v):
            raise ValueError(
                "integrity_hash must be 64 lowercase hex characters"
            )
        return v


class Artifact(BaseModel):
    metadata: ArtifactMetadata
    input_parameters: list[InputParamDefinition] = Field(
        default_factory=list
    )
    credentials: list[CredentialDefinition] = Field(
        default_factory=list,
        description="Values our system supplies from configuration (names only), "
        "never passed by the caller"
    )
    output_definitions: list[OutputParamDefinition] = Field(
        default_factory=list,
        description="Typed outputs this capability returns to the calling agent"
    )
    steps: list[Step] = Field(min_length=1)
    global_assertions: list[GlobalAssertion] = Field(
        default_factory=list,
        description="Terminal assertions verified after all steps complete"
    )

    @model_validator(mode="after")
    def validate_output_definitions_match_extract_steps(self) -> "Artifact":
        declared_keys = [o.key for o in self.output_definitions]
        if len(declared_keys) != len(set(declared_keys)):
            raise ValueError("output_definitions keys must be unique")

        extract_keys = [
            s.output_key for s in self.steps if s.action == ActionType.EXTRACT_TEXT
        ]
        if len(extract_keys) != len(set(extract_keys)):
            raise ValueError(
                "Each declared output must be produced by exactly one EXTRACT_TEXT step "
                "(duplicate output_key found across steps)"
            )

        declared_set = set(declared_keys)
        extract_set = set(extract_keys)

        orphan_declarations = declared_set - extract_set
        if orphan_declarations:
            raise ValueError(
                f"output_definitions declared but never produced by a step: {orphan_declarations}"
            )

        orphan_extractions = extract_set - declared_set
        if orphan_extractions:
            raise ValueError(
                f"EXTRACT_TEXT step(s) reference undeclared output_key: {orphan_extractions}"
            )

        return self

    @model_validator(mode="after")
    def validate_credential_keys_unique(self) -> "Artifact":
        keys = [c.key for c in self.credentials]
        if len(keys) != len(set(keys)):
            raise ValueError("credentials keys must be unique")
        return self

    @model_validator(mode="after")
    def validate_last_step_has_no_next_step_check(self) -> "Artifact":
        last = self.steps[-1]
        if any(c.type == CheckpointType.NEXT_STEP_TARGET for c in last.checkpoints):
            raise ValueError(
                f"step {last.sequence_index} is the last step; "
                "a next_step_target checkpoint has no next step to check"
            )
        return self

    @model_validator(mode="after")
    def validate_placeholders(self) -> "Artifact":
        inputs = {p.key for p in self.input_parameters}
        credentials = {c.key for c in self.credentials}

        def check(text, where, credentials_allowed=False, address=False):
            _check_placeholders(text, where, inputs, credentials, credentials_allowed, address)

        check(self.metadata.description, "description")
        for step in self.steps:
            where = f"step {step.sequence_index}"
            check(step.description, f"{where} description")
            check(
                step.input_value,
                f"{where} input_value",
                credentials_allowed=step.action == ActionType.TYPE,
            )
            if step.option_value and find_placeholders(step.option_value):
                raise ValueError(f"{where} option_value: placeholders are not allowed here")
            for locator in step.locators:
                check(locator.value, f"{where} locator", address=True)
            for checkpoint in step.checkpoints:
                if checkpoint.target_locator:
                    check(checkpoint.target_locator.value, f"{where} checkpoint locator", address=True)
                check(
                    checkpoint.expected_value,
                    f"{where} checkpoint",
                    address=checkpoint.type == CheckpointType.URL_CONTAINS,
                )
        for assertion in self.global_assertions:
            check(
                assertion.value,
                "global assertion",
                address=assertion.type == GlobalAssertionType.FINAL_URL_MATCH,
            )
        return self


def _check_placeholders(
    text: Optional[str],
    where: str,
    inputs: set[str],
    credentials: set[str],
    credentials_allowed: bool,
    address: bool,
) -> None:
    if not text:
        return
    for name, start, end in iter_placeholders(text):
        if ":" in name:
            prefix, key = name.split(":", 1)
            if prefix != _CREDENTIAL_PREFIX:
                raise ValueError(f"{where}: unknown placeholder {{{name}}}")
            if key not in credentials:
                raise ValueError(f"{where}: {{{name}}} is not in the credentials list")
            if not credentials_allowed:
                raise ValueError(f"{where}: credentials may only appear in typed values")
        elif name not in inputs:
            raise ValueError(f"{where}: {{{name}}} is not a declared input")
        if address and not _is_whole_segment(text, start, end):
            raise ValueError(f"{where}: {{{name}}} must be a whole address segment")


def _is_whole_segment(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return before == "/" and (after == "" or after in _ADDRESS_SEGMENT_END)