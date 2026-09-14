from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import uuid
import re
from .placeholders import CREDENTIAL_PREFIX, find_placeholders, iter_placeholders
from .step_schema import ActionType, CheckpointType, Step

_SIMPLE_NAME = r"^[a-z][a-z0-9_]*$"
_ADDRESS_SEGMENT_END = set("/\"'?#&")
# Checkpoints whose expected value is an address, so placeholders there must be whole segments.
_ADDRESS_CHECKS = {CheckpointType.URL_CONTAINS, CheckpointType.PAGE_PATH}


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


class OutputType(str, Enum):
    """What an output holds when the caller receives it. An output is never a secret."""

    STRING = "string"
    NUMBER = "number"
    # An exact amount, returned as plain decimal text ("2450.32") in the output's currency,
    # never as a floating-point number.
    MONEY = "money"


class OutputParamDefinition(BaseModel):
    key: str = Field(min_length=1)
    type: OutputType
    description: str = Field(min_length=1)
    # ISO 4217 code such as "USD": required for money, allowed nowhere else.
    currency: Optional[str] = Field(default=None, pattern=r"^[A-Z]{3}$")

    @model_validator(mode="after")
    def validate_currency(self) -> "OutputParamDefinition":
        if (self.type == OutputType.MONEY) != (self.currency is not None):
            raise ValueError(f"output {self.key}: a money output needs a currency, and only a money output has one")
        return self


class OutcomeSignal(str, Enum):
    """How replay recognises a known outcome."""

    # The page shows this text: the whole phrase, in any case, visible.
    PAGE_TEXT = "page_text"
    # A dropdown offers no option for this input's value (a payee the member doesn't have).
    NO_SUCH_OPTION = "no_such_option"


class KnownOutcome(BaseModel):
    """A legitimate answer other than success, declared by the engineer with a stable code
    the caller can branch on ("no such member" is an answer, not a crash)."""

    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    description: str = Field(min_length=1)
    signal: OutcomeSignal
    # For page_text: the phrase, matched as written.
    text: Optional[str] = None
    # For no_such_option: the input whose value the dropdown doesn't offer.
    input_key: Optional[str] = None

    @model_validator(mode="after")
    def validate_signal_fields(self) -> "KnownOutcome":
        if self.signal == OutcomeSignal.PAGE_TEXT:
            if not (self.text and self.text.strip()) or self.input_key is not None:
                raise ValueError(f"outcome {self.code}: page_text needs the text to look for, and no input_key")
            if find_placeholders(self.text):
                raise ValueError(f"outcome {self.code}: the text is matched as written; placeholders are not allowed")
        elif self.input_key is None or self.text is not None:
            raise ValueError(f"outcome {self.code}: no_such_option needs the input_key, and no text")
        return self


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
    known_outcomes: list[KnownOutcome] = Field(
        default_factory=list,
        description="Legitimate answers other than success, each with a stable code "
        "and how replay recognises it"
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
    def validate_known_outcomes(self) -> "Artifact":
        codes = [outcome.code for outcome in self.known_outcomes]
        if len(codes) != len(set(codes)):
            raise ValueError("known_outcomes codes must be unique")
        inputs = {p.key for p in self.input_parameters}
        for outcome in self.known_outcomes:
            if outcome.input_key is not None and outcome.input_key not in inputs:
                raise ValueError(f"outcome {outcome.code}: {outcome.input_key} is not a declared input")
        return self

    @model_validator(mode="after")
    def validate_navigate_only_first(self) -> "Artifact":
        # A navigate step means "open the start URL", which only makes sense at the start.
        for step in self.steps[1:]:
            if step.action == ActionType.NAVIGATE:
                raise ValueError(
                    f"step {step.sequence_index}: a navigate step may only be the first step"
                )
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
                    address=checkpoint.type in _ADDRESS_CHECKS,
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
            if prefix != CREDENTIAL_PREFIX:
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