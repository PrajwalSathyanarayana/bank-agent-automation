from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import uuid
import re
from urllib.parse import urlparse
from .placeholders import CREDENTIAL_PREFIX, find_placeholders, iter_placeholders
from .routes import route_allowed, valid_route_pattern
from .step_schema import ActionType, CheckpointType, Locator, Step

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


class InterruptionSignal(str, Enum):
    """How replay recognises a known interruption."""

    # The page shows this phrase: the whole phrase, in any case, visible.
    PAGE_TEXT = "page_text"
    # The page's path matches this pattern ("/session-timeout").
    PAGE_PATH = "page_path"
    # This element is showing (an overlay covering the page).
    ELEMENT_VISIBLE = "element_visible"


class RecoveryAction(str, Enum):
    """What replay does about a known interruption; nothing else is ever tried."""

    # Click the stated element, such as a close button.
    CLICK = "click"
    # Run the artifact again from its first step (never after an irreversible step ran).
    START_OVER = "start_over"
    # Wait for it to go away, within the step's check timeout.
    WAIT = "wait"


class KnownInterruption(BaseModel):
    """An obstacle replay clears by itself — a popup, an expired session — declared by the
    engineer with one recovery, so replay never clicks anything nobody approved."""

    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    description: str = Field(min_length=1)
    signal: InterruptionSignal
    # For page_text: the phrase; for page_path: a page pattern. Written as it is.
    text: Optional[str] = None
    # For element_visible: the element that shows the interruption is there.
    locator: Optional[Locator] = None
    recovery: RecoveryAction
    # For a click recovery: what to click.
    target: Optional[Locator] = None

    @model_validator(mode="after")
    def validate_signal_and_recovery(self) -> "KnownInterruption":
        where = f"interruption {self.code}"
        if self.signal == InterruptionSignal.ELEMENT_VISIBLE:
            if self.locator is None or self.text is not None:
                raise ValueError(f"{where}: element_visible needs its locator, and no text")
        else:
            if not (self.text and self.text.strip()) or self.locator is not None:
                raise ValueError(f"{where}: {self.signal.value} needs its text, and no locator")
            if find_placeholders(self.text):
                raise ValueError(f"{where}: the text is matched as written; placeholders are not allowed")
            if self.signal == InterruptionSignal.PAGE_PATH and not valid_route_pattern(self.text):
                raise ValueError(f"{where}: page_path needs a page pattern such as /session-timeout")
        if (self.recovery == RecoveryAction.CLICK) != (self.target is not None):
            raise ValueError(f"{where}: a click recovery needs its target, and only a click has one")
        for locator in (self.locator, self.target):
            if locator is not None and find_placeholders(locator.value):
                raise ValueError(f"{where}: its locators are written as they are; placeholders are not allowed")
        return self


class CompareAs(str, Enum):
    # Word for word, ignoring only extra spaces; case matters.
    TEXT = "text"
    # Read exactly, to the cent, in the check's currency.
    MONEY = "money"


class ConfirmationCheck(BaseModel):
    """Before an irreversible step, the value shown beside this label must equal this run's
    input, or replay doesn't click and a person decides."""

    label: str = Field(min_length=1)
    input_key: str
    compare_as: CompareAs
    # ISO 4217 code for a money check, e.g. "USD"; nothing else has one.
    currency: Optional[str] = Field(default=None, pattern=r"^[A-Z]{3}$")

    @model_validator(mode="after")
    def validate_label_and_currency(self) -> "ConfirmationCheck":
        if not self.label.strip() or find_placeholders(self.label):
            raise ValueError("a confirmation check's label is the page's own text, with no placeholders")
        if (self.compare_as == CompareAs.MONEY) != (self.currency is not None):
            raise ValueError("a money confirmation check needs a currency, and only a money check has one")
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
        description="Ed25519 signature of the artifact's content, as hex; None until signed"
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
        # An Ed25519 signature is 64 bytes: 128 hex characters.
        if v is not None and not re.fullmatch(r"[a-f0-9]{128}", v):
            raise ValueError(
                "integrity_hash must be 128 lowercase hex characters"
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
    allowed_paths: list[str] = Field(
        default_factory=list,
        description="The pages this capability may visit on the bank's host, as path patterns "
        "('/member/*' is one segment); empty means any page on the host"
    )
    known_interruptions: list[KnownInterruption] = Field(
        default_factory=list,
        description="Obstacles replay clears by itself, each with how to spot it and its one recovery"
    )
    confirmation_checks: list[ConfirmationCheck] = Field(
        default_factory=list,
        description="What must match this run's inputs on screen before any irreversible step"
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
    def validate_allowed_paths(self) -> "Artifact":
        # Entries are named by position, not quoted: whatever was typed there stays out of errors.
        for position, pattern in enumerate(self.allowed_paths, start=1):
            if not valid_route_pattern(pattern):
                raise ValueError(f"allowed_paths entry {position} is not a page pattern: "
                                 "'/' or '/segment' parts, where a segment is text or '*'")
        if len(self.allowed_paths) != len(set(self.allowed_paths)):
            raise ValueError("allowed_paths entries must be unique")
        start_path = urlparse(self.metadata.target_url).path or "/"
        if self.allowed_paths and not route_allowed(start_path, self.allowed_paths):
            raise ValueError("allowed_paths: the start page must be one of the allowed pages")
        return self

    @model_validator(mode="after")
    def validate_interruptions_and_checks(self) -> "Artifact":
        codes = [interruption.code for interruption in self.known_interruptions]
        if len(codes) != len(set(codes)):
            raise ValueError("known_interruptions codes must be unique")
        shared = sorted(set(codes) & {outcome.code for outcome in self.known_outcomes})
        if shared:
            raise ValueError(f"{shared[0]} is both a known outcome and a known interruption")
        labels = [check.label for check in self.confirmation_checks]
        if len(labels) != len(set(labels)):
            raise ValueError("confirmation_checks labels must be unique")
        input_types = {p.key: p.type for p in self.input_parameters}
        for position, check in enumerate(self.confirmation_checks, start=1):
            if check.input_key not in input_types:
                raise ValueError(f"confirmation check {position}: {check.input_key} is not a declared input")
            # Money is compared with a number input, text with a text input.
            wanted = ParamType.NUMBER if check.compare_as == CompareAs.MONEY else ParamType.STRING
            if input_types[check.input_key] != wanted:
                raise ValueError(f"confirmation check {position}: a {check.compare_as.value} check "
                                 f"compares a {wanted.value} input")
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