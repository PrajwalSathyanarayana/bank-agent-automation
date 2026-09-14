"""The save-time scan over the whole artifact, after validation and before signing.

Every text field is read, and what the scan may do with it depends on who wrote it:
- the model's words (typed values, checked text, descriptions) may be turned into
  placeholders, or stop the save;
- what our code recorded (locators, page checks, a dropdown's hidden value) was cleared
  of run data when it was recorded and proven on the live page. The scan looks again
  and stops the save if anything got through, but never rewrites it: a rewritten
  locator was never proven;
- the engineer's contract (goal template, URLs, definitions) is checked for secrets
  only; the rest of it is a deliberate human decision;
- identifiers and fixed vocabulary carry no data and are not read.
A text field the scan doesn't know stops the save, so a field added to the schema
later can't pass unread.
"""
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

from src.discovery.locators import Candidate, RunValues, number_pattern, scan, without_placeholders
from src.locating.checks import phrase_pattern
from src.safety.redactor import sensitive_patterns_in
from src.types.artifact_schema import Artifact
from src.types.placeholders import CREDENTIAL_PREFIX, iter_placeholders
from src.types.result_schema import ErrorDetail
from src.types.step_schema import ActionType, CheckpointType, LocatorType, Step

# A field's place in the artifact's JSON, e.g. ("steps", 3, "description").
Path = tuple[Union[str, int], ...]


class FieldKind(str, Enum):
    # The model's words.
    TYPED = "typed value"  # what replay types or selects
    CHECKED = "checked text"  # what replay compares against the page
    DESCRIPTION = "description"  # the model's reasons; only read by people
    # Written by our code, already cleared of run data: a second look only.
    RECORDED = "recorded"
    # Written by the engineer: checked for secrets only.
    CONTRACT = "contract"


class UnclassifiedField(RuntimeError):
    """A text field the scan has no treatment for: a code change is needed, not a retry."""


@dataclass(frozen=True)
class ScannedField:
    """One text field of the artifact: where it is, how the scan treats it, what it says."""

    path: Path
    kind: FieldKind
    text: str
    # The step's sequence_index, for messages; None outside the steps.
    step_index: Optional[int] = None

    @property
    def pattern(self) -> Path:
        return _pattern(self.path)


# Identifiers, names, fixed vocabulary, the version and timestamps. List positions are "*".
SKIPPED_FIELDS: frozenset[Path] = frozenset({
    ("metadata", "artifact_id"),
    ("metadata", "version"),
    ("metadata", "integrity_hash"),
    ("metadata", "created_timestamp"),
    ("metadata", "last_updated_timestamp"),
    ("input_parameters", "*", "key"),
    ("input_parameters", "*", "type"),
    ("credentials", "*", "key"),
    ("credentials", "*", "kind"),
    ("output_definitions", "*", "key"),
    ("output_definitions", "*", "type"),
    ("output_definitions", "*", "currency"),
    ("known_outcomes", "*", "code"),
    ("known_outcomes", "*", "signal"),
    ("known_outcomes", "*", "input_key"),
    ("steps", "*", "step_id"),
    ("steps", "*", "action"),
    ("steps", "*", "safety_tier"),
    ("steps", "*", "output_key"),
    ("steps", "*", "locators", "*", "type"),
    ("steps", "*", "checkpoints", "*", "checkpoint_id"),
    ("steps", "*", "checkpoints", "*", "type"),
    ("steps", "*", "checkpoints", "*", "target_locator", "type"),
    ("global_assertions", "*", "assertion_id"),
    ("global_assertions", "*", "type"),
})

_CONTRACT_METADATA = ("capability", "description", "author", "target_url", "tenant_override_url")
# Page checks our code adds after an action; the other checks compare text someone chose.
_RECORDED_CHECKS = {CheckpointType.PAGE_PATH, CheckpointType.PAGE_TITLE, CheckpointType.URL_CONTAINS}
# What a step's input_value is, by action.
_INPUT_VALUE_KINDS = {
    ActionType.TYPE: FieldKind.TYPED,
    ActionType.SELECT: FieldKind.TYPED,
    ActionType.ASSERT_TEXT: FieldKind.CHECKED,
}


def artifact_fields(artifact: Artifact) -> list[ScannedField]:
    """Every text field the scan reads, each with its treatment.

    The contract comes first, then each step in order, then the final checks.

    Raises UnclassifiedField for a text field that is neither read here nor listed in
    SKIPPED_FIELDS. Empty fields carry no data and are left out.
    """
    fields: list[ScannedField] = []

    def add(path: Path, kind: FieldKind, text: Optional[str], step_index: Optional[int] = None) -> None:
        if text is not None:
            fields.append(ScannedField(path, kind, text, step_index))

    for name in _CONTRACT_METADATA:
        add(("metadata", name), FieldKind.CONTRACT, getattr(artifact.metadata, name))
    for group in ("input_parameters", "credentials", "output_definitions"):
        for position, definition in enumerate(getattr(artifact, group)):
            add((group, position, "description"), FieldKind.CONTRACT, definition.description)
    for position, parameter in enumerate(artifact.input_parameters):
        add(("input_parameters", position, "example_value"), FieldKind.CONTRACT, parameter.example_value)
    for position, outcome in enumerate(artifact.known_outcomes):
        add(("known_outcomes", position, "description"), FieldKind.CONTRACT, outcome.description)
        add(("known_outcomes", position, "text"), FieldKind.CONTRACT, outcome.text)

    for position, step in enumerate(artifact.steps):
        at: Path = ("steps", position)
        index = step.sequence_index
        add((*at, "description"), FieldKind.DESCRIPTION, step.description, index)
        if step.input_value is not None:
            add((*at, "input_value"), _input_value_kind(step), step.input_value, index)
        add((*at, "option_value"), FieldKind.RECORDED, step.option_value, index)
        for number, locator in enumerate(step.locators):
            add((*at, "locators", number, "value"), FieldKind.RECORDED, locator.value, index)
        for number, checkpoint in enumerate(step.checkpoints):
            check_at: Path = (*at, "checkpoints", number)
            if checkpoint.target_locator is not None:
                add((*check_at, "target_locator", "value"), FieldKind.RECORDED,
                    checkpoint.target_locator.value, index)
            kind = FieldKind.RECORDED if checkpoint.type in _RECORDED_CHECKS else FieldKind.CHECKED
            add((*check_at, "expected_value"), kind, checkpoint.expected_value, index)
    for position, assertion in enumerate(artifact.global_assertions):
        add(("global_assertions", position, "value"), FieldKind.CHECKED, assertion.value)

    _check_every_field_is_known(artifact, fields)
    return fields


def _input_value_kind(step: Step) -> FieldKind:
    kind = _INPUT_VALUE_KINDS.get(step.action)
    if kind is None:
        # The recorder writes an input_value only on the actions above.
        raise UnclassifiedField(
            f"step {step.sequence_index}: an input_value on a {step.action.value} step has no treatment"
        )
    return kind


def _check_every_field_is_known(artifact: Artifact, fields: list[ScannedField]) -> None:
    known = {field.pattern for field in fields} | SKIPPED_FIELDS
    for path in _text_paths(artifact.model_dump(mode="json")):
        if _pattern(path) not in known:
            where = "/".join(str(part) for part in _pattern(path))
            raise UnclassifiedField(f"the scan has no treatment for the field {where}; give it one before saving")


def _text_paths(node: Any, path: Path = ()) -> Iterator[Path]:
    # Every place in the JSON that holds text; numbers, booleans and nulls carry none.
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _text_paths(value, (*path, key))
    elif isinstance(node, list):
        for position, value in enumerate(node):
            yield from _text_paths(value, (*path, position))
    elif isinstance(node, str):
        yield path


def _pattern(path: Path) -> Path:
    return tuple("*" if isinstance(part, int) else part for part in path)


# --- this run's values in the model's words, turned into placeholders ---

# What the teller username becomes in a description: descriptions are only read, and
# no placeholder for it is allowed there.
USERNAME_WORDING = "(teller username)"


@dataclass(frozen=True)
class ScanInputs:
    """What the scan needs from the discovery run, besides the artifact."""

    run: RunValues
    # The config credential the username is typed from: {credential:bank_username}.
    username_key: str
    # What this run read with extract_text, by output name, for flagging assertions.
    extracted: Mapping[str, str]


@dataclass(frozen=True)
class Conversion:
    """One piece of this run's data replaced in a field: what replaced it, never the value."""

    path: Path
    kind: FieldKind
    step_index: Optional[int]
    # The placeholder written ({member_id}) or the username's neutral wording.
    replaced_with: str


@dataclass(frozen=True)
class Ambiguity:
    """A value equal to two or more of this run's values, left as it was for the scan to stop on."""

    path: Path
    kind: FieldKind
    step_index: Optional[int]
    # The placeholders it could stand for.
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class Converted:
    artifact: Artifact
    conversions: list[Conversion]
    # Steps whose dropdown choice became an input, so their hidden option value was removed.
    option_values_dropped: list[int]
    ambiguities: list[Ambiguity]


def convert(artifact: Artifact, inputs: ScanInputs) -> Converted:
    """The artifact with this run's values in the model's words replaced by placeholders.

    A typed value is converted only when the whole of it is one input's value (any case
    for text and the username, any common form for a number): replay types exactly what
    is stored. In checked text and descriptions each input's value is replaced wherever
    it stands as whole words, and a description's username becomes neutral wording. A
    value equal to two or more of this run's values is left alone and reported.

    What our code recorded and the engineer's contract are never rewritten, with one
    exception: a dropdown whose choice became an input loses its hidden option value,
    which the schema forbids there and which would select a frozen choice on fallback.
    Returns a new artifact, validated again, so every placeholder written must exist.
    """
    data = artifact.model_dump(mode="json")
    conversions: list[Conversion] = []
    ambiguities: list[Ambiguity] = []
    dropped: list[int] = []

    for scanned in artifact_fields(artifact):
        if scanned.kind == FieldKind.TYPED:
            step = artifact.steps[scanned.path[1]]
            candidates = _typed_candidates(scanned.text, inputs, username_allowed=step.action == ActionType.TYPE)
            if len(candidates) == 1:
                _set(data, scanned.path, candidates[0])
                conversions.append(Conversion(scanned.path, scanned.kind, scanned.step_index, candidates[0]))
                if step.option_value is not None:
                    _set(data, (*scanned.path[:-1], "option_value"), None)
                    dropped.append(step.sequence_index)
            elif candidates:
                ambiguities.append(Ambiguity(scanned.path, scanned.kind, scanned.step_index, tuple(sorted(candidates))))
        elif scanned.kind in (FieldKind.CHECKED, FieldKind.DESCRIPTION):
            new_text, written, ambiguous = _replace_words(
                scanned.text, inputs.run, username=scanned.kind == FieldKind.DESCRIPTION
            )
            if written:
                _set(data, scanned.path, new_text)
                conversions.extend(
                    Conversion(scanned.path, scanned.kind, scanned.step_index, replacement) for replacement in written
                )
            ambiguities.extend(
                Ambiguity(scanned.path, scanned.kind, scanned.step_index, candidates) for candidates in ambiguous
            )

    return Converted(Artifact.model_validate(data), conversions, dropped, ambiguities)


def _typed_candidates(value: str, inputs: ScanInputs, *, username_allowed: bool) -> list[str]:
    """The placeholders this whole typed value could stand for.

    A value already holding a placeholder was written as intended; anything mixed is for
    the scan's later checks, not a conversion. The username's placeholder is only
    allowed in typed text, never as a dropdown choice.
    """
    if _has_placeholder(value):
        return []
    run = inputs.run
    folded = value.casefold()
    candidates = [f"{{{name}}}" for name, text in run.text_inputs.items() if text and folded == text.casefold()]
    candidates += [
        f"{{{name}}}" for name, number in run.number_inputs.items() if number_pattern(number).fullmatch(value)
    ]
    if username_allowed and run.username and folded == run.username.casefold():
        candidates.append(f"{{{CREDENTIAL_PREFIX}:{inputs.username_key}}}")
    return candidates


def _replace_words(text: str, run: RunValues, *, username: bool) -> tuple[str, list[str], list[tuple[str, ...]]]:
    """Each of this run's values standing as whole words in the text, replaced.

    Returns the new text, the replacements written (one per place, in text order) and,
    for each place equal to two or more values, what it could stand for (left as it was).
    Where matches overlap the longest wins, so "Sunbelt Electric Co" isn't split by a
    shorter value inside it. Text inside an existing placeholder is never touched.
    """
    found: dict[tuple[int, int], set[str]] = {}

    def collect(pattern: Optional[re.Pattern[str]], replacement: str) -> None:
        for match in pattern.finditer(text) if pattern else ():
            found.setdefault((match.start(), match.end()), set()).add(replacement)

    for name, value in run.text_inputs.items():
        collect(phrase_pattern(value), f"{{{name}}}")
    for name, number in run.number_inputs.items():
        collect(number_pattern(number), f"{{{name}}}")
    if username and run.username:
        collect(phrase_pattern(run.username), USERNAME_WORDING)

    taken = [(start, end) for _, start, end in iter_placeholders(text)]
    chosen: list[tuple[int, int, str]] = []
    ambiguous: list[tuple[str, ...]] = []
    # Longest first, then leftmost.
    for (start, end), replacements in sorted(found.items(), key=lambda item: (item[0][0] - item[0][1], item[0][0])):
        if any(start < taken_end and taken_start < end for taken_start, taken_end in taken):
            continue
        taken.append((start, end))
        if len(replacements) > 1:
            ambiguous.append(tuple(sorted(replacements)))
        else:
            chosen.append((start, end, next(iter(replacements))))

    chosen.sort()
    new_text = text
    for start, end, replacement in reversed(chosen):
        new_text = new_text[:start] + replacement + new_text[end:]
    return new_text, [replacement for _, _, replacement in chosen], ambiguous


def _set(data: Any, path: Path, value: Any) -> None:
    for part in path[:-1]:
        data = data[part]
    data[path[-1]] = value


# --- what stops the save ---

class AbortCode(str, Enum):
    SECRET_LITERAL = "SECRET_LITERAL"
    RECORDED_FIELD_LITERAL = "RECORDED_FIELD_LITERAL"
    SENSITIVE_LITERAL = "SENSITIVE_LITERAL"
    AMBIGUOUS_LITERAL = "AMBIGUOUS_LITERAL"
    EMBEDDED_INPUT_LITERAL = "EMBEDDED_INPUT_LITERAL"
    USERNAME_LITERAL = "USERNAME_LITERAL"


# Lower is more severe. A secret is the security hole; a recorder bug means our own code
# is wrong and may explain the other findings; apparent member data comes next; the
# model's replay-breaking mistakes share the last rank, so artifact order decides there.
_SEVERITY = {
    AbortCode.SECRET_LITERAL: 0,
    AbortCode.RECORDED_FIELD_LITERAL: 1,
    AbortCode.SENSITIVE_LITERAL: 2,
    AbortCode.AMBIGUOUS_LITERAL: 3,
    AbortCode.EMBEDDED_INPUT_LITERAL: 3,
    AbortCode.USERNAME_LITERAL: 3,
}

# The structure our locator generator writes: position indexes in CSS and in XPath. A
# number there is a place on the page, never an amount.
_POSITION_SYNTAX = re.compile(r":nth-of-type\(\d+\)|\[\d+\]")

_DEFINITION_WORDS = {"input_parameters": "input", "credentials": "credential", "output_definitions": "output",
                     "known_outcomes": "known outcome"}
# Checks that hold text; a next-step check stores none, so it never reaches a message.
_CHECK_WORDS = {
    CheckpointType.PAGE_PATH: "page path check",
    CheckpointType.PAGE_TITLE: "page title check",
    CheckpointType.URL_CONTAINS: "address check",
    CheckpointType.TEXT_MATCH: "text check",
    CheckpointType.VALUE_EQUALS: "value check",
    CheckpointType.ELEMENT_VISIBLE: "visibility check",
}


@dataclass(frozen=True)
class Finding:
    """A reason the artifact can't be saved. The message never includes the value."""

    code: AbortCode
    path: Path
    kind: FieldKind
    step_index: Optional[int]
    message: str


def find_problems(converted: Converted, inputs: ScanInputs) -> list[Finding]:
    """Everything in the converted artifact that stops the save, in artifact order.

    A secret's value stops it wherever it is. What our code recorded gets a second look
    for this run's inputs and the username. The model's words are checked for what the
    conversion couldn't fix: a value equal to two of this run's values, an input inside
    other typed text, the username in checked text, and anything that looks like an
    email address, phone number, SSN or card number. The engineer's contract is checked
    for secrets only.
    """
    artifact = converted.artifact
    ambiguous: dict[Path, list[Ambiguity]] = {}
    for ambiguity in converted.ambiguities:
        ambiguous.setdefault(ambiguity.path, []).append(ambiguity)

    findings: list[Finding] = []
    for scanned in artifact_fields(artifact):
        where = _where(scanned, artifact)

        def found(code: AbortCode, message: str) -> None:
            findings.append(Finding(code, scanned.path, scanned.kind, scanned.step_index, f"{where}: {message}"))

        secret = _secret_in(scanned.text, inputs.run)
        if secret:
            found(AbortCode.SECRET_LITERAL,
                  f"carries the value of the secret {secret}; a secret is never stored in an artifact")
            continue
        if scanned.kind == FieldKind.CONTRACT:
            continue
        if scanned.kind == FieldKind.RECORDED:
            reason = _run_data_in(_recorded_data(scanned), inputs.run)
            if reason:
                found(AbortCode.RECORDED_FIELD_LITERAL,
                      f"{reason}; the recording should have removed it — a recorder bug, not the model's")
            continue

        for ambiguity in ambiguous.get(scanned.path, []):
            found(AbortCode.AMBIGUOUS_LITERAL,
                  f"a value could stand for any of {', '.join(ambiguity.candidates)}; which one can't be known, "
                  "so write the placeholder you meant")
        if scanned.kind == FieldKind.TYPED and scanned.path not in ambiguous:
            reason = _run_data_in(without_placeholders(scanned.text), inputs.run)
            if reason:
                found(AbortCode.EMBEDDED_INPUT_LITERAL,
                      f"{reason} inside other text; type the input alone, as its placeholder")
        if scanned.kind == FieldKind.CHECKED and _has_username(scanned.text, inputs.run):
            found(AbortCode.USERNAME_LITERAL, "contains the teller username; check page state, not the teller's name")
        for pattern_name in sensitive_patterns_in(scanned.text):
            found(AbortCode.SENSITIVE_LITERAL,
                  f"matches the {pattern_name} pattern and no input; remove it or declare it as an input")
    return findings


def abort_error(findings: list[Finding]) -> ErrorDetail:
    """The result's error: the most severe finding (the earliest on a tie) and a count of the rest.

    Every finding is in the run log; the result names the one that matters most.
    """
    if not findings:
        raise ValueError("there are no findings, so nothing stops the save")
    _, worst = min(enumerate(findings), key=lambda item: (_SEVERITY[item[1].code], item[0]))
    message = worst.message
    others = len(findings) - 1
    if others:
        message += f"; {others} more finding{'s' if others > 1 else ''} in the run log"
    return ErrorDetail(code=worst.code.value, message=message)


def _secret_in(text: str, run: RunValues) -> Optional[str]:
    for name, secret in run.secrets.items():
        value = secret.get_secret_value()
        if value and value in text:
            return name
    return None


def _run_data_in(text: str, run: RunValues) -> Optional[str]:
    """Why the text carries this run's data, by the recorder's own scan; None if it doesn't.

    Text inputs anywhere in any case, numbers in their common forms, the username as a
    whole word. The wording names the input, never its value.
    """
    return scan(Candidate("saved field", LocatorType.TEXT_CONTENT, text, data=(text,)), run).reason


def _recorded_data(field: ScannedField) -> str:
    # Placeholders were written on purpose. In a locator, position indexes are structure,
    # which the recorder never read for data either.
    text = without_placeholders(field.text)
    if "locators" in field.path or "target_locator" in field.path:
        text = _POSITION_SYNTAX.sub("", text)
    return text


def _has_username(text: str, run: RunValues) -> bool:
    pattern = phrase_pattern(run.username) if run.username else None
    return pattern is not None and pattern.search(text) is not None


def _where(field: ScannedField, artifact: Artifact) -> str:
    """Where the field is, in words for a message."""
    group, *rest = field.path
    if group == "steps":
        return f"step {field.step_index} {_step_part(field, artifact.steps[int(rest[0])])}"
    if group == "metadata":
        return "the goal template" if rest == ["description"] else f"the {str(rest[0]).replace('_', ' ')}"
    if group == "global_assertions":
        return f"final check {rest[0]}"
    position, name = rest
    entry = getattr(artifact, str(group))[position]
    # A known outcome is named by its code; every other definition by its key.
    label = entry.code if group == "known_outcomes" else entry.key
    return f"{_DEFINITION_WORDS[str(group)]} {label} {str(name).replace('_', ' ')}"


def _step_part(field: ScannedField, step: Step) -> str:
    # Specific enough to find the field: which locator, which kind of check.
    path = field.path
    if "checkpoints" in path:
        words = _CHECK_WORDS[step.checkpoints[int(path[3])].type]
        return f"{words} locator" if "target_locator" in path else words
    if "locators" in path:
        return f"locator {int(path[3]) + 1}"
    if field.path[-1] == "input_value":
        return "typed value" if field.kind == FieldKind.TYPED else "assertion"
    if field.path[-1] == "option_value":
        return "hidden option value"
    return "description"


# --- assertions that may hold only for this member, and the report ---

@dataclass(frozen=True)
class FlaggedAssertion:
    """Checked text holding a value this run read or typed: kept, but listed for review."""

    path: Path
    step_index: Optional[int]
    # Which value, never the value: "contains the value read into checking_balance".
    reason: str


def flag_assertions(artifact: Artifact, inputs: ScanInputs) -> list[FlaggedAssertion]:
    """Checked text containing, as whole words, a value this run read or typed as a literal.

    Such a check may hold only for this member (a balance, an account chosen by name). It
    is kept, since a page label can equal a value by chance, and listed for a person to
    review. Inputs aren't looked for here: their values already became placeholders.
    """
    fields = artifact_fields(artifact)
    handled = [(f"the value read into {name}", value) for name, value in inputs.extracted.items()]
    handled += [
        (f"the value typed at step {field.step_index}", field.text)
        for field in fields
        if field.kind == FieldKind.TYPED and not _has_placeholder(field.text)
    ]
    flagged: list[FlaggedAssertion] = []
    for field in fields:
        if field.kind != FieldKind.CHECKED:
            continue
        for reason, value in handled:
            pattern = phrase_pattern(value)
            if pattern is not None and pattern.search(field.text):
                flagged.append(FlaggedAssertion(field.path, field.step_index, f"contains {reason}"))
    return flagged


@dataclass(frozen=True)
class BackstopReport:
    """What the scan did and found: the run log's one line for this save."""

    # "passed", or the code the result carries.
    outcome: str
    conversions: list[Conversion]
    option_values_dropped: list[int]
    # Typed values left as literals and checked text as stored, for review. A field with a
    # finding is left out of both, so a value that stopped the save is never logged.
    literals_kept: list[ScannedField]
    assertions_recorded: list[ScannedField]
    flagged_assertions: list[FlaggedAssertion]
    findings: list[Finding]

    def log_fields(self) -> dict[str, Any]:
        """The report as RunLogger.backstop_scan's keyword arguments."""
        return {
            "outcome": self.outcome,
            "conversions": [
                {"step": c.step_index, "field": _path_text(c.path), "replaced_with": c.replaced_with}
                for c in self.conversions
            ],
            "option_values_dropped": list(self.option_values_dropped),
            "literals_kept": [
                {"step": f.step_index, "field": _path_text(f.path), "value": f.text} for f in self.literals_kept
            ],
            "assertions_recorded": [
                {"step": f.step_index, "field": _path_text(f.path), "text": f.text} for f in self.assertions_recorded
            ],
            "flagged_assertions": [
                {"step": f.step_index, "field": _path_text(f.path), "reason": f.reason}
                for f in self.flagged_assertions
            ],
            "findings": [
                {"code": f.code.value, "step": f.step_index, "field": _path_text(f.path), "message": f.message}
                for f in self.findings
            ],
        }


@dataclass(frozen=True)
class ScanResult:
    """The scan's verdict: the artifact to sign, or the error that stops the save; always a report."""

    artifact: Optional[Artifact]
    error: Optional[ErrorDetail]
    report: BackstopReport


def scan_artifact(artifact: Artifact, inputs: ScanInputs) -> ScanResult:
    """The whole save-time scan over a validated, unsigned artifact.

    Converts this run's values, looks for anything that stops the save, flags assertions
    that may hold only for this member, and reports all of it. On success the returned
    artifact is the one to sign; on any finding there is none, and the error is the
    result's HARD_ABORT reason.
    """
    converted = convert(artifact, inputs)
    findings = find_problems(converted, inputs)
    error = abort_error(findings) if findings else None
    with_findings = {finding.path for finding in findings}
    fields = [field for field in artifact_fields(converted.artifact) if field.path not in with_findings]
    report = BackstopReport(
        outcome=error.code if error else "passed",
        conversions=converted.conversions,
        option_values_dropped=converted.option_values_dropped,
        literals_kept=[f for f in fields if f.kind == FieldKind.TYPED and not _has_placeholder(f.text)],
        assertions_recorded=[f for f in fields if f.kind == FieldKind.CHECKED],
        flagged_assertions=flag_assertions(converted.artifact, inputs),
        findings=findings,
    )
    return ScanResult(None if error else converted.artifact, error, report)


def _has_placeholder(text: str) -> bool:
    return next(iter_placeholders(text), None) is not None


def _path_text(path: Path) -> str:
    return "/".join(str(part) for part in path)
