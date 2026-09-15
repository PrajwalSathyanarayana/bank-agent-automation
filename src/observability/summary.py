"""A plain-English summary for every result: what was asked, what happened, and whether the
irreversible step (for bill pay: the payment) happened.

Built from fixed wording, never by a model, so it is predictable, testable and never
invents anything. The wording is presentation, not behaviour: it lives here, outside the
signed artifact, so it can be improved without re-signing anything. A capability without
its own wording is summarised from its goal sentence.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

from src.types.artifact_schema import CompareAs, ConfirmationCheck, OutputParamDefinition, OutputType
from src.types.placeholders import MissingValue, fill_text
from src.types.result_schema import BusinessOutcome, ErrorDetail, ExecutionStatus, FailureDetail

_SYMBOLS = {"USD": "$"}


@dataclass(frozen=True)
class Wording:
    """How one capability's results read. Placeholders are its inputs and outputs, as a
    person would read them ({amount} is "$50.00")."""

    action: str
    asked: str
    done: str
    results: str = ""


WORDING = {
    "member_servicing_and_bill_pay": Wording(
        action="payment",
        asked="Pay {amount} to {payee_name} for member {member_id}.",
        done="Paid {amount} to {payee_name} for member {member_id}.",
        results="Checking balance {checking_balance_before} before, {new_checking_balance} after.",
    ),
}

# Each stop code in plain words; the code itself stays in brackets for staff.
REASONS = {
    "NO_ARTIFACT": "this task hasn't been learned yet",
    "UNKNOWN_CAPABILITY": "no task by that name exists",
    "INTEGRITY_CHECK_FAILED": "the saved procedure failed its security check",
    "INPUT_INVALID": "the request was incomplete or invalid",
    "CREDENTIAL_MISSING": "the system's sign-in details aren't configured",
    "LOCATOR_NOT_FOUND": "something on the screen wasn't where it was expected",
    "CHECK_FAILED": "a screen wasn't the one expected",
    "ACTION_FAILED": "an action on the screen didn't work",
    "PAGE_TIMEOUT": "the bank's system didn't respond in time",
    "TIMEOUT": "the task took too long",
    "OUTPUT_UNREADABLE": "a value on the screen couldn't be read",
    "OUTPUT_MISSING": "a value that should have been read wasn't",
    "INTERRUPTION_NOT_CLEARED": "something on the screen blocked the task and couldn't be cleared",
    "ALLOWLIST_VIOLATION": "it would have gone somewhere it isn't allowed to",
    "TYPING_REFUSED": "a value was about to go into the wrong kind of field",
    "UNSUPPORTED_ACTION": "the saved procedure asks for something replay can't do",
    "BROWSER_FAILED": "the browser failed",
    "SANDBOX_NOT_LOCAL": "the test environment isn't on this machine",
    "MAX_STEPS": "it took more steps than allowed",
    "ARTIFACT_INVALID": "the learned procedure couldn't be saved",
    "MODEL_REFUSED": "the model declined to continue",
    "MODEL_UNAVAILABLE": "the model couldn't be reached",
    "MODEL_REQUEST_REJECTED": "the model couldn't take the request",
}
# Families of codes that read alike: the learning agent getting stuck, and the save-time
# scan refusing to store this run's data.
FAMILIES = {"STUCK_": "the learning agent got stuck",
            "_LITERAL": "the learned procedure would have kept data it must not store"}
# Why a person is needed, in plain words.
ESCALATIONS = {
    "OVER_AUTO_LIMIT": "the amount is above the bank's limit for automatic payments",
    "AUTHORIZATION_MISMATCH": "the confirmation screen didn't match the request",
    "NO_CONFIRMATION_CHECKS": "nothing is declared to check before this step",
    "UNDECLARED_RISK": "a step looked riskier than it was recorded as",
    "IRREVERSIBLE_STEP": "the task was learned up to the final confirmation, which a person must make",
    "PERSON_HAD_CONTROL": "a person had control earlier in this run, so the final confirmation is left to a person",
}
# How a handoff that ended the run went, after "A person was needed (why)".
_PERSON_ENDINGS = {
    "stopped": " and stopped the task",
    "timed_out": ", but the time for a person ran out",
    "window_closed": ", and the window was closed",
}


def asked_line(capability: str, goal: str, inputs: Mapping[str, str]) -> str:
    """What was asked, in a sentence ("Pay $50.00 to Sunbelt Electric Co for member 10234.").
    inputs are already as a person reads them; a capability without its own wording is
    described by its goal."""
    wording = WORDING.get(capability)
    return _fill(wording.asked if wording else "", inputs) or (f"Asked: {goal}" if goal else f"Asked: {capability}.")


def reason_for(code: str) -> str:
    """Why a run stopped or needs a person, in plain words, e.g. for the operator's bar."""
    return ESCALATIONS.get(code) or _reason(code)


def summarize(
    capability: str,
    mode: str,
    status: ExecutionStatus,
    *,
    goal: str,
    inputs: Mapping[str, str],
    outputs: Mapping[str, str],
    irreversible_step: Optional[str] = None,
    outcome: Optional[BusinessOutcome] = None,
    failure: Optional[FailureDetail] = None,
    error: Optional[ErrorDetail] = None,
    escalation: Optional[str] = None,
    version: Optional[str] = None,
    person: Optional[str] = None,
) -> str:
    """The result in a sentence or two. inputs and outputs are already as a person reads them.

    person says how a person took part, when one had the run's window: "helped" (handed it
    back and the run went on), "finished", "stopped", "timed_out", "window_closed", or
    "not_saved" (a discovery a person helped whose recording had a gap).
    """
    wording = WORDING.get(capability)
    action = wording.action if wording else "irreversible step"
    values = {**inputs, **outputs}
    asked = asked_line(capability, goal, values)
    done = _fill(wording.done if wording else "", values) or f"Done: {goal}"
    results = _fill(wording.results if wording else "", values) or _listed(outputs)
    reason = escalation or (error.code if error else "")
    parts: list[str] = []
    if status == ExecutionStatus.SUCCESS:
        parts += [done, results]
    elif status == ExecutionStatus.BUSINESS_OUTCOME and outcome is not None:
        parts += [asked, f"Not done: {_lower_first(outcome.description)}."]
    elif status == ExecutionStatus.HUMAN_ESCALATED and person == "finished":
        if irreversible_step in ("completed", None):
            parts += [f"{done.rstrip('.')} (confirmed by a person).", results]
        else:
            # The person's word isn't taken on trust: the page didn't show what should follow.
            parts += [asked, "A person reported it finished, but the receipt wasn't found."]
    elif status == ExecutionStatus.HUMAN_ESCALATED and person in _PERSON_ENDINGS:
        needed = reason_for(reason) if reason else "the system stopped for a review"
        parts += [asked, f"A person was needed ({needed}){_PERSON_ENDINGS[person]}."]
    elif status == ExecutionStatus.HUMAN_ESCALATED:
        parts += [asked, f"A person needs to decide: {ESCALATIONS.get(reason, 'the system stopped for a review')}."]
    else:
        code = error.code if error else "UNKNOWN"
        where = f" at step {failure.step_index} ({failure.step_description.rstrip('.')})" if failure else ""
        parts += [asked, f"Stopped{where}: {_reason(code)} ({code})."]
    if person == "helped":
        parts.append("A person had control during the run.")
    elif person == "not_saved":
        parts.append("Not saved as learned: something a person did couldn't be recorded, so the next request will "
                     "run discovery again.")
    # A completed payment already reads from the "Paid …" sentence.
    if not (irreversible_step == "completed" and (status == ExecutionStatus.SUCCESS or person == "finished")):
        parts.append(_irreversible_sentence(action, irreversible_step))
    if mode == "DISCOVERY" and version:
        parts.append(f"Learned and saved as version {version}.")
    return " ".join(part for part in parts if part)


def readable_values(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
    checks: Sequence[ConfirmationCheck],
    definitions: Sequence[OutputParamDefinition],
) -> tuple[dict[str, str], dict[str, str]]:
    """This run's inputs and read values as a person reads them. An input a money payment
    check compares is money ($50.00); so is a money output ($2,450.32)."""
    money_inputs = {check.input_key: check.currency or "" for check in checks if check.compare_as == CompareAs.MONEY}
    shown_inputs = {}
    for key, value in inputs.items():
        if key in money_inputs:
            shown_inputs[key] = readable_money(value, money_inputs[key])
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            shown_inputs[key] = format(Decimal(str(value)).normalize(), "f")
        else:
            shown_inputs[key] = str(value)
    money_outputs = {d.key: d.currency or "" for d in definitions if d.type == OutputType.MONEY}
    shown_outputs = {key: readable_money(value, money_outputs[key]) if key in money_outputs else str(value)
                     for key, value in outputs.items()}
    return shown_inputs, shown_outputs


def readable_money(value: object, currency: str) -> str:
    """An amount as people write it: $2,450.32, or 50.00 EUR for a currency with no symbol here."""
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return str(value)
    symbol = _SYMBOLS.get(currency)
    return f"{symbol}{amount:,.2f}" if symbol else f"{amount:,.2f} {currency}"


def _reason(code: str) -> str:
    if code in REASONS:
        return REASONS[code]
    for part, reason in FAMILIES.items():
        if code.startswith(part) or code.endswith(part):
            return reason
    return "an internal check stopped it"


def _irreversible_sentence(action: str, state: Optional[str]) -> str:
    if state == "not_reached":
        return f"No {action} was made."
    if state == "unknown":
        return f"The {action} may have gone through: check before trying again."
    if state == "completed":
        return f"The {action} was made."
    return ""


def _fill(template: str, values: Mapping[str, str]) -> str:
    # Empty when the template is missing or names a value this result doesn't have.
    if not template:
        return ""
    try:
        return fill_text(template, values)
    except MissingValue:
        return ""


def _listed(outputs: Mapping[str, str]) -> str:
    return "; ".join(f"{key.replace('_', ' ')}: {value}" for key, value in outputs.items()) + (
        "." if outputs else "")


def _lower_first(text: str) -> str:
    # "No member has this ID" → "no member has this ID", leaving acronyms like "ID" alone.
    if len(text) > 1 and text[0].isupper() and not text[1].isupper():
        return text[0].lower() + text[1:]
    return text
