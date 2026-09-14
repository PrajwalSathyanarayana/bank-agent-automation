"""The payment check: before an irreversible step, what the screen shows must be exactly
what was requested and within the bank's auto-pay limit. Otherwise nothing is clicked and
a person decides.

Pure: the caller reads each declared label's value off the screen, with the same rule
that reads a balance, and hands the readings in. Shared by replay, which clicks only when
this passes, and by a sandbox discovery, which runs it before its own irreversible click.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Union

from src.locating.values import UnreadableValue, read_money
from src.types.artifact_schema import CompareAs, ConfirmationCheck

MISMATCH = "AUTHORIZATION_MISMATCH"
OVER_LIMIT = "OVER_AUTO_LIMIT"
NO_CHECKS = "NO_CONFIRMATION_CHECKS"
_NOT_SHOWN = "(not shown once on this screen)"
_SHOWN_MAX = 80


@dataclass(frozen=True)
class CheckProblem:
    """One reason not to click: the label, what the request says, what the screen says."""

    label: str
    expected: str
    seen: str
    reason: str


@dataclass(frozen=True)
class Authorization:
    """None as the code means the step may run; otherwise the code and every reason."""

    code: Optional[str] = None
    problems: tuple[CheckProblem, ...] = ()

    @property
    def authorized(self) -> bool:
        return self.code is None


def authorize(
    checks: Sequence[ConfirmationCheck],
    readings: Mapping[str, Optional[str]],
    inputs: Mapping[str, Union[str, int, float]],
    limit: Decimal,
    limit_currency: str,
) -> Authorization:
    """Whether the irreversible step may run by itself.

    readings holds, per label, the value shown beside it, or None when the label isn't
    shown exactly once. A mismatch outranks the limit: a wrong payment is never merely
    "too large". No declared checks means no automatic click at all.
    """
    if not checks:
        return Authorization(NO_CHECKS)
    mismatches: list[CheckProblem] = []
    over: list[CheckProblem] = []
    for check in checks:
        seen = readings.get(check.label)
        if check.compare_as == CompareAs.MONEY:
            problem, amount = _money(check, inputs[check.input_key], seen)
        else:
            problem, amount = _text(check, inputs[check.input_key], seen), None
        if problem is not None:
            mismatches.append(problem)
        elif amount is not None:
            beyond = _beyond_limit(check, amount, limit, limit_currency)
            if beyond is not None:
                over.append(beyond)
    if mismatches:
        return Authorization(MISMATCH, tuple(mismatches + over))
    if over:
        return Authorization(OVER_LIMIT, tuple(over))
    return Authorization()


def _text(check: ConfirmationCheck, requested: Union[str, int, float], seen: Optional[str]) -> Optional[CheckProblem]:
    # Word for word, ignoring only extra spaces: a near-match is not a match.
    wanted = _flat(str(requested))
    if seen is None:
        return CheckProblem(check.label, _cap(wanted), _NOT_SHOWN, "the label isn't shown once on this screen")
    shown = _flat(seen)
    if shown != wanted:
        return CheckProblem(check.label, _cap(wanted), _cap(shown), "the screen doesn't show what was requested")
    return None


def _money(
    check: ConfirmationCheck, requested: Union[str, int, float], seen: Optional[str]
) -> tuple[Optional[CheckProblem], Optional[Decimal]]:
    currency = check.currency or ""
    try:
        # The request is read by the same rule as the screen, so both are exact to the cent.
        wanted = read_money(f"{Decimal(str(requested)):f}", currency)
    except (UnreadableValue, ArithmeticError):
        return CheckProblem(check.label, _cap(str(requested)), _cap(_flat(seen or _NOT_SHOWN)),
                            f"the requested amount is not a {currency} amount to the cent"), None
    expected = f"{wanted:f} {currency}"
    if wanted <= 0:
        return CheckProblem(check.label, expected, _cap(_flat(seen or _NOT_SHOWN)),
                            "the requested amount is not a positive amount"), None
    if seen is None:
        return CheckProblem(check.label, expected, _NOT_SHOWN, "the label isn't shown once on this screen"), None
    try:
        amount = read_money(seen, currency)
    except UnreadableValue:
        return CheckProblem(check.label, expected, _cap(_flat(seen)),
                            f"the screen doesn't show a {currency} amount there"), None
    if amount != wanted:
        return CheckProblem(check.label, expected, f"{amount:f} {currency}",
                            "the screen doesn't show what was requested"), None
    return None, amount


def _beyond_limit(check: ConfirmationCheck, amount: Decimal, limit: Decimal, currency: str) -> Optional[CheckProblem]:
    shown = f"{amount:f} {check.currency}"
    if check.currency != currency:
        return CheckProblem(check.label, f"at most {limit:f} {currency}", shown,
                            f"no auto-pay limit is set for {check.currency}")
    if amount > limit:
        return CheckProblem(check.label, f"at most {limit:f} {currency}", shown, "above the bank's auto-pay limit")
    return None


def _flat(text: str) -> str:
    return " ".join(text.split())


def _cap(text: str) -> str:
    return text if len(text) <= _SHOWN_MAX else text[:_SHOWN_MAX - 1] + "…"
