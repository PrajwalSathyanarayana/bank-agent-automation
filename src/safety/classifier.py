import re
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

from src.types.step_schema import SafetyTier, Step

TIER_RANK = {SafetyTier.SAFE: 0, SafetyTier.RISKY: 1, SafetyTier.IRREVERSIBLE: 2}


class SafetyEscalation(Exception):
    """Raised when replay-time re-classification computes a higher tier
    than the artifact declared. An artifact is never trusted to
    self-report a lower risk than independent re-classification finds.
    """


@dataclass(frozen=True)
class ClassificationRule:
    url_pattern: str
    text_pattern: str | None  # matched against description/locator text; None = matches any step at this URL
    tier: SafetyTier


RULES: list[ClassificationRule] = [
    ClassificationRule(r"^/billpay/confirm$", r"confirm payment", SafetyTier.IRREVERSIBLE),
    ClassificationRule(r"^/billpay$", None, SafetyTier.RISKY),
    ClassificationRule(r"^/member/[^/]+/edit$", None, SafetyTier.RISKY),
]


def _step_text_signals(step: Step, element_wording: Sequence[str]) -> list[str]:
    # Every signal can only raise the tier, never lower it. The element's own wording
    # comes first: safety follows what the element is, not which locators survived or
    # how the model described it. Locator values of every type count, so a button's
    # [value="Confirm Payment"] selector is a signal too.
    signals = [*element_wording, step.description, *(locator.value for locator in step.locators)]
    return [signal.lower() for signal in signals if signal]


def classify(step: Step, current_url: str, *, element_wording: Sequence[str]) -> SafetyTier:
    """The step's safety tier on the page it acts on.

    element_wording is what the target element itself says (its text, a button's value,
    aria-label, title, alt), read from the live page; empty for a step with no element.
    It is required so no caller can quietly leave out the strongest signal.
    """
    path = urlparse(current_url).path
    text_signals = _step_text_signals(step, element_wording)

    for rule in RULES:
        if not re.match(rule.url_pattern, path):
            continue
        if rule.text_pattern is None:
            return rule.tier
        if any(re.search(rule.text_pattern, signal) for signal in text_signals):
            return rule.tier

    return SafetyTier.SAFE


def verify_tier(step: Step, current_url: str, *, element_wording: Sequence[str]) -> SafetyTier:
    """Re-classify at replay time, with the wording of the element replay actually found.
    Raises SafetyEscalation if the fresh classification outranks the artifact's declared
    safety_tier.
    """
    recomputed = classify(step, current_url, element_wording=element_wording)
    declared = step.safety_tier
    if TIER_RANK[recomputed] > TIER_RANK[declared]:
        raise SafetyEscalation(
            f"Step {step.step_id} declared safety_tier={declared.value} but "
            f"replay-time re-classification computed {recomputed.value}. "
            f"Refusing to trust the lower declared tier."
        )
    return recomputed
