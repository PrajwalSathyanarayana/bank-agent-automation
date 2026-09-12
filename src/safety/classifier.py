import re
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


def _step_text_signals(step: Step) -> list[str]:
    signals = [step.description]
    for locator in step.locators:
        if locator.type.value == "text_content":
            signals.append(locator.value)
    return [s.lower() for s in signals]


def classify(step: Step, current_url: str) -> SafetyTier:
    path = urlparse(current_url).path
    text_signals = _step_text_signals(step)

    for rule in RULES:
        if not re.match(rule.url_pattern, path):
            continue
        if rule.text_pattern is None:
            return rule.tier
        if any(re.search(rule.text_pattern, signal) for signal in text_signals):
            return rule.tier

    return SafetyTier.SAFE


def verify_tier(step: Step, current_url: str) -> SafetyTier:
    """Re-classify at replay time. Raises SafetyEscalation if the fresh
    classification outranks the artifact's declared safety_tier.
    """
    recomputed = classify(step, current_url)
    declared = step.safety_tier
    if TIER_RANK[recomputed] > TIER_RANK[declared]:
        raise SafetyEscalation(
            f"Step {step.step_id} declared safety_tier={declared.value} but "
            f"replay-time re-classification computed {recomputed.value}. "
            f"Refusing to trust the lower declared tier."
        )
    return recomputed
