import re
from typing import Any

REDACTED = "[REDACTED]"

SENSITIVE_KEYS = {
    # credentials / secrets
    "password", "api_key", "anthropic_api_key", "secret_key",
    "mock_bank_secret_key", "session_token", "token", "auth_token",
    # member PII (regulated financial data)
    "email", "phone", "address", "street", "city", "state", "zip",
    "first_name", "last_name",
}

# Regex backstop for sensitive patterns that leak into free-text fields
# (e.g. ErrorDetail.message, RecoveryAttemptLog.details) rather than
# sitting in a clearly-named field. Known limitation: the card-number
# pattern is a rough 13-16-digit heuristic and could false-positive on
# an unrelated long digit run — accepted rather than
# over-engineering PII detection for this scope.
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

# Named, so a finding can say what a value looks like without showing it. Shared with
# the save-time artifact scan, so logs and artifacts agree on what looks sensitive.
SENSITIVE_PATTERNS: dict[str, re.Pattern[str]] = {
    "email address": _EMAIL_RE,
    "phone number": _PHONE_RE,
    "SSN": _SSN_RE,
    "card number": _CARD_RE,
}


def redact_text(text: str) -> str:
    redacted = text
    for pattern in SENSITIVE_PATTERNS.values():
        redacted = pattern.sub(REDACTED, redacted)
    return redacted


def sensitive_patterns_in(text: str) -> list[str]:
    """The names of the sensitive-looking patterns found in the text, never what matched."""
    return [name for name, pattern in SENSITIVE_PATTERNS.items() if pattern.search(text)]


def _is_sensitive_key(key: str) -> bool:
    return key.lower() in SENSITIVE_KEYS


def redact_value(key: str, value: Any) -> Any:
    if _is_sensitive_key(key):
        return REDACTED
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return redact_dict(value)
    if isinstance(value, list):
        return [redact_value(key, item) for item in value]
    return value


def scrub_known_values(value: Any, secrets: list[str]) -> Any:
    """Replaces exact known secret values anywhere, including inside free text."""
    ordered = sorted((s for s in secrets if s), key=len, reverse=True)
    return _scrub(value, ordered)


def _scrub(value: Any, ordered: list[str]) -> Any:
    if isinstance(value, str):
        for secret in ordered:
            value = value.replace(secret, REDACTED)
        return value
    if isinstance(value, dict):
        return {key: _scrub(item, ordered) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item, ordered) for item in value]
    return value


def redact_dict(data: dict) -> dict:
    """Returns a new dict with sensitive fields redacted. Never mutates
    the input. Recurses into nested dicts/lists so a redacted parent
    key (e.g. "address") collapses its whole nested structure to a
    single placeholder rather than leaving sub-fields exposed.
    """
    return {key: redact_value(key, value) for key, value in data.items()}
