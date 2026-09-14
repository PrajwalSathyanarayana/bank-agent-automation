"""The pages a capability may visit: path patterns such as "/member/*", and the rule that
matches a page's path against them.

A pattern is "/" or one or more "/segment" parts, where a segment is literal text or "*"
for exactly one non-empty segment: "/member/*" matches "/member/10234" but neither
"/member/10234/edit" nor "/member/". Only the path is compared, and case matters.
"""
import re
from collections.abc import Iterable

_PATTERN = re.compile(r"/|(?:/(?:[^/*?#\s{}]+|\*))+")


def valid_route_pattern(pattern: str) -> bool:
    return _PATTERN.fullmatch(pattern) is not None


def route_allowed(path: str, patterns: Iterable[str]) -> bool:
    """Whether the page's path matches one of the patterns."""
    parts = path.split("/")
    return any(_matches(parts, pattern.split("/")) for pattern in patterns)


def _matches(parts: list[str], pattern_parts: list[str]) -> bool:
    return len(parts) == len(pattern_parts) and all(
        part == wanted or (wanted == "*" and part != "") for part, wanted in zip(parts, pattern_parts)
    )
