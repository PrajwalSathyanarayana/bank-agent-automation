import re
from collections.abc import Iterator

# A single-brace placeholder such as {member_id}; doubled braces ({{ }}) are literal text.
_PLACEHOLDER = re.compile(r"(?<!\{)\{([^{}]+)\}(?!\})")
# The prefix of a credential placeholder, {credential:bank_password}; the part after the
# colon is a key in the artifact's credentials list.
CREDENTIAL_PREFIX = "credential"


def find_placeholders(text: str) -> list[str]:
    return _PLACEHOLDER.findall(text)


def iter_placeholders(text: str) -> Iterator[tuple[str, int, int]]:
    """Yields (name, start, end) for each placeholder, end exclusive."""
    for match in _PLACEHOLDER.finditer(text):
        yield match.group(1), match.start(), match.end()
