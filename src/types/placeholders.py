import re

# A single-brace placeholder such as {member_id}; doubled braces ({{ }}) are literal text.
_PLACEHOLDER = re.compile(r"(?<!\{)\{([^{}]+)\}(?!\})")


def find_placeholders(text: str) -> list[str]:
    return _PLACEHOLDER.findall(text)
