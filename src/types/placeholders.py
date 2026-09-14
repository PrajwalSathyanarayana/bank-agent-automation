import re
from collections.abc import Iterator, Mapping

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


class MissingValue(ValueError):
    """A placeholder in the text has no value; the message names it, never a value."""


def fill_text(text: str, values: Mapping[str, str]) -> str:
    """The text with each placeholder replaced by its value and doubled braces made single.

    Plain text, for typing or choosing an option; locators are filled by the resolver,
    which also escapes values for their selector language.
    """
    pieces = []
    position = 0
    for name, start, end in iter_placeholders(text):
        if name not in values:
            raise MissingValue(f"no value for {{{name}}}")
        pieces.append(_single_braces(text[position:start]))
        pieces.append(values[name])
        position = end
    pieces.append(_single_braces(text[position:]))
    return "".join(pieces)


def _single_braces(text: str) -> str:
    return text.replace("{{", "{").replace("}}", "}")
