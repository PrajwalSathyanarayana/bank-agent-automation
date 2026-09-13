"""What a stored locator means on a page.

Discovery proves every locator through resolve() before saving it, and replay finds
elements through the same function, so the two can never read a locator differently.
"""
from collections.abc import Mapping

from playwright.async_api import Locator as PageLocator
from playwright.async_api import Page

from src.types.placeholders import iter_placeholders
from src.types.step_schema import Locator, LocatorType


class UnfillableLocator(ValueError):
    """Raised when a locator's placeholders cannot be filled safely with the given values."""


def resolve(page: Page, locator: Locator, values: Mapping[str, str]) -> PageLocator:
    """The Playwright locator a stored locator stands for, with its placeholders filled.

    - css and xpath: the value as a selector of that kind. The engine is named explicitly,
      so Playwright never guesses it from the text (a value starting with "//" would
      otherwise be read as XPath, and "text=..." as its text engine).
    - text_content: Playwright's exact text lookup. Runs of spaces, &nbsp; and line breaks
      count as one space; submit and button inputs match by their value.
    - aria_label: Playwright's exact label lookup, i.e. the element's accessible name from
      aria-label, aria-labelledby, <label for> or a wrapping <label> - not only the
      attribute the type is named after.

    `values` holds this run's inputs, already formatted as text.
    """
    value = fill(locator, values)
    if locator.type == LocatorType.CSS:
        return page.locator(f"css={value}")
    if locator.type == LocatorType.XPATH:
        return page.locator(f"xpath={value}")
    if locator.type == LocatorType.TEXT_CONTENT:
        return page.get_by_text(value, exact=True)
    if locator.type == LocatorType.ARIA_LABEL:
        return page.get_by_label(value, exact=True)
    raise ValueError(f"unknown locator type {locator.type!r}")


def fill(locator: Locator, values: Mapping[str, str]) -> str:
    """The locator's value with each {placeholder} replaced and doubled braces made single.

    In css and xpath values a placeholder sits inside a double-quoted string (the address
    in a[href="/member/{member_id}/accounts"]), so the value is escaped for that string: a
    member ID containing a quote must not end the string early and change what is matched.
    Text and label values reach Playwright as plain strings and need no escaping.
    """
    text = locator.value
    pieces = []
    position = 0
    for name, start, end in iter_placeholders(text):
        pieces.append(_single_braces(text[position:start]))
        if name not in values:
            raise UnfillableLocator(f"no value for {{{name}}} in a {locator.type.value} locator")
        pieces.append(_escaped(locator.type, values[name], text, start))
        position = end
    pieces.append(_single_braces(text[position:]))
    return "".join(pieces)


def _single_braces(literal: str) -> str:
    # Literal braces are stored doubled so they can't be read as placeholders.
    return literal.replace("{{", "{").replace("}}", "}")


def _escaped(kind: LocatorType, value: str, text: str, start: int) -> str:
    if kind == LocatorType.CSS:
        _require_double_quoted(text, start, backslash_escapes=True)
        return css_string(value)
    if kind == LocatorType.XPATH:
        _require_double_quoted(text, start, backslash_escapes=False)
        if '"' in value:
            # XPath 1.0 strings have no escape character, so this value can't be written
            # here. Failing lets replay move on to the next locator instead of matching
            # something else.
            raise UnfillableLocator("an xpath locator cannot hold a value containing a double quote")
        return value
    return value


def _require_double_quoted(text: str, start: int, backslash_escapes: bool) -> None:
    # The escaping above is only correct inside a double-quoted string, which is how
    # our generated locators quote addresses; refuse anything else rather than guess.
    inside = False
    index = 0
    while index < start:
        character = text[index]
        if backslash_escapes and character == "\\":
            index += 2
            continue
        if character == '"':
            inside = not inside
        index += 1
    if not inside:
        raise UnfillableLocator("a placeholder in a css or xpath locator must be inside a double-quoted string")


def css_string(value: str) -> str:
    """The value escaped to sit inside a double-quoted CSS string.

    Shared with locators.py, so a literal written at discovery and a placeholder filled
    at replay are escaped by the same rule.
    """
    escaped = []
    for character in value:
        if character in '"\\':
            escaped.append("\\" + character)
        elif ord(character) < 0x20 or ord(character) == 0x7F:
            # Control characters (line breaks included) as hex escapes; the trailing space
            # ends the escape so a following hex digit isn't read as part of it.
            escaped.append(f"\\{ord(character):x} ")
        else:
            escaped.append(character)
    return "".join(escaped)
