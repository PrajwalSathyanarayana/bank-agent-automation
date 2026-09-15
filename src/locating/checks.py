"""Reading the page the same way in discovery and replay."""
import re
from collections.abc import Mapping
from typing import Optional, Union

from playwright.async_api import ElementHandle, Page
from playwright.async_api import Locator as PageLocator

from src.locating.values import number_pattern
from src.types.placeholders import MissingValue, iter_placeholders

_SHOWN_TEXT = """(element) => {
  // Visible means rendered, not visibility:hidden, and with a size. innerText can't be
  // trusted alone: for an element that isn't rendered it falls back to the hidden text.
  const box = element.getBoundingClientRect();
  const visible = element.checkVisibility()
    && getComputedStyle(element).visibility === "visible"
    && box.width > 0 && box.height > 0;
  const isButton = element.tagName === "INPUT" && ["submit", "button", "reset"].includes(element.type);
  return { visible: visible, text: isButton ? (element.value || "") : (element.innerText || "") };
}"""

_WORDING = """(element) => {
  const words = [element.textContent || ""];
  // A button input's value is its label. Other inputs' values are typed data, and a
  // password's value is never read.
  if (element.tagName === "INPUT" && ["submit", "button", "reset"].includes(element.type)) {
    words.push(element.value || "");
  }
  for (const name of ["aria-label", "title", "alt"]) {
    words.push(element.getAttribute(name) || "");
  }
  return words.map((word) => word.replace(/\\s+/g, " ").trim()).filter((word) => word);
}"""


async def is_password_box(element: ElementHandle) -> bool:
    """Whether the element is a password box: an <input type="password">.

    Read from the live element when typing is about to happen, never from an earlier
    reading, and without touching its value. The browser reports the type in lower case
    whatever the page wrote. A text box made to look masked only by styling is not a
    password box.
    """
    return await element.evaluate('(element) => element.tagName === "INPUT" && element.type === "password"')


async def element_wording(element: ElementHandle) -> list[str]:
    """What the element itself says, for the safety classifier.

    Its text (hidden parts included, which can only make the tier stricter), a button's
    value, and its aria-label, title and alt. Read from the live element, so the tier
    doesn't depend on which locators survived or how the model described the step.
    """
    return await element.evaluate(_WORDING)


def phrase_pattern(phrase: str) -> Optional[re.Pattern[str]]:
    """The phrase as whole words in any case; None for a phrase with no words.

    Runs of whitespace, &nbsp; included, count as one space on both sides. Also the
    save-time scan's rule for finding an input's value in text, so both read words alike.
    """
    words = phrase.split()
    if not words:
        return None
    body = r"\s+".join(re.escape(word) for word in words)
    # Word boundaries only where the phrase starts or ends with a word character, so a
    # phrase like "Amount:" still matches before a space or the end of the text.
    start = r"(?<!\w)" if re.match(r"\w", words[0]) else ""
    end = r"(?!\w)" if re.search(r"\w$", words[-1]) else ""
    return re.compile(start + body + end, re.IGNORECASE)


def phrase_matches(text: str, phrase: str) -> bool:
    """Whether the phrase appears in the text as whole words, ignoring case.

    "Pay" never matches inside "Payment", but "Payment submitted" matches in "Payment
    submitted - Ref 88121": data before or after the phrase doesn't stop it.
    """
    pattern = phrase_pattern(phrase)
    return pattern is not None and pattern.search(text) is not None


def text_pattern(
    text: str, text_values: Mapping[str, str], number_values: Mapping[str, float]
) -> Optional[re.Pattern[str]]:
    """Checked text with its placeholders filled, matched by the phrase rule above.

    A text placeholder stands for this run's value word for word; a number placeholder
    for any common form of the number ("Amount: ${amount}" with 50 matches "Amount:
    $50.00"). Literal {{braces}} are single braces on the page. None for text with no
    words; MissingValue for a placeholder this run has no value for.
    """
    text = text.strip()
    if not text.split():
        return None
    pieces: list[str] = []
    shown: list[str] = []
    position = 0
    for name, start, end in iter_placeholders(text):
        pieces.append(_literal(text[position:start]))
        shown.append(text[position:start])
        if name in number_values:
            pieces.append(number_pattern(number_values[name]).pattern)
            shown.append("0")
        elif name in text_values:
            pieces.append(_literal(text_values[name]))
            shown.append(text_values[name])
        else:
            raise MissingValue(f"no value for {{{name}}}")
        position = end
    pieces.append(_literal(text[position:]))
    shown.append(text[position:])
    # Word boundaries only where the filled text starts or ends with a word character.
    filled = "".join(shown)
    start = r"(?<!\w)" if re.match(r"\w", filled) else ""
    end = r"(?!\w)" if re.search(r"\w$", filled) else ""
    return re.compile(start + "".join(pieces) + end, re.IGNORECASE)


def _literal(part: str) -> str:
    # The page's own text: doubled braces are single there, and any run of spaces is one.
    part = part.replace("{{", "{").replace("}}", "}")
    return "".join(r"\s+" if piece.isspace() else re.escape(piece) for piece in re.split(r"(\s+)", part) if piece)


async def shows_pattern(element: Union[ElementHandle, PageLocator], pattern: re.Pattern[str]) -> bool:
    """Whether the element is visible and its visible text matches the pattern."""
    shown = await element.evaluate(_SHOWN_TEXT)
    return shown["visible"] and pattern.search(shown["text"]) is not None


async def visible_text(element: Union[ElementHandle, PageLocator]) -> str:
    """The element's visible text (a button input's value); empty when it isn't visible."""
    shown = await element.evaluate(_SHOWN_TEXT)
    return " ".join(shown["text"].split()) if shown["visible"] else ""


async def shows_phrase(element: ElementHandle, phrase: str) -> bool:
    """Whether the element is visible and its visible text shows the phrase.

    The assertion rule for both modes: discovery uses it to find and confirm an
    assertion, replay to check the element its stored locators found. A button input's
    text is its value.
    """
    pattern = phrase_pattern(phrase)
    return pattern is not None and await shows_pattern(element, pattern)


async def find_phrase(page: Page, phrase: str) -> list[ElementHandle]:
    """The innermost visible elements showing the phrase.

    Playwright's loose text lookup (any case, a substring, innermost elements) finds
    every possible element; the stricter rule above keeps only real matches. Handles
    that don't match are released here; the caller releases the ones returned.
    """
    candidates = await page.get_by_text(" ".join(phrase.split())).element_handles()
    found = []
    for handle in candidates:
        if await shows_phrase(handle, phrase):
            found.append(handle)
        else:
            await handle.dispose()
    return found


# The cell after the one holding the label, in the same row: where legacy pages show a value.
VALUE_CELL = """(element) => {
  const cell = element.closest("td, th");
  if (!cell) return null;
  let next = cell.nextElementSibling;
  while (next && !["TD", "TH"].includes(next.tagName)) next = next.nextElementSibling;
  return next;
}"""


async def value_beside(page: Page, label: str) -> Optional[str]:
    """The text in the cell right after the label, read the way discovery reads a balance.

    None when the label isn't shown by exactly one visible element, no cell follows it, or
    that cell is empty: a value nobody can be sure of is no value.
    """
    matches = await find_phrase(page, label)
    try:
        if len(matches) != 1:
            return None
        cell = (await matches[0].evaluate_handle(VALUE_CELL)).as_element()
        if cell is None:
            return None
        try:
            text = " ".join((await cell.inner_text()).split())
        finally:
            await cell.dispose()
        return text or None
    finally:
        for handle in matches:
            await handle.dispose()
