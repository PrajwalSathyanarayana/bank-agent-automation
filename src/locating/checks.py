"""Reading the page the same way in discovery and replay."""
import re
from typing import Optional

from playwright.async_api import ElementHandle, Page

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


async def shows_phrase(element: ElementHandle, phrase: str) -> bool:
    """Whether the element is visible and its visible text shows the phrase.

    The assertion rule for both modes: discovery uses it to find and confirm an
    assertion, replay to check the element its stored locators found. A button input's
    text is its value.
    """
    shown = await element.evaluate(_SHOWN_TEXT)
    return shown["visible"] and phrase_matches(shown["text"], phrase)


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
