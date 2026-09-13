"""What the discovery model sees each step: a screenshot plus a numbered list of elements."""
import json
from dataclasses import dataclass, field
from typing import Optional

from playwright.async_api import ElementHandle

DESCRIPTION_MAX = 80
OPTIONS_MAX = 10
OPTION_MAX = 40
OUTSIDE_MARKER = " (outside the visible area)"
_MIN_QUOTED = 8

_TEXT_INPUT_TYPES = {"", "text", "email", "search", "tel", "url", "number"}
_BUTTON_INPUT_TYPES = {"submit", "button", "reset", "image"}
_ROLE_KINDS = {"button": "button", "link": "link", "menuitem": "menu item"}
_ACTION_KINDS = {"link", "button", "menu item", "clickable cell", "clickable area"}

# How each label source is worded, so the model can tell a real label from a guessed one.
_LABEL_WORDING = {
    "accessible name": "named",
    "left label": "left label",
    "label above": "label above",
    "placeholder": "placeholder",
    "title": "title",
}


class UnknownElement(LookupError):
    """Raised when the model names a number that isn't in the current list."""


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    def intersects(self, width: float, height: float) -> bool:
        return self.x < width and self.y < height and self.x + self.width > 0 and self.y + self.height > 0


@dataclass(frozen=True)
class ElementFacts:
    """Raw facts the browser reports for one element.

    Password inputs report only whether they are filled, never their value.
    """
    tag: str
    box: Box
    input_type: str = ""
    role: str = ""
    text: str = ""
    label: str = ""
    label_source: str = ""
    value: Optional[str] = None
    filled: bool = False
    checked: Optional[bool] = None
    selected: str = ""
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class PageElement:
    number: int
    facts: ElementFacts
    description: str
    in_viewport: bool
    # Live reference for acting now; never sent to the model.
    handle: ElementHandle = field(repr=False, compare=False)


@dataclass
class Observation:
    url: str
    title: str
    screenshot: bytes
    marked_screenshot: bytes
    elements: list[PageElement]
    omitted_count: int = 0

    def element(self, number: int) -> PageElement:
        for element in self.elements:
            if element.number == number:
                return element
        raise UnknownElement(f"no element [{number}] in the current list")

    def element_list_text(self) -> str:
        lines = []
        for element in self.elements:
            line = f"[{element.number}] {element.description}"
            if not element.in_viewport:
                line += OUTSIDE_MARKER
            lines.append(line)
            if element.facts.options:
                lines.append("    " + describe_options(element.facts))
        if self.omitted_count:
            lines.append(f"+{self.omitted_count} more elements not listed")
        return "\n".join(lines)


def element_kind(facts: ElementFacts) -> str:
    if facts.tag == "a":
        return "link"
    if facts.tag == "select":
        return "dropdown"
    if facts.tag == "textarea":
        return "text area"
    if facts.tag == "button":
        return "button"
    if facts.tag == "input":
        input_type = facts.input_type
        if input_type == "password":
            return "password box"
        if input_type in _BUTTON_INPUT_TYPES:
            return "button"
        if input_type == "checkbox":
            return "checkbox"
        if input_type == "radio":
            return "radio button"
        if input_type in _TEXT_INPUT_TYPES:
            return "text box"
        return f"{input_type} input"
    if facts.role in _ROLE_KINDS:
        return _ROLE_KINDS[facts.role]
    return "clickable cell" if facts.tag in ("td", "th") else "clickable area"


def describe(facts: ElementFacts) -> str:
    """One line for the model, without its number. Page text is always quoted, never bare."""
    kind = element_kind(facts)
    # Each part is fixed wording plus, optionally, page text to be quoted after it.
    parts: list[tuple[str, Optional[str]]] = []

    if kind in _ACTION_KINDS and facts.text:
        parts.append((f"{kind} ", facts.text))
    elif facts.label and facts.label_source in _LABEL_WORDING:
        parts.append((f"{kind}, {_LABEL_WORDING[facts.label_source]} ", facts.label))
    else:
        parts.append((f"{kind}, no label", None))

    if kind == "password box":
        # Ignores any value that arrives anyway: a typed password is the real secret.
        parts.append((", filled" if facts.filled else ", empty", None))
    elif kind in ("text box", "text area") or kind.endswith(" input"):
        parts.append((", value ", facts.value) if facts.value else (", empty", None))
    elif kind in ("checkbox", "radio button"):
        parts.append((", checked" if facts.checked else ", unchecked", None))
    elif kind == "dropdown" and facts.selected:
        parts.append((", selected ", facts.selected))

    return _fit(parts, DESCRIPTION_MAX)


def describe_options(facts: ElementFacts) -> str:
    shown = " | ".join(_quote(option, OPTION_MAX) for option in facts.options[:OPTIONS_MAX])
    more = len(facts.options) - OPTIONS_MAX
    return f"options: {shown}" + (f" (+{more} more)" if more > 0 else "")


def choose_elements(
    facts: list[ElementFacts], viewport_width: float, viewport_height: float, max_elements: int
) -> tuple[list[int], int]:
    """Indexes to list, visible ones first, each group in page order; plus how many were cut."""
    visible = [i for i, f in enumerate(facts) if f.box.intersects(viewport_width, viewport_height)]
    outside = [i for i, f in enumerate(facts) if not f.box.intersects(viewport_width, viewport_height)]
    ordered = visible + outside
    return ordered[:max_elements], max(0, len(ordered) - max_elements)


def _fit(parts: list[tuple[str, Optional[str]]], limit: int) -> str:
    # Quoted page text shares whatever room the fixed wording leaves.
    fixed = sum(len(wording) for wording, _ in parts)
    quoted = sum(1 for _, text in parts if text is not None)
    room = max(_MIN_QUOTED, (limit - fixed) // quoted - 2) if quoted else 0
    return "".join(
        wording + (_quote(text, room) if text is not None else "") for wording, text in parts
    )


def _quote(text: str, limit: int) -> str:
    # One line, bounded, and escaped, so page text can't close the quote and pose as our wording.
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return json.dumps(text, ensure_ascii=False)
