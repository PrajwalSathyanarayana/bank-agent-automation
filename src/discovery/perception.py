"""What the discovery model sees each step: a screenshot plus a numbered list of elements."""
import json
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator, Iterable, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import ElementHandle, JSHandle, Page

from src.config.settings import settings

# The in-page fact collector, read once at import so a missing file fails at start-up.
_COLLECTOR_SOURCE = Path(__file__).with_name("collect_elements.js").read_text(encoding="utf-8")

# Marks drawn on the model's copy of the screenshot. Colours cycle by element number so
# neighbours differ; each is dark enough for white digits and clear of the bank's blues
# and greys.
MARK_COLOURS = (
    (194, 24, 91),   # magenta
    (46, 125, 50),   # green
    (106, 27, 154),  # purple
    (191, 54, 12),   # orange
    (0, 121, 107),   # teal
    (121, 85, 72),   # brown
)
TAG_TEXT_COLOUR = (255, 255, 255)
OUTLINE_WIDTH = 2
TAG_FONT_SIZE = 12
_TAG_PAD_X = 3
_TAG_PAD_Y = 2

# Pillow's bundled scalable font: the same glyphs on every machine for the pinned Pillow
# version. Without FreeType, Pillow silently substitutes a tiny bitmap font and ignores
# the size, so that case fails loudly here instead.
_TAG_FONT = ImageFont.load_default(size=TAG_FONT_SIZE)
if not isinstance(_TAG_FONT, ImageFont.FreeTypeFont):
    raise RuntimeError("Pillow was built without FreeType; the numbered marks need its scalable font")
# Every tag is as tall as the digits, whichever digits it holds.
_, _DIGITS_TOP, _, _DIGITS_BOTTOM = _TAG_FONT.getbbox("0123456789")

# (left, top, right, bottom) in screenshot pixels, both ends included, as Pillow draws them.
_Rect = tuple[int, int, int, int]

DESCRIPTION_MAX = 80
OPTIONS_MAX = 10
OPTION_MAX = 40
OUTSIDE_MARKER = " (outside the visible area)"
NO_ELEMENTS = "No interactive elements found on this page."
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


class ObservationReleased(RuntimeError):
    """Raised when an element is asked for after its observation was released.

    Without this, acting on a released handle fails with Playwright's "Target page,
    context or browser has been closed", which reads like a browser crash.
    """


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
    elements: list[PageElement]
    omitted_count: int = 0
    # The copy with numbered boxes drawn on it, which the model receives; None until drawn.
    marked_screenshot: Optional[bytes] = None
    _released: bool = field(default=False, init=False, repr=False)

    def element(self, number: int) -> PageElement:
        if self._released:
            raise ObservationReleased("this element list has been replaced; use the latest one")
        for element in self.elements:
            if element.number == number:
                return element
        raise UnknownElement(f"no element [{number}] in the current list")

    def element_list_text(self) -> str:
        if not self.elements and not self.omitted_count:
            return NO_ELEMENTS
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

    async def release(self) -> None:
        """Let go of every element reference; element() refuses from then on.

        Safe to call twice, and after the page has changed or closed: Playwright's
        dispose raises in none of those cases (checked), so a release at the end of a
        failed turn cannot hide the error that ended it.
        """
        if self._released:
            return
        self._released = True
        for element in self.elements:
            await element.handle.dispose()


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
        # A real accessible name that says something else is shown too, e.g. which of
        # many "Edit" links this is. Guessed labels are never added next to visible text.
        if facts.label_source == "accessible name" and _differs(facts.label, facts.text):
            parts.append((", named ", facts.label))
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


def mark_colour(number: int) -> tuple[int, int, int]:
    return MARK_COLOURS[(number - 1) % len(MARK_COLOURS)]


def mark(screenshot: bytes, elements: Sequence[PageElement]) -> bytes:
    """A copy of the screenshot with each visible element's outline and number drawn on it.

    The screenshot passed in is left as it is: it is the clean evidence copy.
    """
    with Image.open(BytesIO(screenshot)) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    # Elements outside the visible area stay in the list but get no mark.
    shown = []
    for element in elements:
        area = _visible_part(element.facts.box, image.width, image.height) if element.in_viewport else None
        if area is not None:
            shown.append((element, area))

    # Outlines first, then tags, so no outline is ever drawn across a number.
    for element, area in shown:
        draw.rectangle(area, outline=mark_colour(element.number), width=OUTLINE_WIDTH)
    placed: list[_Rect] = []
    for element, area in shown:
        label = str(element.number)
        tag = _place_tag(label, area, placed, image.width, image.height)
        placed.append(tag)
        draw.rectangle(tag, fill=mark_colour(element.number))
        text_left = _TAG_FONT.getbbox(label)[0]
        draw.text(
            (tag[0] + _TAG_PAD_X - text_left, tag[1] + _TAG_PAD_Y - _DIGITS_TOP),
            label,
            fill=TAG_TEXT_COLOUR,
            font=_TAG_FONT,
        )

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _visible_part(box: Box, width: int, height: int) -> Optional[_Rect]:
    # Box coordinates are CSS pixels, equal to screenshot pixels at scale factor 1.
    left = max(0, math.floor(box.x))
    top = max(0, math.floor(box.y))
    right = min(width, math.ceil(box.x + box.width)) - 1
    bottom = min(height, math.ceil(box.y + box.height)) - 1
    if right < left or bottom < top:
        return None
    return (left, top, right, bottom)


def _place_tag(label: str, area: _Rect, placed: list[_Rect], width: int, height: int) -> _Rect:
    text_left, _, text_right, _ = _TAG_FONT.getbbox(label)
    tag_width = text_right - text_left + 2 * _TAG_PAD_X
    tag_height = _DIGITS_BOTTOM - _DIGITS_TOP + 2 * _TAG_PAD_Y
    # Inside the element, at the top-left of its visible part, kept inside the image.
    top = max(0, min(area[1], height - tag_height))
    corner = max(0, min(area[0], width - tag_width))
    left = corner
    while True:
        tag = (left, top, left + tag_width - 1, top + tag_height - 1)
        in_the_way = [other for other in placed if _overlaps(tag, other)]
        if not in_the_way:
            return tag
        # Move right along the box's top edge, past the tags in the way.
        left = max(other[2] for other in in_the_way) + 1
        if left + tag_width > width:
            # No room left in the image: stay at the corner. Drawn after the tag it
            # overlaps, this number stays readable.
            return (corner, top, corner + tag_width - 1, top + tag_height - 1)


def _overlaps(first: _Rect, second: _Rect) -> bool:
    return first[0] <= second[2] and second[0] <= first[2] and first[1] <= second[3] and second[1] <= first[3]


async def observe(page: Page, max_elements: Optional[int] = None) -> Observation:
    """The page as the model will see it now: screenshot plus numbered element list."""
    viewport = page.viewport_size
    if viewport is None:
        raise RuntimeError("the page has no fixed window size, so boxes could not match the screenshot")
    width, height = viewport["width"], viewport["height"]
    limit = settings.discovery_max_elements if max_elements is None else max_elements

    collected = await page.evaluate_handle(_COLLECTOR_SOURCE)
    try:
        # Facts come back as plain data; handles are made only for the elements listed.
        raw_facts = await collected.evaluate("result => result.facts")
        facts = [_to_facts(raw) for raw in raw_facts]
        chosen, omitted = choose_elements(facts, width, height, limit)
        handles = await _handles_for(collected, chosen)
    finally:
        await collected.dispose()
    # Taken straight after the facts, so boxes and pixels describe the same moment.
    screenshot = await page.screenshot()

    elements = [
        PageElement(
            number=number,
            facts=facts[index],
            description=describe(facts[index]),
            in_viewport=facts[index].box.intersects(width, height),
            handle=handle,
        )
        for number, (index, handle) in enumerate(zip(chosen, handles), start=1)
    ]
    return Observation(
        url=page.url,
        title=await page.title(),
        screenshot=screenshot,
        elements=elements,
        omitted_count=omitted,
        marked_screenshot=mark(screenshot, elements),
    )


@asynccontextmanager
async def observing(page: Page, max_elements: Optional[int] = None) -> AsyncIterator[Observation]:
    """observe() for one turn: every element reference is released when the block ends,
    even when the turn ends in an error, so a skipped cleanup cannot happen."""
    observation = await observe(page, max_elements)
    try:
        yield observation
    finally:
        await observation.release()


def _to_facts(raw: dict) -> ElementFacts:
    # An unknown key raises TypeError here; a test checks that none are missing either.
    return ElementFacts(**{**raw, "box": Box(**raw["box"]), "options": tuple(raw["options"])})


async def _handles_for(collected: JSHandle, chosen: list[int]) -> list[ElementHandle]:
    # The page still holds every candidate and hands back only the chosen ones, in list
    # order, so no handle is ever made for an element that is not listed.
    picked = await collected.evaluate_handle(
        "(result, chosen) => chosen.map((index) => result.elements[index])", chosen
    )
    try:
        properties = await picked.get_properties()
    finally:
        await picked.dispose()
    handles = [properties[str(index)].as_element() for index in _element_indexes(properties)]
    if len(handles) != len(chosen) or any(handle is None for handle in handles):
        raise RuntimeError(f"expected {len(chosen)} element handles from the page, got {len(handles)}")
    return handles


def _element_indexes(keys: Iterable[str]) -> list[int]:
    # Array keys arrive as strings, in no promised order. Sorted as text, "10" would
    # come before "2"; so they are sorted as numbers, and non-numeric keys (such as
    # "length", if present) are dropped because they are not elements.
    return sorted(int(key) for key in keys if key.isdigit())


def _differs(first: str, second: str) -> bool:
    return " ".join(first.split()).casefold() != " ".join(second.split()).casefold()


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
