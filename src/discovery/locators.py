"""Locators for the element the model picked: generated, proven on the live page, ranked.

Candidates are built by our code, never the model, in the legacy-aware order: form field
name, visible text or button value (a link's address after its text), the field next to
its label, accessible name, scoped versions of the attribute candidates, and position.
"""
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import ElementHandle, Page
from pydantic import SecretStr

from src.discovery.perception import ACTION_KINDS, ElementFacts, PageElement, element_kind
from src.locating.resolver import UnfillableLocator, css_string, resolve
# Shared with replay's text checks; still importable from here for the backstop and tests.
from src.locating.values import number_pattern
from src.types.placeholders import iter_placeholders
from src.types.step_schema import Locator, LocatorType

# In-page script that reads the element's raw parts; read once at import.
_LOCATOR_PARTS_SOURCE = Path(__file__).with_name("locator_parts.js").read_text(encoding="utf-8")

# Tags whose name attribute is sent to the server with the form.
_NAMED_TAGS = {"input", "select", "textarea", "button"}
_BUTTON_INPUT_TYPES = {"submit", "button", "reset"}
_CELL = "*[self::td or self::th]"
# A path segment: what follows a "/" up to the next "/", query, fragment or quote.
_ADDRESS_SEGMENT = re.compile(r"(?<=/)[^/?#&\"']+")

# Priorities 0, 1 and 2: a primary and two fallbacks.
MAX_LOCATORS = 3
# A proven unscoped candidate makes scoped versions of the same base redundant. A proven
# text candidate already says what a scoped button value would.
_BASE_COVERED_BY = {"name": "name", "address": "address", "text": "value"}


class NoProvenLocator(RuntimeError):
    """Raised when no candidate finds the picked element, not even its position path.

    The full path from the top of the page always exists, so this means the page changed
    while the step was being recorded.
    """


class Verdict(str, Enum):
    """The outcome of proving one locator; worded to read after "the locator"."""

    PROVEN = "finds exactly the picked element"
    NO_MATCH = "matches nothing"
    SEVERAL = "matches more than one element"
    OTHER_ELEMENT = "matches a different element"
    UNFILLABLE = "cannot be filled with this run's values"


@dataclass(frozen=True)
class Candidate:
    """One way to find the picked element again, before it is scanned or proven."""

    kind: str
    locator_type: LocatorType
    value: str
    # For a scoped candidate, the kind it narrows: "name", "address" or "value".
    base: Optional[str] = None
    # The page data the value was built from (attribute values, text, id and class
    # names). The data scan reads only these, never the selector's structure.
    data: tuple[str, ...] = ()
    # For candidates built from a link's address: the address as the page wrote it, and
    # the container selector if scoped, so the selector can be rebuilt with placeholders.
    address: Optional[str] = None
    scope: Optional[str] = None


@dataclass(frozen=True)
class RunValues:
    """This run's values that must never ride along in a saved locator."""

    text_inputs: Mapping[str, str] = field(default_factory=dict)
    number_inputs: Mapping[str, float] = field(default_factory=dict)
    username: str = ""
    # Held as SecretStr so a stray repr or log line can't show them.
    secrets: Mapping[str, SecretStr] = field(default_factory=dict)


@dataclass(frozen=True)
class ScanOutcome:
    """The candidate's value as it would be stored, or why it was discarded.

    Reasons name the input or secret involved, never its value.
    """

    stored_value: Optional[str]
    reason: Optional[str] = None
    # The secret whose value was found, for the run-log warning.
    secret: Optional[str] = None


@dataclass(frozen=True)
class Rejection:
    """A candidate left out, for the run log: its kind and why, never its value."""

    kind: str
    reason: str


@dataclass(frozen=True)
class DerivedLocators:
    """What derive_locators found for one element."""

    # Priorities 0, 1, 2 in this order, as they will be stored.
    locators: list[Locator]
    # The kind of each kept locator ("name", "scoped", "position", ...), for the run log.
    kinds: list[str]
    # Only a position locator survived: flagged for the reviewer; the action still runs.
    weak: bool
    rejected: list[Rejection]
    # Names of secrets whose value appeared in a candidate, for a run-log warning.
    secrets_found: list[str]


async def derive_locators(page: Page, element: PageElement, run: RunValues) -> DerivedLocators:
    """Up to three proven locators for the element the model picked, best first.

    Call before the action runs: afterwards the page may have changed and the element
    may be gone. Candidates are generated, cleared of this run's data, proven through the
    shared resolver, and kept in two passes:

    1. Distinct kinds first, in order. Variants of something already kept are set aside:
       scoped versions of a selector already unique on its own, a second scoped version
       of the same selector, a second position path.
    2. If fewer than three survived, the set-aside variants fill the remaining slots, in
       order. They share a weakness with an earlier locator but survive a renamed
       container, the most likely change on a legacy page.
    """
    candidates = await generate_candidates(element)
    # Scanning is cheap, so every candidate is scanned, and a secret on the page is
    # reported even if the candidate carrying it would never have been reached.
    outcomes = [(candidate, scan(candidate, run)) for candidate in candidates]
    secrets_found = sorted({outcome.secret for _, outcome in outcomes if outcome.secret})

    locators: list[Locator] = []
    kinds: list[str] = []
    rejected: list[Rejection] = []
    covered_bases: set[str] = set()

    async def keep_if_proven(candidate: Candidate, outcome: ScanOutcome) -> None:
        if outcome.stored_value is None:
            rejected.append(Rejection(candidate.kind, outcome.reason or "discarded"))
            return
        locator = Locator(type=candidate.locator_type, value=outcome.stored_value, priority=len(locators))
        verdict = await prove(page, element.handle, locator, run.text_inputs)
        if verdict is not Verdict.PROVEN:
            rejected.append(Rejection(candidate.kind, verdict.value))
            return
        locators.append(locator)
        kinds.append(candidate.kind)
        covered_base = candidate.base if candidate.kind == "scoped" else _BASE_COVERED_BY.get(candidate.kind)
        if covered_base is not None:
            covered_bases.add(covered_base)

    set_aside: list[tuple[Candidate, ScanOutcome]] = []
    for candidate, outcome in outcomes:
        if len(locators) == MAX_LOCATORS:
            break
        is_variant = (candidate.kind == "scoped" and candidate.base in covered_bases) or (
            candidate.kind == "position" and "position" in kinds
        )
        if is_variant:
            set_aside.append((candidate, outcome))
            continue
        await keep_if_proven(candidate, outcome)
    for candidate, outcome in set_aside:
        if len(locators) == MAX_LOCATORS:
            break
        await keep_if_proven(candidate, outcome)

    if not locators:
        raise NoProvenLocator("no candidate locator finds the picked element; the page changed while recording")
    return DerivedLocators(
        locators=locators,
        kinds=kinds,
        weak=set(kinds) == {"position"},
        rejected=rejected,
        secrets_found=secrets_found,
    )


async def generate_candidates(element: PageElement) -> list[Candidate]:
    """Every candidate for the element, best kind first, with literal values."""
    parts = await element.handle.evaluate(_LOCATOR_PARTS_SOURCE)
    return build_candidates(element.facts, parts)


def build_candidates(facts: ElementFacts, parts: dict[str, Any]) -> list[Candidate]:
    tag = parts["tag"]
    is_action = element_kind(facts) in ACTION_KINDS
    candidates: list[Candidate] = []
    # Attribute selectors that scoped candidates narrow to a container:
    # (kind, selector, data it was built from, address if built from one).
    bases: list[tuple[str, str, tuple[str, ...], Optional[str]]] = []

    name = parts["name"]
    if name and tag in _NAMED_TAGS:
        selector = f'{tag}[name="{css_string(name)}"]'
        candidates.append(Candidate("name", LocatorType.CSS, selector, data=(name,)))
        bases.append(("name", selector, (name,), None))

    text = _visible_text(parts) if is_action else ""
    if text:
        candidates.append(Candidate("text", LocatorType.TEXT_CONTENT, text, data=(text,)))

    href = parts["href"]
    if tag == "a" and _is_usable_address(href):
        # After the text: an address can carry data (a member's ID), text rarely does.
        selector = _address_selector(href)
        candidates.append(Candidate("address", LocatorType.CSS, selector, address=href))
        bases.append(("address", selector, (), href))

    value = parts["value"]
    if tag == "input" and parts["type"] in _BUTTON_INPUT_TYPES and value:
        # Only as a base for scoping; unscoped, the text candidate already says it.
        selector = f'input[type="{css_string(parts["type"])}"][value="{css_string(value)}"]'
        bases.append(("value", selector, (value,), None))

    if not is_action and facts.label_source == "left label":
        xpath = _next_to_label_xpath(tag, facts.label)
        if xpath is not None:
            candidates.append(Candidate("label", LocatorType.XPATH, xpath, data=(facts.label,)))

    if facts.label_source == "accessible name":
        candidates.append(
            Candidate("accessible name", LocatorType.ARIA_LABEL, facts.label, data=(facts.label,))
        )

    for base, selector, data, address in bases:
        for scope in parts["scopes"]:
            candidates.append(Candidate(
                "scoped", LocatorType.CSS, f"{scope['selector']} {selector}", base=base,
                data=(scope["token"], *data), address=address, scope=scope["selector"],
            ))

    for position in parts["positions"]:
        data = (position["token"],) if position["token"] else ()
        candidates.append(Candidate("position", LocatorType.CSS, position["path"], data=data))
    return candidates


def scan(candidate: Candidate, run: RunValues) -> ScanOutcome:
    """Keep this run's data out of the candidate, or discard it.

    A link address segment equal to a text input becomes a placeholder
    (/member/{member_id}/accounts). Any other trace of an input, the username or a
    secret discards the candidate. Only the candidate's data parts are read, never its
    structure: the 5 in tr:nth-of-type(5) is not an amount.
    """
    for name, secret in run.secrets.items():
        secret_value = secret.get_secret_value()
        if secret_value and any(secret_value in part for part in _all_data(candidate)):
            return ScanOutcome(None, f"carries the value of the secret {name}", secret=name)

    stored_address = None
    if candidate.address is not None:
        stored_address = parameterize_address(candidate.address, run.text_inputs)
        if stored_address is None:
            return ScanOutcome(None, "its address matches two inputs in one segment")

    texts = list(candidate.data)
    if stored_address is not None:
        texts.append(without_placeholders(stored_address))
    for name, text_input in run.text_inputs.items():
        if text_input and any(text_input.casefold() in text.casefold() for text in texts):
            return ScanOutcome(None, f"carries the value of the input {name}")
    for name, number in run.number_inputs.items():
        pattern = number_pattern(number)
        if any(pattern.search(text) for text in texts):
            return ScanOutcome(None, f"carries the value of the input {name}")
    if run.username:
        word = _whole_word(run.username)
        if any(word.search(text) for text in texts):
            return ScanOutcome(None, "carries the bank username")
    return ScanOutcome(_stored_value(candidate, stored_address))


async def prove(page: Page, target: ElementHandle, locator: Locator, values: Mapping[str, str]) -> Verdict:
    """Whether the locator, as it will be stored, finds exactly the element the model picked.

    It goes through the shared resolver with this run's input values, the same path replay
    takes, so a locator proven here is the locator replay runs. A well-formed locator isn't
    enough: "Bill Pay" is valid but matches two links. Hidden matches count too, because
    replay would see them as well. Must run before the action, while `target` still exists.
    """
    try:
        found = resolve(page, locator, values)
    except UnfillableLocator:
        return Verdict.UNFILLABLE
    # One call, no waiting: count the matches and compare the only match with the target.
    result = await found.evaluate_all(
        "(matches, target) => ({ count: matches.length, same: matches.length === 1 && matches[0] === target })",
        target,
    )
    if result["count"] == 0:
        return Verdict.NO_MATCH
    if result["count"] > 1:
        return Verdict.SEVERAL
    return Verdict.PROVEN if result["same"] else Verdict.OTHER_ELEMENT


def _all_data(candidate: Candidate) -> list[str]:
    return [*candidate.data, *([candidate.address] if candidate.address is not None else [])]


def parameterize_address(address: str, text_inputs: Mapping[str, str]) -> Optional[str]:
    """The address with each whole path segment equal to a text input as a placeholder.

    Shared with the recorder's page-path checkpoint, so both treat addresses alike.

    Literal braces are doubled first so they can't be read as placeholders. Returns None
    when a segment equals two inputs: which one it stands for can't be known.
    """
    ambiguous = False

    def replace(match: re.Match[str]) -> str:
        nonlocal ambiguous
        names = [name for name, value in text_inputs.items() if value and match.group(0) == _literal(value)]
        if len(names) > 1:
            ambiguous = True
        return f"{{{names[0]}}}" if len(names) == 1 else match.group(0)

    stored = _ADDRESS_SEGMENT.sub(replace, _literal(address))
    return None if ambiguous else stored


def _stored_value(candidate: Candidate, stored_address: Optional[str]) -> str:
    if stored_address is None:
        return _literal(candidate.value)
    selector = _address_selector(stored_address)
    return f"{_literal(candidate.scope)} {selector}" if candidate.scope else selector


def _address_selector(address: str) -> str:
    return f'a[href="{css_string(address)}"]'


def _literal(text: str) -> str:
    # Braces in stored text are doubled; single braces mean a placeholder.
    return text.replace("{", "{{").replace("}", "}}")


def without_placeholders(text: str) -> str:
    """The text with its placeholders taken out. Shared with the save-time backstop scan."""
    pieces = []
    position = 0
    for _, start, end in iter_placeholders(text):
        pieces.append(text[position:start])
        position = end
    pieces.append(text[position:])
    return "".join(pieces)


def _whole_word(word: str) -> re.Pattern[str]:
    # "admin" as a word, any case, but not inside "Administration".
    return re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)


def _visible_text(parts: dict[str, Any]) -> str:
    # Playwright's text lookup matches button inputs by their value attribute.
    if parts["tag"] == "input":
        return (parts["value"] or "") if parts["type"] in _BUTTON_INPUT_TYPES else ""
    return parts["text"]


def _is_usable_address(href: Optional[str]) -> bool:
    # "#" and javascript: links go nowhere of their own, so they identify nothing.
    if href is None or not href.strip():
        return False
    return href.strip() != "#" and not href.strip().lower().startswith("javascript:")


def _next_to_label_xpath(tag: str, label: str) -> Optional[str]:
    """The first field of this tag in the nearest following cell after the label's cell.

    translate() turns &nbsp; into a space first, since XPath's normalize-space() doesn't.
    The label comes from the element list, which can differ from what XPath reads (for
    example CSS-added text); the proof then discards the candidate, nothing worse.
    """
    literal = _xpath_literal(label)
    if literal is None:
        return None
    return (
        f"//{_CELL}[normalize-space(translate(., ' ', ' '))={literal}]"
        f"/following-sibling::{_CELL}[.//{tag}][1]//{tag}"
    )


def label_cell_value_xpath(label_text: str) -> Optional[str]:
    """The cell right after the cell whose whole text is this label, in the same row.

    How legacy pages show a value: label and value side by side in a table row. Anchored
    on the label, which is the same for every record, never on the value. None when the
    label holds both kinds of quote.
    """
    literal = _xpath_literal(label_text)
    if literal is None:
        return None
    return (
        f"//{_CELL}[normalize-space(translate(., ' ', ' '))={literal}]"
        f"/following-sibling::{_CELL}[1]"
    )


def _xpath_literal(text: str) -> Optional[str]:
    # XPath 1.0 strings have no escape character: use whichever quote the text lacks,
    # and give up on text containing both rather than guess.
    if '"' not in text:
        return f'"{text}"'
    if "'" not in text:
        return f"'{text}'"
    return None
