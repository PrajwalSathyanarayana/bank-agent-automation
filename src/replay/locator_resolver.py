"""Replay: finding a step's element with the locators discovery proved.

The locators are tried in priority order (0 is the one discovery ranked most robust), and
the first that matches exactly one element wins. Nothing is guessed: a locator matching
nothing, or several elements, is passed over, never narrowed down. When none wins, the page
is looked at again within the step's retry budget, since a slow page may still be drawing
the element. Every try is logged, and the result says which locator worked: a fallback in
use is a sign the page has changed since discovery.
"""
import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Union

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Locator as PageLocator
from playwright.async_api import Page

from src.locating.resolver import UnfillableLocator, resolve
from src.observability.logger import RunLogger
from src.types.step_schema import Locator, Step


@dataclass(frozen=True)
class LocatorTry:
    """One locator on one look at the page."""

    priority: int
    kind: str
    # How many elements it matched; None when it couldn't be used at all.
    matches: Optional[int]
    problem: Optional[str] = None


@dataclass(frozen=True)
class Found:
    element: PageLocator
    priority: int
    attempts: int


@dataclass(frozen=True)
class NotFound:
    attempts: int
    # The last look at the page: one entry per locator, in priority order.
    tries: tuple[LocatorTry, ...]

    def observed(self) -> str:
        """What replay saw, short enough for a result's failure block."""
        seen = ", ".join(f"#{tried.priority} {_what(tried)}" for tried in self.tries)
        return (f"none of its {len(self.tries)} locators matched exactly one element "
                f"after {self.attempts} attempts ({seen})")


async def find_element(page: Page, step: Step, values: Mapping[str, str], logger: RunLogger) -> Union[Found, NotFound]:
    """The step's element, by the first locator that matches exactly one, within its retry budget.

    values fills the locators' placeholders ({member_id}) with this run's inputs.
    """
    if not step.locators:
        raise ValueError(f"step {step.sequence_index} has no locators: there is no element to find")
    ordered = sorted(step.locators, key=lambda locator: locator.priority)
    budget = step.retry_budget
    tries: list[LocatorTry] = []
    for attempt in range(1, budget.max_attempts + 1):
        tries = []
        for locator in ordered:
            started = time.monotonic()
            element, tried = await _try(page, locator, values)
            logger.locator_evaluated(tried.kind, int((time.monotonic() - started) * 1000), attempt,
                                     step_index=step.sequence_index, priority=tried.priority, matches=tried.matches)
            tries.append(tried)
            if element is not None and tried.matches == 1:
                return Found(element, tried.priority, attempt)
        if attempt < budget.max_attempts:
            await asyncio.sleep(budget.poll_interval_ms / 1000)
    return NotFound(budget.max_attempts, tuple(tries))


async def _try(page: Page, locator: Locator, values: Mapping[str, str]) -> tuple[Optional[PageLocator], LocatorTry]:
    kind = locator.type.value
    try:
        element = resolve(page, locator, values)
    except UnfillableLocator:
        return None, LocatorTry(locator.priority, kind, None, "can't be filled")
    try:
        # Hidden matches count too, as they did when discovery proved the locator.
        matches = await element.count()
    except PlaywrightError:
        return None, LocatorTry(locator.priority, kind, None, "not a usable selector")
    return element, LocatorTry(locator.priority, kind, matches)


def _what(tried: LocatorTry) -> str:
    if tried.problem is not None:
        return tried.problem
    return "matched nothing" if tried.matches == 0 else f"matched {tried.matches}"
