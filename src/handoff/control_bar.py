"""The operator's control bar: plain buttons inside the run's own browser window while a
person is needed, and the record of what that person does there.

The bar lives in the page, so every new page loses it: the session manager shows it
again after each load while a person has control, and removes it before the automation
looks at the page again, so it never reaches a screenshot the model reads or an artifact.
Everything the bar reports arrives through the Playwright binding named BINDING, each
report carrying the handoff's token.
"""
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import Page

from src.locating.checks import ELEMENT_WORDING_JS

# The name the page calls to report a choice or an action; exposed by the session manager.
BINDING = "__bankAgentHandoff"
TAKE_OVER = "take_over"
_TITLE = "The automation needs a person"

_BAR_SOURCE = Path(__file__).with_name("control_bar.js").read_text(encoding="utf-8")
# Composed from our own two sources: the bar reads a clicked element's wording with the
# safety classifier's rule.
_SHOW = f"(config) => ({_BAR_SOURCE})({ELEMENT_WORDING_JS}, config)"
_REMOVE = "() => { if (window.__bankAgentHandoffBar) { window.__bankAgentHandoffBar.remove(); } }"


@dataclass(frozen=True)
class Button:
    """One choice offered to the person: what the report says, and what the button says."""

    choice: str
    label: str


@dataclass(frozen=True)
class BarContent:
    """What one handoff's bar shows. deadline is when the person's time runs out, in
    seconds since the epoch (time.time()). taken_over shows the bar as it is once the
    person pressed Take over: no veil, the handoff's own buttons."""

    token: str
    why: str
    context: tuple[str, ...]
    buttons: tuple[Button, ...]
    deadline: float
    taken_over: bool = False


async def show_bar(page: Page, content: BarContent) -> None:
    """Show the bar on the page as it is now, replacing any bar already there."""
    await page.evaluate(_SHOW, {
        "binding": BINDING,
        "token": content.token,
        "title": _TITLE,
        "why": content.why,
        "context": list(content.context),
        "takeOver": {"choice": TAKE_OVER, "label": "Take over"},
        "buttons": [{"choice": button.choice, "label": button.label} for button in content.buttons],
        "deadlineMs": content.deadline * 1000,
        "takenOver": content.taken_over,
    })


async def remove_bar(page: Page) -> None:
    """Remove the bar and stop recording; the page is left as the person left it."""
    await page.evaluate(_REMOVE)
