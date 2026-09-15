"""The browser surface both modes act through: one Chromium page the size of the model's
view, and the only functions that act on it. Discovery and replay share it, so they can
never disagree on how a dialog is answered or a secret is typed.

A secret becomes its real value inside type_text, as the last step before the keystroke,
and nowhere else: never in a log line, a retry record, an error or the reply to the
model. Every action has a time cap, and a dialog nobody expected is dismissed.
"""
from collections.abc import Awaitable, Callable, Mapping
from decimal import Decimal
from typing import Optional, Union
from urllib.parse import urlsplit

from playwright.async_api import Browser, Dialog, ElementHandle, Page, Playwright, async_playwright
from playwright.async_api import Error as PlaywrightError
from pydantic import SecretStr

from src.config.env import env
from src.config.settings import settings
from src.observability.logger import RunLogger
from src.safety.allowlist import check_domain
from src.types.placeholders import CREDENTIAL_PREFIX, fill_text, iter_placeholders

# A placeholder's value: plain text, or a secret kept wrapped until the keystroke.
PlaceholderValue = Union[str, SecretStr]


class ActionFailed(RuntimeError):
    """An action on the page failed. The message never holds a typed value."""


def launch_args(base_url: str) -> list[str]:
    """Chromium's start-up flags for this bank address.

    The mock bank listens on IPv4 only. For "localhost" Chromium tries IPv6 first and
    waits 0.3-0.5 s per request before falling back, so the name is mapped straight to
    127.0.0.1. Any other address is left to normal name resolution.
    """
    if urlsplit(base_url).hostname == "localhost":
        return ["--host-resolver-rules=MAP localhost 127.0.0.1"]
    return []


def action_timeout_ms(time_left_ms: int) -> int:
    """An action's cap: the page action limit or what is left of the run, whichever is less."""
    return max(1, min(settings.discovery_page_action_timeout_ms, time_left_ms))


def number_text(number: float) -> str:
    """How a number input is typed: its plain shortest form (50, 512.75, 1240.5)."""
    return format(Decimal(str(number)).normalize(), "f")


def placeholder_values(
    text_inputs: Mapping[str, str],
    number_inputs: Mapping[str, float],
    credentials: Mapping[str, PlaceholderValue],
) -> dict[str, PlaceholderValue]:
    """Every placeholder this run can fill, by its name in the artifact. Secrets stay wrapped."""
    values: dict[str, PlaceholderValue] = dict(text_inputs)
    values.update({name: number_text(number) for name, number in number_inputs.items()})
    values.update({f"{CREDENTIAL_PREFIX}:{key}": value for key, value in credentials.items()})
    return values


def dismiss_dialogs(
    page: Page, logger: Optional[RunLogger] = None, *, accept_now: Callable[[], bool] = lambda: False
) -> list[str]:
    """Answer every dialog the page opens, and note each one's type and wording.

    A dialog is dismissed, since a confirm usually guards an action nobody meant, unless
    accept_now() says the system is performing an action expected to ask (an
    irreversible step in a test environment). Returns the list the notes go into, for
    the loop to tell the model.
    """
    notes: list[str] = []

    async def on_dialog(dialog: Dialog) -> None:
        # Answered first: an error while noting it must never leave the page blocked,
        # since an open dialog stalls the page until someone answers it.
        accepted = accept_now()
        await (dialog.accept() if accepted else dialog.dismiss())
        notes.append(f"{dialog.type}{' (accepted)' if accepted else ''}: {dialog.message}")
        if logger is not None:
            answer = logger.dialog_accepted if accepted else logger.dialog_dismissed
            answer(dialog.type, dialog.message)

    page.on("dialog", on_dialog)
    return notes


async def click(element: ElementHandle, *, timeout_ms: int) -> None:
    """Click the element. A failure keeps Playwright's reason, e.g. another element on top."""
    try:
        await element.click(timeout=timeout_ms)
    except PlaywrightError as error:
        reason = str(error).splitlines()[0] if str(error) else "no reason given"
        raise ActionFailed(f"clicking the element failed: {reason}") from None


async def type_text(
    element: ElementHandle, text: str, values: Mapping[str, PlaceholderValue], *, timeout_ms: int
) -> None:
    """Type the text into the element, its placeholders filled at the last moment.

    The only place a secret's real value exists: unwrapped here, typed, and dropped. A
    failure becomes ActionFailed with a fixed message and nothing attached.
    """
    await _guarded(element.fill(fill_text(text, _unwrapped(text, values)), timeout=timeout_ms),
                   "typing into the element failed")


async def select_option(
    element: ElementHandle, label: str, values: Mapping[str, PlaceholderValue], *, timeout_ms: int
) -> None:
    """Choose the option with this visible label, its placeholders filled ({payee_name})."""
    await _guarded(element.select_option(label=fill_text(label, _unwrapped(label, values)), timeout=timeout_ms),
                   "choosing the option failed")


class BrowserSession:
    """Chromium with one page the size of the model's view; dialogs are dismissed.

    Use as `async with BrowserSession(logger) as session:`. Everything is closed on the
    way out, after an error too.
    """

    def __init__(self, logger: RunLogger, *, headless: bool = True) -> None:
        self._logger = logger
        self._headless = headless
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.dialogs: list[str] = []
        # Set only while the system performs an action expected to open a dialog.
        self.accepting_dialogs = False

    async def __aenter__(self) -> "BrowserSession":
        self._playwright = await async_playwright().start()
        try:
            self._browser = await self._playwright.chromium.launch(
                headless=self._headless, args=launch_args(env.mock_bank_base_url)
            )
            context = await self._browser.new_context(
                viewport={"width": settings.discovery_viewport_width, "height": settings.discovery_viewport_height},
                device_scale_factor=settings.discovery_device_scale_factor,
            )
            self.page = await context.new_page()
        except BaseException:
            await self._close()
            raise
        self.dialogs = dismiss_dialogs(self.page, self._logger, accept_now=lambda: self.accepting_dialogs)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._close()

    async def open(self, url: str, *, timeout_ms: int) -> None:
        """Open the start page (step 0). Refused before any request if its domain isn't allowed."""
        check_domain(url)
        await self.page.goto(url, timeout=timeout_ms)

    async def _close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()


async def _guarded(action: Awaitable[None], message: str) -> None:
    # The failure is noted inside the except block and raised after it, so the new error
    # has no context chain back to Playwright's error, which could quote what was typed.
    failed = False
    try:
        await action
    except PlaywrightError:
        failed = True
    if failed:
        raise ActionFailed(message)


def _unwrapped(text: str, values: Mapping[str, PlaceholderValue]) -> dict[str, str]:
    # Only the values this text uses, secrets unwrapped here and nowhere else.
    names = {name for name, _, _ in iter_placeholders(text)}
    return {
        name: value.get_secret_value() if isinstance(value, SecretStr) else value
        for name, value in values.items()
        if name in names
    }
