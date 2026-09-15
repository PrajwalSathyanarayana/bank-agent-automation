"""The browser surface both modes act through: one Chromium page the size of the model's
view, and the only functions that act on it. Discovery and replay share it, so they can
never disagree on how a dialog is answered or a secret is typed.

A secret becomes its real value inside type_text, as the last step before the keystroke,
and nowhere else: never in a log line, a retry record, an error or the reply to the
model. Every action has a time cap, and a dialog nobody expected is dismissed — except
while a person has control of the session, when every dialog is left for them to answer.
"""
from collections.abc import Awaitable, Callable, Mapping
from decimal import Decimal
from typing import Any, Optional, Union
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
    page: Page,
    logger: Optional[RunLogger] = None,
    *,
    accept_now: Callable[[], bool] = lambda: False,
    hold: Callable[[Dialog], bool] = lambda dialog: False,
) -> list[str]:
    """Answer every dialog the page opens, and note each one's type and wording.

    A dialog is dismissed, since a confirm usually guards an action nobody meant, unless
    accept_now() says the system is performing an action expected to ask (an
    irreversible step in a test environment). Returns the list the notes go into, for
    the loop to tell the model.

    hold(dialog) returns True when it takes the dialog instead: a person has control and
    answers it on screen themselves. A held dialog is logged but not noted, since the
    model didn't cause it; whoever holds it dismisses it if it is still open later.
    """
    notes: list[str] = []

    async def on_dialog(dialog: Dialog) -> None:
        if hold(dialog):
            if logger is not None:
                logger.dialog_left_for_person(dialog.type, dialog.message)
            return
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
    element: ElementHandle, text: str, values: Mapping[str, PlaceholderValue], *, timeout_ms: int,
    tracing: Any = None,
) -> None:
    """Type the text into the element, its placeholders filled at the last moment.

    The only place a secret's real value exists: unwrapped here, typed, and dropped. When a
    secret is among this text's placeholders and tracing is a context's tracing object
    (BrowserSession.tracing; Playwright doesn't export a public type for it), this one
    keystroke is left out of it - paused just for it (nothing recorded: no snapshot, no
    screenshot, no network - not redacted after the fact), resumed straight after, win or
    lose. A failure becomes ActionFailed with a fixed message and nothing attached.
    """
    has_secret = any(isinstance(values.get(name), SecretStr) for name, _, _ in iter_placeholders(text))
    fill = _guarded(element.fill(fill_text(text, _unwrapped(text, values)), timeout=timeout_ms),
                    "typing into the element failed")
    if has_secret and tracing is not None:
        await tracing.stop_chunk()
        try:
            await fill
        finally:
            await tracing.start_chunk()
    else:
        await fill


async def select_option(
    element: ElementHandle, label: str, values: Mapping[str, PlaceholderValue], *, timeout_ms: int
) -> None:
    """Choose the option with this visible label, its placeholders filled ({payee_name})."""
    await _guarded(element.select_option(label=fill_text(label, _unwrapped(label, values)), timeout=timeout_ms),
                   "choosing the option failed")


class BrowserSession:
    """Chromium with one page the size of the model's view; dialogs are dismissed, or left
    for a person while one has control.

    Use as `async with BrowserSession(logger) as session:`. Everything is closed on the
    way out, after an error too.
    """

    def __init__(self, logger: RunLogger, *, headless: bool = True, fit_window: bool = False,
                 trace: bool = False) -> None:
        """fit_window: the page is the window, whatever its size (a visible replay a person
        watches); otherwise one fixed size and scale, which discovery's screenshots need.
        trace: record a Playwright trace to this run's own evidence folder (off by default -
        real cost per session, so callers opt in; a test suite generally shouldn't)."""
        self._logger = logger
        self._headless = headless
        self._fit_window = fit_window
        self._trace = trace
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.dialogs: list[str] = []
        # Set only while the system performs an action expected to open a dialog.
        self.accepting_dialogs = False
        self._person_in_control = False
        # Dialogs that opened while a person had control, possibly still open.
        self._left_open: list[Dialog] = []
        self._dialogs_for_person = 0
        # Set once tracing.start() succeeds; type_text pauses/resumes chunks through this.
        # Playwright doesn't export a public type for its tracing object (playwright.async_api
        # leaves Tracing out of __all__), so this is left as Any rather than reaching into a
        # private module for a type hint.
        self._tracing: Any = None

    @property
    def person_in_control(self) -> bool:
        return self._person_in_control

    @property
    def tracing(self) -> Any:
        """This session's context.tracing object, when it traces itself; None otherwise.
        type_text pauses it around a secret's real keystroke."""
        return self._tracing

    @property
    def dialogs_for_person(self) -> int:
        """How many dialogs have been left for a person to answer in this session."""
        return self._dialogs_for_person

    def leave_dialogs_to_person(self) -> None:
        """A person has control: from now on every dialog is left open for them to answer."""
        self._person_in_control = True

    async def take_back_dialogs(self) -> list[str]:
        """Control is back with the system: dialogs are answered by code again, and any the
        person left open is dismissed. Returns those, as "type: wording".

        Answering by code resumes first, so a dialog opening meanwhile is handled as usual.
        One the person already answered can't be dismissed again ("No dialog is showing"):
        their answer stands and it is skipped.
        """
        self._person_in_control = False
        left, self._left_open = self._left_open, []
        dismissed: list[str] = []
        for dialog in left:
            try:
                await dialog.dismiss()
            except PlaywrightError:
                continue
            dismissed.append(f"{dialog.type}: {dialog.message}")
            self._logger.dialog_dismissed(dialog.type, dialog.message)
        return dismissed

    def _hold(self, dialog: Dialog) -> bool:
        if not self._person_in_control:
            return False
        self._left_open.append(dialog)
        self._dialogs_for_person += 1
        return True

    async def __aenter__(self) -> "BrowserSession":
        self._playwright = await async_playwright().start()
        try:
            args = launch_args(env.mock_bank_base_url)
            if self._fit_window and not self._headless:
                args = [*args, "--start-maximized"]
            self._browser = await self._playwright.chromium.launch(headless=self._headless, args=args)
            if self._fit_window:
                # Nothing is cut off on a smaller screen: the page follows the window, and the
                # screen's own scaling applies. Never for discovery, whose numbered marks assume
                # one page pixel per screenshot pixel.
                context = await self._browser.new_context(no_viewport=True)
            else:
                context = await self._browser.new_context(
                    viewport={"width": settings.discovery_viewport_width, "height": settings.discovery_viewport_height},
                    device_scale_factor=settings.discovery_device_scale_factor,
                )
            if self._trace:
                await context.tracing.start(screenshots=True, snapshots=True)
                await context.tracing.start_chunk()
                self._tracing = context.tracing
            self.page = await context.new_page()
        except BaseException:
            await self._close()
            raise
        self.dialogs = dismiss_dialogs(self.page, self._logger, accept_now=lambda: self.accepting_dialogs,
                                       hold=self._hold)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._close()

    async def open(self, url: str, *, timeout_ms: int) -> None:
        """Open the start page (step 0). Refused before any request if its domain isn't allowed."""
        check_domain(url)
        await self.page.goto(url, timeout=timeout_ms)

    async def _close(self) -> None:
        if self._tracing is not None:
            # The chunk still open when the run ends: everything since the last secret's
            # keystroke (or the whole run, if none was typed). Best-effort: a tracing
            # failure here must never hide the run's real outcome.
            try:
                await self._tracing.stop_chunk(path=self._logger.run_dir / "trace.zip")
                await self._tracing.stop()
            except PlaywrightError:
                pass
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
