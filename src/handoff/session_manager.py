"""Handing the run's live browser session to a person, and back.

One lock per run says who is in control: the automation or a person. A handoff pauses
the run: the page is photographed as the automation left it, the control bar is shown in
the same window, and the request is announced on the feed. Nothing happens on the page
until the person presses Take over; from then every dialog is theirs to answer and what
they do is recorded — clicks, fields changed (by label), pages visited, tabs opened, never
a typed value. The handoff ends when they choose, when their time runs out, or when they
close the window. Then dialogs they left open are dismissed, the bar is removed, tabs they
opened are closed, the page is photographed as they left it, and control returns.

The caller (replay or discovery) awaits request() and touches nothing meanwhile, so no
second party contends for the lock: it records who holds control, and a report from the
bar counts only with the current handoff's token.
"""
import asyncio
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Protocol
from urllib.parse import urlsplit

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Frame, Page

from src.config.settings import settings
from src.handoff.control_bar import TAKE_OVER, BINDING, BarContent, Button, remove_bar, show_bar
from src.observability.logger import RunLogger
from src.safety.redactor import redact_dict
from src.surface.browser import BrowserSession
from src.types.result_schema import HandoffResolution, HandoffTelemetry

# Page text from the bar is cut to this length, as the bar itself does.
_MAX_TEXT = 80


class Choice(str, Enum):
    """How a person can end a handoff. Each request offers only the ones that fit."""

    HAND_BACK = "hand_back"
    # Discovery, paused at the irreversible step: the person performed it; learning goes on.
    CONFIRMED = "confirmed"
    FINISHED = "finished"
    STOP = "stop"


LABELS = {
    Choice.HAND_BACK: "Hand back",
    Choice.CONFIRMED: "I confirmed it, carry on",
    Choice.FINISHED: "I finished it",
    Choice.STOP: "Stop the task",
}
_RESOLUTIONS = {
    Choice.HAND_BACK: HandoffResolution.RESUMED,
    Choice.CONFIRMED: HandoffResolution.RESUMED,
    Choice.FINISHED: HandoffResolution.MANUAL_COMPLETED,
    Choice.STOP: HandoffResolution.ABORTED,
}


@dataclass(frozen=True)
class HandoffRequest:
    """Why the run needs a person, in a stop code and in plain words, where it paused, and
    the choices that fit there."""

    trigger_reason: str
    why: str
    choices: tuple[Choice, ...]
    step_index: Optional[int] = None
    step_description: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.choices or len(set(self.choices)) != len(self.choices):
            raise ValueError("a handoff offers at least one choice, each once")


@dataclass(frozen=True)
class PersonAction:
    """One thing the person did: "click" (what = the element's wording), "field_changed"
    (what = the field's label), "page_visited" or "tab_opened" (the path alone)."""

    kind: str
    page_path: str
    what: Optional[str] = None
    element_kind: Optional[str] = None


@dataclass(frozen=True)
class HandoffOutcome:
    """How a handoff ended. choice is None when time ran out or the window was closed; the
    telemetry is what the result's handoff record carries."""

    choice: Optional[Choice]
    telemetry: HandoffTelemetry
    actions: tuple[PersonAction, ...]
    # Dialogs the person left open, dismissed by our code at hand-back, as "type: wording".
    dialogs_dismissed: tuple[str, ...]
    window_closed: bool
    pause_screenshot: Optional[str]
    back_screenshot: Optional[str]

    @property
    def resolution(self) -> HandoffResolution:
        return self.telemetry.resolution


class Announcer(Protocol):
    """Where handoff announcements go, for anyone not looking at the run's window."""

    async def announce(self, announcement: dict[str, Any]) -> None:
        ...


@dataclass(frozen=True)
class OperatorSetup:
    """A person is available to this run. Announcements go to announcer, if any;
    timeout_ms replaces the operator timeout (a test's shorter wait)."""

    announcer: Optional[Announcer] = None
    timeout_ms: Optional[int] = None


@dataclass
class _Handoff:
    request: HandoffRequest
    token: str
    deadline: float
    ended: asyncio.Future
    last_path: str
    taken_over: bool = False
    actions: list[PersonAction] = field(default_factory=list)
    extra_pages: list[Page] = field(default_factory=list)


class HandoffManager:
    """One per run: the lock and every handoff the run makes."""

    def __init__(
        self,
        session: BrowserSession,
        logger: RunLogger,
        *,
        capability: str,
        goal: str,
        screenshots_dir: Path,
        announcer: Optional[Announcer] = None,
        timeout_ms: Optional[int] = None,
    ) -> None:
        self._session = session
        self._logger = logger
        self._capability = capability
        self._goal = goal
        self._screenshots_dir = screenshots_dir
        self._announcer = announcer
        self._timeout_s = (timeout_ms if timeout_ms is not None else settings.operator_timeout_ms) / 1000
        self.holder: Literal["automation", "person"] = "automation"
        self._handoffs = 0
        self._bound_page: Optional[Page] = None
        self._active: Optional[_Handoff] = None
        self._tasks: set[asyncio.Task] = set()

    async def request(self, request: HandoffRequest) -> HandoffOutcome:
        """Hand the live session to a person and wait until control comes back."""
        if self._active is not None:
            raise RuntimeError("a handoff is already in progress")
        page = self._session.page
        self._handoffs += 1
        triggered = datetime.now(timezone.utc)
        started = time.monotonic()
        # The page as the automation left it, before the bar covers any of it.
        pause_screenshot = await self._screenshot(page, "pause")
        handoff = _Handoff(request, token=secrets.token_urlsafe(16), deadline=time.time() + self._timeout_s,
                           ended=asyncio.get_running_loop().create_future(), last_path=_path(page.url))
        self._active = handoff
        await self._bind(page)
        buttons = [LABELS[choice] for choice in request.choices]
        self._logger.handoff_requested(request.trigger_reason, request.why, buttons, step_index=request.step_index,
                                       step_description=request.step_description, screenshot_path=pause_screenshot)
        await self._announce("HANDOFF_REQUESTED", why=request.why, trigger_reason=request.trigger_reason,
                             step_index=request.step_index, step_description=request.step_description,
                             buttons=buttons, screenshot_path=pause_screenshot)

        # Each listener is one object, added and removed as the same one: a bound method
        # read twice is two objects, and removal could miss.
        listeners = {"domcontentloaded": self._on_page_loaded, "framenavigated": self._on_navigated,
                     "close": self._on_closed}
        on_new_page = self._on_new_page
        for event, listener in listeners.items():
            page.on(event, listener)
        page.context.on("page", on_new_page)
        try:
            await self._show(page, handoff)
            await asyncio.wait({handoff.ended}, timeout=self._timeout_s)
        finally:
            for event, listener in listeners.items():
                page.remove_listener(event, listener)
            page.context.remove_listener("page", on_new_page)
            # Late reports from the bar are ignored from here on.
            self._active = None
        choice, window_closed = handoff.ended.result() if handoff.ended.done() else (None, False)
        return await self._hand_back(page, handoff, choice, window_closed, triggered, started, pause_screenshot)

    # ------------------------------------------------------------------ hand-back

    async def _hand_back(self, page: Page, handoff: _Handoff, choice: Optional[Choice], window_closed: bool,
                         triggered: datetime, started: float, pause_screenshot: Optional[str]) -> HandoffOutcome:
        # Dialogs first: while one is open the page is frozen, and nothing else could run.
        dismissed = await self._session.take_back_dialogs()
        # Then any bar redraw still running, so none can put the bar back after its removal.
        await self._settle_tasks()
        back_screenshot = None
        if not window_closed:
            await _quietly(remove_bar(page))
            for extra in handoff.extra_pages:
                await _quietly(extra.close())
            back_screenshot = await self._screenshot(page, "back")
        self.holder = "automation"

        if choice is not None:
            resolution = _RESOLUTIONS[choice]
        elif window_closed:
            resolution = HandoffResolution.ABORTED
        else:
            resolution = HandoffResolution.OPERATOR_TIMED_OUT
        duration_ms = int((time.monotonic() - started) * 1000)
        person_actions = len(handoff.actions) if handoff.taken_over else None
        telemetry = HandoffTelemetry(
            triggered_timestamp=triggered, resolved_timestamp=datetime.now(timezone.utc), duration_ms=duration_ms,
            trigger_reason=handoff.request.trigger_reason, step_index=handoff.request.step_index,
            resolution=resolution, session_lock_token=handoff.token, person_actions=person_actions,
        )
        self._logger.handoff_resolved(resolution.value, duration_ms, person_actions=person_actions,
                                      screenshot_path=back_screenshot)
        await self._announce("HANDOFF_RESOLVED", resolution=resolution.value, window_closed=window_closed,
                             person_actions=person_actions, duration_ms=duration_ms)
        return HandoffOutcome(choice, telemetry, tuple(handoff.actions), tuple(dismissed), window_closed,
                              pause_screenshot, back_screenshot)

    # ------------------------------------------------------------------ the bar's reports

    async def _bind(self, page: Page) -> None:
        # A binding outlives page loads but can be exposed only once per page.
        if self._bound_page is not page:
            await page.expose_binding(BINDING, self._on_report)
            self._bound_page = page

    def _on_report(self, source: Any, payload: Any) -> None:
        handoff = self._active
        if handoff is None or not isinstance(payload, dict) or payload.get("token") != handoff.token:
            return
        event = payload.get("event")
        if event == "choice":
            self._on_choice(handoff, payload.get("choice"))
        elif event in ("click", "field_changed") and handoff.taken_over:
            self._record(handoff, PersonAction(kind=event, page_path=_text(payload.get("page_path")) or "",
                                               what=_text(payload.get("what")),
                                               element_kind=_text(payload.get("element_kind"))))

    def _on_choice(self, handoff: _Handoff, choice: Any) -> None:
        if choice == TAKE_OVER:
            if handoff.taken_over:
                return
            handoff.taken_over = True
            self.holder = "person"
            self._session.leave_dialogs_to_person()
            self._logger.handoff_started(handoff.request.trigger_reason, handoff.token)
            self._spawn(self._announce("HANDOFF_STARTED", trigger_reason=handoff.request.trigger_reason))
            return
        offered = {option.value for option in handoff.request.choices}
        if handoff.taken_over and choice in offered and not handoff.ended.done():
            handoff.ended.set_result((Choice(choice), False))

    def _record(self, handoff: _Handoff, action: PersonAction) -> None:
        handoff.actions.append(action)
        self._logger.person_action(action.kind, action.page_path, what=action.what, element_kind=action.element_kind)

    # ------------------------------------------------------------------ page events

    def _on_page_loaded(self, page: Page) -> None:
        # A new page has no bar: show it again, as the person had it.
        handoff = self._active
        if handoff is not None:
            self._spawn(self._show(page, handoff))

    def _on_navigated(self, frame: Frame) -> None:
        handoff = self._active
        if handoff is None or frame.parent_frame is not None:
            return
        path = _path(frame.url)
        if path != handoff.last_path:
            handoff.last_path = path
            if handoff.taken_over:
                self._record(handoff, PersonAction(kind="page_visited", page_path=path))

    def _on_closed(self, page: Page) -> None:
        handoff = self._active
        if handoff is not None and not handoff.ended.done():
            handoff.ended.set_result((None, True))

    def _on_new_page(self, new_page: Page) -> None:
        # Not followed: the run carries on in its own page, and the tab is closed at hand-back.
        handoff = self._active
        if handoff is None:
            return
        handoff.extra_pages.append(new_page)
        if handoff.taken_over:
            self._record(handoff, PersonAction(kind="tab_opened", page_path=_path(new_page.url)))

    # ------------------------------------------------------------------ helpers

    async def _show(self, page: Page, handoff: _Handoff) -> None:
        request = handoff.request
        context = [f"Task: {self._goal}"]
        if request.step_index is not None:
            where = f"Paused at step {request.step_index}"
            context.append(f"{where}: {request.step_description}" if request.step_description else where)
        content = BarContent(token=handoff.token, why=request.why, context=tuple(context),
                             buttons=tuple(Button(choice.value, LABELS[choice]) for choice in request.choices),
                             deadline=handoff.deadline, taken_over=handoff.taken_over)
        # A page still loading may refuse the script; its load event shows the bar again.
        await _quietly(show_bar(page, content))

    async def _screenshot(self, page: Page, moment: str) -> Optional[str]:
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)
        path = self._screenshots_dir / f"{self._logger.trace_id}_handoff{self._handoffs}_{moment}.png"
        try:
            await page.screenshot(path=str(path))
        except PlaywrightError:
            return None
        return str(path)

    async def _announce(self, event: str, **fields: Any) -> None:
        if self._announcer is None:
            return
        announcement = redact_dict({"event": event, "run_id": self._logger.trace_id, "capability": self._capability,
                                    "goal": self._goal, "at": datetime.now(timezone.utc).isoformat(), **fields})
        try:
            await self._announcer.announce(announcement)
        except Exception:
            # The feed only tells others; control stays in the window, so a failed
            # announcement must never stop the run.
            pass

    def _spawn(self, work) -> None:
        task = asyncio.get_running_loop().create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _settle_tasks(self) -> None:
        # Announcements and bar redraws still running finish before the hand-back is told.
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


async def _quietly(action) -> None:
    try:
        await action
    except PlaywrightError:
        pass


def _path(url: str) -> str:
    return urlsplit(url).path or "/"


def _text(value: Any) -> Optional[str]:
    # Page text as the bar sent it, cut to length; nothing or empty becomes None.
    if value is None:
        return None
    return str(value)[:_MAX_TEXT] or None
