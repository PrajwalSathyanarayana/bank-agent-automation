"""The live handoff: the operator's control bar in a real page — what it shows, what it
blocks until a person takes over, and what it reports about their actions (never a typed
value)."""
import asyncio
import json
import re
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from websockets.asyncio.client import connect as ws_connect

from src.config.settings import settings
from src.handoff.control_bar import BINDING, TAKE_OVER, BarContent, Button, remove_bar, show_bar
from src.handoff.session_manager import Choice, HandoffManager, HandoffRequest
from src.handoff.watch import describe
from src.handoff.ws_server import FEED_HOST, FeedUnavailable, HandoffFeed
from src.observability.logger import RunLogger
from src.surface.browser import BrowserSession
from src.types.result_schema import HandoffResolution

BUTTONS = (Button("hand_back", "Hand back"), Button("finished", "I finished it"), Button("stop", "Stop the task"))
PAGE = (
    "<button id=\"pay\" onclick=\"document.title = 'paid'\">Pay</button>"
    '<a href="#more" id="more">See more</a>'
    '<table><tr><td id="label">Amount:</td><td><input name="amount"></td></tr></table>'
)
HOST = "[data-bank-agent-handoff]"


def _content(**overrides) -> BarContent:
    fields = dict(
        token="t-1",
        why="the amount is above the bank's limit for automatic payments",
        context=("Task: pay 1050.00 to Sunbelt Electric Co for member 10234", "Paused at step 14: Confirm the payment"),
        buttons=BUTTONS,
        deadline=time.time() + 600,
    )
    fields.update(overrides)
    return BarContent(**fields)


async def _listening(page) -> list[dict]:
    reports: list[dict] = []
    await page.expose_binding(BINDING, lambda source, payload: reports.append(payload))
    return reports


async def _until(condition, timeout_s=5.0):
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "timed out waiting"
        await asyncio.sleep(0.05)


def _bar_buttons(page):
    return page.locator(f"{HOST} button")


async def _take_over(page, reports):
    await page.locator(f"{HOST} button", has_text="Take over").click()
    await _until(lambda: reports)


@pytest.mark.anyio
async def test_until_a_person_takes_over_the_page_is_paused_behind_the_bar(page):
    reports = await _listening(page)
    await page.set_content(PAGE)
    await show_bar(page, _content())
    shown = await page.locator(f"{HOST} .bar").inner_text()
    for text in ("The automation needs a person", "above the bank's limit", "Paused at step 14"):
        assert text in shown
    assert re.search(r"Time left: \d+:\d\d", shown)
    assert await _bar_buttons(page).all_inner_texts() == ["Take over"]
    # A click on the bank's button lands on the veil: the page does nothing, nothing is recorded.
    box = await page.locator("#pay").bounding_box()
    await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    await asyncio.sleep(0.2)
    assert await page.title() == ""
    assert reports == []


@pytest.mark.anyio
async def test_taking_over_reports_the_choice_and_offers_the_buttons_that_fit(page):
    reports = await _listening(page)
    await page.set_content(PAGE)
    await show_bar(page, _content())
    await _take_over(page, reports)
    assert reports == [{"event": "choice", "choice": TAKE_OVER, "token": "t-1"}]
    assert await _bar_buttons(page).all_inner_texts() == ["Hand back", "I finished it", "Stop the task"]
    await page.click("#pay")
    assert await page.title() == "paid"


@pytest.mark.anyio
async def test_a_persons_click_is_reported_by_the_elements_own_wording(page):
    reports = await _listening(page)
    await page.set_content(PAGE)
    await show_bar(page, _content())
    await _take_over(page, reports)
    path = await page.evaluate("location.pathname")
    await page.click("#pay")
    await page.click("#more")
    await page.click("#label")  # plain text does nothing: not recorded
    await _until(lambda: len(reports) >= 3)
    await asyncio.sleep(0.2)
    assert reports[1:] == [
        {"event": "click", "what": "Pay", "element_kind": "button", "page_path": path, "token": "t-1"},
        {"event": "click", "what": "See more", "element_kind": "link", "page_path": path, "token": "t-1"},
    ]


@pytest.mark.anyio
async def test_a_changed_field_is_reported_once_by_its_label_never_its_value(page, mock_bank_url):
    reports = await _listening(page)
    await page.goto(f"{mock_bank_url}/login")
    await show_bar(page, _content())
    await _take_over(page, reports)
    username, password = page.locator('input[name="username"]'), page.locator('input[name="password"]')
    await username.press_sequentially("someone")
    await username.press("Tab")
    await password.press_sequentially("not-the-real-one")
    await password.press("Tab")
    await username.press_sequentially("-else")
    await username.press("Tab")
    await _until(lambda: len(reports) >= 3)
    await asyncio.sleep(0.2)
    assert [(r["event"], r["what"], r["element_kind"], r["page_path"]) for r in reports[1:]] == [
        ("field_changed", "Username:", "text box", "/login"),
        ("field_changed", "Password:", "password box", "/login"),
    ]
    sent = json.dumps(reports)
    assert "someone" not in sent and "not-the-real-one" not in sent


@pytest.mark.anyio
async def test_a_choice_is_sent_once_and_the_buttons_are_disabled(page):
    reports = await _listening(page)
    await page.set_content(PAGE)
    await show_bar(page, _content())
    await _take_over(page, reports)
    await page.locator(f"{HOST} button", has_text="Hand back").click()
    await _until(lambda: len(reports) == 2)
    assert reports[1] == {"event": "choice", "choice": "hand_back", "token": "t-1"}
    assert [await button.is_disabled() for button in await _bar_buttons(page).all()] == [True, True, True]
    assert "Handing back" in await page.locator(f"{HOST} .status").inner_text()


@pytest.mark.anyio
async def test_showing_the_bar_again_replaces_it(page):
    # After a new page the bar comes back as the person had it: taken over, no veil.
    reports = await _listening(page)
    await page.set_content(PAGE)
    await show_bar(page, _content())
    await show_bar(page, _content(taken_over=True))
    assert await page.locator(HOST).count() == 1
    assert await _bar_buttons(page).all_inner_texts() == ["Hand back", "I finished it", "Stop the task"]
    await page.click("#pay")
    await _until(lambda: reports)
    assert [r["event"] for r in reports] == ["click"]


@pytest.mark.anyio
async def test_removing_the_bar_leaves_the_page_and_stops_recording(page):
    reports = await _listening(page)
    await page.set_content(PAGE)
    await show_bar(page, _content())
    await _take_over(page, reports)
    await remove_bar(page)
    assert await page.locator(HOST).count() == 0
    assert await page.evaluate("window.__bankAgentHandoffBar === undefined")
    await page.click("#pay")
    await asyncio.sleep(0.2)
    assert await page.title() == "paid"
    assert [r["choice"] for r in reports] == [TAKE_OVER]
    await remove_bar(page)  # removing twice is harmless


@pytest.mark.anyio
async def test_text_from_the_run_is_shown_as_text_never_as_markup(page):
    await page.set_content(PAGE)
    await show_bar(page, _content(why="<img src=x onerror=\"document.title='ran'\">"))
    await asyncio.sleep(0.2)
    assert "<img src=x" in await page.locator(f"{HOST} .why").inner_text()
    assert await page.title() == ""


@pytest.mark.anyio
async def test_in_a_short_window_the_bar_stays_in_view_and_the_page_moves_below_it(page):
    # A person's window can show less than the page's full height: the bar is at the top,
    # its buttons on screen, and the page sits below it until it goes.
    await page.set_viewport_size({"width": 1024, "height": 560})
    reports = await _listening(page)
    await page.set_content(PAGE)
    before = (await page.locator("#pay").bounding_box())["y"]
    await show_bar(page, _content())
    await _take_over(page, reports)
    bar = await page.locator(f"{HOST} .bar").bounding_box()
    assert bar["y"] == 0 and bar["height"] < 140
    for button in await _bar_buttons(page).all():
        box = await button.bounding_box()
        assert 0 <= box["y"] and box["y"] + box["height"] <= 560
    # The page moves once the bar has drawn its new buttons: waited for, not assumed.
    await page.wait_for_function(
        "([before, height]) => Math.abs(document.querySelector('#pay').getBoundingClientRect().top"
        " - (before + height)) <= 1", arg=[before, bar["height"]], timeout=5_000)
    await remove_bar(page)
    assert (await page.locator("#pay").bounding_box())["y"] == before


@pytest.mark.anyio
async def test_the_clock_says_when_time_is_up(page):
    await page.set_content(PAGE)
    await show_bar(page, _content(deadline=time.time() - 1))
    assert await page.locator(f"{HOST} .clock").inner_text() == "Time is up: the task will stop."


# --- the session manager: the lock, one handoff from request to hand-back ---

GOAL = "For member 10234, pay 1050.00 to Sunbelt Electric Co."
AT_THE_PAYMENT = HandoffRequest("OVER_AUTO_LIMIT", "the amount is above the bank's limit for automatic payments",
                                (Choice.FINISHED, Choice.STOP), step_index=14, step_description="Confirm the payment")
EARLIER = HandoffRequest("CHECK_FAILED", "a screen wasn't the one expected",
                         (Choice.HAND_BACK, Choice.FINISHED, Choice.STOP), step_index=6,
                         step_description="Run the member search")


@pytest.fixture
def handoff_logger(tmp_path, monkeypatch) -> RunLogger:
    # Each test logs to its own temporary folder, never the project's evidence folder.
    monkeypatch.setattr(settings, "evidence_dir", tmp_path)
    return RunLogger("REPLAY", capability="handoff_test")


def _log_lines(logger) -> list[dict]:
    return [json.loads(line) for line in logger.log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _events(logger) -> list[str]:
    return [line["event_type"] for line in _log_lines(logger)]


class _Feed:
    def __init__(self, fail: bool = False) -> None:
        self.heard: list[dict] = []
        self._fail = fail

    async def announce(self, announcement: dict) -> None:
        if self._fail:
            raise ConnectionError("nobody is listening")
        self.heard.append(announcement)


@asynccontextmanager
async def _handing_off(logger, tmp_path, request, *, url=None, content=None, timeout_ms=10_000, feed=None):
    """A session on a page, a handoff requested on it and its bar up; yields the session,
    the manager and the pending request."""
    async with BrowserSession(logger) as session:
        if url is not None:
            await session.page.goto(url)
        else:
            await session.page.set_content(content or PAGE)
        manager = HandoffManager(session, logger, capability="member_servicing_and_bill_pay", goal=GOAL,
                                 screenshots_dir=tmp_path / "shots", announcer=feed, timeout_ms=timeout_ms)
        pending = asyncio.create_task(manager.request(request))
        await session.page.locator(f"{HOST} button", has_text="Take over").wait_for(timeout=10_000)
        try:
            yield session, manager, pending
        finally:
            if not pending.done():
                pending.cancel()


async def _press(page, label: str) -> None:
    await page.locator(f"{HOST} button", has_text=label).click()


async def _taken_over(session, manager) -> None:
    await _press(session.page, "Take over")
    await _until(lambda: manager.holder == "person")


@pytest.mark.anyio
async def test_a_person_takes_over_acts_and_hands_back(mock_bank_url, handoff_logger, tmp_path):
    async with _handing_off(handoff_logger, tmp_path, EARLIER, url=f"{mock_bank_url}/login") as (session, manager, pending):
        assert manager.holder == "automation" and not session.person_in_control
        await _taken_over(session, manager)
        assert session.person_in_control
        username = session.page.locator('input[name="username"]')
        await username.press_sequentially("someone")
        await username.press("Tab")
        await session.page.get_by_role("link", name="Home", exact=True).first.click()
        # A new page: the bar comes back as the person had it, taken over.
        await session.page.locator(f"{HOST} button", has_text="Hand back").wait_for(timeout=10_000)
        await _press(session.page, "Hand back")
        outcome = await asyncio.wait_for(pending, timeout=10)

        assert (outcome.choice, outcome.resolution) == (Choice.HAND_BACK, HandoffResolution.RESUMED)
        assert [(a.kind, a.what, a.page_path) for a in outcome.actions] == [
            ("field_changed", "Username:", "/login"), ("click", "Home", "/login"), ("page_visited", None, "/")]
        record = outcome.telemetry
        assert (record.trigger_reason, record.step_index, record.person_actions) == ("CHECK_FAILED", 6, 3)
        assert record.duration_ms > 0 and record.session_lock_token and record.resolved_timestamp
        assert Path(outcome.pause_screenshot).is_file() and Path(outcome.back_screenshot).is_file()
        # Control is back: no bar, dialogs answered by code, the lock with the automation.
        assert await session.page.locator(HOST).count() == 0
        assert manager.holder == "automation" and not session.person_in_control
    assert _events(handoff_logger) == ["HANDOFF_REQUESTED", "HANDOFF_STARTED", "PERSON_ACTION", "PERSON_ACTION",
                                       "PERSON_ACTION", "HANDOFF_RESOLVED"]
    assert "someone" not in handoff_logger.log_path.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_only_the_offered_choices_with_this_handoffs_token_count(handoff_logger, tmp_path):
    async with _handing_off(handoff_logger, tmp_path, AT_THE_PAYMENT) as (session, manager, pending):
        await _taken_over(session, manager)
        assert await _bar_buttons(session.page).all_inner_texts() == ["I finished it", "Stop the task"]
        token = next(line for line in _log_lines(handoff_logger) if line["event_type"] == "HANDOFF_STARTED")[
            "session_lock_token"]
        report = f"window.{BINDING}"
        await session.page.evaluate(f"{report}({{event: 'choice', choice: 'stop', token: 'not-this-one'}})")
        await session.page.evaluate(f"{report}({{event: 'choice', choice: 'hand_back', token: '{token}'}})")
        await asyncio.sleep(0.3)
        assert not pending.done()
        await _press(session.page, "I finished it")
        outcome = await asyncio.wait_for(pending, timeout=10)
    assert (outcome.choice, outcome.resolution) == (Choice.FINISHED, HandoffResolution.MANUAL_COMPLETED)
    assert outcome.telemetry.person_actions == 0


@pytest.mark.anyio
async def test_stopping_the_task_ends_the_handoff_as_aborted(handoff_logger, tmp_path):
    async with _handing_off(handoff_logger, tmp_path, AT_THE_PAYMENT) as (session, manager, pending):
        await _taken_over(session, manager)
        await _press(session.page, "Stop the task")
        outcome = await asyncio.wait_for(pending, timeout=10)
    assert (outcome.choice, outcome.resolution, outcome.window_closed) == (
        Choice.STOP, HandoffResolution.ABORTED, False)


@pytest.mark.anyio
async def test_time_running_out_before_anyone_takes_over(handoff_logger, tmp_path):
    async with _handing_off(handoff_logger, tmp_path, AT_THE_PAYMENT, timeout_ms=800) as (session, manager, pending):
        outcome = await asyncio.wait_for(pending, timeout=10)
        assert await session.page.locator(HOST).count() == 0
    assert (outcome.choice, outcome.resolution) == (None, HandoffResolution.OPERATOR_TIMED_OUT)
    # Nobody took control, so nobody's actions are counted.
    assert outcome.telemetry.person_actions is None
    assert manager.holder == "automation"
    assert "HANDOFF_STARTED" not in _events(handoff_logger)


@pytest.mark.anyio
async def test_time_running_out_with_a_dialog_open_dismisses_it_first(handoff_logger, tmp_path):
    page_with_pay = "<button id=\"pay\" onclick=\"document.title = String(confirm('Pay now?'))\">Pay</button>"
    async with _handing_off(handoff_logger, tmp_path, AT_THE_PAYMENT, content=page_with_pay,
                            timeout_ms=2_000) as (session, manager, pending):
        await _taken_over(session, manager)
        # The person presses Pay and walks away with the bank's box still open.
        await session.page.evaluate("setTimeout(() => document.querySelector('#pay').click(), 50)")
        outcome = await asyncio.wait_for(pending, timeout=10)
        await session.page.wait_for_function("document.title === 'false'", timeout=5_000)
        assert await session.page.locator(HOST).count() == 0
    assert outcome.resolution == HandoffResolution.OPERATOR_TIMED_OUT
    assert outcome.dialogs_dismissed == ("confirm: Pay now?",)
    assert Path(outcome.back_screenshot).is_file()
    assert "DIALOG_LEFT_FOR_PERSON" in _events(handoff_logger)


@pytest.mark.anyio
async def test_a_tab_the_person_opens_is_closed_at_hand_back(handoff_logger, tmp_path):
    async with _handing_off(handoff_logger, tmp_path, EARLIER) as (session, manager, pending):
        await _taken_over(session, manager)
        await session.page.evaluate("window.open('about:blank')")
        await _until(lambda: len(session.page.context.pages) == 2)
        await _press(session.page, "Hand back")
        outcome = await asyncio.wait_for(pending, timeout=10)
        assert session.page.context.pages == [session.page]
    assert [action.kind for action in outcome.actions] == ["tab_opened"]


@pytest.mark.anyio
async def test_closing_the_window_ends_the_handoff(handoff_logger, tmp_path):
    async with _handing_off(handoff_logger, tmp_path, AT_THE_PAYMENT) as (session, manager, pending):
        await _taken_over(session, manager)
        await session.page.close()
        outcome = await asyncio.wait_for(pending, timeout=10)
    assert (outcome.choice, outcome.resolution, outcome.window_closed) == (None, HandoffResolution.ABORTED, True)
    assert outcome.back_screenshot is None
    assert manager.holder == "automation"


@pytest.mark.anyio
async def test_the_feed_hears_each_step_with_its_context(handoff_logger, tmp_path):
    feed = _Feed()
    async with _handing_off(handoff_logger, tmp_path, AT_THE_PAYMENT, feed=feed) as (session, manager, pending):
        await _taken_over(session, manager)
        await _press(session.page, "Stop the task")
        await asyncio.wait_for(pending, timeout=10)
    assert [heard["event"] for heard in feed.heard] == ["HANDOFF_REQUESTED", "HANDOFF_STARTED", "HANDOFF_RESOLVED"]
    requested = feed.heard[0]
    assert (requested["run_id"], requested["capability"], requested["goal"]) == (
        handoff_logger.trace_id, "member_servicing_and_bill_pay", GOAL)
    assert (requested["step_index"], requested["why"]) == (14, AT_THE_PAYMENT.why)
    assert requested["buttons"] == ["I finished it", "Stop the task"]
    assert feed.heard[-1]["resolution"] == "ABORTED"


@pytest.mark.anyio
async def test_a_feed_that_fails_never_stops_the_handoff(handoff_logger, tmp_path):
    async with _handing_off(handoff_logger, tmp_path, AT_THE_PAYMENT, feed=_Feed(fail=True)) as (
            session, manager, pending):
        await _taken_over(session, manager)
        await _press(session.page, "I finished it")
        outcome = await asyncio.wait_for(pending, timeout=10)
    assert outcome.resolution == HandoffResolution.MANUAL_COMPLETED


@pytest.mark.anyio
async def test_one_handoff_at_a_time(handoff_logger, tmp_path):
    async with _handing_off(handoff_logger, tmp_path, AT_THE_PAYMENT) as (session, manager, pending):
        with pytest.raises(RuntimeError, match="already in progress"):
            await manager.request(EARLIER)


def test_a_handoff_offers_at_least_one_choice_each_once():
    for choices in ((), (Choice.STOP, Choice.STOP)):
        with pytest.raises(ValueError, match="at least one choice"):
            HandoffRequest("STUCK", "stuck", choices)


# --- the feed: announcements for anyone not looking at the window ---

REQUESTED = {"event": "HANDOFF_REQUESTED", "run_id": "c61de331-b704", "at": "2026-09-15T04:10:06+00:00",
             "capability": "member_servicing_and_bill_pay", "goal": GOAL, "step_index": 12,
             "why": "the amount is above the bank's limit for automatic payments"}
RESOLVED = {"event": "HANDOFF_RESOLVED", "run_id": "c61de331-b704", "at": "2026-09-15T04:10:31+00:00",
            "resolution": "MANUAL_COMPLETED", "window_closed": False}


async def _received(listener) -> dict:
    return json.loads(await asyncio.wait_for(listener.recv(), timeout=5))


@pytest.mark.anyio
async def test_every_listener_hears_each_announcement():
    async with HandoffFeed(0) as feed:
        url = f"ws://{FEED_HOST}:{feed.port}"
        async with ws_connect(url) as first, ws_connect(url) as second:
            await _until(lambda: feed.listening == 2)
            await feed.announce(REQUESTED)
            assert [await _received(first), await _received(second)] == [REQUESTED, REQUESTED]


@pytest.mark.anyio
async def test_a_listener_joining_late_is_told_about_the_open_handoff_only():
    async with HandoffFeed(0) as feed:
        url = f"ws://{FEED_HOST}:{feed.port}"
        await feed.announce(REQUESTED)
        async with ws_connect(url) as late:
            assert await _received(late) == REQUESTED
        await feed.announce(RESOLVED)
        async with ws_connect(url) as after:
            # Nothing is open any more, so a new listener hears nothing until the next handoff.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(after.recv(), timeout=0.5)


@pytest.mark.anyio
async def test_what_a_listener_sends_is_ignored_and_one_leaving_changes_nothing():
    async with HandoffFeed(0) as feed:
        url = f"ws://{FEED_HOST}:{feed.port}"
        async with ws_connect(url) as meddler:
            await _until(lambda: feed.listening == 1)
            await meddler.send(json.dumps({"event": "choice", "choice": "stop"}))
        await _until(lambda: feed.listening == 0)
        await feed.announce(REQUESTED)  # nobody listening: no error


@pytest.mark.anyio
async def test_a_busy_port_is_reported_as_the_feed_being_unavailable():
    with socket.socket() as taken:
        taken.bind((FEED_HOST, 0))
        taken.listen()
        with pytest.raises(FeedUnavailable, match="couldn't listen on port"):
            async with HandoffFeed(taken.getsockname()[1]):
                pass


@pytest.mark.parametrize(
    "announcement, line",
    [
        pytest.param(REQUESTED, f"[04:10:06] A PERSON IS NEEDED at step 12 (run c61de331): the amount is above the "
                                f"bank's limit for automatic payments. Task: {GOAL} Take over in the run's browser "
                                "window.", id="a person is needed"),
        pytest.param({"event": "HANDOFF_STARTED", "run_id": "c61de331-b704", "at": "2026-09-15T04:10:09+00:00"},
                     "[04:10:09] Taken over by a person (run c61de331).", id="taken over"),
        pytest.param(RESOLVED, "[04:10:31] Control is back with the automation (run c61de331): finished by the "
                               "person.", id="finished"),
        pytest.param({**RESOLVED, "resolution": "ABORTED", "window_closed": True},
                     "[04:10:31] Control is back with the automation (run c61de331): stopped: the window was "
                     "closed.", id="window closed"),
    ],
)
def test_the_watcher_prints_each_announcement_as_a_sentence(announcement, line):
    assert describe(announcement) == line
