import json

import pytest

from src.config.env import env
from src.config.settings import settings
from src.observability.logger import RunLogger
from src.replay.checks import CheckFailed, CheckValues, verify_shown_text, verify_step_checks
from src.replay.locator_resolver import Found, NotFound, find_element
from src.types.step_schema import ActionType, CheckpointType, Locator, LocatorType, RetryBudget, Step, StepCheckpoint

QUIRKS_DOCTYPE = '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">'


async def _set_page(page, body) -> None:
    # Same doctype as the bank's pages, so hand-written pages render in quirks mode too.
    await page.set_content(f"{QUIRKS_DOCTYPE}<html><body>{body}</body></html>")


@pytest.fixture
def replay_logger(tmp_path, monkeypatch) -> RunLogger:
    # Each test logs to its own temporary folder, never the project's evidence folder.
    monkeypatch.setattr(settings, "evidence_dir", tmp_path)
    return RunLogger("REPLAY", capability="replay_test")


def _log_lines(logger) -> list[dict]:
    return [json.loads(line) for line in logger.log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- finding a step's element ---

def _step(*values, budget=None) -> Step:
    # A short poll keeps these tests fast; the slow-page test sets its own budget.
    locators = [Locator(type=LocatorType.CSS, value=value, priority=number) for number, value in enumerate(values)]
    return Step(sequence_index=4, action=ActionType.CLICK, description="Open member search", locators=locators,
                retry_budget=budget or RetryBudget(max_attempts=3, poll_interval_ms=50))


@pytest.mark.anyio
async def test_the_primary_locator_wins_when_it_matches_one_element(page, replay_logger):
    await _set_page(page, '<a id="search" href="/search">Member Search</a>')
    found = await find_element(page, _step("#search", "a[href='/search']"), {}, replay_logger)
    assert isinstance(found, Found)
    assert (found.priority, found.attempts) == (0, 1)
    assert await found.element.get_attribute("id") == "search"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "primary",
    [pytest.param("#renamed", id="the primary matches nothing"),
     pytest.param("a", id="the primary matches two elements")],
)
async def test_a_fallback_is_used_when_the_primary_doesnt_match_exactly_one(page, replay_logger, primary):
    await _set_page(page, '<a id="search" href="/search">Member Search</a><a href="/billpay">Bill Pay</a>')
    found = await find_element(page, _step(primary, "a[href='/search']"), {}, replay_logger)
    assert (found.priority, await found.element.get_attribute("id")) == (1, "search")


@pytest.mark.anyio
async def test_a_placeholder_is_filled_with_this_runs_value(page, replay_logger):
    await _set_page(page, '<a id="accounts" href="/member/10234/accounts">View All Accounts</a>')
    step = _step('a[href="/member/{member_id}/accounts"]')
    assert isinstance(await find_element(page, step, {"member_id": "10234"}, replay_logger), Found)
    assert isinstance(await find_element(page, step, {"member_id": "40412"}, replay_logger), NotFound)


@pytest.mark.anyio
async def test_a_locator_that_cant_be_filled_is_passed_over(page, replay_logger):
    await _set_page(page, '<a id="accounts" href="/member/10234/accounts">View All Accounts</a>')
    found = await find_element(page, _step('a[href="/member/{member_id}/accounts"]', "#accounts"), {}, replay_logger)
    assert found.priority == 1


@pytest.mark.anyio
async def test_an_element_drawn_late_is_found_on_a_later_attempt(page, replay_logger):
    await _set_page(page, "<div id='slot'></div><script>setTimeout(() => { document.getElementById('slot')"
                          ".innerHTML = '<a id=\"late\" href=\"/search\">Member Search</a>'; }, 300);</script>")
    found = await find_element(page, _step("#late", budget=RetryBudget(max_attempts=3, poll_interval_ms=500)),
                               {}, replay_logger)
    assert isinstance(found, Found)
    assert found.attempts > 1


@pytest.mark.anyio
async def test_nothing_is_guessed_when_no_locator_matches_exactly_one(page, replay_logger):
    await _set_page(page, '<a href="/a">A</a><a href="/b">B</a>')
    result = await find_element(page, _step("#gone", "a", 'a[href="/member/{member_id}"]'), {}, replay_logger)
    assert isinstance(result, NotFound)
    assert result.observed() == ("none of its 3 locators matched exactly one element after 3 attempts "
                                 "(#0 matched nothing, #1 matched 2, #2 can't be filled)")
    lines = [line for line in _log_lines(replay_logger) if line["event_type"] == "LOCATOR_EVALUATED"]
    assert [(line["attempt"], line["priority"]) for line in lines] == [
        (attempt, priority) for attempt in (1, 2, 3) for priority in (0, 1, 2)]
    assert [line["matches"] for line in lines[:3]] == [0, 2, None]
    assert {line["step_index"] for line in lines} == {4}


@pytest.mark.anyio
async def test_a_step_without_locators_is_not_looked_for(replay_logger):
    step = Step(sequence_index=0, action=ActionType.NAVIGATE, description="Open the start page")
    with pytest.raises(ValueError, match="no element to find"):
        await find_element(None, step, {}, replay_logger)


# --- a step's checks ---

VALUES = CheckValues(text={"member_id": "10234"}, numbers={"amount": 1050.0})


def _css(value) -> Locator:
    return Locator(type=LocatorType.CSS, value=value, priority=0)


def _check(kind, expected=None, target=None, timeout_ms=300) -> StepCheckpoint:
    # A short timeout keeps the failing cases fast; the real default is 10 s.
    return StepCheckpoint(type=kind, expected_value=expected, target_locator=target, timeout_ms=timeout_ms)


def _checked_step(*checkpoints) -> Step:
    return Step(sequence_index=6, action=ActionType.CLICK, description="Run the member search",
                locators=[_css("#go")], checkpoints=list(checkpoints))


async def _sign_in(page) -> None:
    await page.goto("/login")
    await page.fill("input[name='username']", env.mock_bank_username)
    await page.fill("input[name='password']", env.mock_bank_password.get_secret_value())
    await page.click("input[type='submit']")
    await page.wait_for_url("**/dashboard")


@pytest.mark.anyio
async def test_page_checks_hold_on_the_page_reached(page, dashboard_popup):
    dashboard_popup(False)
    await _sign_in(page)
    await page.goto("/member/10234")
    step = _checked_step(_check(CheckpointType.PAGE_PATH, "/member/{member_id}"),
                         _check(CheckpointType.PAGE_TITLE, await page.title()),
                         _check(CheckpointType.URL_CONTAINS, "/member/{member_id}"))
    assert await verify_step_checks(page, step, None, VALUES) is None


@pytest.mark.anyio
async def test_a_page_path_that_doesnt_hold_says_what_was_expected_and_seen(page, dashboard_popup):
    dashboard_popup(False)
    await _sign_in(page)
    await page.goto("/member/99999")  # the bank shows its not-found page instead
    failed = await verify_step_checks(page, _checked_step(_check(CheckpointType.PAGE_PATH, "/member/{member_id}")),
                                      None, CheckValues(text={"member_id": "99999"}, numbers={}))
    assert failed == CheckFailed("page path /member/99999", "page path /member/not-found")


@pytest.mark.anyio
async def test_a_wrong_title_is_reported_with_the_title_seen(page):
    await page.goto("/login")
    failed = await verify_step_checks(page, _checked_step(_check(CheckpointType.PAGE_TITLE, "Dashboard")), None, VALUES)
    assert failed == CheckFailed('page title "Dashboard"', f'page title "{await page.title()}"')


@pytest.mark.anyio
@pytest.mark.parametrize(
    "value, holds",
    [pytest.param("input[name='username']", True, id="present"),
     pytest.param("input[name='member_id']", False, id="not on this page")],
)
async def test_the_next_steps_element_must_be_on_the_page(page, value, holds):
    await page.goto("/login")
    next_step = Step(sequence_index=7, action=ActionType.CLICK, description="Next", locators=[_css(value)])
    failed = await verify_step_checks(page, _checked_step(_check(CheckpointType.NEXT_STEP_TARGET)), next_step, VALUES)
    assert failed == (None if holds else CheckFailed("the element for step 7 on the page",
                                                     "none of its locators found it"))


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body, holds",
    [pytest.param("<div id='r'>Amount: $1,050.00</div>", True, id="the amount in another form"),
     pytest.param("<div id='r'>Amount: $1,050.01</div>", False, id="a different amount"),
     pytest.param("<div id='r' style='display:none'>Amount: $1,050.00</div>", False, id="hidden")],
)
async def test_a_checking_step_needs_its_element_to_show_the_text(page, body, holds):
    await _set_page(page, body)
    failed = await verify_shown_text(page.locator("#r"), "Amount: ${amount}", VALUES, 300)
    assert (failed is None) is holds


@pytest.mark.anyio
async def test_a_checking_step_that_fails_says_what_was_shown(page):
    await _set_page(page, "<div id='r'>Amount: $1,050.01</div>")
    failed = await verify_shown_text(page.locator("#r"), "Amount: ${amount}", VALUES, 300)
    assert failed == CheckFailed('"Amount: $1050" shown', 'shown: "Amount: $1,050.01"')


@pytest.mark.anyio
async def test_a_check_waits_for_a_page_still_drawing(page):
    await _set_page(page, "<div id='r'></div><script>setTimeout(() => { document.getElementById('r')"
                          ".textContent = 'Payment Submitted Successfully'; }, 300);</script>")
    assert await verify_shown_text(page.locator("#r"), "Payment Submitted Successfully", VALUES, 2000) is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "checkpoint, holds",
    [
        pytest.param(_check(CheckpointType.ELEMENT_VISIBLE, target=_css("#notice")), True, id="visible"),
        pytest.param(_check(CheckpointType.ELEMENT_VISIBLE, target=_css("#hidden")), False, id="not visible"),
        pytest.param(_check(CheckpointType.TEXT_MATCH, "member {member_id}", target=_css("#notice")), True,
                     id="text with this run's value"),
        pytest.param(_check(CheckpointType.VALUE_EQUALS, "{amount}", target=_css("input[name='amount']")), True,
                     id="a field holding this run's amount"),
        pytest.param(_check(CheckpointType.VALUE_EQUALS, "50", target=_css("input[name='amount']")), False,
                     id="a field holding another value"),
    ],
)
async def test_element_checks_hold_only_for_what_they_name(page, checkpoint, holds):
    await _set_page(page, "<p id='notice'>Found member 10234</p><p id='hidden' style='display:none'>x</p>"
                          "<input name='amount' value='1050'>")
    failed = await verify_step_checks(page, _checked_step(checkpoint), None, VALUES)
    assert (failed is None) is holds
