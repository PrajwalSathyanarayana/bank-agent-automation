import json

import pytest

from src.config.settings import settings
from src.observability.logger import RunLogger
from src.replay.locator_resolver import Found, NotFound, find_element
from src.types.step_schema import ActionType, Locator, LocatorType, RetryBudget, Step

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
