import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from flask import request

import blueprints.auth as auth_routes  # the mock bank; tests/conftest.py puts its folder on the import path
from src.config.env import env
from src.config.settings import settings
from src.discovery.artifact_builder import write_artifact
from src.main import BILL_PAY, CONTRACTS
from src.replay.executor import ReplayRequest, replay
from src.safety.integrity import sign
from src.types.artifact_schema import Artifact, ArtifactMetadata
from src.types.result_schema import ExecutionStatus, StepStatus
from src.observability.logger import RunLogger
from src.replay.checks import CheckFailed, CheckValues, verify_shown_text, verify_step_checks
from src.replay.locator_resolver import Found, NotFound, find_element
from src.replay.recovery_engine import Recoveries, interruption_showing, missing_option, outcome_showing, recover
from src.types.result_schema import RecoveryTier
from src.types.step_schema import (
    ActionType,
    CheckpointType,
    Locator,
    LocatorType,
    RetryBudget,
    SafetyTier,
    Step,
    StepCheckpoint,
)

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


# --- known outcomes and interruptions, with bill pay's own declarations ---

BILL = CONTRACTS[BILL_PAY]
POPUP, EXPIRED = BILL.known_interruptions
RUN = CheckValues(text={"member_id": "10234", "payee_name": "Sunbelt Electric Co"}, numbers={"amount": 50.0})


async def _recover(page, interruption, logger, recoveries=None, irreversible_done=False):
    return await recover(page, interruption, RUN, BILL.allowed_paths, logger, recoveries or Recoveries(),
                         irreversible_done=irreversible_done)


@pytest.mark.anyio
async def test_a_declared_outcome_is_recognised_by_its_text(page, dashboard_popup):
    dashboard_popup(False)
    await _sign_in(page)
    await page.goto("/member/10234")
    assert await outcome_showing(page, BILL.known_outcomes) is None
    await page.goto("/member/99999")
    assert (await outcome_showing(page, BILL.known_outcomes)).code == "MEMBER_NOT_FOUND"


@pytest.mark.anyio
@pytest.mark.parametrize("payee, code", [pytest.param("Sunbelt Electric Co", None, id="offered"),
                                         pytest.param("Acme Gas", "PAYEE_NOT_FOUND", id="not offered")])
async def test_a_payee_the_dropdown_doesnt_offer_is_the_declared_outcome(page, dashboard_popup, payee, code):
    dashboard_popup(False)
    await _sign_in(page)
    await page.goto("/member/10234")
    await page.goto("/billpay")
    step = Step(sequence_index=9, action=ActionType.SELECT, description="Choose the payee",
                input_value="{payee_name}", locators=[_css("select[name='payee_id']")])
    found = await missing_option(step, page.locator("select[name='payee_id']"), BILL.known_outcomes,
                                 CheckValues(text={"payee_name": payee}, numbers={}))
    assert (found.code if found else None) == code


@pytest.mark.anyio
async def test_the_declared_interruptions_are_recognised_on_the_real_pages(page, dashboard_popup):
    dashboard_popup(True)
    await _sign_in(page)
    assert (await interruption_showing(page, BILL.known_interruptions, RUN)).code == "PROMO_POPUP"
    await page.context.clear_cookies()
    await page.goto("/dashboard")  # the session is gone: the bank shows its timeout page
    assert (await interruption_showing(page, BILL.known_interruptions, RUN)).code == "SESSION_EXPIRED"


@pytest.mark.anyio
async def test_a_clear_page_has_no_interruption(page, dashboard_popup):
    dashboard_popup(False)
    await _sign_in(page)
    assert await interruption_showing(page, BILL.known_interruptions, RUN) is None


@pytest.mark.anyio
async def test_the_popup_is_cleared_by_its_declared_click_and_logged(page, dashboard_popup, replay_logger):
    dashboard_popup(True)
    await _sign_in(page)
    recovery = await _recover(page, POPUP, replay_logger)
    log = recovery.log
    assert (log.resolved, log.interruption_code, log.tier, recovery.start_over) == (
        True, "PROMO_POPUP", RecoveryTier.TIER_1_RULE, False)
    assert await interruption_showing(page, BILL.known_interruptions, RUN) is None
    assert Path(log.screenshot_path).exists()
    [event] = [line for line in _log_lines(replay_logger) if line["event_type"] == "RECOVERY_EVENT"]
    assert (event["interruption_code"], event["recovery"], event["resolved"]) == ("PROMO_POPUP", "click", True)


@pytest.mark.anyio
async def test_a_clearing_click_that_isnt_safe_is_refused_and_nothing_is_clicked(
    page, dashboard_popup, replay_logger, monkeypatch
):
    dashboard_popup(True)
    await _sign_in(page)
    monkeypatch.setattr("src.replay.recovery_engine.classify", lambda *args, **kwargs: SafetyTier.RISKY)
    recovery = await _recover(page, POPUP, replay_logger)
    assert recovery.log.resolved is False
    assert "only a SAFE click clears an interruption" in recovery.log.details
    assert (await interruption_showing(page, BILL.known_interruptions, RUN)).code == "PROMO_POPUP"


@pytest.mark.anyio
async def test_an_expired_session_asks_the_run_to_start_over(page, replay_logger):
    await page.goto("/dashboard")  # signed out: the session-timeout page
    recovery = await _recover(page, EXPIRED, replay_logger)
    assert (recovery.start_over, recovery.log.resolved) == (True, True)


@pytest.mark.anyio
async def test_a_refused_recovery_does_nothing_and_says_why(page, replay_logger):
    await page.goto("/dashboard")
    recovery = await _recover(page, EXPIRED, replay_logger, irreversible_done=True)
    assert (recovery.start_over, recovery.log.resolved) == (False, False)
    assert "could repeat it" in recovery.log.details


def test_the_same_interruption_is_recovered_at_most_twice_per_run():
    recoveries = Recoveries()
    for _ in range(2):
        assert recoveries.refusal(POPUP, irreversible_done=False) is None
        recoveries.note(POPUP)
    assert recoveries.refusal(POPUP, irreversible_done=False) == "PROMO_POPUP came back after 2 recoveries"


def test_starting_over_happens_once_and_never_after_the_irreversible_step():
    assert "could repeat it" in Recoveries().refusal(EXPIRED, irreversible_done=True)
    recoveries = Recoveries()
    recoveries.note(EXPIRED)
    assert recoveries.refusal(EXPIRED, irreversible_done=False) == "the run already started over once"


# --- a whole replay run, with a bill pay artifact written the way discovery records one ---

def _path(value) -> StepCheckpoint:
    return StepCheckpoint(type=CheckpointType.PAGE_PATH, expected_value=value, timeout_ms=5000)


NEXT = StepCheckpoint(type=CheckpointType.NEXT_STEP_TARGET, timeout_ms=5000)


def _label_cell(label) -> Locator:
    return Locator(type=LocatorType.XPATH, value=f'//td[normalize-space(.)="{label}"]/following-sibling::td[1]',
                   priority=0)


def _bill_pay_steps(search_locators=None, sign_in_lands_on="/dashboard") -> list[Step]:
    risky, irreversible = SafetyTier.RISKY, SafetyTier.IRREVERSIBLE
    return [
        Step(sequence_index=0, action=ActionType.NAVIGATE, description="Open the start page",
             checkpoints=[_path("/login"), NEXT]),
        Step(sequence_index=1, action=ActionType.TYPE, description="Enter the username",
             input_value="{credential:bank_username}", locators=[_css("input[name='username']")], checkpoints=[NEXT]),
        Step(sequence_index=2, action=ActionType.TYPE, description="Enter the password",
             input_value="{credential:bank_password}", locators=[_css("input[name='password']")], checkpoints=[NEXT]),
        Step(sequence_index=3, action=ActionType.CLICK, description="Sign in", locators=[_css("input[value='Log In']")],
             checkpoints=[_path(sign_in_lands_on), NEXT]),
        Step(sequence_index=4, action=ActionType.CLICK, description="Open member search",
             locators=search_locators or [_css("a[href='/search']")], checkpoints=[_path("/search"), NEXT]),
        Step(sequence_index=5, action=ActionType.TYPE, description="Enter the member", input_value="{member_id}",
             locators=[_css("input[name='member_id']")], checkpoints=[NEXT]),
        Step(sequence_index=6, action=ActionType.CLICK, description="Search", locators=[_css("input[value='Search']")],
             checkpoints=[_path("/member/{member_id}"), NEXT]),
        Step(sequence_index=7, action=ActionType.EXTRACT_TEXT, description="Read the balance",
             output_key="checking_balance_before", locators=[_label_cell("Primary Account Balance:")],
             checkpoints=[NEXT]),
        Step(sequence_index=8, action=ActionType.CLICK, description="Open Bill Pay",
             locators=[_css("div.actions a[href='/billpay']")], checkpoints=[_path("/billpay"), NEXT]),
        Step(sequence_index=9, action=ActionType.SELECT, description="Choose the payee", input_value="{payee_name}",
             safety_tier=risky, locators=[_css("select[name='payee_id']")], checkpoints=[NEXT]),
        Step(sequence_index=10, action=ActionType.TYPE, description="Enter the amount", input_value="{amount}",
             safety_tier=risky, locators=[_css("input[name='amount']")], checkpoints=[NEXT]),
        Step(sequence_index=11, action=ActionType.CLICK, description="Continue", safety_tier=risky,
             locators=[_css("input[value='Continue']")], checkpoints=[_path("/billpay/confirm"), NEXT]),
        Step(sequence_index=12, action=ActionType.CLICK, description="Confirm the payment", safety_tier=irreversible,
             locators=[_css("input[value='Confirm Payment']")], checkpoints=[_path("/billpay/confirm"), NEXT]),
        Step(sequence_index=13, action=ActionType.EXTRACT_TEXT, description="Read the new balance",
             output_key="new_checking_balance", locators=[_label_cell("New Checking Balance:")], checkpoints=[NEXT]),
        Step(sequence_index=14, action=ActionType.ASSERT_TEXT, description="Check the receipt",
             input_value="Payment Submitted Successfully", locators=[_css("div.msg-ok")]),
    ]


@pytest.fixture
def storage(tmp_path, monkeypatch) -> Path:
    # Each test reads and writes its own temporary store, never the project's artifacts folder.
    folder = tmp_path / "artifacts"
    monkeypatch.setattr(settings, "artifact_storage_dir", folder)
    return folder


@pytest.fixture
def saved_bill_pay(storage, mock_bank_url):
    """Save a signed bill pay artifact (bill pay's real contract) and return its file."""
    def save(steps=None) -> Path:
        now = datetime.now(timezone.utc)
        artifact = Artifact(
            metadata=ArtifactMetadata(capability=BILL_PAY, description=BILL.description, version="3.0.0",
                                      target_url=f"{mock_bank_url}/login", created_timestamp=now,
                                      last_updated_timestamp=now),
            input_parameters=BILL.input_parameters, output_definitions=BILL.output_definitions,
            credentials=BILL.credentials, known_outcomes=BILL.known_outcomes, allowed_paths=BILL.allowed_paths,
            known_interruptions=BILL.known_interruptions, confirmation_checks=BILL.confirmation_checks,
            steps=steps or _bill_pay_steps(),
        )
        return write_artifact(sign(artifact, env.artifact_signing_key))
    return save


def _asked(member="10234", payee="Sunbelt Electric Co", amount=50.0) -> dict:
    return {"member_id": member, "payee_name": payee, "amount": amount}


async def _replay(logger, **asked):
    return await replay(ReplayRequest(BILL_PAY, _asked(**asked)), logger)


def _amount(text) -> Decimal:
    return Decimal(text)


@pytest.mark.anyio
async def test_a_replay_pays_the_bill_and_returns_both_balances(saved_bill_pay, dashboard_popup, replay_logger):
    dashboard_popup(False)
    saved_bill_pay()
    result = await _replay(replay_logger)
    assert result.status == ExecutionStatus.SUCCESS, result.error
    outputs = result.terminal_outputs
    assert _amount(outputs["new_checking_balance"]) == _amount(outputs["checking_balance_before"]) - Decimal("50.00")
    assert (result.integrity_verified, result.artifact_version, result.run_id) == (True, "3.0.0", replay_logger.trace_id)
    assert [trace.status for trace in result.step_traces] == [StepStatus.PASSED] * 15
    [check] = [line for line in _log_lines(replay_logger) if line["event_type"] == "AUTHORIZATION_CHECKED"]
    assert check["authorized"] is True


@pytest.mark.anyio
async def test_the_same_artifact_pays_for_another_member_payee_and_amount(saved_bill_pay, dashboard_popup,
                                                                        replay_logger):
    dashboard_popup(False)
    saved_bill_pay()
    result = await _replay(replay_logger, member="40412", payee="Desert Valley Water Utility", amount=25.5)
    assert result.status == ExecutionStatus.SUCCESS, result.error
    outputs = result.terminal_outputs
    assert _amount(outputs["new_checking_balance"]) == _amount(outputs["checking_balance_before"]) - Decimal("25.50")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "asked, code",
    [
        pytest.param({"member": "99999"}, "MEMBER_NOT_FOUND", id="no such member"),
        pytest.param({"member": "20567", "amount": 1000.0}, "INSUFFICIENT_FUNDS", id="more than the balance"),
        pytest.param({"member": "30891"}, "ACCOUNT_RESTRICTED", id="a restricted account"),
        pytest.param({"payee": "Acme Gas"}, "PAYEE_NOT_FOUND", id="a payee not in the list"),
    ],
)
async def test_each_known_outcome_is_an_answer_not_a_failure(saved_bill_pay, dashboard_popup, replay_logger,
                                                            asked, code):
    dashboard_popup(False)
    saved_bill_pay()
    result = await _replay(replay_logger, **asked)
    assert result.status == ExecutionStatus.BUSINESS_OUTCOME, result.error
    assert result.outcome.code == code
    assert (result.error, result.failure) == (None, None)
    assert "new_checking_balance" not in (result.terminal_outputs or {})


@pytest.mark.anyio
async def test_the_popup_is_cleared_and_the_run_carries_on(saved_bill_pay, dashboard_popup, replay_logger):
    dashboard_popup(True)
    saved_bill_pay()
    result = await _replay(replay_logger)
    assert result.status == ExecutionStatus.SUCCESS, result.error
    [recovered] = [trace for trace in result.step_traces if trace.status == StepStatus.RECOVERED]
    assert recovered.sequence_index == 4
    assert [log.interruption_code for log in recovered.recovery_logs] == ["PROMO_POPUP"]


@pytest.mark.anyio
async def test_an_expired_session_starts_the_run_over_once(saved_bill_pay, dashboard_popup, replay_logger,
                                                          monkeypatch):
    dashboard_popup(False)
    saved_bill_pay()
    # The bank forgets the session once, when Bill Pay opens; the second time round it doesn't.
    real_check, forgotten = auth_routes.has_valid_session, []

    def forgets_once() -> bool:
        if request.path == "/billpay" and not forgotten:
            forgotten.append(True)
            return False
        return real_check()

    monkeypatch.setattr(auth_routes, "has_valid_session", forgets_once)
    result = await _replay(replay_logger)
    assert result.status == ExecutionStatus.SUCCESS, result.error
    assert any(trace.error_message == "started over: SESSION_EXPIRED" for trace in result.step_traces)


@pytest.mark.anyio
async def test_a_payment_over_the_limit_goes_to_a_person_and_isnt_made(saved_bill_pay, dashboard_popup,
                                                                       replay_logger, monkeypatch):
    dashboard_popup(False)
    saved_bill_pay()
    monkeypatch.setattr(env, "auto_execute_limit", Decimal("10.00"))
    result = await _replay(replay_logger)
    assert (result.status, result.error.code) == (ExecutionStatus.HUMAN_ESCALATED, "OVER_AUTO_LIMIT")
    assert (result.failure.step_index, result.handoff_events[0].trigger_reason) == (12, "OVER_AUTO_LIMIT")
    assert result.failure.expected == "Amount: at most 10.00 USD"
    assert "new_checking_balance" not in result.terminal_outputs


@pytest.mark.anyio
async def test_a_renamed_element_is_found_by_its_fallback_locator(saved_bill_pay, dashboard_popup, replay_logger):
    dashboard_popup(False)
    saved_bill_pay(_bill_pay_steps(search_locators=[_css("#member-search-renamed"),
                                                    Locator(type=LocatorType.CSS, value="a[href='/search']",
                                                            priority=1)]))
    result = await _replay(replay_logger)
    assert result.status == ExecutionStatus.SUCCESS, result.error
    assert result.step_traces[4].locator_priority == 1


@pytest.mark.anyio
async def test_a_check_that_fails_for_no_known_reason_is_a_failure_that_says_where(saved_bill_pay, dashboard_popup,
                                                                                  replay_logger):
    dashboard_popup(False)
    saved_bill_pay(_bill_pay_steps(sign_in_lands_on="/elsewhere"))
    result = await _replay(replay_logger)
    assert (result.status, result.error.code) == (ExecutionStatus.TECHNICAL_FAIL, "CHECK_FAILED")
    failure = result.failure
    assert (failure.step_index, failure.expected, failure.observed) == (3, "page path /elsewhere",
                                                                      "page path /dashboard")
    assert Path(failure.screenshot_path).exists()


@pytest.mark.anyio
async def test_a_hand_edited_artifact_is_refused_before_any_browser_opens(saved_bill_pay, replay_logger, monkeypatch):
    path = saved_bill_pay()
    data = json.loads(path.read_text(encoding="utf-8"))
    data["steps"][10]["input_value"] = "5000"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr("src.replay.executor.BrowserSession", _no_browser)
    result = await _replay(replay_logger)
    assert (result.status, result.error.code) == (ExecutionStatus.HARD_ABORT, "INTEGRITY_CHECK_FAILED")
    assert result.integrity_verified is False


@pytest.mark.anyio
async def test_a_capability_never_discovered_has_nothing_to_replay(storage, replay_logger, monkeypatch):
    monkeypatch.setattr("src.replay.executor.BrowserSession", _no_browser)
    result = await _replay(replay_logger)
    assert (result.status, result.error.code) == (ExecutionStatus.HARD_ABORT, "NO_ARTIFACT")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "inputs, message",
    [
        pytest.param({"member_id": "10234", "payee_name": "Sunbelt Electric Co"}, "missing inputs: amount",
                     id="an input missing"),
        pytest.param({**_asked(), "amount": "50"}, "amount must be a number", id="a number given as text"),
        pytest.param({**_asked(), "memo": "rent"}, "not inputs of this capability: memo", id="an unknown input"),
        pytest.param({**_asked(), "payee_name": "  "}, "payee_name must be non-empty text", id="empty text"),
    ],
)
async def test_inputs_that_dont_fit_the_contract_are_refused_before_any_browser_opens(
    saved_bill_pay, replay_logger, monkeypatch, inputs, message
):
    saved_bill_pay()
    monkeypatch.setattr("src.replay.executor.BrowserSession", _no_browser)
    result = await replay(ReplayRequest(BILL_PAY, inputs), replay_logger)
    assert (result.status, result.error.code, result.error.message) == (ExecutionStatus.HARD_ABORT, "INPUT_INVALID",
                                                                        message)


@pytest.mark.anyio
async def test_only_a_plain_capability_name_is_looked_up(replay_logger, monkeypatch):
    monkeypatch.setattr("src.replay.executor.BrowserSession", _no_browser)
    result = await replay(ReplayRequest("../../etc", _asked()), replay_logger)
    assert (result.status, result.error.code) == (ExecutionStatus.HARD_ABORT, "UNKNOWN_CAPABILITY")


def _no_browser(*args, **kwargs):
    raise AssertionError("no browser should open")
