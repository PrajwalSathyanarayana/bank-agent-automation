import asyncio
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
from src.handoff.session_manager import OperatorSetup
from src.main import BILL_PAY, CONTRACTS
from src.replay.executor import ReplayRequest, replay
from src.safety.integrity import sign
from src.surface.browser import BrowserSession
from src.types.artifact_schema import Artifact, ArtifactMetadata
from src.types.result_schema import ExecutionStatus, HandoffResolution, StepStatus
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
                                                     "none of its locators found it", next_step=True))


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


def _bill_pay_steps(search_locators=None, sign_in_lands_on="/dashboard",
                    confirm_lands_on="/billpay/confirm") -> list[Step]:
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
             locators=[_css("input[value='Confirm Payment']")], checkpoints=[_path(confirm_lands_on), NEXT]),
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
    def save(steps=None, bank_url=None) -> Path:
        now = datetime.now(timezone.utc)
        artifact = Artifact(
            metadata=ArtifactMetadata(capability=BILL_PAY, description=BILL.description, version="3.0.0",
                                      target_url=f"{bank_url or mock_bank_url}/login", created_timestamp=now,
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


def _lower_first(text: str) -> str:
    return text[0].lower() + text[1:]


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
    assert result.irreversible_step == "completed"
    assert result.summary == (f"Paid $50.00 to Sunbelt Electric Co for member 10234. Checking balance "
                              f"${_amount(outputs['checking_balance_before']):,.2f} before, "
                              f"${_amount(outputs['new_checking_balance']):,.2f} after.")


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
    assert result.irreversible_step == "not_reached"
    assert result.summary.endswith(f"Not done: {_lower_first(result.outcome.description)}. No payment was made.")


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
    assert result.irreversible_step == "not_reached"
    assert result.summary == ("Pay $50.00 to Sunbelt Electric Co for member 10234. A person needs to decide: the "
                              "amount is above the bank's limit for automatic payments. No payment was made.")


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
    assert result.irreversible_step is None  # the artifact was never trusted, so nothing is known about it
    assert result.summary.endswith("Stopped: the saved procedure failed its security check (INTEGRITY_CHECK_FAILED).")


@pytest.mark.anyio
async def test_a_run_that_stops_after_the_payment_click_says_it_may_have_gone_through(saved_bill_pay,
                                                                                    dashboard_popup,
                                                                                    replay_logger):
    # The page after Confirm Payment isn't the one the artifact expects, so what followed the
    # click is never confirmed: the payment may or may not have gone through.
    dashboard_popup(False)
    saved_bill_pay(_bill_pay_steps(confirm_lands_on="/elsewhere"))
    result = await _replay(replay_logger, member="40412", payee="Horizon Credit Card Services", amount=5.0)
    assert (result.status, result.error.code, result.failure.step_index) == (
        ExecutionStatus.TECHNICAL_FAIL, "CHECK_FAILED", 12)
    assert result.irreversible_step == "unknown"
    assert result.summary.endswith("The payment may have gone through: check before trying again.")


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


# --- repeatability: the same request, the same path and answer every time ---

def _path_taken(result) -> list[tuple]:
    # Each step replay ran, how it ended and which of its locators found the element.
    return [(trace.sequence_index, trace.status, trace.locator_priority) for trace in result.step_traces]


@pytest.mark.anyio
async def test_the_same_request_replays_to_the_same_answer_every_time(saved_bill_pay, dashboard_popup):
    dashboard_popup(False)
    saved_bill_pay()
    results = [await _replay(RunLogger("REPLAY", capability=BILL_PAY), payee="Acme Gas") for _ in range(3)]
    answers = {(result.status, result.outcome.code, result.summary, result.irreversible_step) for result in results}
    assert answers == {(ExecutionStatus.BUSINESS_OUTCOME, "PAYEE_NOT_FOUND", results[0].summary, "not_reached")}
    assert all(_path_taken(result) == _path_taken(results[0]) for result in results)


@pytest.mark.anyio
async def test_repeated_payments_take_the_same_path_and_move_exactly_the_amount_each_time(saved_bill_pay,
                                                                                         dashboard_popup):
    dashboard_popup(False)
    saved_bill_pay()
    results = [await _replay(RunLogger("REPLAY", capability=BILL_PAY), member="40412", amount=10.0)
               for _ in range(3)]
    assert [result.status for result in results] == [ExecutionStatus.SUCCESS] * 3
    assert all(_path_taken(result) == _path_taken(results[0]) for result in results)
    balances = [(_amount(result.terminal_outputs["checking_balance_before"]),
                 _amount(result.terminal_outputs["new_checking_balance"])) for result in results]
    for before, after in balances:
        assert after == before - Decimal("10.00")  # exactly the amount, never twice
    for (_, after), (next_before, _) in zip(balances, balances[1:]):
        assert next_before == after  # each run starts where the last one ended


# --- a changed bank: the mock bank's test switches ---

# Step 4 exactly as discovery recorded it in v3.0.0: the link's text, its address, its menu position.
DISCOVERED_SEARCH_LOCATORS = [
    Locator(type=LocatorType.TEXT_CONTENT, value="Member Search", priority=0),
    Locator(type=LocatorType.CSS, value='a[href="/search"]', priority=1),
    Locator(type=LocatorType.CSS, value="td.nav > div:nth-of-type(3) > a:nth-of-type(1)", priority=2),
]


@pytest.mark.anyio
async def test_the_discovered_locators_survive_a_relabelled_menu(saved_bill_pay, switched_bank, dashboard_popup,
                                                                replay_logger):
    dashboard_popup(False)
    saved_bill_pay(_bill_pay_steps(search_locators=DISCOVERED_SEARCH_LOCATORS),
                   bank_url=switched_bank(renamed_menu=True))
    result = await _replay(replay_logger)
    assert result.status == ExecutionStatus.SUCCESS, result.error
    assert result.step_traces[4].locator_priority == 1
    tried = [(line["priority"], line["matches"]) for line in _log_lines(replay_logger)
             if line["event_type"] == "LOCATOR_EVALUATED" and line["step_index"] == 4]
    assert tried == [(0, 0), (1, 1)]  # the text no longer matches; the address still does


@pytest.mark.anyio
async def test_a_slow_bank_is_waited_for(saved_bill_pay, switched_bank, dashboard_popup, replay_logger):
    dashboard_popup(False)
    saved_bill_pay(bank_url=switched_bank(slow_pages_ms=1000))
    result = await _replay(replay_logger, member="40412", payee="Canyon Ridge Mortgage Co", amount=10.0)
    assert result.status == ExecutionStatus.SUCCESS, result.error
    assert result.duration_ms >= 5000  # every page arrived a second late, and replay waited


@pytest.mark.anyio
async def test_a_bank_slower_than_the_page_limit_ends_in_a_clear_failure(saved_bill_pay, switched_bank,
                                                                        dashboard_popup, replay_logger,
                                                                        monkeypatch):
    # A click waits for the page it leads to, so what limits a slow bank is the page limit
    # (30 s); here it is 1.5 s and every page comes 2.5 s late, so the test stays quick.
    dashboard_popup(False)
    monkeypatch.setattr(settings, "discovery_page_action_timeout_ms", 1500)
    saved_bill_pay(bank_url=switched_bank(slow_pages_ms=2500))
    result = await _replay(replay_logger)
    assert (result.status, result.error.code) == (ExecutionStatus.TECHNICAL_FAIL, "PAGE_TIMEOUT")
    assert (result.failure.step_index, result.failure.expected, result.failure.observed) == (
        0, "the start page within 1.5 s", "it didn't arrive in time")


# --- a person in the loop: the live handoff during a replay ---

HANDOFF_BAR = "[data-bank-agent-handoff]"


@pytest.fixture
def replay_sessions(monkeypatch) -> list:
    """Every browser session replay opens, so a test can act as the person at its window."""
    sessions = []

    class _Kept(BrowserSession):
        async def __aenter__(self):
            session = await super().__aenter__()
            sessions.append(session)
            return session

    monkeypatch.setattr("src.replay.executor.BrowserSession", _Kept)
    return sessions


async def _until(condition, timeout_s=30.0):
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "timed out waiting"
        await asyncio.sleep(0.05)


async def _at_the_bar(sessions, button):
    """The person, at the paused run's window once its bar offers this button."""
    await _until(lambda: sessions)
    page = sessions[-1].page
    await page.locator(f"{HANDOFF_BAR} button", has_text=button).wait_for(timeout=30_000)
    return page


async def _press(page, label):
    await page.locator(f"{HANDOFF_BAR} button", has_text=label).click()


async def _bar_offers(page, label) -> list[str]:
    await page.locator(f"{HANDOFF_BAR} button", has_text=label).wait_for(timeout=10_000)
    return await page.locator(f"{HANDOFF_BAR} button").all_inner_texts()


def _with_person(logger, timeout_ms=60_000, **asked):
    return asyncio.create_task(replay(ReplayRequest(BILL_PAY, _asked(**asked)), logger,
                                      operator=OperatorSetup(timeout_ms=timeout_ms)))


async def _confirm_payment_as_the_person(page):
    # The person clicks Confirm Payment and presses OK in the bank's own box; the run's own
    # dialog handler leaves that box to them.
    async def ok(dialog):
        await dialog.accept()

    page.once("dialog", ok)
    await page.click("input[value='Confirm Payment']")
    await page.locator("div.msg-ok").wait_for(timeout=10_000)


@pytest.mark.anyio
async def test_a_person_confirms_a_payment_over_the_limit_and_replay_reads_the_receipt(
    saved_bill_pay, dashboard_popup, replay_logger, replay_sessions, monkeypatch
):
    dashboard_popup(False)
    saved_bill_pay()
    monkeypatch.setattr(env, "auto_execute_limit", Decimal("10.00"))
    run = _with_person(replay_logger)
    page = await _at_the_bar(replay_sessions, "Take over")
    await _press(page, "Take over")
    # At the payment there is nothing left to hand back: only finishing or stopping fits.
    assert await _bar_offers(page, "I finished it") == ["I finished it", "Stop the task"]
    await _confirm_payment_as_the_person(page)
    await _at_the_bar(replay_sessions, "I finished it")
    await _press(page, "I finished it")
    result = await asyncio.wait_for(run, timeout=60)

    assert (result.status, result.error, result.irreversible_step) == (
        ExecutionStatus.HUMAN_ESCALATED, None, "completed")
    [handoff] = result.handoff_events
    assert (handoff.trigger_reason, handoff.step_index, handoff.resolution, handoff.person_actions) == (
        "OVER_AUTO_LIMIT", 12, HandoffResolution.MANUAL_COMPLETED, 1)
    before, after = (_amount(result.terminal_outputs[key]) for key in ("checking_balance_before",
                                                                      "new_checking_balance"))
    assert after == before - Decimal("50.00")
    last = result.step_traces[-1]
    assert (last.sequence_index, last.status, last.recovery_logs[0].details) == (
        12, StepStatus.RECOVERED, "finished by a person")
    assert result.summary == (f"Paid $50.00 to Sunbelt Electric Co for member 10234 (confirmed by a person). "
                              f"Checking balance ${before:,.2f} before, ${after:,.2f} after.")


@pytest.mark.anyio
async def test_a_person_who_stops_the_task_leaves_the_payment_unmade(saved_bill_pay, dashboard_popup, replay_logger,
                                                                     replay_sessions, monkeypatch):
    dashboard_popup(False)
    saved_bill_pay()
    monkeypatch.setattr(env, "auto_execute_limit", Decimal("10.00"))
    run = _with_person(replay_logger)
    page = await _at_the_bar(replay_sessions, "Take over")
    await _press(page, "Take over")
    await _bar_offers(page, "Stop the task")
    await _press(page, "Stop the task")
    result = await asyncio.wait_for(run, timeout=60)
    assert (result.status, result.error.code, result.failure.step_index) == (
        ExecutionStatus.HUMAN_ESCALATED, "OVER_AUTO_LIMIT", 12)
    assert [(h.resolution, h.person_actions) for h in result.handoff_events] == [(HandoffResolution.ABORTED, 0)]
    assert result.irreversible_step == "not_reached"
    assert result.summary == ("Pay $50.00 to Sunbelt Electric Co for member 10234. A person was needed (the amount "
                              "is above the bank's limit for automatic payments) and stopped the task. "
                              "No payment was made.")


@pytest.mark.anyio
async def test_when_no_one_takes_over_in_time_the_run_ends_and_says_so(saved_bill_pay, dashboard_popup,
                                                                      replay_logger, monkeypatch):
    dashboard_popup(False)
    saved_bill_pay()
    monkeypatch.setattr(env, "auto_execute_limit", Decimal("10.00"))
    result = await asyncio.wait_for(_with_person(replay_logger, timeout_ms=1_500), timeout=60)
    assert (result.status, result.error.code) == (ExecutionStatus.HUMAN_ESCALATED, "OVER_AUTO_LIMIT")
    [handoff] = result.handoff_events
    assert (handoff.resolution, handoff.person_actions) == (HandoffResolution.OPERATOR_TIMED_OUT, None)
    assert result.summary == ("Pay $50.00 to Sunbelt Electric Co for member 10234. A person was needed (the amount "
                              "is above the bank's limit for automatic payments), but the time for a person ran out. "
                              "No payment was made.")


@pytest.mark.anyio
async def test_after_a_person_does_a_step_replay_carries_on_and_leaves_the_payment_to_them(
    saved_bill_pay, dashboard_popup, replay_logger, replay_sessions
):
    dashboard_popup(False)
    saved_bill_pay(_bill_pay_steps(search_locators=[_css("#member-search-gone")]))
    run = _with_person(replay_logger)
    page = await _at_the_bar(replay_sessions, "Take over")
    await _press(page, "Take over")
    # Paused early, so all three fit.
    assert await _bar_offers(page, "Hand back") == ["Hand back", "I finished it", "Stop the task"]
    await page.click("a[href='/search']")  # the person opens member search themselves
    await _at_the_bar(replay_sessions, "Hand back")  # the bar, back on the new page
    await _press(page, "Hand back")
    # Replay carries on by itself up to the payment, which it now leaves to a person.
    await _at_the_bar(replay_sessions, "Take over")
    await _press(page, "Take over")
    assert await _bar_offers(page, "I finished it") == ["I finished it", "Stop the task"]
    await _confirm_payment_as_the_person(page)
    await _at_the_bar(replay_sessions, "I finished it")
    await _press(page, "I finished it")
    result = await asyncio.wait_for(run, timeout=60)

    assert [(h.trigger_reason, h.step_index, h.resolution) for h in result.handoff_events] == [
        ("LOCATOR_NOT_FOUND", 4, HandoffResolution.RESUMED),
        ("PERSON_HAD_CONTROL", 12, HandoffResolution.MANUAL_COMPLETED)]
    # Step 3 did its part; only step 4's element was missing, so the pause was at step 4.
    sign_in = result.step_traces[3]
    assert (sign_in.status, sign_in.error_message) == (
        StepStatus.PASSED, "step 4's element wasn't found here; left to that step")
    search = result.step_traces[4]
    assert (search.sequence_index, search.status) == (4, StepStatus.RECOVERED)
    assert [(log.tier, log.details) for log in search.recovery_logs] == [
        (RecoveryTier.TIER_3_HANDOFF, "done by a person")]
    # Replay did the steps between the two handoffs itself.
    assert [trace.status for trace in result.step_traces[5:12]] == [StepStatus.PASSED] * 7
    assert (result.status, result.irreversible_step) == (ExecutionStatus.HUMAN_ESCALATED, "completed")
    assert "(confirmed by a person)" in result.summary


@pytest.mark.anyio
async def test_a_step_handed_back_undone_is_tried_once_more_then_fails_as_usual(
    saved_bill_pay, dashboard_popup, replay_logger, replay_sessions
):
    dashboard_popup(False)
    saved_bill_pay(_bill_pay_steps(search_locators=[_css("#member-search-gone")]))
    run = _with_person(replay_logger)
    page = await _at_the_bar(replay_sessions, "Take over")
    await _press(page, "Take over")
    await _bar_offers(page, "Hand back")
    await _press(page, "Hand back")  # handed back without doing anything
    result = await asyncio.wait_for(run, timeout=60)
    assert (result.status, result.error.code, result.failure.step_index) == (
        ExecutionStatus.TECHNICAL_FAIL, "LOCATOR_NOT_FOUND", 4)
    # One handoff per step: the second failure there ends the run.
    assert [h.resolution for h in result.handoff_events] == [HandoffResolution.RESUMED]
    assert "A person had control during the run." in result.summary
    assert result.irreversible_step == "not_reached"


@pytest.mark.anyio
async def test_unattended_a_missing_element_is_reported_by_the_check_before_it(saved_bill_pay, dashboard_popup,
                                                                              replay_logger):
    # With no person the run stops where the check failed: step 3's check that step 4's
    # element is on the page. With a person, the pause is at step 4 itself (tests above).
    dashboard_popup(False)
    saved_bill_pay(_bill_pay_steps(search_locators=[_css("#member-search-gone")]))
    result = await _replay(replay_logger)
    assert (result.status, result.error.code, result.failure.step_index) == (
        ExecutionStatus.TECHNICAL_FAIL, "CHECK_FAILED", 3)
    assert result.failure.expected == "the element for step 4 on the page"
