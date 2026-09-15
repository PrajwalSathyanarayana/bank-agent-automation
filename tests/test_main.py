import html
import re
from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import urlparse

import pytest

import src.main as main_module
from app import create_app  # the mock bank; tests/conftest.py puts its folder on the import path
from src.config.env import env
from src.config.settings import settings
from src.locating.checks import find_phrase, phrase_matches, value_beside
from src.locating.resolver import resolve
from src.catalog import CHECKING, EMAIL, PHONE, SAVINGS
from src.locating.values import read_output
from src.handoff.ws_server import FeedUnavailable, HandoffFeed
from src.intake import IntakeAnswer, IntakeUnavailable
from src.main import BILL_PAY, CONTRACTS, main, parse_args
from src.observability.logger import RunLogger
from src.router import Handled
from src.types.result_schema import (
    BusinessOutcome,
    ErrorDetail,
    EvidencePaths,
    ExecutionResult,
    ExecutionStatus,
    HandoffResolution,
    HandoffTelemetry,
)
from src.safety.authorization import authorize
from src.types.artifact_schema import CompareAs, OutcomeSignal, OutputType, RecoveryAction
from src.types.routes import route_allowed


def test_the_bill_pay_contract_declares_its_inputs_outputs_and_credentials():
    contract = CONTRACTS[BILL_PAY]
    assert [parameter.key for parameter in contract.input_parameters] == ["member_id", "amount", "payee_name"]
    assert [output.key for output in contract.output_definitions] == ["checking_balance_before", "new_checking_balance"]
    assert [credential.key for credential in contract.credentials] == ["bank_username", "bank_password"]
    assert contract.target_url == settings.mock_bank_login_url
    # The goal itself asks for the value, so the model plans to read it.
    assert "read the checking balance" in contract.description


def test_the_bill_pay_balances_are_money_in_us_dollars():
    outputs = CONTRACTS[BILL_PAY].output_definitions
    assert [(output.type, output.currency) for output in outputs] == [(OutputType.MONEY, "USD")] * 2


def test_the_bill_pay_contract_declares_its_four_known_outcomes():
    contract = CONTRACTS[BILL_PAY]
    assert [outcome.code for outcome in contract.known_outcomes] == [
        "MEMBER_NOT_FOUND", "INSUFFICIENT_FUNDS", "ACCOUNT_RESTRICTED", "PAYEE_NOT_FOUND"]
    [payee] = [outcome for outcome in contract.known_outcomes if outcome.signal == OutcomeSignal.NO_SUCH_OPTION]
    assert payee.input_key in {parameter.key for parameter in contract.input_parameters}


@pytest.fixture
def bank():
    # The mock bank in-process, signed in with the configured teller credentials.
    client = create_app().test_client()
    client.post("/login", data={"username": env.mock_bank_username,
                                "password": env.mock_bank_password.get_secret_value()})
    return client


def _visible_text(response) -> str:
    # The page's words without its markup, the way a reader sees them.
    return html.unescape(re.sub(r"<[^>]+>", " ", response.get_data(as_text=True)))


def _trigger(bank, code):
    # Each outcome as the bank produces it: 99999 doesn't exist, 20567 has $512.75, 30891 is restricted.
    if code == "MEMBER_NOT_FOUND":
        return bank.get("/member/99999", follow_redirects=True)
    if code == "INSUFFICIENT_FUNDS":
        bank.get("/member/20567")
        return bank.post("/billpay", data={"payee_id": "P001", "amount": "1000.00"})
    bank.get("/member/30891")
    return bank.get("/billpay")


@pytest.mark.parametrize("code", ["MEMBER_NOT_FOUND", "INSUFFICIENT_FUNDS", "ACCOUNT_RESTRICTED"])
def test_each_outcomes_wording_is_what_the_bank_shows(bank, code):
    [outcome] = [outcome for outcome in CONTRACTS[BILL_PAY].known_outcomes if outcome.code == code]
    assert phrase_matches(_visible_text(_trigger(bank, code)), outcome.text)


def test_no_outcome_wording_appears_on_the_normal_path(bank):
    # Member page, Bill Pay form and the confirm page for a payment that can go through.
    pages = [bank.get("/member/10234"), bank.get("/billpay"),
             bank.post("/billpay", data={"payee_id": "P001", "amount": "50.00"}, follow_redirects=True)]
    texts = [_visible_text(page) for page in pages]
    for outcome in CONTRACTS[BILL_PAY].known_outcomes:
        if outcome.text:
            assert not any(phrase_matches(text, outcome.text) for text in texts), outcome.code


def _pages_passed(response) -> set[str]:
    # Every page a request went through, redirects included.
    return {step.request.path for step in (*response.history, response)}


def test_the_phone_contract_declares_its_inputs_outcomes_and_pages():
    contract = CONTRACTS[PHONE]
    assert [parameter.key for parameter in contract.input_parameters] == ["member_id", "new_phone"]
    # Nothing to read and nothing irreversible: no outputs, no payment check.
    assert (contract.output_definitions, contract.confirmation_checks) == ([], [])
    assert [outcome.code for outcome in contract.known_outcomes] == ["MEMBER_NOT_FOUND", "INVALID_PHONE"]
    assert contract.target_url == settings.mock_bank_login_url


def test_the_phone_task_runs_on_its_own_pages_and_the_bank_answers_as_declared(bank):
    contract = CONTRACTS[PHONE]
    flow = [("GET", "/", None), ("GET", "/dashboard", None), ("GET", "/search", None),
            ("POST", "/search", {"member_id": "40412"}), ("GET", "/member/40412/edit", None)]
    visited = set()
    for method, path, data in flow:
        visited |= _pages_passed(bank.open(path, method=method, data=data, follow_redirects=True))
    saved = bank.post("/member/40412/edit", data={"email": "grace@example.com", "phone": "(520) 555-0199"})
    refused = bank.post("/member/40412/edit", data={"email": "grace@example.com", "phone": "520-555-0199"})
    visited |= _pages_passed(saved) | _pages_passed(bank.get("/member/99999", follow_redirects=True))
    assert sorted(path for path in visited if not route_allowed(path, contract.allowed_paths)) == []
    assert not route_allowed("/billpay", contract.allowed_paths)
    # The bank's words: a good save confirms itself; the refusal is the declared outcome.
    [invalid] = [outcome for outcome in contract.known_outcomes if outcome.code == "INVALID_PHONE"]
    assert phrase_matches(_visible_text(saved), "Profile updated successfully.")
    assert not phrase_matches(_visible_text(saved), invalid.text)
    assert phrase_matches(_visible_text(refused), invalid.text)


def test_every_page_of_the_flow_and_its_outcomes_is_allowed_but_profile_edit_is_not(bank):
    allowed = CONTRACTS[BILL_PAY].allowed_paths
    flow = [("GET", "/", None), ("GET", "/dashboard", None), ("GET", "/search", None),
            ("POST", "/search", {"member_id": "10234"}), ("GET", "/member/10234/accounts", None),
            ("GET", "/billpay", None), ("POST", "/billpay", {"payee_id": "P001", "amount": "50.00"}),
            ("POST", "/billpay/confirm", None)]
    visited = set()
    for method, path, data in flow:
        visited |= _pages_passed(bank.open(path, method=method, data=data, follow_redirects=True))
    for code in ("MEMBER_NOT_FOUND", "INSUFFICIENT_FUNDS", "ACCOUNT_RESTRICTED"):
        visited |= _pages_passed(_trigger(bank, code))
    signed_out = create_app().test_client()
    for path in ("/login", "/dashboard"):
        visited |= _pages_passed(signed_out.get(path, follow_redirects=True))

    assert {"/member/10234", "/member/not-found", "/billpay/confirm", "/session-timeout"} <= visited
    assert sorted(path for path in visited if not route_allowed(path, allowed)) == []
    assert not route_allowed("/member/10234/edit", allowed)
    assert route_allowed(urlparse(CONTRACTS[BILL_PAY].target_url).path, allowed)


def test_the_discover_command_reads_typed_inputs_a_step_limit_and_a_window_option():
    args = parse_args(["discover", "--member-id", "10234", "--amount", "50", "--payee", "Sunbelt Electric Co",
                       "--max-steps", "25", "--headed"])
    assert (args.member_id, args.amount, args.payee, args.max_steps, args.headed) == (
        "10234", 50.0, "Sunbelt Electric Co", 25, True)


def test_the_step_limit_defaults_to_the_setting_and_the_window_stays_hidden():
    args = parse_args(["discover", "--member-id", "10234", "--amount", "50", "--payee", "Sunbelt Electric Co"])
    assert (args.max_steps, args.headed) == (None, False)


@pytest.mark.parametrize(
    "argv", [pytest.param(["run", "For member 10234, pay 50 to Sunbelt Electric Co"], id="run"),
             pytest.param(["discover", "--member-id", "10234", "--amount", "50", "--payee", "Sunbelt Electric Co"],
                          id="discover"),
             pytest.param(["replay", "--member-id", "10234", "--amount", "50", "--payee", "Sunbelt Electric Co"],
                          id="replay")],
)
def test_slow_mo_defaults_to_none_on_every_command(argv):
    assert parse_args(argv).slow_mo is None


def test_slow_mo_takes_a_value_in_milliseconds():
    args = parse_args(["replay", "--member-id", "10234", "--amount", "50", "--payee", "Sunbelt Electric Co",
                       "--slow-mo", "250"])
    assert args.slow_mo == 250


@pytest.mark.parametrize(
    "extra, note_shown",
    [pytest.param([], True, id="neither headed nor operator"),
     pytest.param(["--headed"], False, id="headed"),
     pytest.param(["--operator"], False, id="operator")],
)
def test_slow_mo_without_a_visible_window_prints_a_note(capsys, extra, note_shown):
    args = parse_args(["replay", "--member-id", "10234", "--amount", "50", "--payee", "Sunbelt Electric Co",
                       "--slow-mo", "250", *extra])
    main_module._slow_mo_note(args)
    assert ("Note:" in capsys.readouterr().out) is note_shown


def test_no_note_when_slow_mo_isnt_given(capsys):
    args = parse_args(["replay", "--member-id", "10234", "--amount", "50", "--payee", "Sunbelt Electric Co"])
    main_module._slow_mo_note(args)
    assert capsys.readouterr().out == ""


# --- the replay command ---

REPLAY_ARGS = ["replay", "--member-id", "10234", "--amount", "50", "--payee", "Sunbelt Electric Co"]


def test_the_replay_command_reads_the_same_typed_inputs_and_a_window_option():
    args = parse_args(["replay", "--member-id", "40412", "--amount", "25.5", "--payee", "Desert Valley Water Utility",
                       "--headed"])
    assert (args.command, args.member_id, args.amount, args.payee, args.headed) == (
        "replay", "40412", 25.5, "Desert Valley Water Utility", True)


def test_the_replay_command_has_no_step_limit_since_no_model_takes_steps():
    assert not hasattr(parse_args(REPLAY_ARGS), "max_steps")


def _result(status: ExecutionStatus, logger) -> ExecutionResult:
    # The smallest result each status allows.
    now = datetime.now(timezone.utc)
    carries = {
        ExecutionStatus.BUSINESS_OUTCOME: {"outcome": BusinessOutcome(code="MEMBER_NOT_FOUND", description="No member")},
        ExecutionStatus.HUMAN_ESCALATED: {"handoff_events": [HandoffTelemetry(triggered_timestamp=now,
                                                                              trigger_reason="OVER_AUTO_LIMIT")]},
        ExecutionStatus.TECHNICAL_FAIL: {"error": ErrorDetail(code="CHECK_FAILED", message="a check failed")},
    }.get(status, {})
    return ExecutionResult(capability=BILL_PAY, mode="REPLAY", status=status, start_time=now, end_time=now,
                           duration_ms=0, evidence_paths=EvidencePaths(log_file=str(logger.log_path),
                                                                       screenshots_dir="screenshots"), **carries)


@pytest.mark.parametrize(
    "status, exit_code",
    [
        pytest.param(ExecutionStatus.SUCCESS, 0, id="success"),
        pytest.param(ExecutionStatus.BUSINESS_OUTCOME, 0, id="an answer, not a failure"),
        pytest.param(ExecutionStatus.HUMAN_ESCALATED, 1, id="a person is needed"),
        pytest.param(ExecutionStatus.TECHNICAL_FAIL, 1, id="a failure"),
    ],
)
def test_the_replay_command_asks_for_bill_pay_and_exits_by_status(monkeypatch, capsys, tmp_path, status, exit_code):
    # The replay itself is covered in test_replay.py; here only what the command asks and reports.
    monkeypatch.setattr(settings, "evidence_dir", tmp_path)
    monkeypatch.setattr(main_module, "_bank_is_up", lambda: True)
    asked = []

    async def fake_replay(request, logger, *, headless, operator=None, trace=False, slow_mo_ms=None):
        asked.append((request, headless, operator, trace))
        return _result(status, logger)

    monkeypatch.setattr(main_module, "replay", fake_replay)
    assert main(REPLAY_ARGS) == exit_code
    [(request, headless, operator, trace)] = asked
    assert (request.capability, dict(request.inputs), headless, operator, trace) == (
        BILL_PAY, {"member_id": "10234", "amount": 50.0, "payee_name": "Sunbelt Electric Co"}, True, None, True)
    assert f'"status": "{status.value}"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "irreversible, exit_code",
    [
        pytest.param("completed", 0, id="a person finished it and the receipt was seen"),
        pytest.param("unknown", 1, id="a person said it was finished but no receipt was seen"),
    ],
)
def test_with_an_operator_the_window_is_shown_and_a_finished_task_exits_by_its_receipt(
    monkeypatch, capsys, tmp_path, irreversible, exit_code
):
    monkeypatch.setattr(settings, "evidence_dir", tmp_path)
    monkeypatch.setattr(main_module, "_bank_is_up", lambda: True)
    monkeypatch.setattr(env, "ws_handoff_port", 0)  # a free port, never a real run's
    asked = []

    async def fake_replay(request, logger, *, headless, operator=None, trace=False, slow_mo_ms=None):
        asked.append((headless, operator))
        finished = HandoffTelemetry(triggered_timestamp=datetime.now(timezone.utc), trigger_reason="OVER_AUTO_LIMIT",
                                    resolution=HandoffResolution.MANUAL_COMPLETED)
        return _result(ExecutionStatus.HUMAN_ESCALATED, logger).model_copy(
            update={"handoff_events": [finished], "irreversible_step": irreversible})

    monkeypatch.setattr(main_module, "replay", fake_replay)
    assert main([*REPLAY_ARGS, "--operator"]) == exit_code
    [(headless, operator)] = asked
    # A person can only take over a window they can see; the feed announces to anyone watching.
    assert headless is False and isinstance(operator.announcer, HandoffFeed)
    out = capsys.readouterr().out
    assert "take over" in out and "Handoff announcements: ws://127.0.0.1:" in out


def test_a_busy_feed_port_is_reported_and_the_run_goes_on_without_it(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(settings, "evidence_dir", tmp_path)
    monkeypatch.setattr(main_module, "_bank_is_up", lambda: True)
    asked = []

    class _Busy:
        def __init__(self, port):
            self.port = port

        async def __aenter__(self):
            raise FeedUnavailable(f"the handoff feed couldn't listen on port {self.port} (in use)")

        async def __aexit__(self, *exc_info):
            return None

    async def fake_replay(request, logger, *, headless, operator=None, trace=False, slow_mo_ms=None):
        asked.append(operator)
        return _result(ExecutionStatus.SUCCESS, logger)

    monkeypatch.setattr(main_module, "HandoffFeed", _Busy)
    monkeypatch.setattr(main_module, "replay", fake_replay)
    assert main([*REPLAY_ARGS, "--operator"]) == 0
    [operator] = asked
    assert operator is not None and operator.announcer is None
    assert "carrying on without announcements" in capsys.readouterr().out


DISCOVER_ARGS = ["discover", "--member-id", "10234", "--amount", "50", "--payee", "Sunbelt Electric Co"]
RUN_ARGS = ["run", "For member 10234, pay 50 to Sunbelt Electric Co"]


def test_the_run_command_takes_the_request_in_words_and_the_window_options():
    args = parse_args([*RUN_ARGS, "--headed", "--operator"])
    assert (args.command, args.request, args.headed, args.operator) == (
        "run", "For member 10234, pay 50 to Sunbelt Electric Co", True, True)


def _stub_run(monkeypatch, tmp_path, handled_or_error):
    monkeypatch.setattr(settings, "evidence_dir", tmp_path)
    monkeypatch.setattr(main_module, "_bank_is_up", lambda: True)
    # The engines are stubbed below: no model is called.
    monkeypatch.setattr(main_module, "ClaudeModel", lambda: object())
    monkeypatch.setattr(main_module, "ClaudeIntakeModel", lambda: object())
    asked = []

    async def fake_handle(request, **options):
        asked.append((request, options))
        if isinstance(handled_or_error, Exception):
            raise handled_or_error
        # A function builds the answer only now, after the evidence folder points at tmp_path.
        return handled_or_error() if callable(handled_or_error) else handled_or_error

    monkeypatch.setattr(main_module, "handle", fake_handle)
    return asked


def test_a_request_that_isnt_run_prints_why_and_exits_3(monkeypatch, capsys, tmp_path):
    answer = IntakeAnswer("not_supported", "I can't do that yet. The tasks I know are:\n- …")
    asked = _stub_run(monkeypatch, tmp_path, Handled(answer))
    assert main(RUN_ARGS) == 3
    assert asked[0][0] == "For member 10234, pay 50 to Sunbelt Electric Co"
    assert capsys.readouterr().out.startswith("I can't do that yet.")


def test_a_request_that_is_run_prints_what_was_understood_then_the_result(monkeypatch, capsys, tmp_path):
    answer = IntakeAnswer("run", "Understood as: Pay $50.00 to Sunbelt Electric Co for member 10234.",
                          capability=BILL_PAY, inputs={"member_id": "10234", "amount": 50.0,
                                                       "payee_name": "Sunbelt Electric Co"})
    asked = _stub_run(monkeypatch, tmp_path, lambda: Handled(
        answer, _result(ExecutionStatus.SUCCESS, RunLogger("REPLAY", capability=BILL_PAY))))
    assert main(RUN_ARGS) == 0
    assert asked[0][1]["headless"] is True and asked[0][1]["operator"] is None
    out = capsys.readouterr().out
    assert out.startswith("Understood as: Pay $50.00") and '"status": "SUCCESS"' in out


def test_an_intake_that_cant_be_reached_runs_nothing_and_says_so(monkeypatch, capsys, tmp_path):
    _stub_run(monkeypatch, tmp_path, IntakeUnavailable("APIConnectionError"))
    assert main(RUN_ARGS) == 2
    assert "nothing was run" in capsys.readouterr().out


def test_both_commands_take_an_operator():
    assert (parse_args(REPLAY_ARGS).operator, parse_args(DISCOVER_ARGS).operator) == (False, False)
    assert parse_args([*DISCOVER_ARGS, "--operator"]).operator is True


def test_discover_with_an_operator_shows_the_window_and_passes_the_feed(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(settings, "evidence_dir", tmp_path)
    monkeypatch.setattr(main_module, "_bank_is_up", lambda: True)
    monkeypatch.setattr(env, "ws_handoff_port", 0)  # a free port, never a real run's
    monkeypatch.setattr(main_module, "ClaudeModel", lambda: object())  # the run is faked: no model is called
    asked = []

    async def fake_discover(request, model, logger, *, headless, max_steps, operator=None, trace=False,
                            slow_mo_ms=None):
        asked.append((headless, operator))
        return _result(ExecutionStatus.SUCCESS, logger).model_copy(update={"mode": "DISCOVERY"})

    monkeypatch.setattr(main_module, "discover", fake_discover)
    assert main([*DISCOVER_ARGS, "--operator"]) == 0
    [(headless, operator)] = asked
    assert headless is False and isinstance(operator.announcer, HandoffFeed)
    assert "take over" in capsys.readouterr().out


def test_the_replay_command_says_when_the_bank_isnt_running(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "_bank_is_up", lambda: False)
    assert main(REPLAY_ARGS) == 2
    assert "isn't answering" in capsys.readouterr().out


# --- the bill pay interruptions and payment checks, tried on the real pages ---

def test_the_bill_pay_contract_declares_its_interruptions_and_payment_checks():
    contract = CONTRACTS[BILL_PAY]
    assert [(interruption.code, interruption.recovery) for interruption in contract.known_interruptions] == [
        ("PROMO_POPUP", RecoveryAction.CLICK), ("SESSION_EXPIRED", RecoveryAction.START_OVER)]
    assert [(check.label, check.input_key, check.compare_as) for check in contract.confirmation_checks] == [
        ("Payee:", "payee_name", CompareAs.TEXT), ("Amount:", "amount", CompareAs.MONEY)]


def _interruption(code):
    return next(interruption for interruption in CONTRACTS[BILL_PAY].known_interruptions if interruption.code == code)


async def _sign_in(page) -> None:
    await page.goto("/login")
    await page.fill("input[name='username']", env.mock_bank_username)
    await page.fill("input[name='password']", env.mock_bank_password.get_secret_value())
    await page.click("input[type='submit']")
    await page.wait_for_url("**/dashboard")


@pytest.mark.anyio
async def test_the_promo_popup_is_spotted_and_its_declared_click_clears_it(page, dashboard_popup):
    dashboard_popup(True)
    await _sign_in(page)
    popup = _interruption("PROMO_POPUP")
    assert await resolve(page, popup.locator, {}).is_visible()
    await resolve(page, popup.target, {}).click()
    assert not await resolve(page, popup.locator, {}).is_visible()


@pytest.mark.anyio
async def test_an_expired_session_shows_the_declared_text_once(page):
    # Signed out, the dashboard sends the browser to the session-timeout page.
    await page.goto("/dashboard")
    assert len(await find_phrase(page, _interruption("SESSION_EXPIRED").text)) == 1


# --- the three read and profile tasks ---

def test_the_email_and_balance_contracts_declare_their_inputs_outputs_answers_and_pages():
    declared = {
        EMAIL: (["member_id", "new_email"], [], ["MEMBER_NOT_FOUND", "INVALID_EMAIL"], "/member/10234/edit"),
        CHECKING: (["member_id"], ["checking_balance"], ["MEMBER_NOT_FOUND"], "/member/10234"),
        SAVINGS: (["member_id"], ["savings_balance"], ["MEMBER_NOT_FOUND", "NO_SAVINGS_ACCOUNT"],
                  "/member/10234/accounts"),
    }
    for capability, (inputs, outputs, codes, page_needed) in declared.items():
        contract = CONTRACTS[capability]
        assert [parameter.key for parameter in contract.input_parameters] == inputs, capability
        assert [output.key for output in contract.output_definitions] == outputs, capability
        assert [outcome.code for outcome in contract.known_outcomes] == codes, capability
        # A balance is money, read exactly; nothing here is irreversible.
        assert all((output.type, output.currency) == (OutputType.MONEY, "USD") for output in contract.output_definitions)
        assert contract.confirmation_checks == []
        assert route_allowed(page_needed, contract.allowed_paths), capability
        assert not route_allowed("/billpay", contract.allowed_paths), capability


def test_the_new_answers_are_worded_as_the_bank_shows_them(bank):
    [invalid_email] = [outcome for outcome in CONTRACTS[EMAIL].known_outcomes if outcome.code == "INVALID_EMAIL"]
    refused = bank.post("/member/40412/edit", data={"email": "g@example", "phone": "(480) 555-0117"})
    assert phrase_matches(_visible_text(refused), invalid_email.text)
    [no_savings] = [outcome for outcome in CONTRACTS[SAVINGS].known_outcomes if outcome.code == "NO_SAVINGS_ACCOUNT"]
    assert phrase_matches(_visible_text(bank.get("/member/20567/accounts")), no_savings.text)
    assert not phrase_matches(_visible_text(bank.get("/member/10234/accounts")), no_savings.text)


@pytest.mark.anyio
async def test_both_balances_are_read_by_their_labels_on_the_real_pages(page, dashboard_popup):
    # The cell right after each label, as discovery's reading step and replay read it.
    dashboard_popup(False)
    await _sign_in(page)
    await page.goto("/member/40412")
    checking = await value_beside(page, "Primary Account Balance:")
    await page.goto("/member/40412/accounts")
    savings = await value_beside(page, "savings")
    for shown, capability in ((checking, CHECKING), (savings, SAVINGS)):
        assert shown is not None and shown.startswith("$"), capability
        read_output(shown, CONTRACTS[capability].output_definitions[0])  # a USD amount, read exactly


@pytest.mark.anyio
async def test_the_confirm_screen_shows_each_declared_check_and_the_request_passes(page, dashboard_popup):
    dashboard_popup(False)
    await _sign_in(page)
    await page.goto("/search")
    await page.fill("input[name='member_id']", "10234")
    await page.click("input[value='Search']")
    await page.goto("/billpay")
    await page.select_option("select[name='payee_id']", label="Sunbelt Electric Co")
    await page.fill("input[name='amount']", "50.00")
    await page.click("input[value='Continue']")
    await page.wait_for_url("**/billpay/confirm")

    checks = CONTRACTS[BILL_PAY].confirmation_checks
    readings = {check.label: await value_beside(page, check.label) for check in checks}
    assert readings == {"Payee:": "Sunbelt Electric Co", "Amount:": "$50.00"}
    asked = {"member_id": "10234", "payee_name": "Sunbelt Electric Co", "amount": 50.0}
    assert authorize(checks, readings, asked, Decimal("1000.00"), "USD").authorized
