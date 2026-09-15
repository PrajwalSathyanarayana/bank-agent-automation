import json

import pytest
from pydantic import SecretStr

from src.config import settings as settings_module
from src.config.env import env
from src.observability.logger import RunLogger
from src.observability.summary import readable_money, readable_values, summarize
from src.types.artifact_schema import CompareAs, ConfirmationCheck, OutputParamDefinition, OutputType
from src.types.result_schema import BusinessOutcome, ErrorDetail, ExecutionStatus, FailureDetail


def _make_logger(tmp_path, monkeypatch, mode="DISCOVERY"):
    monkeypatch.setattr(settings_module.settings, "evidence_dir", tmp_path)
    return RunLogger(mode=mode, capability="member_servicing_and_bill_pay")


def _read_lines(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_run_logger_creates_a_folder_of_its_own_under_evidence_runs(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch, mode="DISCOVERY")
    logger.execution_started(goal="look up member balance")
    assert logger.run_dir.parent == tmp_path / "runs"
    assert logger.run_dir.name.endswith(f"member_servicing_and_bill_pay_discovery_{logger.trace_id[:8]}")
    assert logger.log_path == logger.run_dir / "log.json"
    assert logger.log_path.exists()


def test_two_runs_in_the_same_mode_and_capability_get_different_folders(tmp_path, monkeypatch):
    logger_a = _make_logger(tmp_path, monkeypatch)
    logger_b = _make_logger(tmp_path, monkeypatch)
    assert logger_a.run_dir != logger_b.run_dir


def test_write_result_saves_result_json_beside_the_runs_own_log(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    logger.write_result('{"status": "SUCCESS"}')
    result_path = logger.run_dir / "result.json"
    assert result_path.exists()
    assert result_path.read_text(encoding="utf-8") == '{"status": "SUCCESS"}'
    assert logger.screenshots_dir == logger.run_dir / "screenshots"


def test_trace_path_is_none_until_a_trace_is_actually_saved_there(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    assert logger.trace_path is None
    (logger.run_dir / "trace.zip").write_bytes(b"PK\x03\x04")
    assert logger.trace_path == logger.run_dir / "trace.zip"


def test_each_call_appends_one_line(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    logger.execution_started(goal="test goal")
    logger.step_fetched(step_id="s1", index=0, action="click")
    lines = _read_lines(logger.log_path)
    assert len(lines) == 2


def test_every_line_is_valid_json_with_expected_event_type(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    logger.step_executed(step_id="s1", status="PASSED")
    lines = _read_lines(logger.log_path)
    assert lines[0]["event_type"] == "STEP_EXECUTED"
    assert lines[0]["step_id"] == "s1"
    assert lines[0]["status"] == "PASSED"


def test_trace_id_is_consistent_across_events_in_one_run(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    logger.execution_started(goal="test goal")
    logger.step_fetched(step_id="s1", index=0, action="click")
    logger.summary_metrics(duration_ms=1000, step_count=1, retry_count=0, human_interventions=0)
    lines = _read_lines(logger.log_path)
    trace_ids = {line["trace_id"] for line in lines}
    assert len(trace_ids) == 1
    assert trace_ids.pop() == logger.trace_id


def test_two_separate_loggers_get_different_trace_ids(tmp_path, monkeypatch):
    logger_a = _make_logger(tmp_path, monkeypatch)
    logger_b = _make_logger(tmp_path, monkeypatch)
    assert logger_a.trace_id != logger_b.trace_id


def test_redaction_is_applied_at_the_emit_chokepoint(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    # Exercises _emit directly to prove redaction happens for ANY field
    # passed through it, not just the 9 named tap points.
    logger._emit("TEST_EVENT", {"password": "hunter2", "step_id": "s1"})
    lines = _read_lines(logger.log_path)
    assert lines[0]["password"] == "[REDACTED]"
    assert lines[0]["step_id"] == "s1"


def test_bank_password_scrubbed_from_free_text(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    password = env.mock_bank_password.get_secret_value()
    # "detail" isn't a sensitive field name and the password matches no
    # pattern, so only the exact-value scrub can catch it.
    logger._emit("TEST_EVENT", {"detail": f"typing failed near {password} on step 3"})
    raw = logger.log_path.read_text(encoding="utf-8")
    assert password not in raw
    assert "[REDACTED]" in raw


def test_old_signing_key_scrubbed_from_free_text_while_set(tmp_path, monkeypatch):
    key = "old-shared-signing-key-" + "x" * 32
    monkeypatch.setattr(env, "artifact_signing_key", SecretStr(key))
    logger = _make_logger(tmp_path, monkeypatch)
    logger._emit("TEST_EVENT", {"detail": f"signature check used {key}"})
    raw = logger.log_path.read_text(encoding="utf-8")
    assert key not in raw
    assert "[REDACTED]" in raw


def test_logging_works_once_the_old_signing_key_is_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "artifact_signing_key", None)
    logger = _make_logger(tmp_path, monkeypatch)
    logger._emit("TEST_EVENT", {"detail": "saved"})
    assert _read_lines(logger.log_path)[0]["detail"] == "saved"


def test_execution_started_records_resolved_settings(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    logger.execution_started(goal="test goal")
    logged = _read_lines(logger.log_path)[0]["resolved_settings"]
    assert logged["discovery_max_steps"] == settings_module.settings.discovery_max_steps
    assert logged["evidence_dir"] == str(tmp_path)
    assert logged["anthropic_model"] == env.anthropic_model
    assert logged["mock_bank_base_url"] == env.mock_bank_base_url
    assert logged["target_environment"] == env.target_environment
    assert (logged["auto_execute_limit"], logged["auto_execute_currency"]) == (
        str(env.auto_execute_limit), env.auto_execute_currency)


def test_resolved_settings_hold_only_settings_and_named_env_values(tmp_path, monkeypatch):
    # Fails closed: any other env value (username, secrets) must stay out.
    logger = _make_logger(tmp_path, monkeypatch)
    logger.execution_started(goal="test goal")
    logged = _read_lines(logger.log_path)[0]["resolved_settings"]
    expected = set(settings_module.Settings.model_fields) | {
        "anthropic_model", "mock_bank_base_url", "target_environment", "auto_execute_limit", "auto_execute_currency"}
    assert set(logged) == expected


def test_first_line_contains_no_secret_values(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    logger.execution_started(goal="test goal")
    raw = logger.log_path.read_text(encoding="utf-8")
    for secret in (
        env.anthropic_api_key,
        env.mock_bank_secret_key,
        env.mock_bank_password,
        env.artifact_signing_key,
    ):
        assert secret is None or secret.get_secret_value() not in raw


def _record_a_step(logger, **overrides):
    fields = dict(
        index=1,
        action="type",
        description="Enter the username",
        safety_tier="SAFE",
        acted=True,
        input_value="{credential:bank_username}",
        locators=[{"priority": 0, "kind": "name", "type": "css", "value": 'input[name="username"]'}],
        weak=False,
        rejected=[{"kind": "text", "reason": "matches more than one element"}],
        checkpoints=[{"type": "page_path", "expected_value": "/login"}],
        is_assertion=False,
        next_step_check_added_to=0,
    )
    fields.update(overrides)
    logger.step_recorded(**fields)
    return fields


def test_step_recorded_is_one_line_with_the_whole_step(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    fields = _record_a_step(logger)
    lines = _read_lines(logger.log_path)
    assert len(lines) == 1
    assert lines[0]["event_type"] == "STEP_RECORDED"
    for name, value in fields.items():
        assert lines[0][name] == value


def test_step_recorded_still_scrubs_a_secret_that_slips_into_it(tmp_path, monkeypatch):
    # Every event goes through the chokepoint; a password in a description is scrubbed.
    logger = _make_logger(tmp_path, monkeypatch)
    password = env.mock_bank_password.get_secret_value()
    _record_a_step(logger, description=f"Typed {password} into the box")
    raw = logger.log_path.read_text(encoding="utf-8")
    assert password not in raw
    assert "[REDACTED]" in raw


def test_secret_on_page_names_the_secret_and_where_it_was_found(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    logger.secret_on_page(step_index=3, secrets=["bank_password"], found_in="locator candidates")
    line = _read_lines(logger.log_path)[0]
    assert line["event_type"] == "SECRET_ON_PAGE"
    assert line["secrets"] == ["bank_password"]
    assert line["step_index"] == 3
    assert line["found_in"] == "locator candidates"


def test_recovery_event_handles_optional_fields(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    logger.recovery_event(tier="TIER_1_RULE")
    lines = _read_lines(logger.log_path)
    assert lines[0]["screenshot_path"] is None
    assert lines[0]["dom_snapshot"] is None


# --- a person in control of the run ---

def test_handoff_requested_carries_the_context_a_person_needs(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch, mode="REPLAY")
    logger.handoff_requested("OVER_AUTO_LIMIT", "the amount is above the bank's limit for automatic payments",
                             ["I finished it", "Stop the task"], step_index=14,
                             step_description="Confirm the payment", screenshot_path="shots/pause.png")
    line = _read_lines(logger.log_path)[0]
    assert line["event_type"] == "HANDOFF_REQUESTED"
    assert (line["trigger_reason"], line["step_index"], line["step_description"], line["screenshot_path"]) == (
        "OVER_AUTO_LIMIT", 14, "Confirm the payment", "shots/pause.png")
    assert line["why"] == "the amount is above the bank's limit for automatic payments"
    assert line["buttons"] == ["I finished it", "Stop the task"]


def test_handoff_resolved_counts_what_the_person_did(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch, mode="REPLAY")
    logger.handoff_resolved("MANUAL_COMPLETED", 41_000, person_actions=3, screenshot_path="shots/back.png")
    line = _read_lines(logger.log_path)[0]
    assert (line["event_type"], line["resolution"], line["duration_ms"]) == ("HANDOFF_RESOLVED", "MANUAL_COMPLETED", 41_000)
    assert (line["person_actions"], line["screenshot_path"]) == (3, "shots/back.png")


@pytest.mark.parametrize(
    "kind, what, element_kind",
    [("click", "Confirm Payment", "button"), ("field_changed", "Amount:", "text box"), ("page_visited", None, None)],
)
def test_each_person_action_is_one_line(tmp_path, monkeypatch, kind, what, element_kind):
    logger = _make_logger(tmp_path, monkeypatch, mode="REPLAY")
    logger.person_action(kind, "/billpay/confirm", what=what, element_kind=element_kind)
    line = _read_lines(logger.log_path)[0]
    assert line["event_type"] == "PERSON_ACTION"
    assert (line["kind"], line["page_path"], line["what"], line["element_kind"]) == (
        kind, "/billpay/confirm", what, element_kind)


def test_a_person_action_still_scrubs_a_secret_that_slips_into_it(tmp_path, monkeypatch):
    # A button's wording is page text, never typed text; the chokepoint still catches a secret.
    logger = _make_logger(tmp_path, monkeypatch, mode="REPLAY")
    password = env.mock_bank_password.get_secret_value()
    logger.person_action("click", "/login", what=f"Sign on as {password}", element_kind="button")
    raw = logger.log_path.read_text(encoding="utf-8")
    assert password not in raw
    assert "[REDACTED]" in raw


def test_a_dialog_left_for_the_person_is_noted_with_its_wording(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch, mode="REPLAY")
    logger.dialog_left_for_person("confirm", "This action cannot be undone. Continue?")
    line = _read_lines(logger.log_path)[0]
    assert (line["event_type"], line["dialog_type"], line["dialog_message"]) == (
        "DIALOG_LEFT_FOR_PERSON", "confirm", "This action cannot be undone. Continue?")


# --- the plain-English summary on every result ---

BILL_PAY = "member_servicing_and_bill_pay"
ASKED = {"member_id": "10234", "payee_name": "Sunbelt Electric Co", "amount": "$50.00"}
READ = {"checking_balance_before": "$2,450.32", "new_checking_balance": "$2,400.32"}
ASKED_LINE = "Pay $50.00 to Sunbelt Electric Co for member 10234."


def _summary(status, mode="REPLAY", inputs=ASKED, outputs=None, **facts):
    return summarize(BILL_PAY, mode, status, goal="For member 10234, pay 50 to Sunbelt Electric Co.",
                     inputs=inputs, outputs=outputs or {}, **facts)


@pytest.mark.parametrize(
    "status, facts, expected",
    [
        pytest.param(ExecutionStatus.SUCCESS, {"outputs": READ, "irreversible_step": "completed"},
                     "Paid $50.00 to Sunbelt Electric Co for member 10234. "
                     "Checking balance $2,450.32 before, $2,400.32 after.", id="paid"),
        pytest.param(ExecutionStatus.BUSINESS_OUTCOME,
                     {"inputs": {**ASKED, "member_id": "99999"}, "irreversible_step": "not_reached",
                      "outcome": BusinessOutcome(code="MEMBER_NOT_FOUND", description="No member has this ID")},
                     "Pay $50.00 to Sunbelt Electric Co for member 99999. Not done: no member has this ID. "
                     "No payment was made.", id="no such member"),
        pytest.param(ExecutionStatus.HUMAN_ESCALATED,
                     {"error": ErrorDetail(code="OVER_AUTO_LIMIT", message="over"), "irreversible_step": "not_reached"},
                     f"{ASKED_LINE} A person needs to decide: the amount is above the bank's limit for automatic "
                     "payments. No payment was made.", id="over the limit"),
        pytest.param(ExecutionStatus.TECHNICAL_FAIL,
                     {"error": ErrorDetail(code="CHECK_FAILED", message="failed"), "irreversible_step": "unknown",
                      "failure": FailureDetail(step_index=12, step_description="Submit the payment for final processing.",
                                               expected="page path /billpay/confirm", observed="page path /login")},
                     f"{ASKED_LINE} Stopped at step 12 (Submit the payment for final processing): a screen wasn't the "
                     "one expected (CHECK_FAILED). The payment may have gone through: check before trying again.",
                     id="stopped after the payment was clicked"),
        pytest.param(ExecutionStatus.HARD_ABORT, {"error": ErrorDetail(code="NO_ARTIFACT", message="none")},
                     f"{ASKED_LINE} Stopped: this task hasn't been learned yet (NO_ARTIFACT).", id="not learned yet"),
        pytest.param(ExecutionStatus.SUCCESS,
                     {"mode": "DISCOVERY", "outputs": READ, "irreversible_step": "completed", "version": "3.0.0"},
                     "Paid $50.00 to Sunbelt Electric Co for member 10234. Checking balance $2,450.32 before, "
                     "$2,400.32 after. Learned and saved as version 3.0.0.", id="learned"),
        pytest.param(ExecutionStatus.HUMAN_ESCALATED,
                     {"mode": "DISCOVERY", "escalation": "IRREVERSIBLE_STEP", "irreversible_step": "not_reached",
                      "version": "2.0.0"},
                     f"{ASKED_LINE} A person needs to decide: the task was learned up to the final confirmation, which "
                     "a person must make. No payment was made. Learned and saved as version 2.0.0.",
                     id="learned up to the payment"),
    ],
)
def test_a_result_reads_as_plain_english(status, facts, expected):
    assert _summary(status, **facts) == expected


@pytest.mark.parametrize(
    "status, facts, expected",
    [
        pytest.param(ExecutionStatus.HUMAN_ESCALATED,
                     {"person": "finished", "outputs": READ, "irreversible_step": "completed"},
                     "Paid $50.00 to Sunbelt Electric Co for member 10234 (confirmed by a person). "
                     "Checking balance $2,450.32 before, $2,400.32 after.", id="finished by a person"),
        pytest.param(ExecutionStatus.HUMAN_ESCALATED, {"person": "finished", "irreversible_step": "unknown"},
                     f"{ASKED_LINE} A person reported it finished, but the receipt wasn't found. "
                     "The payment may have gone through: check before trying again.", id="finished, no receipt seen"),
        pytest.param(ExecutionStatus.HUMAN_ESCALATED,
                     {"person": "stopped", "irreversible_step": "not_reached",
                      "error": ErrorDetail(code="OVER_AUTO_LIMIT", message="over")},
                     f"{ASKED_LINE} A person was needed (the amount is above the bank's limit for automatic payments) "
                     "and stopped the task. No payment was made.", id="stopped by a person"),
        pytest.param(ExecutionStatus.HUMAN_ESCALATED,
                     {"person": "timed_out", "irreversible_step": "not_reached",
                      "error": ErrorDetail(code="PERSON_HAD_CONTROL", message="earlier")},
                     f"{ASKED_LINE} A person was needed (a person had control earlier in this run, so the final "
                     "confirmation is left to a person), but the time for a person ran out. No payment was made.",
                     id="the time for a person ran out"),
        pytest.param(ExecutionStatus.HUMAN_ESCALATED,
                     {"person": "window_closed", "irreversible_step": "not_reached",
                      "error": ErrorDetail(code="CHECK_FAILED", message="failed")},
                     f"{ASKED_LINE} A person was needed (a screen wasn't the one expected), and the window was "
                     "closed. No payment was made.", id="the window closed"),
        pytest.param(ExecutionStatus.SUCCESS, {"person": "helped", "outputs": READ, "irreversible_step": "completed"},
                     "Paid $50.00 to Sunbelt Electric Co for member 10234. Checking balance $2,450.32 before, "
                     "$2,400.32 after. A person had control during the run.", id="helped along the way"),
    ],
)
def test_a_persons_part_in_the_run_reads_as_plain_english(status, facts, expected):
    assert _summary(status, **facts) == expected


@pytest.mark.parametrize(
    "capability, inputs, outputs, expected",
    [
        pytest.param("look_up_checking_balance", {"member_id": "10234"}, {"checking_balance": "$2,450.32"},
                     "Checking balance for member 10234: $2,450.32. Learned and saved as version 1.0.0.",
                     id="a balance lookup names its value once"),
        pytest.param("update_member_phone", {"member_id": "10234", "new_phone": "(520) 555-0199"}, {},
                     "Changed the phone number of member 10234 to (520) 555-0199. Learned and saved as version 1.0.0.",
                     id="a change with nothing to read"),
    ],
)
def test_a_task_with_its_own_wording_isnt_followed_by_a_list_of_its_values(capability, inputs, outputs, expected):
    assert summarize(capability, "DISCOVERY", ExecutionStatus.SUCCESS, goal="", inputs=inputs, outputs=outputs,
                     version="1.0.0") == expected


def test_a_request_missing_an_input_is_described_by_its_goal():
    text = _summary(ExecutionStatus.HARD_ABORT, inputs={"member_id": "10234"},
                    error=ErrorDetail(code="INPUT_INVALID", message="missing inputs: amount"))
    assert text == ("Asked: For member 10234, pay 50 to Sunbelt Electric Co. "
                    "Stopped: the request was incomplete or invalid (INPUT_INVALID).")


def test_an_unfamiliar_code_still_reads_as_a_sentence():
    text = _summary(ExecutionStatus.TECHNICAL_FAIL, error=ErrorDetail(code="SOMETHING_NEW", message="new"))
    assert text == f"{ASKED_LINE} Stopped: an internal check stopped it (SOMETHING_NEW)."


def test_a_capability_without_its_own_wording_is_summarised_from_its_goal():
    text = summarize("update_contact", "REPLAY", ExecutionStatus.SUCCESS, goal="For member 10234, update the email.",
                     inputs={"member_id": "10234"}, outputs={"confirmation_number": "88121"})
    assert text == "Done: For member 10234, update the email. confirmation number: 88121."


@pytest.mark.parametrize(
    "value, currency, shown",
    [("2450.32", "USD", "$2,450.32"), (50.0, "USD", "$50.00"), ("50", "EUR", "50.00 EUR"), ("n/a", "USD", "n/a")],
)
def test_amounts_are_shown_the_way_people_write_them(value, currency, shown):
    assert readable_money(value, currency) == shown


@pytest.mark.parametrize(
    "code, reason",
    [pytest.param("STUCK_NO_PROGRESS", "the learning agent got stuck", id="a stuck learning agent"),
     pytest.param("SECRET_LITERAL", "the learned procedure would have kept data it must not store",
                  id="the save-time scan refusing run data")],
)
def test_families_of_codes_read_alike(code, reason):
    assert _summary(ExecutionStatus.HARD_ABORT, error=ErrorDetail(code=code, message=code)) == (
        f"{ASKED_LINE} Stopped: {reason} ({code}).")


def test_values_are_shown_as_money_only_where_the_contract_says_money():
    checks = [ConfirmationCheck(label="Amount:", input_key="amount", compare_as=CompareAs.MONEY, currency="USD")]
    definitions = [OutputParamDefinition(key="balance", type=OutputType.MONEY, description="Balance", currency="USD"),
                   OutputParamDefinition(key="count", type=OutputType.NUMBER, description="Count")]
    inputs, outputs = readable_values({"member_id": "10234", "amount": 1050.0, "copies": 2.0},
                                      {"balance": "2450.32", "count": 3}, checks, definitions)
    assert inputs == {"member_id": "10234", "amount": "$1,050.00", "copies": "2"}
    assert outputs == {"balance": "$2,450.32", "count": "3"}
