import json

from src.config import settings as settings_module
from src.config.env import env
from src.observability.logger import RunLogger


def _make_logger(tmp_path, monkeypatch, mode="DISCOVERY"):
    monkeypatch.setattr(settings_module.settings, "evidence_dir", tmp_path)
    return RunLogger(mode=mode, capability="member_servicing_and_bill_pay")


def _read_lines(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_run_logger_creates_log_file_under_evidence_mode_dir(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch, mode="DISCOVERY")
    logger.execution_started(goal="look up member balance")
    assert logger.log_path == tmp_path / "discovery" / "run_log.json"
    assert logger.log_path.exists()


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


def test_signing_key_scrubbed_from_free_text(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    key = env.artifact_signing_key.get_secret_value()
    logger._emit("TEST_EVENT", {"detail": f"signature check used {key}"})
    raw = logger.log_path.read_text(encoding="utf-8")
    assert key not in raw
    assert "[REDACTED]" in raw


def test_execution_started_records_resolved_settings(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    logger.execution_started(goal="test goal")
    logged = _read_lines(logger.log_path)[0]["resolved_settings"]
    assert logged["discovery_max_steps"] == settings_module.settings.discovery_max_steps
    assert logged["evidence_dir"] == str(tmp_path)
    assert logged["anthropic_model"] == env.anthropic_model
    assert logged["mock_bank_base_url"] == env.mock_bank_base_url


def test_resolved_settings_hold_only_settings_and_named_env_values(tmp_path, monkeypatch):
    # Fails closed: any other env value (username, secrets) must stay out.
    logger = _make_logger(tmp_path, monkeypatch)
    logger.execution_started(goal="test goal")
    logged = _read_lines(logger.log_path)[0]["resolved_settings"]
    expected = set(settings_module.Settings.model_fields) | {"anthropic_model", "mock_bank_base_url"}
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
        assert secret.get_secret_value() not in raw


def test_recovery_event_handles_optional_fields(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path, monkeypatch)
    logger.recovery_event(tier="TIER_1_RULE")
    lines = _read_lines(logger.log_path)
    assert lines[0]["screenshot_path"] is None
    assert lines[0]["dom_snapshot"] is None
