from datetime import datetime, timezone

import pytest

from src.evidence.report import render_report, write_report
from src.types.result_schema import (
    BusinessOutcome,
    ErrorDetail,
    EvidencePaths,
    ExecutionResult,
    ExecutionStatus,
    FailureDetail,
    HandoffTelemetry,
    RecoveryTier,
    StepExecutionTrace,
    StepStatus,
)
from src.types.step_schema import SafetyTier

NOW = datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc)
PATHS = EvidencePaths(log_file="log.json", screenshots_dir="screenshots")


def _result(status, **overrides) -> ExecutionResult:
    fields = dict(capability="member_servicing_and_bill_pay", mode="REPLAY", status=status,
                 start_time=NOW, end_time=NOW, duration_ms=1500, evidence_paths=PATHS,
                 summary="Paid $50.00 to Sunbelt Electric Co for member 10234.")
    if status == ExecutionStatus.BUSINESS_OUTCOME:
        fields["outcome"] = BusinessOutcome(code="MEMBER_NOT_FOUND", description="No member has this ID")
    elif status in (ExecutionStatus.TECHNICAL_FAIL, ExecutionStatus.HARD_ABORT):
        fields["error"] = ErrorDetail(code="CHECK_FAILED", message="a check failed")
    elif status == ExecutionStatus.HUMAN_ESCALATED:
        fields["handoff_events"] = [HandoffTelemetry(triggered_timestamp=NOW, trigger_reason="OVER_AUTO_LIMIT")]
    fields.update(overrides)
    return ExecutionResult(**fields)


def _step(index, status=StepStatus.PASSED, priority=0, recoveries=0) -> StepExecutionTrace:
    return StepExecutionTrace(step_id=f"s{index}", sequence_index=index, status=status,
                              safety_tier=SafetyTier.SAFE, attempt_count=1, duration_ms=100,
                              locator_priority=priority,
                              recovery_logs=[{"timestamp": NOW, "tier": RecoveryTier.TIER_1_RULE, "resolved": True}]
                              * recoveries)


def test_write_report_creates_report_html_beside_the_runs_own_evidence(tmp_path):
    result = _result(ExecutionStatus.SUCCESS)
    path = write_report(result, tmp_path)
    assert path == tmp_path / "report.html"
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert "member servicing and bill pay" in html.lower()
    assert result.summary in html


@pytest.mark.parametrize(
    "status, badge_class",
    [
        pytest.param(ExecutionStatus.SUCCESS, "ok", id="success"),
        pytest.param(ExecutionStatus.BUSINESS_OUTCOME, "answer", id="a business outcome"),
        pytest.param(ExecutionStatus.HUMAN_ESCALATED, "person", id="a person was needed"),
        pytest.param(ExecutionStatus.TECHNICAL_FAIL, "fail", id="a technical failure"),
        pytest.param(ExecutionStatus.HARD_ABORT, "fail", id="a hard abort"),
    ],
)
def test_the_status_badge_matches_the_results_status(tmp_path, status, badge_class):
    html = render_report(_result(status), tmp_path)
    assert f'badge {badge_class}' in html


def test_a_business_outcomes_code_and_description_are_shown(tmp_path):
    html = render_report(_result(ExecutionStatus.BUSINESS_OUTCOME), tmp_path)
    assert "MEMBER_NOT_FOUND" in html
    assert "No member has this ID" in html


def test_an_errors_code_and_message_are_shown(tmp_path):
    html = render_report(_result(ExecutionStatus.TECHNICAL_FAIL), tmp_path)
    assert "CHECK_FAILED" in html
    assert "a check failed" in html


def test_a_failures_step_and_what_was_seen_are_shown(tmp_path):
    failure = FailureDetail(step_index=12, step_description="Submit the payment", expected="page /confirm",
                            observed="page /login")
    result = _result(ExecutionStatus.TECHNICAL_FAIL, failure=failure)
    html = render_report(result, tmp_path)
    assert "Submit the payment" in html
    assert "page /confirm" in html and "page /login" in html


def test_terminal_outputs_are_listed_as_a_table(tmp_path):
    result = _result(ExecutionStatus.SUCCESS, terminal_outputs={"checking_balance_before": "2450.32"})
    html = render_report(result, tmp_path)
    assert "checking_balance_before" in html
    assert "2450.32" in html


@pytest.mark.parametrize(
    "priority, shown",
    [pytest.param(0, "primary", id="the primary locator"),
     pytest.param(1, "fallback #1", id="a fallback locator"),
     pytest.param(None, "—", id="no element to find (navigate)")],
)
def test_a_steps_locator_priority_reads_in_words(tmp_path, priority, shown):
    result = _result(ExecutionStatus.SUCCESS, step_traces=[_step(0, priority=priority)])
    html = render_report(result, tmp_path)
    assert shown in html


def test_a_step_with_recoveries_shows_their_count(tmp_path):
    result = _result(ExecutionStatus.SUCCESS, step_traces=[_step(0, status=StepStatus.RECOVERED, recoveries=2)])
    html = render_report(result, tmp_path)
    assert ">2<" in html


def test_no_steps_is_shown_plainly_not_as_an_empty_table(tmp_path):
    html = render_report(_result(ExecutionStatus.HARD_ABORT), tmp_path)
    assert "No steps ran." in html


def test_a_handoffs_trigger_and_resolution_are_shown(tmp_path):
    html = render_report(_result(ExecutionStatus.HUMAN_ESCALATED), tmp_path)
    assert "OVER_AUTO_LIMIT" in html


def test_screenshots_in_the_runs_folder_are_listed_and_linked(tmp_path):
    shots = tmp_path / "screenshots"
    shots.mkdir()
    (shots / "b.png").write_bytes(b"")
    (shots / "a.png").write_bytes(b"")
    html = render_report(_result(ExecutionStatus.SUCCESS), tmp_path)
    assert html.index('src="screenshots/a.png"') < html.index('src="screenshots/b.png"')


def test_no_screenshots_folder_is_handled_without_a_crash_or_an_empty_section(tmp_path):
    html = render_report(_result(ExecutionStatus.SUCCESS), tmp_path)
    assert "<h2>Screenshots</h2>" not in html


def test_a_value_that_looks_like_markup_is_escaped_not_rendered(tmp_path):
    result = _result(ExecutionStatus.TECHNICAL_FAIL,
                     error=ErrorDetail(code="CHECK_FAILED", message="<script>alert(1)</script>"))
    html = render_report(result, tmp_path)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_the_reports_links_point_at_the_runs_own_result_and_log(tmp_path):
    html = render_report(_result(ExecutionStatus.SUCCESS), tmp_path)
    assert 'href="result.json"' in html
    assert 'href="log.json"' in html
