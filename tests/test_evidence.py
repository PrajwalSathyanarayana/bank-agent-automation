from datetime import datetime, timezone

import pytest

from src.config.settings import settings
from src.evidence.index import RunEntry, discover_runs, main, render_index_html, render_index_markdown, write_index
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


def _step(index, status=StepStatus.PASSED, priority=0, recoveries=0, attempts=1,
         description="Open Bill Pay for this member.") -> StepExecutionTrace:
    return StepExecutionTrace(step_id=f"s{index}", sequence_index=index, description=description, status=status,
                              safety_tier=SafetyTier.SAFE, attempt_count=attempts, duration_ms=100,
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


def test_the_outcomes_screenshot_is_linked_when_one_was_taken(tmp_path):
    outcome = BusinessOutcome(code="MEMBER_NOT_FOUND", description="No member has this ID",
                              screenshot_path=str(tmp_path / "screenshots" / "run_outcome_step06_X.png"))
    html = render_report(_result(ExecutionStatus.BUSINESS_OUTCOME, outcome=outcome), tmp_path)
    assert 'src="screenshots/run_outcome_step06_X.png"' in html


def test_no_outcome_screenshot_link_when_none_was_taken(tmp_path):
    # The default outcome in _result() carries no screenshot_path.
    html = render_report(_result(ExecutionStatus.BUSINESS_OUTCOME), tmp_path)
    assert "<img" not in html


def test_the_outcomes_screenshot_isnt_also_listed_in_the_general_gallery(tmp_path):
    shots = tmp_path / "screenshots"
    shots.mkdir()
    (shots / "run_outcome_step06_X.png").write_bytes(b"")
    (shots / "run_other.png").write_bytes(b"")
    outcome = BusinessOutcome(code="MEMBER_NOT_FOUND", description="No member has this ID",
                              screenshot_path=str(shots / "run_outcome_step06_X.png"))
    html = render_report(_result(ExecutionStatus.BUSINESS_OUTCOME, outcome=outcome), tmp_path)
    assert "<figcaption>run_outcome_step06_X.png</figcaption>" not in html  # shown inline, not in the gallery too
    assert "<figcaption>run_other.png</figcaption>" in html  # everything else still shows there


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
    assert "<img" not in html  # this failure carries no screenshot_path


def test_the_failures_screenshot_is_linked_when_one_was_taken(tmp_path):
    failure = FailureDetail(step_index=12, step_description="Submit the payment", expected="page /confirm",
                            observed="page /login",
                            screenshot_path=str(tmp_path / "screenshots" / "run_failure_step12.png"))
    html = render_report(_result(ExecutionStatus.TECHNICAL_FAIL, failure=failure), tmp_path)
    assert 'src="screenshots/run_failure_step12.png"' in html


def test_terminal_outputs_are_listed_as_a_table_with_a_readable_label(tmp_path):
    result = _result(ExecutionStatus.SUCCESS, terminal_outputs={"checking_balance_before": "2450.32"})
    html = render_report(result, tmp_path)
    assert "Checking balance before" in html
    assert "checking_balance_before" not in html
    assert "2450.32" in html


@pytest.mark.parametrize(
    "priority, shown",
    [pytest.param(0, "the saved locator worked as-is", id="the primary locator"),
     pytest.param(1, "a backup locator was needed (fallback #1)", id="a fallback locator"),
     pytest.param(None, "—", id="no element to find (navigate)")],
)
def test_a_steps_locator_priority_reads_in_words(tmp_path, priority, shown):
    result = _result(ExecutionStatus.SUCCESS, step_traces=[_step(0, priority=priority)])
    html = render_report(result, tmp_path)
    assert shown in html


def test_a_steps_own_description_is_shown_not_just_its_number(tmp_path):
    step = _step(0, description="Open Bill Pay for this member.")
    html = render_report(_result(ExecutionStatus.SUCCESS, step_traces=[step]), tmp_path)
    assert "Open Bill Pay for this member." in html


def test_a_step_with_no_saved_description_says_so_plainly(tmp_path):
    step = _step(0, description=None)
    html = render_report(_result(ExecutionStatus.SUCCESS, step_traces=[step]), tmp_path)
    assert "(no description saved for this step)" in html


def test_a_step_with_recoveries_shows_their_count(tmp_path):
    result = _result(ExecutionStatus.SUCCESS, step_traces=[_step(0, status=StepStatus.RECOVERED, recoveries=2)])
    html = render_report(result, tmp_path)
    assert "2 interruption(s) cleared first" in html


def test_a_retried_step_says_how_many_times(tmp_path):
    step = _step(0, attempts=3)
    html = render_report(_result(ExecutionStatus.SUCCESS, step_traces=[step]), tmp_path)
    assert "(retried 2x)" in html


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


# --- the front page: evidence/index.html and index.md ---

def _write_result(folder, result) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.json").write_text(result.to_json(), encoding="utf-8")


def test_discover_runs_finds_every_folder_with_a_result_json(tmp_path):
    runs = tmp_path / "runs"
    _write_result(runs / "a", _result(ExecutionStatus.SUCCESS))
    _write_result(runs / "b", _result(ExecutionStatus.SUCCESS))
    (runs / "still_going").mkdir()  # no result.json yet: a run in progress
    entries = discover_runs(runs)
    assert {entry.folder for entry in entries} == {"runs/a", "runs/b"}


def test_discover_runs_on_a_missing_folder_is_empty_not_an_error(tmp_path):
    assert discover_runs(tmp_path / "nope") == []


def test_an_unreadable_result_json_is_skipped_not_fatal_to_the_rest(tmp_path, capsys):
    runs = tmp_path / "runs"
    broken = runs / "broken"
    broken.mkdir(parents=True)
    (broken / "result.json").write_text("not json", encoding="utf-8")
    _write_result(runs / "readable", _result(ExecutionStatus.SUCCESS))
    entries = discover_runs(runs)
    assert [entry.folder for entry in entries] == ["runs/readable"]
    assert "broken" in capsys.readouterr().err


def test_runs_are_listed_newest_first(tmp_path):
    runs = tmp_path / "runs"
    _write_result(runs / "older", _result(ExecutionStatus.SUCCESS, start_time=NOW.replace(hour=1)))
    _write_result(runs / "newer", _result(ExecutionStatus.SUCCESS, start_time=NOW.replace(hour=5)))
    entries = discover_runs(runs)
    assert [entry.folder for entry in entries] == ["runs/newer", "runs/older"]


def test_write_index_renders_every_listed_runs_report(tmp_path):
    _write_result(tmp_path / "runs" / "a", _result(ExecutionStatus.SUCCESS))
    write_index(tmp_path)
    assert (tmp_path / "runs" / "a" / "report.html").exists()


def test_write_index_writes_both_the_html_and_markdown_front_pages(tmp_path):
    _write_result(tmp_path / "runs" / "a", _result(ExecutionStatus.SUCCESS))
    html_path, md_path = write_index(tmp_path)
    assert (html_path, md_path) == (tmp_path / "index.html", tmp_path / "index.md")
    assert html_path.exists() and md_path.exists()


def test_the_front_page_links_to_each_runs_own_report(tmp_path):
    _write_result(tmp_path / "runs" / "a", _result(ExecutionStatus.SUCCESS))
    html_path, _ = write_index(tmp_path)
    assert 'href="runs/a/report.html"' in html_path.read_text(encoding="utf-8")


def test_the_front_page_can_be_filtered_by_mode(tmp_path):
    _write_result(tmp_path / "runs" / "a", _result(ExecutionStatus.SUCCESS, mode="REPLAY"))
    _write_result(tmp_path / "runs" / "b", _result(ExecutionStatus.SUCCESS, mode="DISCOVERY"))
    html_path, _ = write_index(tmp_path)
    html = html_path.read_text(encoding="utf-8")
    assert '<option value="Replay">Replay</option>' in html
    assert '<option value="Discovery">Discovery</option>' in html
    assert 'data-mode="Replay"' in html and 'data-mode="Discovery"' in html


def test_an_empty_run_list_says_so_plainly_in_html(tmp_path):
    assert "No runs yet." in render_index_html([])


def test_the_markdown_table_names_the_run_by_its_short_id_and_links_its_report():
    result = _result(ExecutionStatus.SUCCESS, run_id="deadbeef-0000-0000-0000-000000000000")
    md = render_index_markdown([RunEntry(result, "runs/a")])
    assert "| Member Servicing And Bill Pay | Replay |" in md
    assert "[deadbeef](runs/a/report.html)" in md


def test_a_pipe_in_the_summary_is_escaped_in_the_markdown_table():
    result = _result(ExecutionStatus.SUCCESS, summary="Paid $50 | done")
    md = render_index_markdown([RunEntry(result, "runs/a")])
    assert "Paid $50 \\| done" in md


def test_an_empty_run_list_is_still_a_valid_markdown_table():
    md = render_index_markdown([])
    assert md.startswith("| Capability |")
    assert "\n|---|" in md


def test_main_writes_the_index_to_the_folder_it_is_given(tmp_path, capsys):
    _write_result(tmp_path / "runs" / "a", _result(ExecutionStatus.SUCCESS))
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert str(tmp_path / "index.html") in out
    assert (tmp_path / "index.html").exists()


def test_main_defaults_to_the_configured_evidence_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "evidence_dir", tmp_path)
    _write_result(tmp_path / "runs" / "a", _result(ExecutionStatus.SUCCESS))
    assert main([]) == 0
    assert (tmp_path / "index.html").exists()
