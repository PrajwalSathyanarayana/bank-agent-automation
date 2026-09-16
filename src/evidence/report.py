"""Evidence stage: turning a run's saved result.json into a plain-English report.html a
non-technical reviewer can open directly, beside that same run's log.json, result.json and
screenshots/ (all written by pieces 1-3 of the evidence stage already).

Never generated automatically inside a run (src/replay/executor.py, src/discovery/agent.py):
rendering one costs nothing to skip for the hundreds of test-suite and throwaway dev runs
that don't want one, and an old run's report (including a real, paid-for discovery run) can
be re-rendered later without running it again.
"""
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.observability.summary import reason_for
from src.types.result_schema import ExecutionResult, ExecutionStatus, HandoffTelemetry, RecoveryAttemptLog

_TEMPLATES = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATES), autoescape=select_autoescape(["html"]))

# A plain word for each status, and the badge colour it gets: shared with index.py, so a
# run's badge reads the same way on the front page and on its own report.
_STATUS = {
    "SUCCESS": ("Done", "ok"),
    "BUSINESS_OUTCOME": ("The bank's answer", "answer"),
    "HUMAN_ESCALATED": ("A person was needed", "person"),
    "TECHNICAL_FAIL": ("Stopped", "fail"),
    "HARD_ABORT": ("Stopped before it could run", "fail"),
}

# How a handoff ended, in plain words - the raw HandoffResolution value otherwise.
_RESOLUTION = {
    "RESUMED": "handed back; the run continued",
    "MANUAL_COMPLETED": "completed by the person",
    "OPERATOR_TIMED_OUT": "no one responded in time",
    "ABORTED": "no person was available",
}


def status_badge(result: ExecutionResult) -> tuple[str, str]:
    """The run's main badge. 'A person was needed' whenever a person had any part in the run
    (handoff_events non-empty) - whether the task they helped with went on to finish or not.
    A reviewer weighing how autonomous the system was wants that fact first, ahead of whatever
    the run's own status says. The raw ExecutionStatus otherwise."""
    if result.handoff_events:
        return _STATUS["HUMAN_ESCALATED"]
    return _STATUS.get(result.status.value, (result.status.value, "fail"))


def _screenshot_name(path: Optional[str]) -> Optional[str]:
    # Stored as the full path the run wrote it to; the report links it relative to its own
    # folder, so only the filename (in screenshots/) is needed.
    return Path(path).name if path else None


def _handoff_row(h: HandoffTelemetry) -> dict:
    resolution = h.resolution.value if h.resolution else None
    return {
        "trigger": reason_for(h.trigger_reason).capitalize(),
        "resolution": _RESOLUTION.get(resolution, resolution or "—").capitalize(),
        "duration": f"{h.duration_ms / 1000:.1f} s" if h.duration_ms is not None else "—",
        "person_actions": h.person_actions if h.person_actions is not None else "—",
    }


def artifact_not_saved(result: ExecutionResult) -> bool:
    """True for a discovery run that reached an end worth learning from (it finished, or a
    person was asked to help) but has no artifact to show for it - a stopped handoff, or a
    recording gap (see agent.py's _save). The next request for this capability discovers it
    again from scratch."""
    return (result.mode == "DISCOVERY" and result.artifact_version is None
            and result.status in (ExecutionStatus.SUCCESS, ExecutionStatus.HUMAN_ESCALATED))


def secondary_notes(result: ExecutionResult) -> list[tuple[str, str]]:
    """Small tags next to a run's main badge for something it doesn't say on its own: whether
    discovery left an artifact behind."""
    return [("Artifact not saved", "warn")] if artifact_not_saved(result) else []


def _recovery_text(log: RecoveryAttemptLog) -> str:
    """One recovery-log entry, in plain words: a person completing a held step reads very
    differently from the system clearing a popup by itself, even though both are stored the
    same way (a step's recovery_logs) - the report should say which happened."""
    if log.tier.value == "TIER_3_HANDOFF":
        return f"A person: {log.details}" if log.details else "Done by a person"
    if log.interruption_code:
        return f"Recovered from {log.interruption_code}" + (f": {log.details}" if log.details else "")
    return log.details or "Recovered"


def render_report(result: ExecutionResult, run_dir: Path) -> str:
    """The report's HTML, without writing anything. screenshots are read from
    run_dir/screenshots/ (whatever the run itself already put there)."""
    screenshots_dir = run_dir / "screenshots"
    all_screenshots = sorted(p.name for p in screenshots_dir.glob("*.png")) if screenshots_dir.is_dir() else []
    label, status_class = status_badge(result)
    outcome_screenshot = _screenshot_name(result.outcome.screenshot_path) if result.outcome else None
    failure_screenshot = _screenshot_name(result.failure.screenshot_path) if result.failure else None
    # Shown once, inline next to the claim it belongs to - left out of the general
    # gallery below so the same image isn't shown twice on the same page.
    already_shown = {name for name in (outcome_screenshot, failure_screenshot) if name}
    screenshots = [name for name in all_screenshots if name not in already_shown]
    return _env.get_template("report.html").render(
        result=result, status_label=label, status_class=status_class,
        duration_s=result.duration_ms / 1000, screenshots=screenshots,
        outcome_screenshot=outcome_screenshot, failure_screenshot=failure_screenshot,
        handoffs=[_handoff_row(h) for h in result.handoff_events], recovery_text=_recovery_text,
        notes=secondary_notes(result), artifact_not_saved=artifact_not_saved(result),
    )


def write_report(result: ExecutionResult, run_dir: Path) -> Path:
    """Render report.html into run_dir, beside this run's log.json and result.json.
    Returns the file written."""
    path = run_dir / "report.html"
    path.write_text(render_report(result, run_dir), encoding="utf-8")
    return path
