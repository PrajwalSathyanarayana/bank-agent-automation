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

from src.types.result_schema import ExecutionResult

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


def status_badge(status_value: str) -> tuple[str, str]:
    """A plain label and a CSS class (ok/answer/person/fail) for a result's status value."""
    return _STATUS.get(status_value, (status_value, "fail"))


def _screenshot_name(path: Optional[str]) -> Optional[str]:
    # Stored as the full path the run wrote it to; the report links it relative to its own
    # folder, so only the filename (in screenshots/) is needed.
    return Path(path).name if path else None


def render_report(result: ExecutionResult, run_dir: Path) -> str:
    """The report's HTML, without writing anything. screenshots are read from
    run_dir/screenshots/ (whatever the run itself already put there)."""
    screenshots_dir = run_dir / "screenshots"
    all_screenshots = sorted(p.name for p in screenshots_dir.glob("*.png")) if screenshots_dir.is_dir() else []
    label, status_class = status_badge(result.status.value)
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
    )


def write_report(result: ExecutionResult, run_dir: Path) -> Path:
    """Render report.html into run_dir, beside this run's log.json and result.json.
    Returns the file written."""
    path = run_dir / "report.html"
    path.write_text(render_report(result, run_dir), encoding="utf-8")
    return path
