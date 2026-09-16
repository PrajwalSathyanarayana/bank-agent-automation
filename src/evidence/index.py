"""Evidence stage: the front page. Scans evidence/runs/ for every run that saved a
result.json, makes sure each one has a report.html (src/evidence/report.py), and writes
evidence/index.html (linking to each) and evidence/index.md (the same list as a Markdown
table, for pasting into REPORT.md or a PR description).

On demand only, like report.py: never wired into a run itself. Doesn't decide which runs
are "the showcase" - it lists whatever's actually in evidence/runs/ when it's run; pruning
throwaway runs first is a separate, manual step.
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.config.settings import settings
from src.evidence.report import secondary_notes, status_badge, write_report
from src.types.result_schema import ExecutionResult

_TEMPLATES = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATES), autoescape=select_autoescape(["html"]))


@dataclass(frozen=True)
class RunEntry:
    """One run's saved result, plus where its own folder sits relative to evidence/."""

    result: ExecutionResult
    folder: str  # e.g. "runs/2026-09-16_member_servicing_and_bill_pay_replay_b063002b"


def discover_runs(runs_dir: Path) -> list[RunEntry]:
    """Every subfolder of runs_dir with a readable result.json: every DISCOVERY run first
    (newest first within that group), then every REPLAY run (newest first within that
    group) - grouped by mode rather than left to whichever order they happened to run in,
    so a later replay never pushes an earlier discovery further down the page.

    A folder without one (a run still going, or one that never got that far) is left out
    quietly. One with a result.json that won't parse is left out and named on stderr - not
    fatal to the rest of the index.
    """
    if not runs_dir.is_dir():
        return []
    entries = []
    for folder in sorted(runs_dir.iterdir()):
        result_path = folder / "result.json"
        if not result_path.is_file():
            continue
        try:
            result = ExecutionResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        except ValueError as invalid:
            print(f"evidence index: skipping {folder.name} ({invalid})", file=sys.stderr)
            continue
        entries.append(RunEntry(result, f"runs/{folder.name}"))
    entries.sort(key=lambda entry: (entry.result.mode != "DISCOVERY", -entry.result.start_time.timestamp()))
    return entries


def write_index(evidence_dir: Path) -> tuple[Path, Path]:
    """Render every listed run's report.html - always, even one it already has, so a
    template change (a new legend, a reworded outcome box) reaches every past run too -
    then write evidence_dir/index.html and index.md. Returns the two files written."""
    runs = discover_runs(evidence_dir / "runs")
    for entry in runs:
        write_report(entry.result, evidence_dir / entry.folder)
    html_path = evidence_dir / "index.html"
    html_path.write_text(render_index_html(runs), encoding="utf-8")
    md_path = evidence_dir / "index.md"
    md_path.write_text(render_index_markdown(runs), encoding="utf-8")
    return html_path, md_path


def render_index_html(runs: Sequence[RunEntry]) -> str:
    rows = [_row(entry) for entry in runs]
    # discover_runs() already sorts discovery before replay; splitting here keeps that one
    # sort as the single source of order, rather than a second rule the template would need.
    discovery_rows = [row for row in rows if row["mode"] == "Discovery"]
    replay_rows = [row for row in rows if row["mode"] == "Replay"]
    return _env.get_template("index.html").render(
        rows=rows, discovery_rows=discovery_rows, replay_rows=replay_rows)


def render_index_markdown(runs: Sequence[RunEntry]) -> str:
    header = "| Capability | Mode | Status | Summary | Started | Duration | Report |"
    divider = "|---|---|---|---|---|---|---|"
    if not runs:
        return f"{header}\n{divider}\n"
    lines = [header, divider]
    for entry in runs:
        row = _row(entry)
        summary = row["summary"].replace("|", "\\|") or "—"
        status = row["status_label"] + "".join(f", {text}" for text, _ in row["notes"])
        lines.append(f"| {row['capability']} | {row['mode']} | {status} | {summary} | "
                     f"{row['started']} | {row['duration']} | [{row['run_id']}]({row['report_link']}) |")
    return "\n".join(lines) + "\n"


def _row(entry: RunEntry) -> dict:
    result = entry.result
    label, status_class = status_badge(result)
    return {
        "capability": result.capability.replace("_", " ").title(),
        "mode": result.mode.title(),
        "status_label": label,
        "status_class": status_class,
        "notes": secondary_notes(result),
        "summary": result.summary or "",
        "started": result.start_time.strftime("%Y-%m-%d %H:%M UTC"),
        "duration": f"{result.duration_ms / 1000:.1f} s",
        "run_id": result.run_id[:8],
        "report_link": f"{entry.folder}/report.html",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.evidence.index")
    parser.add_argument("evidence_dir", nargs="?", default=None,
                        help=f"default: {settings.evidence_dir}")
    args = parser.parse_args(argv)
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else settings.evidence_dir
    html_path, md_path = write_index(evidence_dir)
    print(f"Wrote {html_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
