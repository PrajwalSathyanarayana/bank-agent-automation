import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pythonjsonlogger import json as jsonlogger

from src.config.env import env
from src.config.settings import settings
from src.safety.redactor import redact_dict, scrub_known_values

# Named, not all of env: a secret mistyped as a plain str must not reach the evidence.
_LOGGED_ENV_FIELDS = ("anthropic_model", "mock_bank_base_url")


def _resolved_settings() -> dict:
    resolved = settings.model_dump(mode="json")
    resolved.update({name: getattr(env, name) for name in _LOGGED_ENV_FIELDS})
    return resolved


class RunLogger:
    """Structured JSON evidence logger, one instance per discovery or
    replay run. Writes newline-delimited JSON to evidence/{mode}/run_log.json,
    appending as each event happens (crash-resilient — evidence survives
    up to the point of a hard failure).
    """

    def __init__(self, mode: str, capability: Optional[str] = None):
        self.mode = mode
        self.trace_id = str(uuid4())
        self.capability = capability
        self._secrets = (
            env.anthropic_api_key,
            env.mock_bank_secret_key,
            env.mock_bank_password,
            env.artifact_signing_key,
        )

        log_dir = settings.evidence_dir / mode.lower()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "run_log.json"

        self._logger = logging.getLogger(f"run.{self.trace_id}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            handler = logging.FileHandler(self.log_path, mode="a", encoding="utf-8")
            handler.setFormatter(jsonlogger.JsonFormatter())
            self._logger.addHandler(handler)

    def _emit(self, event_type: str, fields: dict) -> None:
        """Single chokepoint every tap point funnels through — redaction
        happens here so it cannot be forgotten at a call site.
        """
        safe_fields = scrub_known_values(
            redact_dict(fields),
            [secret.get_secret_value() for secret in self._secrets],
        )
        self._logger.info(
            event_type,
            extra={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "trace_id": self.trace_id,
                **safe_fields,
            },
        )

    def execution_started(self, goal: str) -> None:
        self._emit(
            "EXECUTION_STARTED",
            {
                "goal": goal,
                "mode": self.mode,
                "capability": self.capability,
                "resolved_settings": _resolved_settings(),
            },
        )

    def step_fetched(self, step_id: str, index: int, action: str) -> None:
        self._emit("STEP_FETCHED", {"step_id": step_id, "index": index, "action": action})

    def locator_evaluated(self, strategy: str, duration_ms: int, attempt: int) -> None:
        self._emit(
            "LOCATOR_EVALUATED",
            {"strategy": strategy, "duration_ms": duration_ms, "attempt": attempt},
        )

    def step_executed(self, step_id: str, status: str) -> None:
        self._emit("STEP_EXECUTED", {"step_id": step_id, "status": status})

    def recovery_event(
        self,
        tier: str,
        screenshot_path: Optional[str] = None,
        dom_snapshot: Optional[str] = None,
    ) -> None:
        self._emit(
            "RECOVERY_EVENT",
            {"tier": tier, "screenshot_path": screenshot_path, "dom_snapshot": dom_snapshot},
        )

    def handoff_started(self, trigger_reason: str, session_lock_token: str) -> None:
        self._emit(
            "HANDOFF_STARTED",
            {"trigger_reason": trigger_reason, "session_lock_token": session_lock_token},
        )

    def handoff_resolved(self, resolution: str, duration_ms: int) -> None:
        self._emit("HANDOFF_RESOLVED", {"resolution": resolution, "duration_ms": duration_ms})

    def artifact_saved(self, artifact_id: str, version: str, sha256_hash: str) -> None:
        self._emit(
            "ARTIFACT_SAVED",
            {"artifact_id": artifact_id, "version": version, "sha256_hash": sha256_hash},
        )

    def step_recorded(
        self,
        index: int,
        action: str,
        description: str,
        safety_tier: str,
        acted: bool,
        input_value: Optional[str],
        locators: list[dict],
        weak: bool,
        rejected: list[dict],
        checkpoints: list[dict],
        is_assertion: bool,
        next_step_check_added_to: Optional[int],
    ) -> None:
        """Discovery: one line per recorded step, telling its whole story.

        Weak locators ("weak": true) and assertions ("is_assertion": true) are one search
        away. Locator values are the stored ones, already cleared of this run's data.
        Rejections carry a kind and a reason, never a value. acted is false for the
        irreversible step, recorded but never clicked.
        """
        self._emit(
            "STEP_RECORDED",
            {
                "index": index,
                "action": action,
                "description": description,
                "safety_tier": safety_tier,
                "acted": acted,
                "input_value": input_value,
                "locators": locators,
                "weak": weak,
                "rejected": rejected,
                "checkpoints": checkpoints,
                "is_assertion": is_assertion,
                "next_step_check_added_to": next_step_check_added_to,
            },
        )

    def secret_on_page(self, step_index: int, secrets: list[str], found_in: str) -> None:
        """Discovery warning: a secret's value appeared on the page. Names only, never values.

        Its own event so a security issue never sits unnoticed inside a routine line.
        """
        self._emit("SECRET_ON_PAGE", {"step_index": step_index, "secrets": secrets, "found_in": found_in})

    def dialog_dismissed(self, dialog_type: str, message: str) -> None:
        """A browser dialog nobody expected was dismissed: its type and wording.

        The wording goes under dialog_message: "message" is a reserved attribute of
        Python's log records, and logging refuses it as an extra field.
        """
        self._emit("DIALOG_DISMISSED", {"dialog_type": dialog_type, "dialog_message": message})

    def backstop_scan(
        self,
        outcome: str,
        conversions: list[dict],
        option_values_dropped: list[int],
        literals_kept: list[dict],
        assertions_recorded: list[dict],
        flagged_assertions: list[dict],
        findings: list[dict],
    ) -> None:
        """Discovery: the save-time scan's whole report, one line per save, on pass and on abort.

        Conversions say what replaced a value, never the value; findings never carry one.
        Literals kept and assertions recorded are shown for review; the scan has already
        left out any field with a finding.
        """
        self._emit(
            "BACKSTOP_SCAN",
            {
                "outcome": outcome,
                "conversions": conversions,
                "option_values_dropped": option_values_dropped,
                "literals_kept": literals_kept,
                "assertions_recorded": assertions_recorded,
                "flagged_assertions": flagged_assertions,
                "findings": findings,
            },
        )

    def summary_metrics(
        self, duration_ms: int, step_count: int, retry_count: int, human_interventions: int
    ) -> None:
        self._emit(
            "SUMMARY_METRICS",
            {
                "duration_ms": duration_ms,
                "step_count": step_count,
                "retry_count": retry_count,
                "human_interventions": human_interventions,
            },
        )
