import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pythonjsonlogger import json as jsonlogger

from src.config.env import env
from src.config.settings import settings
from src.safety.redactor import redact_dict, scrub_known_values

# Named, not all of env: a secret mistyped as a plain str must not reach the evidence.
# Named on purpose: every other env value (the username, every secret) stays out of the log.
_LOGGED_ENV_FIELDS = ("anthropic_model", "mock_bank_base_url", "target_environment",
                      "auto_execute_limit", "auto_execute_currency")


def _resolved_settings() -> dict:
    resolved = settings.model_dump(mode="json")
    # As text: the limit is an exact Decimal, and JSON has no exact number type.
    resolved.update({name: str(getattr(env, name)) for name in _LOGGED_ENV_FIELDS})
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

    def locator_evaluated(
        self, strategy: str, duration_ms: int, attempt: int, *,
        step_index: Optional[int] = None, priority: Optional[int] = None, matches: Optional[int] = None,
    ) -> None:
        """Replay tried one locator: its kind and priority, and how many elements it matched on
        which attempt (None: it couldn't be used at all)."""
        self._emit(
            "LOCATOR_EVALUATED",
            {"strategy": strategy, "duration_ms": duration_ms, "attempt": attempt,
             "step_index": step_index, "priority": priority, "matches": matches},
        )

    def step_executed(self, step_id: str, status: str) -> None:
        self._emit("STEP_EXECUTED", {"step_id": step_id, "status": status})

    def recovery_event(
        self,
        tier: str,
        screenshot_path: Optional[str] = None,
        dom_snapshot: Optional[str] = None,
        *,
        interruption_code: Optional[str] = None,
        recovery: Optional[str] = None,
        resolved: Optional[bool] = None,
        details: Optional[str] = None,
    ) -> None:
        """Replay met a known interruption: which one, the recovery tried, whether it worked
        and why not, and the screenshot taken before it."""
        self._emit(
            "RECOVERY_EVENT",
            {"tier": tier, "screenshot_path": screenshot_path, "dom_snapshot": dom_snapshot,
             "interruption_code": interruption_code, "recovery": recovery, "resolved": resolved,
             "details": details},
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

    def artifact_unchanged(self, artifact_id: str, version: str) -> None:
        """A rediscovery recorded exactly the latest saved version: no new file was written."""
        self._emit("ARTIFACT_UNCHANGED", {"artifact_id": artifact_id, "version": version})

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

    def overlay_dismissed(self, reason: str) -> None:
        """Discovery: the model closed an overlay. The click is deliberately not a step."""
        self._emit("OVERLAY_DISMISSED", {"reason": reason})

    def model_fallback(self, from_model: str, to_model: str) -> None:
        """Discovery: a model declined a turn and the fallback model answered it."""
        self._emit("MODEL_FALLBACK", {"from_model": from_model, "to_model": to_model})

    def model_action(self, turn: int, tool: str, element: Optional[int], reason: str) -> None:
        """Discovery: what the model chose this turn: the tool, the element number and its reason."""
        self._emit("MODEL_ACTION", {"turn": turn, "tool": tool, "element": element, "reason": reason})

    def action_refused(self, turn: int, tool: str, refusal: str) -> None:
        """Discovery: the model's action was refused or failed; what it was told."""
        self._emit("ACTION_REFUSED", {"turn": turn, "tool": tool, "refusal": refusal})

    def model_usage(
        self, turn: int, input_tokens: int, cache_write_tokens: int, cache_read_tokens: int, output_tokens: int
    ) -> None:
        """Discovery: the tokens one model call used (uncached input, cache writes, cache reads, output)."""
        self._emit("MODEL_USAGE", {
            "turn": turn, "input_tokens": input_tokens, "cache_write_tokens": cache_write_tokens,
            "cache_read_tokens": cache_read_tokens, "output_tokens": output_tokens,
        })

    def run_usage(
        self, input_tokens: int, cache_write_tokens: int, cache_read_tokens: int, output_tokens: int,
        estimated_cost_usd: Optional[float],
    ) -> None:
        """Discovery: the run's token totals and its estimated cost at list price (None if unpriced)."""
        self._emit("RUN_USAGE", {
            "input_tokens": input_tokens, "cache_write_tokens": cache_write_tokens,
            "cache_read_tokens": cache_read_tokens, "output_tokens": output_tokens,
            "estimated_cost_usd": estimated_cost_usd,
        })

    def execution_ended(self, status: str, error_code: Optional[str], error_message: Optional[str]) -> None:
        """How the run ended: its status and, for anything but success, the code and reason."""
        self._emit("EXECUTION_ENDED", {"status": status, "error_code": error_code, "error_message": error_message})

    def dialog_accepted(self, dialog_type: str, message: str) -> None:
        """A dialog the system expected (an irreversible step in a test environment) was accepted."""
        self._emit("DIALOG_ACCEPTED", {"dialog_type": dialog_type, "dialog_message": message})

    def irreversible_executed(self, step_index: int) -> None:
        """Discovery in a test environment performed an irreversible step, to learn what follows it."""
        self._emit("IRREVERSIBLE_EXECUTED", {"step_index": step_index, "environment": "sandbox"})

    def authorization_checked(self, code: Optional[str], problems: list[dict[str, str]]) -> None:
        """The payment check before an irreversible step: authorized, or its code and every
        problem (label, expected, seen, reason). The values pass through the redaction chokepoint."""
        self._emit("AUTHORIZATION_CHECKED", {"authorized": code is None, "code": code, "problems": problems})

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
