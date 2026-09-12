import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pythonjsonlogger import json as jsonlogger

from src.config.settings import settings
from src.safety.redactor import redact_dict


class RunLogger:
    """Structured JSON evidence logger, one instance per discovery or
    replay run. Writes newline-delimited JSON to evidence/{mode}/run_log.json,
    appending as each event happens (crash-resilient — evidence survives
    up to the point of a hard failure, per D024).
    """

    def __init__(self, mode: str, capability: Optional[str] = None):
        self.mode = mode
        self.trace_id = str(uuid4())
        self.capability = capability

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
        happens here so it cannot be forgotten at a call site (D024,
        same defense-in-depth principle as D022's verify_tier).
        """
        safe_fields = redact_dict(fields)
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
            {"goal": goal, "mode": self.mode, "capability": self.capability},
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
