from pathlib import Path
from pydantic import BaseModel
from src.config.env import env


class Settings(BaseModel):
    # Resolved absolute paths (prevents cwd-relative bugs)
    artifact_storage_dir: Path = Path(env.artifact_storage_dir).resolve()
    evidence_dir: Path = Path(env.evidence_dir).resolve()
    artifact_private_key_path: Path = Path(env.artifact_private_key_path).resolve()
    artifact_trusted_keys_dir: Path = Path(env.artifact_trusted_keys_dir).resolve()

    # System-level execution timeouts (not per-step)
    # Steps catch fast loops, wall-clock catches slow hangs
    discovery_max_steps: int = 40
    discovery_timeout_ms: int = 900_000        # 15 minutes total

    # Each call gets min(cap, time left); SDK auto-retry is off in discovery
    discovery_llm_call_timeout_ms: int = 90_000
    discovery_page_action_timeout_ms: int = 30_000
    discovery_llm_retries: int = 1

    # What the model sees each step: every mock bank page fits this window.
    # Scale 1 keeps element boxes (CSS pixels) aligned with screenshot pixels.
    discovery_viewport_width: int = 1280
    discovery_viewport_height: int = 800
    discovery_device_scale_factor: int = 1
    discovery_max_elements: int = 100           # bounds the element list sent to the model
    replay_checkpoint_timeout_ms: int = 10_000  # 10 seconds per checkpoint
    replay_total_timeout_ms: int = 120_000      # 2 minutes total

    # Human handoff
    operator_timeout_ms: int = 600_000          # 10 minutes then OPERATOR_TIMED_OUT

    # Recovery engine
    tier1_max_dismiss_attempts: int = 2
    tier2_llm_timeout_ms: int = 15_000          # Bounded LLM call timeout

    # Computed URLs
    mock_bank_login_url: str = f"{env.mock_bank_base_url}/login"
    mock_bank_search_url: str = f"{env.mock_bank_base_url}/search"


settings = Settings()