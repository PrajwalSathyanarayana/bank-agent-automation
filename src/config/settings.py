from pathlib import Path
from pydantic import BaseModel
from src.config.env import env


class Settings(BaseModel):
    # Resolved absolute paths (prevents cwd-relative bugs)
    artifact_storage_dir: Path = Path(env.artifact_storage_dir).resolve()
    evidence_dir: Path = Path(env.evidence_dir).resolve()

    # System-level execution timeouts (not per-step)
    discovery_max_steps: int = 20
    discovery_timeout_ms: int = 300_000        # 5 minutes total
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