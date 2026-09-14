from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets print as **********; read them only via .get_secret_value() where used
    anthropic_api_key: SecretStr
    anthropic_model: str
    mock_bank_base_url: str = "http://localhost:5000"
    # What the bank address points at. In a sandbox (a test copy) discovery may perform
    # an irreversible step to learn what follows it; in production it never does.
    target_environment: Literal["sandbox", "production"] = "production"
    mock_bank_secret_key: SecretStr
    mock_bank_username: str
    mock_bank_password: SecretStr
    artifact_signing_key: SecretStr = Field(min_length=32)
    ws_handoff_port: int = 8765
    artifact_storage_dir: str = "./artifacts"
    evidence_dir: str = "./evidence"


env = Env()


def configured_credentials() -> dict[str, str | SecretStr]:
    """Credential values by the names artifacts use ({credential:bank_password}).

    The one place a credential's name meets its .env variable; each bank's runtime
    supplies its own values for the same artifact. Secrets stay SecretStr.
    """
    return {"bank_username": env.mock_bank_username, "bank_password": env.mock_bank_password}
