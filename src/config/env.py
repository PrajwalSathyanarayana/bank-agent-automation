from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets print as **********; read them only via .get_secret_value() where used (D038)
    anthropic_api_key: SecretStr
    anthropic_model: str
    mock_bank_base_url: str = "http://localhost:5000"
    mock_bank_secret_key: SecretStr
    mock_bank_username: str
    mock_bank_password: SecretStr
    artifact_signing_key: SecretStr = Field(min_length=32)
    ws_handoff_port: int = 8765
    artifact_storage_dir: str = "./artifacts"
    evidence_dir: str = "./evidence"


env = Env()
