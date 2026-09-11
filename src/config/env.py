from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str
    anthropic_model: str
    mock_bank_base_url: str = "http://localhost:5000"
    ws_handoff_port: int = 8765
    artifact_storage_dir: str = "./artifacts"
    evidence_dir: str = "./evidence"


env = Env()
