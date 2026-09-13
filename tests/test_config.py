import pytest
from pydantic import ValidationError

from src.config.env import Env, env
from src.config.settings import Settings, settings


def test_env_loads_all_required_fields():
    assert env.anthropic_api_key.get_secret_value()
    assert env.anthropic_model
    assert env.mock_bank_base_url
    assert env.mock_bank_secret_key.get_secret_value()
    assert env.mock_bank_username
    assert env.mock_bank_password.get_secret_value()
    assert env.artifact_signing_key.get_secret_value()
    assert env.ws_handoff_port
    assert env.artifact_storage_dir
    assert env.evidence_dir


def test_env_ws_handoff_port_is_int():
    assert isinstance(env.ws_handoff_port, int)


def test_env_missing_required_var_raises_clear_error():
    with pytest.raises(ValidationError) as exc_info:
        Env(_env_file="nonexistent.env", anthropic_api_key="x")

    errors = exc_info.value.errors()
    assert any(e["loc"] == ("anthropic_model",) for e in errors)
    assert any(e["type"] == "missing" for e in errors)


def test_settings_resolves_absolute_paths():
    assert settings.artifact_storage_dir.is_absolute()
    assert settings.evidence_dir.is_absolute()


def test_settings_computed_urls_match_base_url():
    assert settings.mock_bank_login_url == f"{env.mock_bank_base_url}/login"
    assert settings.mock_bank_search_url == f"{env.mock_bank_base_url}/search"


def test_settings_operator_timeout_matches_architecture_spec():
    # ARCHITECTURE.md Module 2: operator timeout is 10 minutes.
    assert settings.operator_timeout_ms == 600_000


def test_settings_is_pydantic_model():
    assert isinstance(settings, Settings)
    dumped = settings.model_dump()
    assert "artifact_storage_dir" in dumped


def test_settings_discovery_limits_match_d033():
    assert settings.discovery_max_steps == 40
    assert settings.discovery_timeout_ms == 900_000
    assert settings.discovery_llm_call_timeout_ms == 90_000
    assert settings.discovery_page_action_timeout_ms == 30_000
    assert settings.discovery_llm_retries == 1


# --- D038: secrets held as SecretStr ---

def test_secrets_are_masked_when_printed():
    for secret in (
        env.anthropic_api_key,
        env.mock_bank_secret_key,
        env.mock_bank_password,
        env.artifact_signing_key,
    ):
        assert str(secret) == "**********"
    printed = repr(env)
    assert env.mock_bank_password.get_secret_value() not in printed
    assert env.artifact_signing_key.get_secret_value() not in printed


def test_username_is_plain_config_not_a_secret():
    assert isinstance(env.mock_bank_username, str)


def test_short_signing_key_is_rejected():
    with pytest.raises(ValidationError) as exc_info:
        Env(
            _env_file="nonexistent.env",
            anthropic_api_key="x",
            anthropic_model="m",
            mock_bank_secret_key="x",
            mock_bank_username="u",
            mock_bank_password="p",
            artifact_signing_key="too-short",
        )
    assert any(e["loc"] == ("artifact_signing_key",) for e in exc_info.value.errors())
