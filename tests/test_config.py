import pytest
from pydantic import ValidationError

from src.config.env import Env, env
from src.config.settings import Settings, settings


def test_env_loads_all_required_fields():
    assert env.anthropic_api_key
    assert env.anthropic_model
    assert env.mock_bank_base_url
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
