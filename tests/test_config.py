from decimal import Decimal

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


def test_settings_operator_timeout_is_ten_minutes():
    # An operator gets 10 minutes to act on a handoff before it times out.
    assert settings.operator_timeout_ms == 600_000


def test_settings_is_pydantic_model():
    assert isinstance(settings, Settings)
    dumped = settings.model_dump()
    assert "artifact_storage_dir" in dumped


def test_settings_discovery_limits():
    assert settings.discovery_max_steps == 40
    assert settings.discovery_timeout_ms == 900_000
    assert settings.discovery_llm_call_timeout_ms == 90_000
    assert settings.discovery_page_action_timeout_ms == 30_000
    assert settings.discovery_llm_retries == 1


def test_settings_discovery_perception():
    assert settings.discovery_viewport_width == 1280
    assert settings.discovery_viewport_height == 800
    assert settings.discovery_device_scale_factor == 1
    assert settings.discovery_max_elements == 100


# --- secrets held as SecretStr ---

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


REQUIRED = {"anthropic_api_key": "x", "anthropic_model": "m", "mock_bank_secret_key": "x",
            "mock_bank_username": "u", "mock_bank_password": "p", "artifact_signing_key": "k" * 32}


def test_the_auto_pay_limit_defaults_to_a_thousand_dollars_held_exactly():
    configured = Env(_env_file="nonexistent.env", **REQUIRED)
    assert (configured.auto_execute_limit, configured.auto_execute_currency) == (Decimal("1000.00"), "USD")
    assert isinstance(configured.auto_execute_limit, Decimal)


@pytest.mark.parametrize(
    "setting",
    [
        pytest.param({"auto_execute_limit": "0"}, id="a limit of zero"),
        pytest.param({"auto_execute_limit": "-50"}, id="a negative limit"),
        pytest.param({"auto_execute_limit": "10.005"}, id="finer than a cent"),
        pytest.param({"auto_execute_limit": "a lot"}, id="not an amount"),
        pytest.param({"auto_execute_currency": "usd"}, id="currency not in capitals"),
    ],
)
def test_an_unclear_auto_pay_limit_is_rejected(setting):
    with pytest.raises(ValidationError):
        Env(_env_file="nonexistent.env", **REQUIRED, **setting)


def test_the_mock_bank_test_switches_are_off_unless_set():
    configured = Env(_env_file="nonexistent.env", **REQUIRED)
    assert (configured.mock_bank_renamed_menu, configured.mock_bank_slow_pages_ms) == (False, 0)


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
