from collections.abc import Sequence
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from src.config.env import env
from src.types.routes import route_allowed
from src.types.step_schema import ActionType


class AllowlistViolation(Exception):
    """Raised when a step attempts to act outside the configured allowlist."""


def _default_allowed_domains() -> list[str]:
    hostname = urlparse(env.mock_bank_base_url).hostname
    return [hostname] if hostname else []


class AllowlistConfig(BaseModel):
    allowed_domains: list[str] = Field(default_factory=_default_allowed_domains)
    allowed_action_types: list[ActionType] = Field(
        default_factory=lambda: list(ActionType)
    )


ALLOWLIST = AllowlistConfig()


def check_domain(url: str, config: AllowlistConfig = ALLOWLIST) -> None:
    hostname = urlparse(url).hostname
    if hostname not in config.allowed_domains:
        raise AllowlistViolation(
            f"Domain '{hostname}' is not in the allowlist {config.allowed_domains}. "
            f"URL: {url}"
        )


def check_action_type(action: ActionType, config: AllowlistConfig = ALLOWLIST) -> None:
    if action not in config.allowed_action_types:
        raise AllowlistViolation(
            f"Action type '{action}' is not in the allowlist {config.allowed_action_types}."
        )


def check_route(url: str, allowed_paths: Sequence[str]) -> None:
    """The page must be one the capability declared it may visit. The domain check keeps the
    agent inside the bank; this keeps each capability inside its own pages. An empty list
    allows any page on the host."""
    if not allowed_paths:
        return
    path = urlparse(url).path or "/"
    if not route_allowed(path, allowed_paths):
        raise AllowlistViolation(f"The page '{path}' is not one this capability may visit.")


def enforce_safety(
    url: str, action: ActionType, config: AllowlistConfig = ALLOWLIST, allowed_paths: Sequence[str] = ()
) -> None:
    check_domain(url, config)
    check_route(url, allowed_paths)
    check_action_type(action, config)
