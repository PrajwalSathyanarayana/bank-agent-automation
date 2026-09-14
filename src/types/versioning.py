"""Artifact versions: reading them, comparing them, and which part a new recording bumps.

When a capability that already has an artifact is discovered again, the new one gets
the next version, bumped by what changed against the latest one:
- major: the contract — what the capability takes, returns and can answer, and where it
  starts. A calling agent must not assume the new version behaves like the old;
- minor: the flow — the sequence of actions, their risk, what they type or read;
- patch: only details — how steps find their elements and check the page.
The model's wording of a step (its description) is not a change: two runs of the same
flow phrase their steps differently, and a new version for that would be noise.
Versions only go up, so no two saves of a capability share one.
"""
import re
from enum import Enum
from typing import Any

from .artifact_schema import Artifact, ArtifactMetadata

Version = tuple[int, int, int]
FIRST_VERSION: Version = (1, 0, 0)
_VERSION = re.compile(r"([0-9]+)\.([0-9]+)\.([0-9]+)")

# What the engineer declared: a change here is a major change.
_CONTRACT_METADATA = ("capability", "description", "target_url", "tenant_override_url")
_CONTRACT_GROUPS = ("input_parameters", "output_definitions", "credentials", "known_outcomes", "allowed_paths")
# Every top-level field has a rule: the contract above, or the steps and final checks.
_COMPARED = {"metadata", *_CONTRACT_GROUPS, "steps", "global_assertions"}
# Metadata that says which file this is, not what the capability does.
_METADATA_IDENTITY = {"artifact_id", "version", "integrity_hash", "author",
                      "created_timestamp", "last_updated_timestamp"}
# New on every save, or the model's wording of a step: neither says anything about behaviour.
_IGNORED = {"step_id", "checkpoint_id", "assertion_id", "description"}


class Change(str, Enum):
    NONE = "none"
    PATCH = "patch"
    MINOR = "minor"
    MAJOR = "major"


class UnruledField(RuntimeError):
    """An artifact field with no versioning rule: a code change is needed, not a retry."""


def parse_version(text: str) -> Version:
    """"1.10.0" → (1, 10, 0), so versions compare as numbers: 1.10.0 is above 1.9.0."""
    found = _VERSION.fullmatch(text)
    if found is None:
        raise ValueError(f"not a version: {text!r}")
    major, minor, patch = (int(part) for part in found.groups())
    return major, minor, patch


def version_text(version: Version) -> str:
    return ".".join(str(part) for part in version)


def bump(version: Version, change: Change) -> Version:
    """The version after a change of this kind; unchanged for Change.NONE."""
    major, minor, patch = version
    if change == Change.MAJOR:
        return major + 1, 0, 0
    if change == Change.MINOR:
        return major, minor + 1, 0
    if change == Change.PATCH:
        return major, minor, patch + 1
    return version


def change_between(old: Artifact, new: Artifact) -> Change:
    """The largest kind of change from old to new. Generated ids, the model's step wording,
    timestamps, the version and the signature are ignored: two recordings of the same
    flow are Change.NONE.

    Raises UnruledField for a field the rules don't cover, so a field added to the schema
    later can't slip through as "no change".
    """
    _check_every_field_has_a_rule()
    if _contract(old) != _contract(new):
        return Change.MAJOR
    if _flow(old) != _flow(new):
        return Change.MINOR
    if _details(old) != _details(new):
        return Change.PATCH
    return Change.NONE


def _check_every_field_has_a_rule() -> None:
    unruled = sorted(set(Artifact.model_fields) - _COMPARED)
    unruled += [f"metadata.{name}" for name in sorted(
        set(ArtifactMetadata.model_fields) - set(_CONTRACT_METADATA) - _METADATA_IDENTITY)]
    if unruled:
        raise UnruledField(f"no versioning rule for {', '.join(unruled)}; give each one before saving")


def _contract(artifact: Artifact) -> tuple[dict[str, Any], dict[str, Any]]:
    data = artifact.model_dump(mode="json")
    return ({name: data["metadata"][name] for name in _CONTRACT_METADATA},
            {group: data[group] for group in _CONTRACT_GROUPS})


def _flow(artifact: Artifact) -> list[tuple[Any, ...]]:
    # What each step does, in order; how it finds its element or checks the page is detail.
    return [(step.action, step.safety_tier, step.input_value, step.output_key) for step in artifact.steps]


def _details(artifact: Artifact) -> Any:
    return _behaviour_only(artifact.model_dump(mode="json", include={"steps", "global_assertions"}))


def _behaviour_only(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _behaviour_only(value) for key, value in node.items() if key not in _IGNORED}
    if isinstance(node, list):
        return [_behaviour_only(value) for value in node]
    return node
