import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.config.settings import settings
from src.safety.integrity import sign
from src.storage.artifacts import latest_saved, saved_versions
from src.types.artifact_schema import Artifact, ArtifactMetadata
from src.types.step_schema import ActionType, Step

CAPABILITY = "member_servicing_and_bill_pay"


@pytest.fixture
def store(tmp_path, monkeypatch) -> Path:
    # Each test reads its own temporary store, never the project's artifacts folder.
    monkeypatch.setattr(settings, "artifact_storage_dir", tmp_path)
    return tmp_path / CAPABILITY


def _artifact(version="1.0.0", created=None, capability=CAPABILITY) -> Artifact:
    when = created or datetime.now(timezone.utc)
    artifact = Artifact(
        metadata=ArtifactMetadata(capability=capability, description="Open the bank.", version=version,
                                  target_url="http://localhost:5000/login",
                                  created_timestamp=when, last_updated_timestamp=when),
        steps=[Step(sequence_index=0, action=ActionType.NAVIGATE, description="Open the start page")],
    )
    return sign(artifact)


def _save(store: Path, artifact: Artifact, name=None) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    path = store / (name or f"{artifact.metadata.artifact_id}_v{artifact.metadata.version}.json")
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return path


def _tamper(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["metadata"]["description"] = "Changed by hand."
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_capability_with_nothing_saved_has_no_latest_version(store):
    assert saved_versions(CAPABILITY) == []
    assert latest_saved(CAPABILITY) is None


def test_versions_are_ordered_as_numbers(store):
    for version in ("1.9.0", "1.10.0", "1.2.0"):
        _save(store, _artifact(version))
    assert [version for version, _ in saved_versions(CAPABILITY)] == [(1, 2, 0), (1, 9, 0), (1, 10, 0)]
    latest = latest_saved(CAPABILITY)
    assert (latest.version, latest.trusted, latest.artifact.metadata.version) == ((1, 10, 0), True, "1.10.0")
    assert latest.signed_by == "tests"


def test_files_that_are_not_artifacts_are_ignored(store):
    _save(store, _artifact("1.0.0"))
    (store / "0a1b_v9.0.0.json.tmp").write_text("half written", encoding="utf-8")
    (store / "notes.txt").write_text("not an artifact", encoding="utf-8")
    assert latest_saved(CAPABILITY).version == (1, 0, 0)


def test_a_tampered_highest_version_is_returned_untrusted_never_an_older_one(store):
    _save(store, _artifact("1.0.0"))
    _tamper(_save(store, _artifact("2.0.0")))
    latest = latest_saved(CAPABILITY)
    assert (latest.version, latest.trusted) == ((2, 0, 0), False)
    assert latest.artifact is not None


def test_an_unreadable_file_still_holds_its_version(store):
    _save(store, _artifact("1.0.0"))
    (store / "abc_v3.0.0.json").write_text("not json", encoding="utf-8")
    latest = latest_saved(CAPABILITY)
    assert (latest.version, latest.artifact, latest.trusted) == ((3, 0, 0), None, False)


def test_a_file_whose_name_disagrees_with_its_content_is_not_trusted(store):
    artifact = _artifact("1.0.0")
    _save(store, artifact, name=f"{artifact.metadata.artifact_id}_v4.0.0.json")
    assert (latest_saved(CAPABILITY).version, latest_saved(CAPABILITY).trusted) == ((4, 0, 0), False)


def test_another_capabilitys_artifact_in_this_folder_is_not_trusted(store):
    _save(store, _artifact("1.0.0", capability="other_capability"))
    assert latest_saved(CAPABILITY).trusted is False


def test_a_signature_no_trusted_key_matches_is_not_trusted(store):
    _save(store, _artifact("1.0.0"))
    someone_else = {"someone_else": Ed25519PrivateKey.generate().public_key()}
    latest = latest_saved(CAPABILITY, trusted=someone_else)
    assert (latest.trusted, latest.signed_by) == (False, None)


def test_with_no_trusted_keys_nothing_is_trusted(store):
    _save(store, _artifact("1.0.0"))
    assert latest_saved(CAPABILITY, trusted={}).trusted is False


def test_among_files_sharing_a_version_the_newest_trusted_one_wins(store):
    now = datetime.now(timezone.utc)
    older = _artifact("1.0.0", created=now - timedelta(days=1))
    newer = _artifact("1.0.0", created=now)
    _save(store, older)
    _save(store, newer)
    assert latest_saved(CAPABILITY).artifact.metadata.artifact_id == newer.metadata.artifact_id
    # A still newer file that fails its signature doesn't displace a trusted one.
    _tamper(_save(store, _artifact("1.0.0", created=now + timedelta(days=1))))
    assert latest_saved(CAPABILITY).artifact.metadata.artifact_id == newer.metadata.artifact_id
