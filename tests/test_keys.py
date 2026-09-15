"""The key commands: make a key pair, and re-sign artifacts saved with the old shared key.
Every test uses its own temporary key locations and store, never the project's."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import SecretStr

from src.config.env import env
from src.config.settings import settings
from src.keys import DONE, NOT_STARTED, SOME_LEFT, main, old_signature
from src.safety.integrity import canonical_bytes, sign, verify
from src.safety.keys import load_private_key, trusted_keys
from src.types.artifact_schema import Artifact, ArtifactMetadata
from src.types.step_schema import ActionType, Step

CAPABILITY = "member_servicing_and_bill_pay"
OLD_KEY = SecretStr("old-shared-signing-key-" + "x" * 32)


@pytest.fixture
def places(tmp_path, monkeypatch) -> Path:
    """Empty key locations and store of this test's own, with the old shared key set."""
    monkeypatch.setattr(settings, "artifact_private_key_path", tmp_path / "secrets" / "signing_key.pem")
    monkeypatch.setattr(settings, "artifact_trusted_keys_dir", tmp_path / "trusted")
    monkeypatch.setattr(settings, "artifact_storage_dir", tmp_path / "artifacts")
    monkeypatch.setattr(env, "artifact_signing_key", OLD_KEY)
    return tmp_path


def _artifact(version="1.0.0") -> Artifact:
    now = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    return Artifact(
        metadata=ArtifactMetadata(capability=CAPABILITY, description="Open the bank.", version=version,
                                  target_url="http://localhost:5000/login",
                                  created_timestamp=now, last_updated_timestamp=now),
        steps=[Step(sequence_index=0, action=ActionType.NAVIGATE, description="Open the start page")],
    )


def _file_for(artifact: Artifact) -> Path:
    folder = settings.artifact_storage_dir / CAPABILITY
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{artifact.metadata.artifact_id}_v{artifact.metadata.version}.json"


def _save_old(artifact: Artifact, key: SecretStr = OLD_KEY) -> Path:
    """Saved as artifacts were before key pairs: with the old keyed signature."""
    data = artifact.model_dump(mode="json")
    data["metadata"]["integrity_hash"] = old_signature(artifact, key)
    path = _file_for(artifact)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _load(path: Path) -> Artifact:
    return Artifact.model_validate_json(path.read_text(encoding="utf-8"))


# --- generate ---

def test_generate_writes_a_private_key_and_a_trusted_public_key(places, capsys):
    assert main(["generate"]) == DONE
    assert load_private_key().public_key() == trusted_keys()["discovery"]
    printed = capsys.readouterr().out
    assert str(settings.artifact_private_key_path) in printed and "discovery.pub" in printed


def test_generate_never_replaces_a_key(places):
    main(["generate"])
    before = settings.artifact_private_key_path.read_bytes()
    assert main(["generate", "--name", "second"]) == NOT_STARTED
    assert settings.artifact_private_key_path.read_bytes() == before
    assert list(trusted_keys()) == ["discovery"]


@pytest.mark.parametrize("name", ["../outside", "Discovery", "", ".hidden"])
def test_a_key_name_that_isnt_a_simple_name_is_refused(places, name):
    assert main(["generate", "--name", name]) == NOT_STARTED
    assert not settings.artifact_private_key_path.exists()


# --- resign ---

def test_an_artifact_whose_old_signature_matches_is_re_signed_in_place(places, capsys):
    main(["generate"])
    artifact = _artifact()
    path = _save_old(artifact)
    assert main(["resign"]) == DONE
    resigned = _load(path)
    assert verify(resigned) == "discovery"
    assert canonical_bytes(resigned) == canonical_bytes(artifact)  # nothing but the signature changed
    assert "1 re-signed, 0 already signed, 0 left as they are." in capsys.readouterr().out


def _edit_description(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["metadata"]["description"] = "Changed by hand."
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    "save",
    [
        pytest.param(lambda artifact: _edit_description(_save_old(artifact)) or _file_for(artifact),
                     id="edited after it was signed"),
        pytest.param(lambda artifact: _save_old(artifact, SecretStr("another-shared-key-" + "y" * 32)),
                     id="signed with another shared key"),
    ],
)
def test_an_artifact_whose_old_signature_doesnt_match_is_left_as_it_is(places, capsys, save):
    main(["generate"])
    path = save(_artifact())
    before = path.read_bytes()
    assert main(["resign"]) == SOME_LEFT
    assert path.read_bytes() == before
    assert "its old signature doesn't match its content" in capsys.readouterr().out


def test_an_artifact_already_signed_with_a_trusted_key_is_left_as_it_is(places, capsys):
    main(["generate"])
    artifact = sign(_artifact())
    path = _file_for(artifact)
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    before = path.read_bytes()
    assert main(["resign"]) == DONE
    assert path.read_bytes() == before
    assert "0 re-signed, 1 already signed, 0 left as they are." in capsys.readouterr().out


def test_a_file_that_isnt_an_artifact_is_left_as_it_is(places, capsys):
    main(["generate"])
    folder = settings.artifact_storage_dir / CAPABILITY
    folder.mkdir(parents=True)
    (folder / "abc_v2.0.0.json").write_text("not json", encoding="utf-8")
    assert main(["resign"]) == SOME_LEFT
    assert "it can't be read as an artifact" in capsys.readouterr().out


def test_resign_with_nothing_saved_is_done(places):
    main(["generate"])
    assert main(["resign"]) == DONE


def test_resign_needs_the_old_shared_key(places, monkeypatch):
    main(["generate"])
    path = _save_old(_artifact())
    monkeypatch.setattr(env, "artifact_signing_key", None)
    before = path.read_bytes()
    assert main(["resign"]) == NOT_STARTED
    assert path.read_bytes() == before


def test_resign_needs_the_private_key(places, capsys):
    assert main(["resign"]) == NOT_STARTED
    assert "python -m src.keys generate" in capsys.readouterr().out


def test_resign_refuses_a_private_key_whose_public_key_isnt_trusted(places, capsys):
    main(["generate"])
    (settings.artifact_trusted_keys_dir / "discovery.pub").unlink()
    path = _save_old(_artifact())
    before = path.read_bytes()
    assert main(["resign"]) == NOT_STARTED
    assert path.read_bytes() == before
    assert "wouldn't be trusted" in capsys.readouterr().out
