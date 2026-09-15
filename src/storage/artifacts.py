"""The saved artifacts on disk: each capability's versions, and which one is the latest.

Read by discovery, to number a new save, and by replay, to run the highest version.
Nothing here writes or changes a file.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from collections.abc import Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import ValidationError

from src.config.settings import settings
from src.safety.integrity import IntegrityCheckFailed, verify
from src.safety.keys import trusted_keys
from src.types.artifact_schema import Artifact
from src.types.versioning import Version, parse_version, version_text

# {artifact_id}_v{version}.json. Anything else in the folder, such as a half-written
# .json.tmp, is not an artifact.
_FILE_NAME = re.compile(r"([0-9a-f-]+)_v([0-9]+\.[0-9]+\.[0-9]+)\.json")


@dataclass(frozen=True)
class SavedArtifact:
    """One saved file: its version, the artifact if it could be read, and whether it can be
    trusted — it reads as an artifact, its signature holds, and its content agrees with its
    file name and folder. signed_by names the trusted key whose signature it carries."""

    path: Path
    version: Version
    artifact: Optional[Artifact]
    trusted: bool
    signed_by: Optional[str] = None


def capability_folder(capability: str) -> Path:
    return settings.artifact_storage_dir / capability


def saved_versions(capability: str) -> list[tuple[Version, Path]]:
    """Every saved file of the capability with the version its name gives, lowest first.

    Numbering comes from the names, so a file that can't be read still keeps its version
    taken: no later save reuses it.
    """
    folder = capability_folder(capability)
    if not folder.is_dir():
        return []
    found = []
    for path in folder.iterdir():
        name = _FILE_NAME.fullmatch(path.name)
        if name is not None and path.is_file():
            found.append((parse_version(name.group(2)), path))
    return sorted(found)


def latest_saved(capability: str, trusted: Optional[Mapping[str, Ed25519PublicKey]] = None
                 ) -> Optional[SavedArtifact]:
    """The capability's highest version, or None if nothing is saved. Signatures are checked
    against trusted (by default the configured trusted public keys).

    The highest version is returned even when it can't be trusted, never an older one in
    its place: replay refuses it, discovery numbers past it. Two files share a version only
    if they were saved before versioning existed; then the newest trusted one wins.
    """
    files = saved_versions(capability)
    if not files:
        return None
    highest = files[-1][0]
    keys = trusted_keys() if trusted is None else trusted
    tied = [_read(path, version, capability, keys) for version, path in files if version == highest]
    return max(tied, key=_preference)


def _read(path: Path, version: Version, capability: str, trusted: Mapping[str, Ed25519PublicKey]) -> SavedArtifact:
    try:
        artifact = Artifact.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return SavedArtifact(path, version, None, trusted=False)
    try:
        signed_by = verify(artifact, trusted)
    except IntegrityCheckFailed:
        return SavedArtifact(path, version, artifact, trusted=False)
    # A renamed or moved file is signed content in the wrong place: not trusted either.
    metadata = artifact.metadata
    agrees = (metadata.version == version_text(version) and metadata.capability == capability
              and path.name.startswith(f"{metadata.artifact_id}_v"))
    return SavedArtifact(path, version, artifact, trusted=agrees, signed_by=signed_by)


def _preference(saved: SavedArtifact) -> tuple[bool, float]:
    created = saved.artifact.metadata.created_timestamp.timestamp() if saved.artifact else float("-inf")
    return saved.trusted, created
