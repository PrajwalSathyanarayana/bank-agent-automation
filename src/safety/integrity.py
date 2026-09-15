"""Artifact signatures: Ed25519 over the artifact's canonical content.

Discovery signs with the private key; replay checks with the trusted public keys. A public
key can check a signature but can't make one, so a machine that only replays can't produce
an artifact replay would trust.
"""
import json
from collections.abc import Mapping
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from src.safety.keys import load_private_key, trusted_keys
from src.types.artifact_schema import Artifact

# Everything else is signed, so a field added later is protected by default.
_UNSIGNED_FIELDS = {
    "metadata": {"integrity_hash", "created_timestamp", "last_updated_timestamp"}
}


class IntegrityCheckFailed(Exception):
    """Raised when an artifact is unsigned or no trusted key's signature matches its content."""


def canonical_bytes(artifact: Artifact) -> bytes:
    content = artifact.model_dump(mode="json", exclude=_UNSIGNED_FIELDS)
    return json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sign(artifact: Artifact, private_key: Optional[Ed25519PrivateKey] = None) -> Artifact:
    """A signed copy of the artifact, by default with the configured private key.
    Raises SigningKeyMissing when there is none."""
    key = private_key or load_private_key()
    signature = key.sign(canonical_bytes(artifact)).hex()
    metadata = artifact.metadata.model_copy(update={"integrity_hash": signature})
    return artifact.model_copy(update={"metadata": metadata})


def verify(artifact: Artifact, trusted: Optional[Mapping[str, Ed25519PublicKey]] = None) -> str:
    """The name of the trusted key that signed the artifact, by default among the configured
    trusted keys. Raises IntegrityCheckFailed when none did."""
    signature = artifact.metadata.integrity_hash
    if signature is None:
        raise IntegrityCheckFailed("artifact is unsigned")
    keys = trusted_keys() if trusted is None else trusted
    if not keys:
        raise IntegrityCheckFailed("no trusted public keys are configured")
    content = canonical_bytes(artifact)
    for name, key in sorted(keys.items()):
        try:
            key.verify(bytes.fromhex(signature), content)
            return name
        except InvalidSignature:
            continue
    raise IntegrityCheckFailed("signature does not match the artifact's content")
