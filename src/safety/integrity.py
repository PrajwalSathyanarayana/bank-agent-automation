import hashlib
import hmac
import json

from pydantic import SecretStr

from src.types.artifact_schema import Artifact

# Everything else is signed, so a field added later is protected by default.
_UNSIGNED_FIELDS = {
    "metadata": {"integrity_hash", "created_timestamp", "last_updated_timestamp"}
}


class IntegrityCheckFailed(Exception):
    """Raised when an artifact is unsigned or its content no longer matches its signature."""


def canonical_bytes(artifact: Artifact) -> bytes:
    content = artifact.model_dump(mode="json", exclude=_UNSIGNED_FIELDS)
    return json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def compute_signature(artifact: Artifact, key: SecretStr) -> str:
    return hmac.new(
        key.get_secret_value().encode("utf-8"), canonical_bytes(artifact), hashlib.sha256
    ).hexdigest()


def sign(artifact: Artifact, key: SecretStr) -> Artifact:
    signature = compute_signature(artifact, key)
    metadata = artifact.metadata.model_copy(update={"integrity_hash": signature})
    return artifact.model_copy(update={"metadata": metadata})


def verify(artifact: Artifact, key: SecretStr) -> None:
    signature = artifact.metadata.integrity_hash
    if signature is None:
        raise IntegrityCheckFailed("artifact is unsigned")
    # Constant-time compare; the expected signature is never put in the message,
    # since it is the valid signature for the tampered content.
    if not hmac.compare_digest(signature, compute_signature(artifact, key)):
        raise IntegrityCheckFailed("signature does not match the artifact's content")
