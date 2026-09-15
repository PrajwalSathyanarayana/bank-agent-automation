"""The key files: the private key that signs artifacts, and the public keys that check them.

The private key stays on the machine that runs discovery, outside the repository: whoever
holds it can sign. The public keys are committed with the code: any copy of the project can
check that an artifact came from a trusted signer, but no public key can sign one.
"""
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from src.config.settings import settings

PUBLIC_KEY_SUFFIX = ".pub"


class SigningKeyMissing(Exception):
    """No usable private key at the configured path, so nothing can be signed."""


def load_private_key(path: Optional[Path] = None) -> Ed25519PrivateKey:
    """The private key at path (by default the configured one). The message names the path
    but never the key."""
    path = path or settings.artifact_private_key_path
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except FileNotFoundError:
        raise SigningKeyMissing(f"no private key at {path}") from None
    # A file that isn't a key, or is protected by a password: not usable here.
    except (OSError, ValueError, TypeError):
        raise SigningKeyMissing(f"the file at {path} isn't a usable private key") from None
    if not isinstance(key, Ed25519PrivateKey):
        raise SigningKeyMissing(f"the key at {path} isn't an Ed25519 key")
    return key


def trusted_keys(folder: Optional[Path] = None) -> dict[str, Ed25519PublicKey]:
    """Each trusted public key in folder (by default the configured one), by its file name
    without .pub. A file that isn't an Ed25519 public key is left out: it can only make
    fewer artifacts trusted, never more."""
    folder = folder or settings.artifact_trusted_keys_dir
    if not folder.is_dir():
        return {}
    keys = {}
    for path in sorted(folder.glob(f"*{PUBLIC_KEY_SUFFIX}")):
        try:
            key = serialization.load_pem_public_key(path.read_bytes())
        except (OSError, ValueError):
            continue
        if isinstance(key, Ed25519PublicKey):
            keys[path.stem] = key
    return keys


def write_key_pair(private_key: Ed25519PrivateKey, private_path: Path, public_path: Path) -> None:
    """Write the private key and its public key as PEM files. Never overwrites: a lost private
    key can't be recovered, and replacing a public key stops artifacts it signed from being
    trusted."""
    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    for path in (private_path, public_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists")
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    with open(private_path, "xb") as file:
        file.write(private_pem)
    with open(public_path, "xb") as file:
        file.write(public_pem)
