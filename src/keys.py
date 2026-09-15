"""Key commands: make the key pair that signs artifacts, and move artifacts signed with the
old shared key onto it.

    python -m src.keys generate [--name NAME]
    python -m src.keys resign

generate writes the private key (kept out of git) and its public key into the trusted
folder (committed). resign re-signs each saved artifact whose old keyed signature still
matches its content; one that doesn't match, or can't be read, is listed and left as it is,
so an edited artifact never becomes trusted by being re-signed.

Exit codes: 0 done; 1 some artifacts were left as they are; 2 couldn't start.
"""
import argparse
import hashlib
import hmac
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import SecretStr, ValidationError

from src.config.env import env
from src.config.settings import settings
from src.safety.integrity import IntegrityCheckFailed, canonical_bytes, sign, verify
from src.safety.keys import PUBLIC_KEY_SUFFIX, SigningKeyMissing, load_private_key, trusted_keys, write_key_pair
from src.storage.artifacts import saved_versions
from src.types.artifact_schema import Artifact

DONE, SOME_LEFT, NOT_STARTED = 0, 1, 2
# A file name in the trusted folder: no path separators, nothing hidden.
_KEY_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
_OLD_SIGNATURE = re.compile(r"[a-f0-9]{64}")
_NEW_SIGNATURE = re.compile(r"[a-f0-9]{128}")


@dataclass
class ResignReport:
    resigned: list[Path] = field(default_factory=list)
    already_signed: list[Path] = field(default_factory=list)
    left_alone: list[tuple[Path, str]] = field(default_factory=list)


def old_signature(artifact: Artifact, key: SecretStr) -> str:
    """The keyed HMAC-SHA256 artifacts carried before key pairs, over the same content."""
    return hmac.new(key.get_secret_value().encode("utf-8"), canonical_bytes(artifact), hashlib.sha256).hexdigest()


def generate(name: str) -> Path:
    """A new key pair: the private key at the configured path, the public key named name in
    the trusted folder. Returns the public key's path; never replaces an existing file."""
    public_path = settings.artifact_trusted_keys_dir / f"{name}{PUBLIC_KEY_SUFFIX}"
    write_key_pair(Ed25519PrivateKey.generate(), settings.artifact_private_key_path, public_path)
    return public_path


def resign(old_key: SecretStr, private_key: Ed25519PrivateKey) -> ResignReport:
    """Re-sign every saved artifact whose old signature matches its content, in place."""
    report = ResignReport()
    store = settings.artifact_storage_dir
    trusted = trusted_keys()
    folders = sorted(path for path in store.iterdir() if path.is_dir()) if store.is_dir() else []
    for folder in folders:
        for _, path in saved_versions(folder.name):
            _resign_one(path, old_key, private_key, trusted, report)
    return report


def _resign_one(path: Path, old_key: SecretStr, private_key: Ed25519PrivateKey,
                trusted: Mapping[str, Ed25519PublicKey], report: ResignReport) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        signature = data["metadata"].get("integrity_hash")
        data["metadata"]["integrity_hash"] = None
        artifact = Artifact.model_validate(data)
    except (OSError, ValueError, ValidationError, KeyError, TypeError, AttributeError):
        report.left_alone.append((path, "it can't be read as an artifact"))
        return
    if isinstance(signature, str) and _NEW_SIGNATURE.fullmatch(signature):
        metadata = artifact.metadata.model_copy(update={"integrity_hash": signature})
        try:
            verify(artifact.model_copy(update={"metadata": metadata}), trusted)
        except IntegrityCheckFailed:
            report.left_alone.append((path, "its signature matches no trusted key"))
            return
        report.already_signed.append(path)
        return
    if not (isinstance(signature, str) and _OLD_SIGNATURE.fullmatch(signature)
            and hmac.compare_digest(signature, old_signature(artifact, old_key))):
        report.left_alone.append((path, "its old signature doesn't match its content"))
        return
    # A temporary file moved into place, so a crash never leaves half an artifact.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(sign(artifact, private_key).model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, path)
    report.resigned.append(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.keys", description="Manage the keys that sign artifacts.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("generate", help="make the key pair that signs artifacts")
    make.add_argument("--name", default="discovery", help="the public key's name in the trusted folder")
    commands.add_parser("resign", help="re-sign artifacts saved with the old shared key")
    args = parser.parse_args(argv)
    return _generate(args.name) if args.command == "generate" else _resign()


def _generate(name: str) -> int:
    if not _KEY_NAME.fullmatch(name):
        print("The name must be lowercase letters, digits, - or _, starting with a letter or digit.")
        return NOT_STARTED
    try:
        public_path = generate(name)
    except FileExistsError as error:
        print(f"Nothing written: {error}. A key is never replaced: a lost private key can't be recovered.")
        return NOT_STARTED
    print(f"Private key: {settings.artifact_private_key_path}")
    print("  Keep it out of git and off shared drives: anyone holding it can sign artifacts.")
    print(f"Public key: {public_path}")
    print("  Commit it: replay trusts the artifacts it checks.")
    return DONE


def _resign() -> int:
    if env.artifact_signing_key is None:
        print("ARTIFACT_SIGNING_KEY isn't set: it's needed once, to check each artifact's old signature.")
        return NOT_STARTED
    try:
        private_key = load_private_key()
    except SigningKeyMissing as error:
        print(f"{error}: run 'python -m src.keys generate' first.")
        return NOT_STARTED
    if private_key.public_key() not in trusted_keys().values():
        print(f"The private key's public key isn't in {settings.artifact_trusted_keys_dir}, "
              "so re-signed artifacts wouldn't be trusted. Nothing was changed.")
        return NOT_STARTED
    report = resign(env.artifact_signing_key, private_key)
    for path in report.resigned:
        print(f"Re-signed: {path}")
    for path in report.already_signed:
        print(f"Already signed with a trusted key: {path}")
    for path, why in report.left_alone:
        print(f"Left as it is ({why}): {path}")
    print(f"{len(report.resigned)} re-signed, {len(report.already_signed)} already signed, "
          f"{len(report.left_alone)} left as they are.")
    return SOME_LEFT if report.left_alone else DONE


if __name__ == "__main__":
    sys.exit(main())
