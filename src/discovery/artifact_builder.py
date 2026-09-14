"""Turns a discovery's recording into a saved artifact: validate, scan, sign, write.

Nothing is written unless every stage passes. A failure ends discovery with a HARD_ABORT
reason instead: an engineer fixes an invalid artifact, not a person at the browser.
"""
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from src.config.env import env
from src.config.settings import settings
from src.discovery.backstop import ScanInputs, scan_artifact
from src.observability.logger import RunLogger
from src.safety.integrity import sign
from src.types.artifact_schema import (
    Artifact,
    ArtifactMetadata,
    CredentialDefinition,
    InputParamDefinition,
    KnownOutcome,
    OutputParamDefinition,
)
from src.types.result_schema import ErrorDetail
from src.types.step_schema import Step

# Every discovery saves a new artifact at the first version; later versions come from edits.
FIRST_VERSION = "1.0.0"
INVALID_CODE = "ARTIFACT_INVALID"
# The capability names a folder, so it must be a plain name that can't leave the store.
_FOLDER_NAME = re.compile(r"[a-z][a-z0-9_]*")


class UnsignedArtifact(RuntimeError):
    """Raised when asked to write an artifact that carries no signature."""


@dataclass(frozen=True)
class ArtifactContract:
    """What the engineer declared for the capability in the discovery request."""

    capability: str
    # The goal template, e.g. "For member {member_id}, pay {amount} to {payee_name}."
    description: str
    target_url: str
    input_parameters: list[InputParamDefinition]
    output_definitions: list[OutputParamDefinition]
    credentials: list[CredentialDefinition]
    # The legitimate answers other than success; a capability may have none.
    known_outcomes: list[KnownOutcome] = field(default_factory=list)


@dataclass(frozen=True)
class BuildResult:
    """The saved artifact and its file, or the reason nothing was saved."""

    artifact: Optional[Artifact] = None
    path: Optional[Path] = None
    error: Optional[ErrorDetail] = None


def build_and_save(
    contract: ArtifactContract, steps: list[Step], inputs: ScanInputs, logger: RunLogger
) -> BuildResult:
    """Validate, scan, sign and write the artifact, logging the scan's report and the save.

    Every stage must pass before anything is written. The scan's report is logged on
    pass and on abort, so a failed save still shows what was converted and where it
    stopped.
    """
    now = datetime.now(timezone.utc)
    try:
        artifact = Artifact(
            metadata=ArtifactMetadata(
                capability=contract.capability,
                description=contract.description,
                version=FIRST_VERSION,
                target_url=contract.target_url,
                created_timestamp=now,
                last_updated_timestamp=now,
            ),
            input_parameters=list(contract.input_parameters),
            output_definitions=list(contract.output_definitions),
            credentials=list(contract.credentials),
            known_outcomes=list(contract.known_outcomes),
            steps=list(steps),
        )
    except ValidationError as error:
        return BuildResult(error=ErrorDetail(code=INVALID_CODE, message=_validation_message(error)))
    if not _has_a_check(artifact):
        return BuildResult(error=ErrorDetail(
            code=INVALID_CODE,
            message="the artifact has no checkpoint and no final check, so replay could never tell a step failed",
        ))

    scanned = scan_artifact(artifact, inputs)
    logger.backstop_scan(**scanned.report.log_fields())
    if scanned.error is not None:
        return BuildResult(error=scanned.error)

    signed = sign(scanned.artifact, env.artifact_signing_key)
    path = write_artifact(signed)
    logger.artifact_saved(signed.metadata.artifact_id, signed.metadata.version, signed.metadata.integrity_hash)
    return BuildResult(artifact=signed, path=path)


def write_artifact(artifact: Artifact) -> Path:
    """Write a signed artifact to {store}/{capability}/{artifact_id}_v{version}.json.

    Refuses an unsigned artifact. Writes a temporary file first and moves it into place,
    so a crash never leaves half an artifact for replay to find.
    """
    metadata = artifact.metadata
    if metadata.integrity_hash is None:
        raise UnsignedArtifact("an artifact is signed before it is written")
    if not _FOLDER_NAME.fullmatch(metadata.capability):
        raise ValueError("the capability must be a simple lowercase name to be stored as a folder")
    folder = settings.artifact_storage_dir / metadata.capability
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{metadata.artifact_id}_v{metadata.version}.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def _has_a_check(artifact: Artifact) -> bool:
    # Either a checkpoint on some step or a final check satisfies it.
    return any(step.checkpoints for step in artifact.steps) or bool(artifact.global_assertions)


def _validation_message(error: ValidationError) -> str:
    # Where and what, never the offending value: it could be member data or a secret.
    problems = [
        f"{'.'.join(str(part) for part in problem['loc']) or 'artifact'}: {problem['msg']}"
        for problem in error.errors(include_input=False, include_url=False, include_context=False)
    ]
    return "the artifact is invalid: " + "; ".join(problems)
