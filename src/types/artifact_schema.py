from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import uuid
import re
from .step_schema import Step


class ParamType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SECRET = "secret"


class InputParamDefinition(BaseModel):
    key: str = Field(min_length=1)
    type: ParamType
    required: bool = True
    description: str = Field(min_length=1)
    example_value: Optional[str] = None


class GlobalAssertionType(str, Enum):
    FINAL_URL_MATCH = "final_url_match"
    SUCCESS_BANNER_TEXT = "success_banner_text"
    TERMINAL_DOM_STATE = "terminal_dom_state"


class GlobalAssertion(BaseModel):
    assertion_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    type: GlobalAssertionType
    value: str = Field(min_length=1)


class ArtifactMetadata(BaseModel):
    artifact_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    capability: str = Field(min_length=1)
    version: str = Field(
        description="SemVer format e.g. 1.0.0"
    )
    integrity_hash: str = Field(
        min_length=64,
        max_length=64,
        description="SHA-256 hash of the step sequence"
    )
    author: str = "DiscoveryEngine"
    target_url: str = Field(
        min_length=1,
        description="Entry point URL this artifact was recorded against"
    )
    tenant_override_url: Optional[str] = Field(
        default=None,
        description="Per-tenant URL override for multi-tenant reuse"
    )
    created_timestamp: datetime
    last_updated_timestamp: datetime

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        if not re.match(r"^\d+\.\d+\.\d+$", v):
            raise ValueError(
                "Version must follow SemVer format e.g. 1.0.0"
            )
        return v

    @field_validator("integrity_hash")
    @classmethod
    def validate_hex(cls, v: str) -> str:
        if not re.match(r"^[a-f0-9]{64}$", v):
            raise ValueError(
                "integrity_hash must be a valid SHA-256 hex string"
            )
        return v


class Artifact(BaseModel):
    metadata: ArtifactMetadata
    input_parameters: list[InputParamDefinition] = Field(
        default_factory=list
    )
    steps: list[Step] = Field(min_length=1)
    global_assertions: list[GlobalAssertion] = Field(
        default_factory=list,
        description="Terminal assertions verified after all steps complete"
    )