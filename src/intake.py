"""The intake: a request in plain words → one of the declared tasks and its inputs, or the
reason it can't be run.

One model call, answered with a strict tool: one tool per declared task with its inputs
typed as the contract declares them (null when the request doesn't state one), plus one for
a request that is none of them. The model only reads the sentence; it never acts on the
bank. What it returns is checked again here, against the contract and against the request
itself: a value the request doesn't state is never used, so a missing or invented value
becomes "needs input", naming what's missing.
"""
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Protocol, Union

import anthropic

from src.config.env import env
from src.discovery.artifact_builder import ArtifactContract
from src.locating.checks import phrase_matches
from src.locating.values import number_pattern
from src.observability.summary import asked_line, readable_values
from src.types.artifact_schema import InputParamDefinition, ParamType
from src.types.placeholders import MissingValue, fill_text

NONE_OF_THEM = "not_a_known_task"
_JSON_TYPES = {ParamType.STRING: "string", ParamType.NUMBER: "number", ParamType.BOOLEAN: "boolean"}

INTAKE_PROMPT = """You turn a staff member's request into one of the tasks offered as tools, for a credit union's back-office system. Call exactly one tool.
- Choose the task the request asks for, and fill each input only with a value the request states: numbers as plain numbers (50, 1050.5), names exactly as the request writes them.
- If the request doesn't state a value, set that input to null. Never guess, infer or supply a default.
- If the request asks for anything other than one of these tasks, or for several tasks at once, call not_a_known_task."""

Value = Union[str, float, bool]


@dataclass(frozen=True)
class IntakeAnswer:
    """What the intake made of a request. kind "run": capability and inputs are ready for the
    router. "needs_input": missing names the inputs to ask for. "not_supported": no declared
    task fits. message is what the person reads."""

    kind: Literal["run", "needs_input", "not_supported"]
    message: str
    capability: Optional[str] = None
    inputs: Mapping[str, Value] = field(default_factory=dict)
    missing: tuple[str, ...] = ()

    def to_json(self) -> str:
        return json.dumps({key: value for key, value in asdict(self).items() if value not in (None, (), {})},
                          indent=2, ensure_ascii=False)


class IntakeUnavailable(RuntimeError):
    """The model couldn't be asked (network, rate limit, server error); nothing was run."""


class IntakeModel(Protocol):
    async def choose(self, request: str, tools: list[dict[str, Any]]) -> Optional[tuple[str, dict[str, Any]]]:
        """The tool the model called and its input; None when it called none or declined."""
        ...


class ClaudeIntakeModel:
    """The real intake model: one short call, low effort; a declined request is re-run once
    on a fallback model inside the same call."""

    def __init__(self, client: Optional[anthropic.AsyncAnthropic] = None) -> None:
        self._client = client or anthropic.AsyncAnthropic(api_key=env.anthropic_api_key.get_secret_value())

    async def choose(self, request: str, tools: list[dict[str, Any]]) -> Optional[tuple[str, dict[str, Any]]]:
        try:
            response = await self._client.beta.messages.create(
                model=env.anthropic_model,
                max_tokens=4_000,
                system=INTAKE_PROMPT,
                tools=tools,
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                thinking={"type": "adaptive"},
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": request}],
            )
        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError) as error:
            raise IntakeUnavailable(type(error).__name__) from None
        if response.stop_reason == "refusal":
            return None
        call = next((block for block in response.content if getattr(block, "type", None) == "tool_use"), None)
        return (call.name, dict(call.input)) if call is not None else None


def intake_tools(contracts: Mapping[str, ArtifactContract]) -> list[dict[str, Any]]:
    """One strict tool per declared task (inputs typed as declared, null when not stated),
    then the tool for a request that is none of them. Secrets are never taken from a request."""
    tools = []
    for capability in sorted(contracts):
        contract = contracts[capability]
        properties = {
            parameter.key: {"type": [_JSON_TYPES[parameter.type], "null"], "description": parameter.description}
            for parameter in contract.input_parameters if parameter.type in _JSON_TYPES
        }
        tools.append({
            "name": capability,
            "description": f"The task: {task_line(contract)} Set an input to null when the request doesn't state it.",
            "strict": True,
            "input_schema": {"type": "object", "properties": properties, "required": list(properties),
                             "additionalProperties": False},
        })
    tools.append({
        "name": NONE_OF_THEM,
        "description": "The request is none of the tasks above, or asks for several at once.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"],
                         "additionalProperties": False},
    })
    return tools


def task_line(contract: ArtifactContract) -> str:
    """A task as a person reads it: its goal with each input named ("For member <member id>, …")."""
    names = {parameter.key: f"<{parameter.key.replace('_', ' ')}>" for parameter in contract.input_parameters}
    try:
        return fill_text(contract.description, names)
    except MissingValue:
        return contract.description


async def interpret(request: str, contracts: Mapping[str, ArtifactContract], model: IntakeModel) -> IntakeAnswer:
    """The intake's answer for one request. Never raises for what the model says; only an
    unreachable model raises IntakeUnavailable."""
    if not request.strip():
        return IntakeAnswer("needs_input", "The request is empty: say what you'd like done.")
    chosen = await model.choose(request, intake_tools(contracts))
    if chosen is None or chosen[0] not in contracts:
        return _not_supported(contracts)
    capability, given = chosen
    contract = contracts[capability]
    inputs: dict[str, Value] = {}
    missing: list[InputParamDefinition] = []
    for parameter in contract.input_parameters:
        value = _stated(given.get(parameter.key), parameter, request)
        if value is None:
            if parameter.required:
                missing.append(parameter)
            continue
        inputs[parameter.key] = value
    if missing:
        needed = "; ".join(parameter.description.lower() for parameter in missing)
        return IntakeAnswer("needs_input", f"To do that I also need: {needed}. Please say it in the request, "
                            "with any amount written in figures.", capability=capability, inputs=inputs,
                            missing=tuple(parameter.key for parameter in missing))
    shown, _ = readable_values(inputs, {}, contract.confirmation_checks, contract.output_definitions)
    return IntakeAnswer("run", f"Understood as: {asked_line(capability, contract.description, shown)}",
                        capability=capability, inputs=inputs)


def _stated(value: Any, parameter: InputParamDefinition, request: str) -> Optional[Value]:
    """The value typed as declared, and only if the request states it; None otherwise."""
    if value is None:
        return None
    if parameter.type == ParamType.NUMBER:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        # Stated in figures somewhere in the request, in any common form ("50", "$50.00").
        return float(value) if number_pattern(float(value)).search(request) else None
    if parameter.type == ParamType.BOOLEAN:
        return value if isinstance(value, bool) else None
    if not isinstance(value, str) or not value.strip():
        return None
    # Words the request itself contains, whole words in any case: never a name the model supplied.
    return value.strip() if phrase_matches(request, value.strip()) else None


def _not_supported(contracts: Mapping[str, ArtifactContract]) -> IntakeAnswer:
    known = "\n".join(f"- {task_line(contracts[capability])}" for capability in sorted(contracts))
    return IntakeAnswer("not_supported", f"I can't do that yet. The tasks I know are:\n{known}")
