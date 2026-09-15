"""The router: the system's one entry point. A request in plain words goes to the intake; an
understood request goes to replay when its task has a saved artifact, and to discovery when
it has none (discovery by default).

A saved artifact that fails its signature still goes to replay, which refuses it with its
own reason: nothing is quietly relearned over a tampered artifact.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

from src.catalog import CONTRACTS
from src.discovery.agent import DiscoveryRequest, Model, discover
from src.discovery.artifact_builder import ArtifactContract
from src.handoff.session_manager import OperatorSetup
from src.intake import IntakeAnswer, IntakeModel, interpret
from src.observability.logger import RunLogger
from src.replay.executor import ReplayRequest, replay
from src.storage.artifacts import latest_saved
from src.types.result_schema import ExecutionResult


@dataclass(frozen=True)
class Handled:
    """What became of a request: the intake's answer and, when it was run, the run's result."""

    intake: IntakeAnswer
    result: Optional[ExecutionResult] = None


async def handle(
    request: str,
    *,
    intake_model: IntakeModel,
    discovery_model: Model,
    contracts: Mapping[str, ArtifactContract] = CONTRACTS,
    headless: bool = True,
    operator: Optional[OperatorSetup] = None,
    trace: bool = False,
    slow_mo_ms: Optional[int] = None,
) -> Handled:
    """Interpret the request and, if it's understood, run its task the way it can be run now."""
    answer = await interpret(request, contracts, intake_model)
    if answer.kind != "run":
        return Handled(answer)
    capability = answer.capability
    if latest_saved(capability) is None:
        logger = RunLogger("DISCOVERY", capability=capability)
        result = await discover(DiscoveryRequest(contracts[capability], answer.inputs), discovery_model, logger,
                                headless=headless, operator=operator, trace=trace, slow_mo_ms=slow_mo_ms)
    else:
        logger = RunLogger("REPLAY", capability=capability)
        result = await replay(ReplayRequest(capability, answer.inputs), logger, headless=headless, operator=operator,
                              trace=trace, slow_mo_ms=slow_mo_ms)
    return Handled(answer, result)
