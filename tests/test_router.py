"""The router: an understood request → replay when its task has a saved artifact, discovery
when it has none; a request that isn't understood runs nothing. The engines are stubbed
here: each is tested on its own elsewhere."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import src.router as router_module
from src.catalog import BILL_PAY
from src.config.settings import settings
from src.handoff.session_manager import OperatorSetup
from src.router import handle
from src.types.result_schema import EvidencePaths, ExecutionResult, ExecutionStatus

REQUEST = "For member 10234, pay 50 to Sunbelt Electric Co"
UNDERSTOOD = (BILL_PAY, {"member_id": "10234", "amount": 50, "payee_name": "Sunbelt Electric Co"})


class _Intake:
    def __init__(self, choice):
        self.choice = choice

    async def choose(self, request, tools):
        return self.choice


def _result(mode) -> ExecutionResult:
    now = datetime.now(timezone.utc)
    return ExecutionResult(capability=BILL_PAY, mode=mode, status=ExecutionStatus.SUCCESS, start_time=now,
                           end_time=now, duration_ms=0,
                           evidence_paths=EvidencePaths(log_file="run_log.json", screenshots_dir="screenshots"))


@pytest.fixture
def engines(monkeypatch, tmp_path):
    """Both engines stubbed; records which one ran and with what."""
    monkeypatch.setattr(settings, "evidence_dir", tmp_path)
    ran = []

    async def fake_discover(request, model, logger, *, headless, operator):
        ran.append(("discovery", request.contract.capability, dict(request.input_values), headless, operator))
        return _result("DISCOVERY")

    async def fake_replay(request, logger, *, headless, operator):
        ran.append(("replay", request.capability, dict(request.inputs), headless, operator))
        return _result("REPLAY")

    monkeypatch.setattr(router_module, "discover", fake_discover)
    monkeypatch.setattr(router_module, "replay", fake_replay)
    return ran


def _saved(monkeypatch, saved):
    monkeypatch.setattr(router_module, "latest_saved", lambda capability: saved)


@pytest.mark.anyio
async def test_a_task_never_learned_is_discovered(engines, monkeypatch):
    _saved(monkeypatch, None)
    handled = await handle(REQUEST, intake_model=_Intake(UNDERSTOOD), discovery_model=object())
    assert [run[:3] for run in engines] == [
        ("discovery", BILL_PAY, {"member_id": "10234", "amount": 50.0, "payee_name": "Sunbelt Electric Co"})]
    assert handled.result.mode == "DISCOVERY" and handled.intake.kind == "run"


@pytest.mark.anyio
async def test_a_learned_task_is_replayed_with_no_model(engines, monkeypatch):
    _saved(monkeypatch, SimpleNamespace(trusted=True))
    handled = await handle(REQUEST, intake_model=_Intake(UNDERSTOOD), discovery_model=object())
    assert [run[0] for run in engines] == ["replay"]
    assert handled.result.mode == "REPLAY"


@pytest.mark.anyio
async def test_a_tampered_artifact_goes_to_replay_to_be_refused_not_relearned(engines, monkeypatch):
    _saved(monkeypatch, SimpleNamespace(trusted=False))
    await handle(REQUEST, intake_model=_Intake(UNDERSTOOD), discovery_model=object())
    assert [run[0] for run in engines] == ["replay"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "request_text, choice, kind",
    [pytest.param("Pay Sunbelt Electric Co for member 10234", (BILL_PAY, {**UNDERSTOOD[1], "amount": None}),
                  "needs_input", id="an input missing"),
     pytest.param("Close the account of member 10234", None, "not_supported", id="an unknown task")],
)
async def test_a_request_that_isnt_understood_runs_nothing(engines, monkeypatch, request_text, choice, kind):
    _saved(monkeypatch, None)
    handled = await handle(request_text, intake_model=_Intake(choice), discovery_model=object())
    assert (handled.intake.kind, handled.result, engines) == (kind, None, [])


@pytest.mark.anyio
async def test_the_window_and_the_person_are_passed_to_the_engine(engines, monkeypatch):
    _saved(monkeypatch, SimpleNamespace(trusted=True))
    operator = OperatorSetup()
    await handle(REQUEST, intake_model=_Intake(UNDERSTOOD), discovery_model=object(), headless=False,
                 operator=operator)
    assert engines[0][3:] == (False, operator)
