"""Command-line entry point. The mock bank must already be running (python mock_bank/app.py).

    python -m src.main run "For member 10234, pay 50 to Sunbelt Electric Co"
    python -m src.main discover --capability member_servicing_and_bill_pay \\
        --input member_id=10234 --input amount=50 --input payee_name="Sunbelt Electric Co"
    python -m src.main replay --capability look_up_checking_balance --input member_id=10234

run is the system's entry point: the request is read by one model call (the intake), then
replayed if its task has been learned, or discovered if not. discover and replay run one
engine directly for a named capability, for development - including re-discovering a
capability that already has a saved artifact, which run's router would never do on its own
(it always prefers replay once something is learned).

discover runs one real discovery of the named capability; it calls the Claude API, which
costs money. replay runs that capability's latest trusted artifact with no model at all.
--input KEY=VALUE (repeatable) supplies each of the capability's own declared inputs (see
src/catalog.py), typed and checked against what it declares - an unknown key or a bad
number is refused, never guessed. With --operator (either command), a run that needs a
person hands them its own window instead of stopping; in discovery, what they do is
recorded as steps of the artifact.
"""
import argparse
import asyncio
import sys
import urllib.error
import urllib.request
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional, Sequence, Union

from src.config.env import env
from src.config.settings import settings
from src.catalog import CONTRACTS
from src.discovery.agent import ClaudeModel, DiscoveryRequest, discover
from src.evidence.index import write_index
from src.handoff.session_manager import OperatorSetup
from src.handoff.ws_server import FEED_HOST, FeedUnavailable, HandoffFeed
from src.intake import ClaudeIntakeModel, IntakeUnavailable
from src.observability.logger import RunLogger
from src.replay.executor import ReplayRequest, replay
from src.router import handle
from src.types.artifact_schema import ParamType
from src.types.result_schema import ExecutionResult, ExecutionStatus, HandoffResolution

# The request wasn't run: the intake needs more, or the task isn't one the system knows.
NOT_RUN = 3

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.main")
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("run", help="do a task described in words: replayed if learned, else discovered")
    ask.add_argument("request", help='e.g. "For member 10234, pay 50 to Sunbelt Electric Co"')
    ask.add_argument("--headed", action="store_true", help="show the browser window")
    ask.add_argument("--operator", action="store_true",
                     help="when the run needs a person, hand them this run's window (shows the window)")
    ask.add_argument("--slow-mo", type=int, default=None, metavar="MS",
                     help="pause this long after every action, for a person watching --headed to follow along")
    run = commands.add_parser("discover", help="discover a named capability with the real model (costs money)")
    _capability_inputs(run)
    run.add_argument("--max-steps", type=int, default=None,
                     help=f"a lower step limit for this run (default {settings.discovery_max_steps})")
    again = commands.add_parser("replay", help="replay a named capability's latest trusted artifact (no model)")
    _capability_inputs(again)
    return parser.parse_args(argv)


def _capability_inputs(command: argparse.ArgumentParser) -> None:
    # The same shape for both commands, so a replay is asked exactly as a discovery was.
    command.add_argument("--capability", required=True, choices=sorted(CONTRACTS), help="which task to run")
    command.add_argument("--input", action="append", default=[], metavar="KEY=VALUE",
                         help="a declared input for --capability, e.g. --input member_id=10234 (repeatable)")
    command.add_argument("--headed", action="store_true", help="show the browser window")
    command.add_argument("--operator", action="store_true",
                         help="when the run needs a person, hand them this run's window (shows the window)")
    command.add_argument("--slow-mo", type=int, default=None, metavar="MS",
                         help="pause this long after every action, for a person watching --headed to follow along")


class InputsInvalid(Exception):
    """--input didn't match what --capability declares. The message is ready to print as-is."""


def _parse_inputs(capability: str, raw: list[str]) -> dict[str, Union[str, float, bool]]:
    """--input KEY=VALUE strings, typed and checked against the capability's own declared
    input_parameters (src/catalog.py) - an unknown key, a missing one, or a bad number is
    refused with a clear message, never guessed."""
    declared = {p.key: p for p in CONTRACTS[capability].input_parameters}
    values: dict[str, Union[str, float, bool]] = {}
    for item in raw:
        if "=" not in item:
            raise InputsInvalid(f"--input must be KEY=VALUE, got: {item!r}")
        key, _, raw_value = item.partition("=")
        param = declared.get(key)
        if param is None:
            raise InputsInvalid(f"{capability} has no input named {key!r}; declared inputs: "
                                f"{', '.join(sorted(declared)) or '(none)'}")
        if param.type == ParamType.NUMBER:
            try:
                values[key] = float(raw_value)
            except ValueError:
                raise InputsInvalid(f"{key} must be a number, got: {raw_value!r}") from None
        elif param.type == ParamType.BOOLEAN:
            if raw_value.lower() not in ("true", "false"):
                raise InputsInvalid(f"{key} must be true or false, got: {raw_value!r}")
            values[key] = raw_value.lower() == "true"
        else:
            values[key] = raw_value
    missing = [key for key, param in declared.items() if param.required and key not in values]
    if missing:
        raise InputsInvalid(f"{capability} needs --input for: {', '.join(missing)}")
    return values


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    commands = {"run": _run, "replay": _replay, "discover": _discover}
    return asyncio.run(commands[args.command](args))


def _slow_mo_note(args: argparse.Namespace) -> None:
    if args.slow_mo is not None and not (args.headed or args.operator):
        print("Note: --slow-mo has no visible effect without --headed or --operator.")


def _write_evidence(run_dir: Path) -> None:
    # Every real command line run updates this run's report and the evidence front page -
    # never the test suite, which never reaches this function (it calls the engines
    # directly, or mocks them). write_index renders this run's own report.html too (it had
    # none until now), so the path below is real by the time it's printed.
    html_path, _ = write_index(settings.evidence_dir)
    print(f"Report: {run_dir / 'report.html'}")
    print(f"Evidence index: {html_path}")


async def _run(args: argparse.Namespace) -> int:
    # Checked before anything else, so no model call is paid for when the bank is down.
    if not _bank_is_up():
        print(f"The mock bank isn't answering at {env.mock_bank_base_url}; start it with: python mock_bank/app.py")
        return 2
    _slow_mo_note(args)
    try:
        async with AsyncExitStack() as stack:
            operator = await _operator(stack, args.operator)
            handled = await handle(args.request, intake_model=ClaudeIntakeModel(), discovery_model=ClaudeModel(),
                                   headless=not (args.headed or args.operator), operator=operator, trace=True,
                                   slow_mo_ms=args.slow_mo)
    except IntakeUnavailable as unavailable:
        print(f"The request couldn't be read right now ({unavailable}); nothing was run. Please try again.")
        return 2
    print(handled.intake.message)
    if handled.result is None:
        return NOT_RUN
    print(handled.result.to_json())
    print(f"Run log: {handled.result.evidence_paths.log_file}")
    _write_evidence(Path(handled.result.evidence_paths.log_file).parent)
    return _exit_code(handled.result)


def _exit_code(result: ExecutionResult) -> int:
    # Done, or the bank's own answer (no such member …), or a task a person finished whose
    # receipt was seen; anything else needs someone's attention.
    last_handoff = result.handoff_events[-1] if result.handoff_events else None
    finished = (last_handoff is not None and last_handoff.resolution == HandoffResolution.MANUAL_COMPLETED
                and result.irreversible_step != "unknown")
    return 0 if result.status in (ExecutionStatus.SUCCESS, ExecutionStatus.BUSINESS_OUTCOME) or finished else 1


async def _discover(args: argparse.Namespace) -> int:
    if not _bank_is_up():
        print(f"The mock bank isn't answering at {env.mock_bank_base_url}; start it with: python mock_bank/app.py")
        return 2
    try:
        inputs = _parse_inputs(args.capability, args.input)
    except InputsInvalid as invalid:
        print(str(invalid))
        return 2
    _slow_mo_note(args)
    logger = RunLogger("DISCOVERY", capability=args.capability)
    request = DiscoveryRequest(CONTRACTS[args.capability], inputs)
    async with AsyncExitStack() as stack:
        operator = await _operator(stack, args.operator)
        # A person can only take over a window they can see.
        result = await discover(request, ClaudeModel(), logger, headless=not (args.headed or args.operator),
                                max_steps=args.max_steps, operator=operator, trace=True, slow_mo_ms=args.slow_mo)
    print(result.to_json())
    print(f"Run log: {logger.log_path}")
    _write_evidence(logger.run_dir)
    return 0 if result.status in (ExecutionStatus.SUCCESS, ExecutionStatus.HUMAN_ESCALATED) else 1


async def _replay(args: argparse.Namespace) -> int:
    if not _bank_is_up():
        print(f"The mock bank isn't answering at {env.mock_bank_base_url}; start it with: python mock_bank/app.py")
        return 2
    try:
        inputs = _parse_inputs(args.capability, args.input)
    except InputsInvalid as invalid:
        print(str(invalid))
        return 2
    _slow_mo_note(args)
    logger = RunLogger("REPLAY", capability=args.capability)
    request = ReplayRequest(args.capability, inputs)
    async with AsyncExitStack() as stack:
        operator = await _operator(stack, args.operator)
        # A person can only take over a window they can see.
        result = await replay(request, logger, headless=not (args.headed or args.operator), operator=operator,
                             trace=True, slow_mo_ms=args.slow_mo)
    print(result.to_json())
    print(f"Run log: {logger.log_path}")
    _write_evidence(logger.run_dir)
    return _exit_code(result)


async def _operator(stack: AsyncExitStack, wanted: bool) -> Optional[OperatorSetup]:
    # A person at this machine, if asked for: the feed opened for the run, and a word on what to expect.
    if not wanted:
        return None
    setup = OperatorSetup(announcer=await _open_feed(stack))
    print("If the run needs a person, the browser window will show a bar asking them to take over.")
    return setup


async def _open_feed(stack: AsyncExitStack) -> Optional[HandoffFeed]:
    # The feed only announces; the bar in the window works without it, so a busy port is
    # reported and the run goes on.
    try:
        feed = await stack.enter_async_context(HandoffFeed(env.ws_handoff_port))
    except FeedUnavailable as unavailable:
        print(f"Note: {unavailable}; carrying on without announcements (the bar in the window still works).")
        return None
    print(f"Handoff announcements: ws://{FEED_HOST}:{feed.port} (watch them with: python -m src.handoff.watch)")
    return feed


def _bank_is_up() -> bool:
    # Generous on purpose: a bank that isn't running at all refuses the connection
    # immediately regardless of this ceiling, so it only matters for one real case - the
    # mock bank's own slow-page test switch (D109), which must read as "up, just slow",
    # never as "not answering", or the very scenario it exists to test can't be reached.
    try:
        with urllib.request.urlopen(settings.mock_bank_login_url, timeout=40):
            return True
    except (urllib.error.URLError, OSError):
        return False


if __name__ == "__main__":
    sys.exit(main())
