"""Command-line entry point. The mock bank must already be running (python mock_bank/app.py).

    python -m src.main run "For member 10234, pay 50 to Sunbelt Electric Co"
    python -m src.main discover --member-id 10234 --amount 50 --payee "Sunbelt Electric Co"
    python -m src.main replay   --member-id 10234 --amount 50 --payee "Sunbelt Electric Co"

run is the system's entry point: the request is read by one model call (the intake), then
replayed if its task has been learned, or discovered if not. discover and replay run one
engine directly, for development.

discover runs one real discovery of the bill pay capability; it calls the Claude API,
which costs money. replay runs the capability's latest trusted artifact with no model at
all. With --operator (either command), a run that needs a person hands them its own window
instead of stopping; in discovery, what they do is recorded as steps of the artifact.
Choosing between them by itself, from a goal sentence, comes with the router.
"""
import argparse
import asyncio
import sys
import urllib.error
import urllib.request
from contextlib import AsyncExitStack
from typing import Optional, Sequence

from src.config.env import env
from src.config.settings import settings
from src.catalog import BILL_PAY, CONTRACTS
from src.discovery.agent import ClaudeModel, DiscoveryRequest, discover
from src.handoff.session_manager import OperatorSetup
from src.handoff.ws_server import FEED_HOST, FeedUnavailable, HandoffFeed
from src.intake import ClaudeIntakeModel, IntakeUnavailable
from src.observability.logger import RunLogger
from src.replay.executor import ReplayRequest, replay
from src.router import handle
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
    run = commands.add_parser("discover", help="discover the bill pay capability with the real model (costs money)")
    _bill_pay_inputs(run)
    run.add_argument("--max-steps", type=int, default=None,
                     help=f"a lower step limit for this run (default {settings.discovery_max_steps})")
    again = commands.add_parser("replay", help="replay the bill pay capability's latest trusted artifact (no model)")
    _bill_pay_inputs(again)
    return parser.parse_args(argv)


def _bill_pay_inputs(command: argparse.ArgumentParser) -> None:
    # The same typed inputs for both commands, so a replay is asked exactly as a discovery was.
    command.add_argument("--member-id", required=True)
    command.add_argument("--amount", required=True, type=float)
    command.add_argument("--payee", required=True, help='the payee as named on the page, e.g. "Sunbelt Electric Co"')
    command.add_argument("--headed", action="store_true", help="show the browser window")
    command.add_argument("--operator", action="store_true",
                         help="when the run needs a person, hand them this run's window (shows the window)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    commands = {"run": _run, "replay": _replay, "discover": _discover}
    return asyncio.run(commands[args.command](args))


async def _run(args: argparse.Namespace) -> int:
    # Checked before anything else, so no model call is paid for when the bank is down.
    if not _bank_is_up():
        print(f"The mock bank isn't answering at {env.mock_bank_base_url}; start it with: python mock_bank/app.py")
        return 2
    try:
        async with AsyncExitStack() as stack:
            operator = await _operator(stack, args.operator)
            handled = await handle(args.request, intake_model=ClaudeIntakeModel(), discovery_model=ClaudeModel(),
                                   headless=not (args.headed or args.operator), operator=operator)
    except IntakeUnavailable as unavailable:
        print(f"The request couldn't be read right now ({unavailable}); nothing was run. Please try again.")
        return 2
    print(handled.intake.message)
    if handled.result is None:
        return NOT_RUN
    print(handled.result.to_json())
    print(f"Run log: {handled.result.evidence_paths.log_file}")
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
    logger = RunLogger("DISCOVERY", capability=BILL_PAY)
    request = DiscoveryRequest(
        CONTRACTS[BILL_PAY], {"member_id": args.member_id, "amount": args.amount, "payee_name": args.payee}
    )
    async with AsyncExitStack() as stack:
        operator = await _operator(stack, args.operator)
        # A person can only take over a window they can see.
        result = await discover(request, ClaudeModel(), logger, headless=not (args.headed or args.operator),
                                max_steps=args.max_steps, operator=operator)
    print(result.to_json())
    print(f"Run log: {logger.log_path}")
    return 0 if result.status in (ExecutionStatus.SUCCESS, ExecutionStatus.HUMAN_ESCALATED) else 1


async def _replay(args: argparse.Namespace) -> int:
    if not _bank_is_up():
        print(f"The mock bank isn't answering at {env.mock_bank_base_url}; start it with: python mock_bank/app.py")
        return 2
    logger = RunLogger("REPLAY", capability=BILL_PAY)
    request = ReplayRequest(BILL_PAY, {"member_id": args.member_id, "amount": args.amount, "payee_name": args.payee})
    async with AsyncExitStack() as stack:
        operator = await _operator(stack, args.operator)
        # A person can only take over a window they can see.
        result = await replay(request, logger, headless=not (args.headed or args.operator), operator=operator)
    print(result.to_json())
    print(f"Run log: {logger.log_path}")
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
    try:
        with urllib.request.urlopen(settings.mock_bank_login_url, timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


if __name__ == "__main__":
    sys.exit(main())
