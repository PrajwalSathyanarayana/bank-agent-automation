"""Command-line entry point.

    python -m src.main discover --member-id 10234 --amount 50 --payee "Sunbelt Electric Co"

Runs one real discovery against the mock bank, which must already be running
(python mock_bank/app.py). It calls the Claude API, which costs money. Choosing between
discovery and replay for a capability comes with replay.
"""
import argparse
import asyncio
import sys
import urllib.error
import urllib.request
from typing import Optional, Sequence

from src.config.env import env
from src.config.settings import settings
from src.discovery.agent import ClaudeModel, DiscoveryRequest, discover
from src.discovery.artifact_builder import ArtifactContract
from src.observability.logger import RunLogger
from src.types.artifact_schema import (
    CredentialDefinition,
    CredentialKind,
    InputParamDefinition,
    KnownOutcome,
    OutcomeSignal,
    OutputParamDefinition,
    OutputType,
    ParamType,
)
from src.types.result_schema import ExecutionStatus

BILL_PAY = "member_servicing_and_bill_pay"

# What the engineer declares for each capability: the goal template, where to start, and
# the contract. The model never invents any of it.
CONTRACTS = {
    BILL_PAY: ArtifactContract(
        capability=BILL_PAY,
        description=("For member {member_id}, read the checking balance, pay {amount} to {payee_name}, "
                     "then read the new balance."),
        target_url=settings.mock_bank_login_url,
        input_parameters=[
            InputParamDefinition(key="member_id", type=ParamType.STRING, description="Member ID"),
            InputParamDefinition(key="amount", type=ParamType.NUMBER, description="Amount to pay, in dollars"),
            InputParamDefinition(key="payee_name", type=ParamType.STRING, description="Payee, as named in the payee list"),
        ],
        output_definitions=[
            OutputParamDefinition(key="checking_balance_before", type=OutputType.MONEY, currency="USD",
                                  description="The member's checking balance as shown before paying"),
            OutputParamDefinition(key="new_checking_balance", type=OutputType.MONEY, currency="USD",
                                  description="The checking balance shown on the receipt after paying"),
        ],
        credentials=[
            CredentialDefinition(key="bank_username", kind=CredentialKind.CONFIG, description="Teller username"),
            CredentialDefinition(key="bank_password", kind=CredentialKind.SECRET, description="Teller password"),
        ],
        # The answers other than a completed payment, each with the bank's own wording.
        known_outcomes=[
            KnownOutcome(code="MEMBER_NOT_FOUND", description="No member has this ID",
                         signal=OutcomeSignal.PAGE_TEXT, text="No member found with that ID."),
            KnownOutcome(code="INSUFFICIENT_FUNDS", description="The checking balance doesn't cover the amount",
                         signal=OutcomeSignal.PAGE_TEXT, text="Insufficient funds for this payment amount."),
            KnownOutcome(code="ACCOUNT_RESTRICTED", description="The paying account is restricted; Bill Pay is refused",
                         signal=OutcomeSignal.PAGE_TEXT, text="the paying account is restricted"),
            KnownOutcome(code="PAYEE_NOT_FOUND", description="The payee isn't in the payee list",
                         signal=OutcomeSignal.NO_SUCH_OPTION, input_key="payee_name"),
        ],
    ),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.main")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("discover", help="discover the bill pay capability with the real model (costs money)")
    run.add_argument("--member-id", required=True)
    run.add_argument("--amount", required=True, type=float)
    run.add_argument("--payee", required=True, help='the payee as named on the page, e.g. "Sunbelt Electric Co"')
    run.add_argument("--max-steps", type=int, default=None,
                     help=f"a lower step limit for this run (default {settings.discovery_max_steps})")
    run.add_argument("--headed", action="store_true", help="show the browser window")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return asyncio.run(_discover(args))


async def _discover(args: argparse.Namespace) -> int:
    if not _bank_is_up():
        print(f"The mock bank isn't answering at {env.mock_bank_base_url}; start it with: python mock_bank/app.py")
        return 2
    logger = RunLogger("DISCOVERY", capability=BILL_PAY)
    request = DiscoveryRequest(
        CONTRACTS[BILL_PAY], {"member_id": args.member_id, "amount": args.amount, "payee_name": args.payee}
    )
    result = await discover(request, ClaudeModel(), logger, headless=not args.headed, max_steps=args.max_steps)
    print(result.model_dump_json(indent=2))
    print(f"Run log: {logger.log_path}")
    return 0 if result.status in (ExecutionStatus.SUCCESS, ExecutionStatus.HUMAN_ESCALATED) else 1


def _bank_is_up() -> bool:
    try:
        with urllib.request.urlopen(settings.mock_bank_login_url, timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


if __name__ == "__main__":
    sys.exit(main())
