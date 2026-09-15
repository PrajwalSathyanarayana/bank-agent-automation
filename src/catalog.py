"""The tasks this system knows: each capability's contract, declared once by a person.

A contract is what the engineer declares for a task: the goal template, where to start,
the inputs and outputs, and the rules for the run (known outcomes, allowed pages, known
interruptions, the payment checks). The model never invents any of it. The intake offers
only these tasks; a request for anything else is answered as not supported.
"""
from src.config.settings import settings
from src.discovery.artifact_builder import ArtifactContract
from src.types.artifact_schema import (
    CompareAs,
    ConfirmationCheck,
    CredentialDefinition,
    CredentialKind,
    InputParamDefinition,
    InterruptionSignal,
    KnownInterruption,
    KnownOutcome,
    OutcomeSignal,
    OutputParamDefinition,
    OutputType,
    ParamType,
    RecoveryAction,
)
from src.types.step_schema import Locator, LocatorType

BILL_PAY = "member_servicing_and_bill_pay"
PHONE = "update_member_phone"

# Shared by the tasks: the teller's sign-in, the bank's answer for an unknown member, and the
# obstacles replay clears by itself, each with the one recovery a person approved.
_TELLER = [
    CredentialDefinition(key="bank_username", kind=CredentialKind.CONFIG, description="Teller username"),
    CredentialDefinition(key="bank_password", kind=CredentialKind.SECRET, description="Teller password"),
]
_MEMBER_NOT_FOUND = KnownOutcome(code="MEMBER_NOT_FOUND", description="No member has this ID",
                                 signal=OutcomeSignal.PAGE_TEXT, text="No member found with that ID.")
_PROMO_POPUP = KnownInterruption(
    code="PROMO_POPUP", description="A promotion covers the dashboard; its Close button hides it",
    signal=InterruptionSignal.ELEMENT_VISIBLE, locator=Locator(type=LocatorType.CSS, value="div.overlay", priority=0),
    recovery=RecoveryAction.CLICK,
    target=Locator(type=LocatorType.CSS, value="div.overlay input[value='Close']", priority=0))
_SESSION_EXPIRED = KnownInterruption(
    code="SESSION_EXPIRED", description="The session expired; the bank forgot the member and any pending payment",
    signal=InterruptionSignal.PAGE_TEXT, text="Your session has expired.", recovery=RecoveryAction.START_OVER)

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
        credentials=_TELLER,
        # The answers other than a completed payment, each with the bank's own wording.
        known_outcomes=[
            _MEMBER_NOT_FOUND,
            KnownOutcome(code="INSUFFICIENT_FUNDS", description="The checking balance doesn't cover the amount",
                         signal=OutcomeSignal.PAGE_TEXT, text="Insufficient funds for this payment amount."),
            KnownOutcome(code="ACCOUNT_RESTRICTED", description="The paying account is restricted; Bill Pay is refused",
                         signal=OutcomeSignal.PAGE_TEXT, text="the paying account is restricted"),
            KnownOutcome(code="PAYEE_NOT_FOUND", description="The payee isn't in the payee list",
                         signal=OutcomeSignal.NO_SUCH_OPTION, input_key="payee_name"),
        ],
        # Only the pages bill pay needs, its outcome and session-timeout pages included. The
        # profile edit page (/member/<id>/edit) changes member data and is left out on purpose.
        allowed_paths=["/", "/login", "/dashboard", "/search", "/member/*", "/member/*/accounts",
                       "/billpay", "/billpay/confirm", "/session-timeout"],
        known_interruptions=[_PROMO_POPUP, _SESSION_EXPIRED],
        # Before Confirm Payment, the screen must show exactly this run's payee and amount.
        confirmation_checks=[
            ConfirmationCheck(label="Payee:", input_key="payee_name", compare_as=CompareAs.TEXT),
            ConfirmationCheck(label="Amount:", input_key="amount", compare_as=CompareAs.MONEY, currency="USD"),
        ],
    ),
    # The second task: it changes member data, but nothing irreversible (the old number can be
    # put back), so it needs no payment check. The edit page shows the phone only inside its
    # box, so there is no value to read: the bank's "Profile updated successfully." confirms it.
    PHONE: ArtifactContract(
        capability=PHONE,
        description="For member {member_id}, change the phone number to {new_phone}.",
        target_url=settings.mock_bank_login_url,
        input_parameters=[
            InputParamDefinition(key="member_id", type=ParamType.STRING, description="Member ID"),
            InputParamDefinition(key="new_phone", type=ParamType.STRING,
                                 description="New phone number, as (NNN) NNN-NNNN"),
        ],
        output_definitions=[],
        credentials=_TELLER,
        known_outcomes=[
            _MEMBER_NOT_FOUND,
            KnownOutcome(code="INVALID_PHONE", description="The phone number isn't in the form (NNN) NNN-NNNN",
                         signal=OutcomeSignal.PAGE_TEXT, text="Phone must be in the form (NNN) NNN-NNNN."),
        ],
        # The member's pages and the profile editor; Bill Pay is not one of this task's pages.
        allowed_paths=["/", "/login", "/dashboard", "/search", "/member/*", "/member/*/edit", "/session-timeout"],
        known_interruptions=[_PROMO_POPUP, _SESSION_EXPIRED],
    ),
}
