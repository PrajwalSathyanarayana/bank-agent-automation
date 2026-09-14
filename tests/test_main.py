import html
import re

import pytest

from app import create_app  # the mock bank; tests/conftest.py puts its folder on the import path
from src.config.env import env
from src.config.settings import settings
from src.locating.checks import phrase_matches
from src.main import BILL_PAY, CONTRACTS, parse_args
from src.types.artifact_schema import OutcomeSignal, OutputType


def test_the_bill_pay_contract_declares_its_inputs_outputs_and_credentials():
    contract = CONTRACTS[BILL_PAY]
    assert [parameter.key for parameter in contract.input_parameters] == ["member_id", "amount", "payee_name"]
    assert [output.key for output in contract.output_definitions] == ["checking_balance_before", "new_checking_balance"]
    assert [credential.key for credential in contract.credentials] == ["bank_username", "bank_password"]
    assert contract.target_url == settings.mock_bank_login_url
    # The goal itself asks for the value, so the model plans to read it.
    assert "read the checking balance" in contract.description


def test_the_bill_pay_balances_are_money_in_us_dollars():
    outputs = CONTRACTS[BILL_PAY].output_definitions
    assert [(output.type, output.currency) for output in outputs] == [(OutputType.MONEY, "USD")] * 2


def test_the_bill_pay_contract_declares_its_four_known_outcomes():
    contract = CONTRACTS[BILL_PAY]
    assert [outcome.code for outcome in contract.known_outcomes] == [
        "MEMBER_NOT_FOUND", "INSUFFICIENT_FUNDS", "ACCOUNT_RESTRICTED", "PAYEE_NOT_FOUND"]
    [payee] = [outcome for outcome in contract.known_outcomes if outcome.signal == OutcomeSignal.NO_SUCH_OPTION]
    assert payee.input_key in {parameter.key for parameter in contract.input_parameters}


@pytest.fixture
def bank():
    # The mock bank in-process, signed in with the configured teller credentials.
    client = create_app().test_client()
    client.post("/login", data={"username": env.mock_bank_username,
                                "password": env.mock_bank_password.get_secret_value()})
    return client


def _visible_text(response) -> str:
    # The page's words without its markup, the way a reader sees them.
    return html.unescape(re.sub(r"<[^>]+>", " ", response.get_data(as_text=True)))


def _trigger(bank, code):
    # Each outcome as the bank produces it: 99999 doesn't exist, 20567 has $512.75, 30891 is restricted.
    if code == "MEMBER_NOT_FOUND":
        return bank.get("/member/99999", follow_redirects=True)
    if code == "INSUFFICIENT_FUNDS":
        bank.get("/member/20567")
        return bank.post("/billpay", data={"payee_id": "P001", "amount": "1000.00"})
    bank.get("/member/30891")
    return bank.get("/billpay")


@pytest.mark.parametrize("code", ["MEMBER_NOT_FOUND", "INSUFFICIENT_FUNDS", "ACCOUNT_RESTRICTED"])
def test_each_outcomes_wording_is_what_the_bank_shows(bank, code):
    [outcome] = [outcome for outcome in CONTRACTS[BILL_PAY].known_outcomes if outcome.code == code]
    assert phrase_matches(_visible_text(_trigger(bank, code)), outcome.text)


def test_no_outcome_wording_appears_on_the_normal_path(bank):
    # Member page, Bill Pay form and the confirm page for a payment that can go through.
    pages = [bank.get("/member/10234"), bank.get("/billpay"),
             bank.post("/billpay", data={"payee_id": "P001", "amount": "50.00"}, follow_redirects=True)]
    texts = [_visible_text(page) for page in pages]
    for outcome in CONTRACTS[BILL_PAY].known_outcomes:
        if outcome.text:
            assert not any(phrase_matches(text, outcome.text) for text in texts), outcome.code


def test_the_discover_command_reads_typed_inputs_a_step_limit_and_a_window_option():
    args = parse_args(["discover", "--member-id", "10234", "--amount", "50", "--payee", "Sunbelt Electric Co",
                       "--max-steps", "25", "--headed"])
    assert (args.member_id, args.amount, args.payee, args.max_steps, args.headed) == (
        "10234", 50.0, "Sunbelt Electric Co", 25, True)


def test_the_step_limit_defaults_to_the_setting_and_the_window_stays_hidden():
    args = parse_args(["discover", "--member-id", "10234", "--amount", "50", "--payee", "Sunbelt Electric Co"])
    assert (args.max_steps, args.headed) == (None, False)
