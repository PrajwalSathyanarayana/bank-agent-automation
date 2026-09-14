from src.config.settings import settings
from src.main import BILL_PAY, CONTRACTS, parse_args


def test_the_bill_pay_contract_declares_its_inputs_outputs_and_credentials():
    contract = CONTRACTS[BILL_PAY]
    assert [parameter.key for parameter in contract.input_parameters] == ["member_id", "amount", "payee_name"]
    assert [output.key for output in contract.output_definitions] == ["checking_balance_before", "new_checking_balance"]
    assert [credential.key for credential in contract.credentials] == ["bank_username", "bank_password"]
    assert contract.target_url == settings.mock_bank_login_url
    # The goal itself asks for the value, so the model plans to read it.
    assert "read the checking balance" in contract.description


def test_the_discover_command_reads_typed_inputs_a_step_limit_and_a_window_option():
    args = parse_args(["discover", "--member-id", "10234", "--amount", "50", "--payee", "Sunbelt Electric Co",
                       "--max-steps", "25", "--headed"])
    assert (args.member_id, args.amount, args.payee, args.max_steps, args.headed) == (
        "10234", 50.0, "Sunbelt Electric Co", 25, True)


def test_the_step_limit_defaults_to_the_setting_and_the_window_stays_hidden():
    args = parse_args(["discover", "--member-id", "10234", "--amount", "50", "--payee", "Sunbelt Electric Co"])
    assert (args.max_steps, args.headed) == (None, False)
