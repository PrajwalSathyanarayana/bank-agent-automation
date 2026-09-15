"""The intake: a request in plain words → a declared task and its inputs, or why not. The
model is scripted here; one real call is made only on purpose."""
import json
from types import SimpleNamespace

import pytest

from src.catalog import BILL_PAY, CONTRACTS
from src.config.env import env
from src.intake import NONE_OF_THEM, ClaudeIntakeModel, IntakeAnswer, intake_tools, interpret, task_line

REQUEST = "For member 10234, pay 50 to Sunbelt Electric Co"
BILL_PAY_TASK = ("For member <member id>, read the checking balance, pay <amount> to <payee name>, "
                 "then read the new balance.")


class _Scripted:
    """Answers with a fixed tool call (or none), and keeps what it was asked."""

    def __init__(self, choice):
        self.choice = choice
        self.asked = []

    async def choose(self, request, tools):
        self.asked.append((request, tools))
        return self.choice


def _bill_pay(**inputs):
    values = {"member_id": "10234", "amount": 50, "payee_name": "Sunbelt Electric Co"}
    values.update(inputs)
    return (BILL_PAY, values)


@pytest.mark.anyio
async def test_a_complete_request_is_understood_and_ready_to_run():
    answer = await interpret(REQUEST, CONTRACTS, _Scripted(_bill_pay()))
    assert (answer.kind, answer.capability) == ("run", BILL_PAY)
    assert answer.inputs == {"member_id": "10234", "amount": 50.0, "payee_name": "Sunbelt Electric Co"}
    assert isinstance(answer.inputs["amount"], float)
    assert answer.message == "Understood as: Pay $50.00 to Sunbelt Electric Co for member 10234."


@pytest.mark.anyio
async def test_a_value_the_request_doesnt_state_is_asked_for_never_guessed():
    answer = await interpret("Pay Sunbelt Electric Co for member 10234", CONTRACTS,
                             _Scripted(_bill_pay(amount=None)))
    assert (answer.kind, answer.missing) == ("needs_input", ("amount",))
    assert "amount to pay, in dollars" in answer.message
    assert answer.inputs == {"member_id": "10234", "payee_name": "Sunbelt Electric Co"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "request_text, invented, missing",
    [
        pytest.param("For member 10234, pay 50 to Sunbelt", {"payee_name": "Sunbelt Electric Co"}, ("payee_name",),
                     id="a payee name the request doesn't contain"),
        pytest.param("For member 10234, pay 50 to Sunbelt Electric Co", {"amount": 75}, ("amount",),
                     id="an amount the request doesn't contain"),
        pytest.param("For member 10234, pay fifty dollars to Sunbelt Electric Co", {"amount": 50}, ("amount",),
                     id="an amount in words, not figures"),
    ],
)
async def test_a_value_the_model_supplies_but_the_request_doesnt_state_is_not_used(request_text, invented, missing):
    answer = await interpret(request_text, CONTRACTS, _Scripted(_bill_pay(**invented)))
    assert (answer.kind, answer.missing) == ("needs_input", missing)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "choice",
    [pytest.param((NONE_OF_THEM, {"reason": "closing an account"}), id="none of the tasks"),
     pytest.param(None, id="no tool called, or declined"),
     pytest.param(("close_account", {}), id="a task nobody declared")],
)
async def test_a_request_for_an_unknown_task_gets_a_plain_message(choice):
    answer = await interpret("Close the account of member 10234", CONTRACTS, _Scripted(choice))
    assert (answer.kind, answer.capability) == ("not_supported", None)
    assert answer.message == f"I can't do that yet. The tasks I know are:\n- {BILL_PAY_TASK}"


@pytest.mark.anyio
async def test_an_empty_request_asks_what_to_do_without_calling_the_model():
    model = _Scripted(_bill_pay())
    answer = await interpret("   ", CONTRACTS, model)
    assert answer.kind == "needs_input" and model.asked == []


def test_each_declared_task_is_a_strict_tool_with_its_inputs_typed_and_nullable():
    tools = {tool["name"]: tool for tool in intake_tools(CONTRACTS)}
    assert list(tools) == [BILL_PAY, NONE_OF_THEM]
    bill_pay = tools[BILL_PAY]
    assert bill_pay["strict"] is True and bill_pay["input_schema"]["additionalProperties"] is False
    properties = bill_pay["input_schema"]["properties"]
    assert {key: spec["type"] for key, spec in properties.items()} == {
        "member_id": ["string", "null"], "amount": ["number", "null"], "payee_name": ["string", "null"]}
    assert bill_pay["input_schema"]["required"] == ["member_id", "amount", "payee_name"]
    assert task_line(CONTRACTS[BILL_PAY]) == BILL_PAY_TASK


def test_the_answer_prints_without_empty_fields():
    printed = json.loads(IntakeAnswer("not_supported", "I can't do that yet.").to_json())
    assert printed == {"kind": "not_supported", "message": "I can't do that yet."}


class _FakeClient:
    """Stands in for the Anthropic client: records the request, answers as told."""

    def __init__(self, response):
        self.sent = {}

        async def create(**request):
            self.sent = request
            return response

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


@pytest.mark.anyio
async def test_the_real_model_is_asked_once_with_the_tools_and_the_request():
    call = SimpleNamespace(type="tool_use", name=BILL_PAY, input={"member_id": "10234", "amount": 50,
                                                                  "payee_name": "Sunbelt Electric Co"})
    client = _FakeClient(SimpleNamespace(stop_reason="tool_use", content=[call]))
    tools = intake_tools(CONTRACTS)
    assert await ClaudeIntakeModel(client).choose(REQUEST, tools) == (BILL_PAY, dict(call.input))
    sent = client.sent
    assert (sent["model"], sent["tools"], sent["messages"]) == (
        env.anthropic_model, tools, [{"role": "user", "content": REQUEST}])
    # The model chooses among the tools; it isn't forced, and makes one call at most.
    assert sent["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


@pytest.mark.anyio
async def test_a_declined_request_is_no_choice():
    client = _FakeClient(SimpleNamespace(stop_reason="refusal", content=[]))
    assert await ClaudeIntakeModel(client).choose(REQUEST, intake_tools(CONTRACTS)) is None
