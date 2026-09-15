"""The discovery loop: observe the page, let the model choose one action, check it, record
it, act, and repeat until the goal is done, the run has to stop, or the next step is
irreversible.

The loop owns every control point: the step and time limits, the safety gate, what gets
recorded, and how the run ends. The model only ever chooses.
"""
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Protocol, Union

import anthropic
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from src.config.env import configured_credentials, env
from src.config.settings import settings
from src.discovery.artifact_builder import ArtifactContract, build_and_save
from src.discovery.backstop import ScanInputs
from src.discovery.person_steps import PersonStepRecorder
from src.handoff.session_manager import Choice, HandoffManager, HandoffRequest, OperatorSetup
from src.surface.browser import (
    ActionFailed,
    BrowserSession,
    action_timeout_ms,
    click,
    number_text,
    placeholder_values,
    select_option,
    type_text,
)
from src.discovery.locators import NoProvenLocator, RunValues, derive_locators
from src.discovery.perception import Observation, PageElement, UnknownElement, observing
from src.locating.checks import value_beside
from src.locating.values import UnreadableValue, read_output
from src.discovery.prompts import (
    ASSERT_FIRST,
    NO_ACTION_REPROMPT,
    SYSTEM_PROMPT,
    allowlist_refusal,
    goal_message,
    output_kind,
    outputs_first,
    page_blocks,
    person_handed_back,
    progress,
    tool_definitions,
)
from src.discovery.recorder import Action, AssertionRefused, ExtractionRefused, Recorder, TypingRefused
from src.observability.logger import RunLogger
from src.observability.summary import readable_values, reason_for, summarize
from src.safety.allowlist import AllowlistViolation, check_domain, check_route, enforce_safety
from src.safety.authorization import MISMATCH, authorize
from src.safety.redactor import redact_text, scrub_known_values
from src.safety.keys import signing_refusal
from src.safety.sandbox import sandbox_refusal
from src.types.artifact_schema import CredentialKind, ParamType
from src.types.placeholders import fill_text
from src.types.result_schema import (
    ErrorDetail,
    EvidencePaths,
    ExecutionResult,
    ExecutionStatus,
    HandoffResolution,
    HandoffTelemetry,
)
from src.types.step_schema import ActionType, SafetyTier, Step

# The model's element tools and the step each one records.
_TOOL_ACTIONS = {
    "click": ActionType.CLICK,
    "dismiss_overlay": ActionType.CLICK,
    "type_text": ActionType.TYPE,
    "select_option": ActionType.SELECT,
}
_UNLOCATABLE = "Refused: that element can't be found again reliably on a later run; choose another way."
_NOT_AN_OVERLAY = "Refused: that control does more than close an overlay; use click if the goal needs it."


class ModelCallFailed(RuntimeError):
    """A model call failed in a way worth one retry: a timeout, a lost connection, a rate
    limit or a server error. The message names the kind of failure, nothing more."""


@dataclass(frozen=True)
class ModelReply:
    """One reply, as the loop needs it. content is appended to the history unchanged."""

    stop_reason: str
    content: list[Any]
    # Tokens the call used: input_tokens (uncached), cache_write_tokens, cache_read_tokens,
    # output_tokens.
    usage: Optional[Mapping[str, int]] = None


class Model(Protocol):
    async def reply(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, timeout_s: float) -> ModelReply:
        ...


class ClaudeModel:
    """The real model: Claude through the Anthropic API, with discovery's settings.

    The SDK's own retries are off: the loop retries a failed call once itself, so every
    wait stays inside the run's time limit.
    """

    def __init__(self, client: Optional[anthropic.AsyncAnthropic] = None) -> None:
        self._client = client or anthropic.AsyncAnthropic(
            api_key=env.anthropic_api_key.get_secret_value(), max_retries=0
        )

    async def reply(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, timeout_s: float) -> ModelReply:
        try:
            response = await self._client.with_options(timeout=timeout_s).beta.messages.create(
                model=env.anthropic_model,
                max_tokens=16_000,
                system=[{"type": "text", "text": SYSTEM_PROMPT}],
                tools=tools,
                # One action per turn, always on a fresh screenshot and element list.
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
                # The history only grows, so the cached prefix is reused every turn.
                cache_control={"type": "ephemeral"},
                # A declined turn is re-run once on a fallback model inside the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=messages,
            )
        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError) as error:
            raise ModelCallFailed(type(error).__name__) from None
        usage = response.usage
        return ModelReply(response.stop_reason, list(response.content), {
            "input_tokens": usage.input_tokens or 0,
            "cache_write_tokens": usage.cache_creation_input_tokens or 0,
            "cache_read_tokens": usage.cache_read_input_tokens or 0,
            "output_tokens": usage.output_tokens or 0,
        })


# US dollars per million tokens (input, output) at list price. A cache write costs 1.25x
# the input price, a cache read 0.1x.
_PRICES = {"claude-opus-5": (5.0, 25.0)}
_USAGE_KEYS = ("input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens")


def estimated_cost_usd(model: str, usage: Mapping[str, int]) -> Optional[float]:
    """What the tokens cost at list price, for the run log; None for a model with no known price."""
    if model not in _PRICES:
        return None
    input_price, output_price = _PRICES[model]
    dollars = (
        usage["input_tokens"] * input_price
        + usage["cache_write_tokens"] * input_price * 1.25
        + usage["cache_read_tokens"] * input_price * 0.1
        + usage["output_tokens"] * output_price
    ) / 1_000_000
    return round(dollars, 4)


@dataclass(frozen=True)
class DiscoveryRequest:
    """What starts a discovery: the engineer's contract (its description is the goal
    template) and this run's input values."""

    contract: ArtifactContract
    input_values: Mapping[str, Union[str, float]]


async def discover(
    request: DiscoveryRequest,
    model: Model,
    logger: RunLogger,
    *,
    headless: bool = True,
    max_steps: Optional[int] = None,
    sandbox: Optional[bool] = None,
    operator: Optional[OperatorSetup] = None,
    trace: bool = False,
) -> ExecutionResult:
    """Run one discovery and return its result.

    Every way the run can end becomes a result, never an exception: SUCCESS or
    HUMAN_ESCALATED with a saved artifact, HARD_ABORT with a reason, or TECHNICAL_FAIL.
    max_steps lowers the step limit for one run (e.g. a first, cautious real run).
    sandbox says whether the bank is a test copy, where an irreversible step is
    performed to learn what follows it; None reads TARGET_ENVIRONMENT. A sandbox run
    whose start address isn't on this machine is refused before the browser opens.
    operator says a person is available: when the agent is stuck, and outside a sandbox
    at the irreversible step, the run hands them its window; what they do is recorded
    as the next steps, and after they hand back the agent carries on. trace records a
    Playwright trace to this run's own evidence folder, paused around the password's
    keystroke (off by default: real cost per run, so callers opt in).
    """
    if sandbox is None:
        sandbox = env.target_environment == "sandbox"
    return await _Discovery(request, model, logger, headless, max_steps, sandbox, operator, trace).execute()


@dataclass(frozen=True)
class _Next:
    """What the model is told next, before the new page: an action's outcome or a nudge."""

    text: str
    tool_use_id: Optional[str] = None
    is_error: bool = False


class _OutOfTime(Exception):
    pass


class _ModelUnavailable(Exception):
    pass


class _Discovery:
    def __init__(self, request: DiscoveryRequest, model: Model, logger: RunLogger, headless: bool,
                 max_steps: Optional[int], sandbox: bool, operator: Optional[OperatorSetup],
                 trace: bool = False) -> None:
        contract = request.contract
        self._contract = contract
        self._model = model
        self._logger = logger
        self._headless = headless
        self._sandbox = sandbox
        self._operator = operator
        self._trace_enabled = trace
        self._handoff: Optional[HandoffManager] = None
        self._person: Optional[PersonStepRecorder] = None
        self._handoffs: list[HandoffTelemetry] = []
        # Actions of a person's the handoff couldn't pass on to be recorded.
        self._unrecorded: list[str] = []
        # Time spent with a person, which doesn't count against the run's own limit.
        self._paused_ms = 0
        self._people = 0
        # A person gets one chance to unblock a stuck agent.
        self._stuck_handed_off = False
        # How a person took part, for the summary ("helped", "stopped", "not_saved", …).
        self._person_end: Optional[str] = None
        self._max_steps = max_steps or settings.discovery_max_steps
        self._usage = dict.fromkeys(_USAGE_KEYS, 0)

        configured = configured_credentials()
        missing = [credential.key for credential in contract.credentials if credential.key not in configured]
        if missing:
            raise ValueError(f"no configured value for the credential(s): {', '.join(missing)}")
        credentials = {credential.key: configured[credential.key] for credential in contract.credentials}
        number_keys = {p.key for p in contract.input_parameters if p.type == ParamType.NUMBER}
        text_inputs = {key: str(value) for key, value in request.input_values.items() if key not in number_keys}
        number_inputs = {key: float(value) for key, value in request.input_values.items() if key in number_keys}
        # The run's inputs as the caller gave them: what the payment check compares the screen with.
        self._input_values = dict(request.input_values)
        # Whether an irreversible step happened: None until one is met; "unknown" from the
        # moment it is clicked until what follows is recorded.
        self._irreversible_step: Optional[str] = None
        config_keys = [c.key for c in contract.credentials if c.kind == CredentialKind.CONFIG]
        self._username_key = config_keys[0] if config_keys else ""
        self._run = RunValues(
            text_inputs=text_inputs,
            number_inputs=number_inputs,
            username=str(credentials[self._username_key]) if self._username_key else "",
            secrets={c.key: credentials[c.key] for c in contract.credentials if c.kind == CredentialKind.SECRET},
        )
        self._values = placeholder_values(text_inputs, number_inputs, credentials)
        self._goal = fill_text(
            contract.description, {**text_inputs, **{key: number_text(n) for key, n in number_inputs.items()}}
        )
        self._first_message = goal_message(
            self._goal, contract.input_parameters, request.input_values, contract.credentials,
            contract.output_definitions,
        )
        self._tools = tool_definitions([output.key for output in contract.output_definitions])

        self._recorder = Recorder(logger)
        self._messages: list[dict[str, Any]] = []
        # Each output as its declared type, for the caller; the page's text as shown, for the
        # save-time scan, which looks for that text in what the model asked to check.
        self._outputs: dict[str, Union[str, int, float]] = {}
        self._read_text: dict[str, str] = {}
        self._output_definitions = {output.key: output for output in contract.output_definitions}
        self._turns = 0
        self._retries = 0
        self._violations = 0
        self._no_action = 0
        self._dialogs_seen = 0
        self._asserted_here = False
        self._started = time.monotonic()
        self._start_time = datetime.now(timezone.utc)
        self._screenshots = logger.screenshots_dir
        self._session: Optional[BrowserSession] = None

    async def execute(self) -> ExecutionResult:
        self._logger.execution_started(self._goal)
        # A run that may confirm a payment must start on this machine; the allowlist
        # then keeps every page it acts on at the bank's own host.
        if self._sandbox and (refusal := sandbox_refusal(self._contract.target_url)):
            return self._end(ExecutionStatus.HARD_ABORT, "SANDBOX_NOT_LOCAL", refusal)
        # What's learned is saved signed: without a trusted signing key it couldn't be kept,
        # so stop before anything is spent on the model.
        if refused := signing_refusal():
            return self._end(ExecutionStatus.HARD_ABORT, *refused)
        try:
            async with BrowserSession(self._logger, headless=self._headless, trace=self._trace_enabled) as session:
                self._session = session
                try:
                    check_route(self._contract.target_url, self._contract.allowed_paths)
                    await session.open(self._contract.target_url, timeout_ms=action_timeout_ms(self._time_left_ms()))
                    # The bank may redirect the start page elsewhere: where it landed counts.
                    check_route(session.page.url, self._contract.allowed_paths)
                except AllowlistViolation as violation:
                    return self._end(ExecutionStatus.HARD_ABORT, "ALLOWLIST_VIOLATION", str(violation))
                start = self._recorder.draft_start(session.page.url)
                await self._recorder.commit(start, session.page, self._run, derived=None)
                if self._operator is not None:
                    # A person's actions continue this same recording.
                    self._person = PersonStepRecorder(
                        self._recorder, session, self._run, self._contract.allowed_paths,
                        username_key=self._username_key, known_interruptions=self._contract.known_interruptions,
                        logger=self._logger)
                    self._handoff = HandoffManager(
                        session, self._logger, capability=self._contract.capability, goal=self._goal,
                        screenshots_dir=self._screenshots, announcer=self._operator.announcer,
                        timeout_ms=self._operator.timeout_ms, person_steps=self._person)
                return await self._loop()
        except _OutOfTime:
            return self._end(ExecutionStatus.HARD_ABORT, "TIMEOUT", "the time limit was reached")
        except _ModelUnavailable as failure:
            return self._end(ExecutionStatus.TECHNICAL_FAIL, "MODEL_UNAVAILABLE", str(failure))
        except anthropic.APIStatusError as error:
            # The API's own explanation names the offending field; redacted and scrubbed.
            reason = self._clean(str(error.message))[:500]
            return self._end(ExecutionStatus.TECHNICAL_FAIL, "MODEL_REQUEST_REJECTED",
                             f"the model API rejected the request (HTTP {error.status_code}): {reason}")
        except PlaywrightError as error:
            return self._end(ExecutionStatus.TECHNICAL_FAIL, "BROWSER_FAILED", _first_line(error))

    @property
    def _page(self):
        return self._session.page

    async def _loop(self) -> ExecutionResult:
        told = _Next(self._first_message)
        while True:
            if self._turns >= self._max_steps:
                return self._end(ExecutionStatus.HARD_ABORT, "MAX_STEPS",
                                 f"the step limit ({self._max_steps}) was reached")
            if self._time_left_ms() <= 0:
                raise _OutOfTime()
            async with observing(self._page) as observation:
                self._messages.append({"role": "user", "content": self._user_content(told, observation)})
                reply = await self._ask()
                self._turns += 1
                self._count_usage(reply)
                self._messages.append({"role": "assistant", "content": reply.content})
                self._log_fallbacks(reply)

                if reply.stop_reason == "refusal":
                    return self._end(ExecutionStatus.HARD_ABORT, "MODEL_REFUSED", "the model declined to continue")
                call = next((block for block in reply.content if getattr(block, "type", None) == "tool_use"), None)
                if call is None:
                    self._no_action += 1
                    if self._no_action >= 2:
                        self._no_action = 0
                        outcome = await self._stuck("STUCK_NO_PROGRESS", "two replies in a row had no action")
                        if isinstance(outcome, ExecutionResult):
                            return outcome
                        told = outcome
                        continue
                    told = _Next(NO_ACTION_REPROMPT)
                    continue
                self._no_action = 0
                self._log_decision(call)
                outcome = await self._handle(call, observation)
                if isinstance(outcome, ExecutionResult):
                    return outcome
                if outcome.is_error:
                    self._logger.action_refused(self._turns, call.name, self._clean(outcome.text))
                told = outcome

    def _user_content(self, told: _Next, observation: Observation) -> list[dict[str, Any]]:
        image = observation.marked_screenshot or observation.screenshot
        self._save_screenshot(image)
        page = [
            *page_blocks(image, observation.element_list_text()),
            {"type": "text", "text": progress(self._turns + 1, self._max_steps, self._time_left_ms())},
        ]
        if told.tool_use_id is None:
            return [{"type": "text", "text": told.text}, *page]
        # The tool result carries only its outcome as text; the new page follows it in the
        # same message. An error result holding the screenshot was rejected by the API.
        return [{
            "type": "tool_result",
            "tool_use_id": told.tool_use_id,
            "content": told.text,
            "is_error": told.is_error,
        }, *page]

    async def _ask(self) -> ModelReply:
        attempts = settings.discovery_llm_retries + 1
        failure: Optional[ModelCallFailed] = None
        for attempt in range(attempts):
            timeout_ms = min(settings.discovery_llm_call_timeout_ms, self._time_left_ms())
            if timeout_ms <= 0:
                raise _OutOfTime()
            try:
                return await self._model.reply(self._messages, self._tools, timeout_s=timeout_ms / 1000)
            except ModelCallFailed as error:
                failure = error
                if attempt + 1 < attempts:
                    self._retries += 1
        raise _ModelUnavailable(f"the model call failed {attempts} times ({failure})")

    async def _handle(self, call: Any, observation: Observation) -> Union[_Next, ExecutionResult]:
        name, args = call.name, dict(call.input)
        if name == "report_stuck":
            return await self._stuck(f"STUCK_{args['category']}",
                                     f"the model reported it can't continue: {self._clean(args['detail'])}", call.id)
        if name == "mark_goal_complete":
            if not self._asserted_here:
                return _Next(ASSERT_FIRST, call.id, is_error=True)
            if self._unread_outputs():
                return _Next(outputs_first(self._unread_outputs()), call.id, is_error=True)
            return self._save(ExecutionStatus.SUCCESS)
        if name == "assert_visible":
            try:
                drafted = await self._recorder.draft_assertion(args["expected_text"], args["reason"], self._page, self._run)
            except AssertionRefused as refused:
                return _Next(f"Refused: {refused}.", call.id, is_error=True)
            await self._recorder.commit(drafted.step, self._page, self._run, derived=drafted.derived)
            self._asserted_here = True
            return _Next("Done: the check passed and was recorded.", call.id)
        if name == "extract_text":
            return await self._extract(args, call.id)
        return await self._act(name, args, call.id, observation)

    async def _extract(self, args: dict[str, Any], call_id: str) -> Union[_Next, ExecutionResult]:
        # Read by the label beside the value, so replay reads the same place for anyone.
        try:
            drafted = await self._recorder.draft_extraction(
                args["label"], args["output_key"], args["reason"], self._page, self._run
            )
            enforce_safety(self._page.url, drafted.step.action, allowed_paths=self._contract.allowed_paths)
        except ExtractionRefused as refused:
            return _Next(f"Refused: {refused}.", call_id, is_error=True)
        except AllowlistViolation:
            return self._violation(call_id)
        key = args["output_key"]
        definition = self._output_definitions[key]
        try:
            # Checked before recording: a value of the wrong kind usually means the wrong label.
            value = read_output(drafted.value, definition)
        except UnreadableValue as unreadable:
            return _Next(f"Refused: {key} must be {output_kind(definition)}, but {unreadable}; "
                         "read it by the label right before that value.", call_id, is_error=True)
        await self._recorder.commit(drafted.step, self._page, self._run, derived=drafted.derived)
        self._outputs[key] = value
        self._read_text[key] = drafted.value
        return _Next(self._with_dialogs(f'Done: read "{drafted.value}" into {key}.'), call_id)

    async def _act(self, name: str, args: dict[str, Any], call_id: str, observation: Observation) -> Union[_Next, ExecutionResult]:
        kind = _TOOL_ACTIONS.get(name)
        if kind is None:
            return _Next(f"Refused: there is no tool named {name}.", call_id, is_error=True)
        try:
            element = observation.element(args["element"])
        except UnknownElement as unknown:
            return _Next(f"Refused: {unknown}.", call_id, is_error=True)
        try:
            derived = await derive_locators(self._page, element, self._run)
        except NoProvenLocator:
            return _Next(_UNLOCATABLE, call_id, is_error=True)
        value = args.get("text", args.get("option_label"))
        action = Action(kind, args["reason"], value=value, output_key=args.get("output_key"))
        try:
            step = await self._recorder.draft_step(action, element.handle, derived, self._page.url, run=self._run)
            enforce_safety(self._page.url, step.action, allowed_paths=self._contract.allowed_paths)
        except TypingRefused as refused:
            return _Next(f"Refused: {refused}.", call_id, is_error=True)
        except AllowlistViolation:
            return self._violation(call_id)
        if kind == ActionType.CLICK and await self._leads_off_the_allowlist(element):
            return self._violation(call_id)

        if name == "dismiss_overlay":
            return await self._dismiss(step, element, args["reason"], call_id)
        irreversible = step.safety_tier == SafetyTier.IRREVERSIBLE
        if irreversible and not self._sandbox and self._handoff is not None:
            refusal = await self._confirmation_refusal()
            self._irreversible_step = self._irreversible_step or "not_reached"
            if refusal is not None:
                return _Next(refusal, call_id, is_error=True)
            # A person makes the final confirmation. Their click is recorded as this step, and
            # values that exist only afterwards are read once they hand back.
            return await self._with_person("IRREVERSIBLE_STEP", "the final confirmation is left to a person", call_id)
        if irreversible and not self._sandbox:
            if self._unread_outputs():
                # Stopping now would lose the run: an artifact must produce every declared
                # output. The model goes back for them; the step isn't recorded yet.
                return _Next(outputs_first(self._unread_outputs()), call_id, is_error=True)
            # Recorded from the screen, never performed: the run stops here for a person.
            await self._recorder.commit(step, self._page, self._run, derived=derived, acted=False)
            self._irreversible_step = "not_reached"
            return self._save(ExecutionStatus.HUMAN_ESCALATED)
        if irreversible:
            refusal = await self._confirmation_refusal()
            if refusal is not None:
                # Not clicked and not recorded: the model can go back and put it right.
                self._irreversible_step = self._irreversible_step or "not_reached"
                return _Next(refusal, call_id, is_error=True)
            # A test environment: the step is performed, so what follows it (the receipt,
            # values that exist only afterwards) is learned too. Its dialog is accepted.
            self._logger.irreversible_executed(step.sequence_index)
            self._session.accepting_dialogs = True
            # From here it may have happened, even if the click seems to fail.
            self._irreversible_step = "unknown"

        try:
            await self._perform(kind, element, value)
        except ActionFailed as failed:
            return _Next(f"Failed: {failed}.", call_id, is_error=True)
        except PlaywrightTimeoutError:
            return _Next("Failed: the page did not respond in time.", call_id, is_error=True)
        finally:
            self._session.accepting_dialogs = False
        if not self._on_allowlist():
            await self._page.go_back()
            return self._violation(call_id)
        await self._recorder.commit(step, self._page, self._run, derived=derived)
        if irreversible:
            self._irreversible_step = "completed"
        if kind == ActionType.CLICK:
            self._asserted_here = False
        done = "Done: this irreversible step was performed (test environment)." if irreversible else "Done."
        return _Next(self._with_dialogs(done), call_id)

    async def _confirmation_refusal(self) -> Optional[str]:
        """Why the irreversible click must not happen, worded for the model; None if it may.

        The same payment check replay makes, so a wrong label in the contract or a wrong
        choice on the way is caught in discovery, not on the first real payment. The limit
        guards automatic payments; here a sandbox (no real money) or a person decides, so
        only a mismatch stops the click, before a person is ever asked.
        """
        checks = self._contract.confirmation_checks
        if not checks:
            return None
        readings = {check.label: await value_beside(self._page, check.label) for check in checks}
        result = authorize(checks, readings, self._input_values, env.auto_execute_limit, env.auto_execute_currency)
        self._logger.authorization_checked(result.code, [asdict(problem) for problem in result.problems])
        if result.code != MISMATCH:
            return None
        found = "; ".join(f'{problem.label} expected "{problem.expected}", seen "{problem.seen}"'
                          for problem in result.problems)
        return (f"Refused: the confirm screen doesn't match this run's request, so it was not confirmed: {found}. "
                "Go back and correct it, or use report_stuck if the screen can't show it.")

    async def _perform(self, kind: ActionType, element: PageElement, value: Optional[str]) -> None:
        timeout_ms = action_timeout_ms(self._time_left_ms())
        if kind == ActionType.CLICK:
            await click(element.handle, timeout_ms=timeout_ms)
            await self._page.wait_for_load_state("load", timeout=timeout_ms)
        elif kind == ActionType.TYPE:
            await type_text(element.handle, value or "", self._values, timeout_ms=timeout_ms,
                            tracing=self._session.tracing)
        else:
            await select_option(element.handle, value or "", self._values, timeout_ms=timeout_ms)

    async def _dismiss(self, step: Step, element: PageElement, reason: str, call_id: str) -> Union[_Next, ExecutionResult]:
        # Closing an overlay is never recorded: it may not appear on the next run, and
        # replay's recovery handles it whenever it does.
        if step.safety_tier != SafetyTier.SAFE:
            return _Next(_NOT_AN_OVERLAY, call_id, is_error=True)
        timeout_ms = action_timeout_ms(self._time_left_ms())
        try:
            await click(element.handle, timeout_ms=timeout_ms)
            await self._page.wait_for_load_state("load", timeout=timeout_ms)
        except ActionFailed as failed:
            return _Next(f"Failed: {failed}.", call_id, is_error=True)
        except PlaywrightTimeoutError:
            return _Next("Failed: the page did not respond in time.", call_id, is_error=True)
        if not self._on_allowlist():
            await self._page.go_back()
            return self._violation(call_id)
        self._logger.overlay_dismissed(self._clean(reason))
        self._asserted_here = False
        return _Next(self._with_dialogs("Done: closed. It was not recorded as a step."), call_id)

    async def _stuck(self, code: str, message: str, call_id: Optional[str] = None) -> Union[_Next, ExecutionResult]:
        # With a person available the agent gets one chance to be unblocked; stuck again, the
        # run ends as it would without one.
        if self._handoff is None or self._stuck_handed_off:
            return self._end(ExecutionStatus.HARD_ABORT, code, message)
        self._stuck_handed_off = True
        return await self._with_person(code, message, call_id)

    async def _with_person(self, code: str, message: str, call_id: Optional[str]) -> Union[_Next, ExecutionResult]:
        """Hand the window to a person; what they do is recorded as the next steps. Hand back →
        the model is told what was recorded and carries on. Anything else ends the run with
        nothing saved, since the task wasn't completed."""
        recorded_before = len(self._recorder.steps)
        outcome = await self._handoff.request(HandoffRequest(
            code, reason_for(code), (Choice.HAND_BACK, Choice.STOP), step_index=recorded_before))
        telemetry = outcome.telemetry
        self._handoffs.append(telemetry)
        self._paused_ms += telemetry.duration_ms or 0
        if telemetry.person_actions is not None:
            self._people += 1
        self._unrecorded.extend(outcome.unrecorded)
        if self._person.irreversible_done:
            self._irreversible_step = "completed"
        if outcome.choice != Choice.HAND_BACK:
            if outcome.window_closed:
                self._person_end = "window_closed"
            elif telemetry.resolution == HandoffResolution.OPERATOR_TIMED_OUT:
                self._person_end = "timed_out"
            else:
                self._person_end = "stopped"
            return self._end(ExecutionStatus.HUMAN_ESCALATED, code, message)
        self._person_end = "helped"
        self._asserted_here = False
        recorded = [step.description for step in self._recorder.steps[recorded_before:]]
        return _Next(self._clean(person_handed_back(recorded, gap=bool(self._problems()))), call_id)

    def _problems(self) -> list[str]:
        # What a person did that couldn't be recorded: the recording has a gap.
        return [*self._person.problems, *self._unrecorded] if self._person is not None else []

    def _violation(self, call_id: str) -> Union[_Next, ExecutionResult]:
        self._violations += 1
        if self._violations >= 2:
            return self._end(ExecutionStatus.HARD_ABORT, "ALLOWLIST_VIOLATION",
                             "a second action outside the allowlist was refused")
        return _Next(allowlist_refusal(first=True), call_id, is_error=True)

    def _save(self, status: ExecutionStatus) -> ExecutionResult:
        if self._problems():
            # A recording with a gap would break every replay at that point: the task is done,
            # and the result says why nothing was saved.
            self._person_end = "not_saved"
            return self._end(status, outputs=dict(self._outputs) or None)
        inputs = ScanInputs(self._run, username_key=self._username_key, extracted=dict(self._read_text))
        built = build_and_save(self._contract, self._recorder.steps, inputs, self._logger)
        if built.error is not None:
            return self._end(ExecutionStatus.HARD_ABORT, error=built.error)
        handoff = None
        if status == ExecutionStatus.HUMAN_ESCALATED:
            handoff = HandoffTelemetry(triggered_timestamp=datetime.now(timezone.utc),
                                       trigger_reason="IRREVERSIBLE_STEP", resolution=HandoffResolution.ABORTED)
        return self._end(status, artifact_version=built.artifact.metadata.version, handoff=handoff,
                         outputs=dict(self._outputs) or None)

    def _end(self, status: ExecutionStatus, code: Optional[str] = None, message: Optional[str] = None, *,
             error: Optional[ErrorDetail] = None, artifact_version: Optional[str] = None,
             handoff: Optional[HandoffTelemetry] = None,
             outputs: Optional[dict[str, Union[str, int, float]]] = None) -> ExecutionResult:
        if error is None and code is not None:
            error = ErrorDetail(code=code, message=message or code)
        duration_ms = int((time.monotonic() - self._started) * 1000)
        self._logger.run_usage(**self._usage, estimated_cost_usd=estimated_cost_usd(env.anthropic_model, self._usage))
        self._logger.execution_ended(status.value, error.code if error else None, error.message if error else None)
        self._logger.summary_metrics(duration_ms, len(self._recorder.steps), self._retries, self._people)
        inputs, shown_outputs = readable_values(self._input_values, outputs or {}, self._contract.confirmation_checks,
                                                self._contract.output_definitions)
        summary = self._clean(summarize(
            self._contract.capability, "DISCOVERY", status, goal=self._goal, inputs=inputs, outputs=shown_outputs,
            irreversible_step=self._irreversible_step, error=error,
            escalation=handoff.trigger_reason if handoff else None, version=artifact_version,
            person=self._person_end))
        result = ExecutionResult(
            # The run log's trace id, so a result leads straight to its log lines.
            run_id=self._logger.trace_id,
            capability=self._contract.capability,
            artifact_version=artifact_version,
            mode="DISCOVERY",
            status=status,
            summary=summary,
            irreversible_step=self._irreversible_step,
            start_time=self._start_time,
            end_time=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            handoff_events=[*self._handoffs, *([handoff] if handoff else [])],
            evidence_paths=EvidencePaths(
                log_file=str(self._logger.log_path), screenshots_dir=str(self._screenshots),
                playwright_trace_zip=self._expected_trace_path()),
            terminal_outputs=outputs,
            error=error,
        )
        self._logger.write_result(result.to_json())
        return result

    def _expected_trace_path(self) -> Optional[str]:
        """Where this run's trace will be once BrowserSession._close() saves it - called
        while still inside the `async with BrowserSession(...)` block, before it has, so
        the path is predicted rather than checked. `async with`'s __aexit__ always runs
        before this coroutine's result reaches its caller, so the file is there by the
        time anyone can look. None when tracing was never asked for, or no session ever
        opened (an early HARD_ABORT before the browser starts)."""
        if not self._trace_enabled or self._session is None:
            return None
        return str(self._logger.run_dir / "trace.zip")

    def _unread_outputs(self) -> list[str]:
        return [output.key for output in self._contract.output_definitions if output.key not in self._outputs]

    def _on_allowlist(self) -> bool:
        return _allowed(self._page.url, self._contract.allowed_paths)

    async def _leads_off_the_allowlist(self, element: PageElement) -> bool:
        # A link's destination is known before the click, so it is refused there rather than
        # visited and undone. Anything else is judged by where it lands, after the action.
        try:
            destination = await element.handle.evaluate("element => element.tagName === 'A' ? element.href : ''")
        except PlaywrightError:
            return False
        if not destination.startswith(("http://", "https://")):
            return False  # no link, "#", or a script link: nothing to judge before the click
        return not _allowed(destination, self._contract.allowed_paths)

    def _with_dialogs(self, text: str) -> str:
        dialogs = self._session.dialogs[self._dialogs_seen:]
        self._dialogs_seen = len(self._session.dialogs)
        return f"{text} A dialog was dismissed: {'; '.join(dialogs)}." if dialogs else text

    def _log_decision(self, call: Any) -> None:
        # What the model chose and why, every turn: a refused or failed action records no
        # step, so without this line the turn would be invisible in the evidence.
        args = dict(call.input)
        why = args.get("reason") or args.get("summary") or args.get("detail") or ""
        self._logger.model_action(self._turns, call.name, args.get("element"), self._clean(str(why)))

    def _count_usage(self, reply: ModelReply) -> None:
        # Every call's tokens are logged, so the real cost of a run (and whether caching
        # works) shows in the evidence, not in an estimate.
        if reply.usage is None:
            return
        self._logger.model_usage(self._turns, **{key: reply.usage[key] for key in _USAGE_KEYS})
        for key in _USAGE_KEYS:
            self._usage[key] += reply.usage[key]

    def _log_fallbacks(self, reply: ModelReply) -> None:
        # Every turn another model answered is recorded with both models' names.
        for block in reply.content:
            if getattr(block, "type", None) == "fallback":
                self._logger.model_fallback(block.from_.model, block.to.model)

    def _clean(self, text: str) -> str:
        # Model-written text in a result or log: patterns redacted and any secret scrubbed.
        secrets = [secret.get_secret_value() for secret in self._run.secrets.values()]
        return scrub_known_values(redact_text(text), secrets)

    def _save_screenshot(self, image: bytes) -> None:
        self._screenshots.mkdir(parents=True, exist_ok=True)
        (self._screenshots / f"{self._logger.trace_id}_turn{self._turns + 1:02d}.png").write_bytes(image)

    def _time_left_ms(self) -> int:
        return int(settings.discovery_timeout_ms - (time.monotonic() - self._started) * 1000 + self._paused_ms)


def _allowed(url: str, allowed_paths: list[str]) -> bool:
    # The bank's host, and one of the capability's pages when it declared any.
    try:
        check_domain(url)
        check_route(url, allowed_paths)
    except AllowlistViolation:
        return False
    return True


def _first_line(error: Exception) -> str:
    text = str(error)
    return text.splitlines()[0] if text else type(error).__name__
