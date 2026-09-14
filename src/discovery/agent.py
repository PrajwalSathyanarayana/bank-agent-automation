"""The discovery loop: observe the page, let the model choose one action, check it, record
it, act, and repeat until the goal is done, the run has to stop, or the next step is
irreversible.

The loop owns every control point: the step and time limits, the safety gate, what gets
recorded, and how the run ends. The model only ever chooses.
"""
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Protocol, Union

import anthropic
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from src.config.env import configured_credentials, env
from src.config.settings import settings
from src.discovery.artifact_builder import ArtifactContract, build_and_save
from src.discovery.backstop import ScanInputs
from src.discovery.browser import (
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
from src.discovery.prompts import (
    ASSERT_FIRST,
    NO_ACTION_REPROMPT,
    SYSTEM_PROMPT,
    allowlist_refusal,
    goal_message,
    page_blocks,
    progress,
    tool_definitions,
)
from src.discovery.recorder import Action, AssertionRefused, Recorder, TypingRefused
from src.observability.logger import RunLogger
from src.safety.allowlist import AllowlistViolation, check_domain, enforce_safety
from src.safety.redactor import redact_text, scrub_known_values
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
    "extract_text": ActionType.EXTRACT_TEXT,
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
        return ModelReply(response.stop_reason, list(response.content))


@dataclass(frozen=True)
class DiscoveryRequest:
    """What starts a discovery: the engineer's contract (its description is the goal
    template) and this run's input values."""

    contract: ArtifactContract
    input_values: Mapping[str, Union[str, float]]


async def discover(
    request: DiscoveryRequest, model: Model, logger: RunLogger, *, headless: bool = True
) -> ExecutionResult:
    """Run one discovery and return its result.

    Every way the run can end becomes a result, never an exception: SUCCESS or
    HUMAN_ESCALATED with a saved artifact, HARD_ABORT with a reason, or TECHNICAL_FAIL.
    """
    return await _Discovery(request, model, logger, headless).execute()


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
    def __init__(self, request: DiscoveryRequest, model: Model, logger: RunLogger, headless: bool) -> None:
        contract = request.contract
        self._contract = contract
        self._model = model
        self._logger = logger
        self._headless = headless

        configured = configured_credentials()
        missing = [credential.key for credential in contract.credentials if credential.key not in configured]
        if missing:
            raise ValueError(f"no configured value for the credential(s): {', '.join(missing)}")
        credentials = {credential.key: configured[credential.key] for credential in contract.credentials}
        number_keys = {p.key for p in contract.input_parameters if p.type == ParamType.NUMBER}
        text_inputs = {key: str(value) for key, value in request.input_values.items() if key not in number_keys}
        number_inputs = {key: float(value) for key, value in request.input_values.items() if key in number_keys}
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
        self._outputs: dict[str, str] = {}
        self._turns = 0
        self._retries = 0
        self._violations = 0
        self._no_action = 0
        self._dialogs_seen = 0
        self._asserted_here = False
        self._started = time.monotonic()
        self._start_time = datetime.now(timezone.utc)
        self._screenshots = settings.evidence_dir / "discovery" / "screenshots"
        self._session: Optional[BrowserSession] = None

    async def execute(self) -> ExecutionResult:
        self._logger.execution_started(self._goal)
        try:
            async with BrowserSession(self._logger, headless=self._headless) as session:
                self._session = session
                try:
                    await session.open(self._contract.target_url, timeout_ms=action_timeout_ms(self._time_left_ms()))
                except AllowlistViolation as violation:
                    return self._end(ExecutionStatus.HARD_ABORT, "ALLOWLIST_VIOLATION", str(violation))
                start = self._recorder.draft_start(session.page.url)
                await self._recorder.commit(start, session.page, self._run, derived=None)
                return await self._loop()
        except _OutOfTime:
            return self._end(ExecutionStatus.HARD_ABORT, "TIMEOUT", "the time limit was reached")
        except _ModelUnavailable as failure:
            return self._end(ExecutionStatus.TECHNICAL_FAIL, "MODEL_UNAVAILABLE", str(failure))
        except anthropic.APIStatusError as error:
            return self._end(ExecutionStatus.TECHNICAL_FAIL, "MODEL_REQUEST_REJECTED",
                             f"the model API rejected the request (HTTP {error.status_code})")
        except PlaywrightError as error:
            return self._end(ExecutionStatus.TECHNICAL_FAIL, "BROWSER_FAILED", _first_line(error))

    @property
    def _page(self):
        return self._session.page

    async def _loop(self) -> ExecutionResult:
        told = _Next(self._first_message)
        while True:
            if self._turns >= settings.discovery_max_steps:
                return self._end(ExecutionStatus.HARD_ABORT, "MAX_STEPS",
                                 f"the step limit ({settings.discovery_max_steps}) was reached")
            if self._time_left_ms() <= 0:
                raise _OutOfTime()
            async with observing(self._page) as observation:
                self._messages.append({"role": "user", "content": self._user_content(told, observation)})
                reply = await self._ask()
                self._turns += 1
                self._messages.append({"role": "assistant", "content": reply.content})
                self._log_fallbacks(reply)

                if reply.stop_reason == "refusal":
                    return self._end(ExecutionStatus.HARD_ABORT, "MODEL_REFUSED", "the model declined to continue")
                call = next((block for block in reply.content if getattr(block, "type", None) == "tool_use"), None)
                if call is None:
                    self._no_action += 1
                    if self._no_action >= 2:
                        return self._end(ExecutionStatus.HARD_ABORT, "STUCK_NO_PROGRESS",
                                         "two replies in a row had no action")
                    told = _Next(NO_ACTION_REPROMPT)
                    continue
                self._no_action = 0
                outcome = await self._handle(call, observation)
                if isinstance(outcome, ExecutionResult):
                    return outcome
                told = outcome

    def _user_content(self, told: _Next, observation: Observation) -> list[dict[str, Any]]:
        image = observation.marked_screenshot or observation.screenshot
        self._save_screenshot(image)
        page = [
            *page_blocks(image, observation.element_list_text()),
            {"type": "text", "text": progress(self._turns + 1, settings.discovery_max_steps, self._time_left_ms())},
        ]
        if told.tool_use_id is None:
            return [{"type": "text", "text": told.text}, *page]
        return [{
            "type": "tool_result",
            "tool_use_id": told.tool_use_id,
            "content": [{"type": "text", "text": told.text}, *page],
            "is_error": told.is_error,
        }]

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
            return self._end(ExecutionStatus.HARD_ABORT, f"STUCK_{args['category']}",
                             f"the model reported it can't continue: {self._clean(args['detail'])}")
        if name == "mark_goal_complete":
            if not self._asserted_here:
                return _Next(ASSERT_FIRST, call.id, is_error=True)
            return self._save(ExecutionStatus.SUCCESS)
        if name == "assert_visible":
            try:
                drafted = await self._recorder.draft_assertion(args["expected_text"], args["reason"], self._page, self._run)
            except AssertionRefused as refused:
                return _Next(f"Refused: {refused}.", call.id, is_error=True)
            await self._recorder.commit(drafted.step, self._page, self._run, derived=drafted.derived)
            self._asserted_here = True
            return _Next("Done: the check passed and was recorded.", call.id)
        return await self._act(name, args, call.id, observation)

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
            enforce_safety(self._page.url, step.action)
        except TypingRefused as refused:
            return _Next(f"Refused: {refused}.", call_id, is_error=True)
        except AllowlistViolation:
            return self._violation(call_id)

        if name == "dismiss_overlay":
            return await self._dismiss(step, element, args["reason"], call_id)
        if step.safety_tier == SafetyTier.IRREVERSIBLE:
            # Recorded from the screen, never performed: the run stops here for a person.
            await self._recorder.commit(step, self._page, self._run, derived=derived, acted=False)
            return self._save(ExecutionStatus.HUMAN_ESCALATED)

        try:
            read = await self._perform(kind, element, value, args.get("output_key"))
        except ActionFailed as failed:
            return _Next(f"Failed: {failed}.", call_id, is_error=True)
        except PlaywrightTimeoutError:
            return _Next("Failed: the page did not respond in time.", call_id, is_error=True)
        if not self._on_allowlist():
            await self._page.go_back()
            return self._violation(call_id)
        await self._recorder.commit(step, self._page, self._run, derived=derived)
        if kind == ActionType.CLICK:
            self._asserted_here = False
        done = f'Done: read "{read}".' if kind == ActionType.EXTRACT_TEXT else "Done."
        return _Next(self._with_dialogs(done), call_id)

    async def _perform(self, kind: ActionType, element: PageElement, value: Optional[str],
                       output_key: Optional[str]) -> Optional[str]:
        timeout_ms = action_timeout_ms(self._time_left_ms())
        if kind == ActionType.CLICK:
            await click(element.handle, timeout_ms=timeout_ms)
            await self._page.wait_for_load_state("load", timeout=timeout_ms)
        elif kind == ActionType.TYPE:
            await type_text(element.handle, value or "", self._values, timeout_ms=timeout_ms)
        elif kind == ActionType.SELECT:
            await select_option(element.handle, value or "", self._values, timeout_ms=timeout_ms)
        else:
            read = " ".join((await element.handle.inner_text()).split())
            self._outputs[output_key] = read
            return read
        return None

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

    def _violation(self, call_id: str) -> Union[_Next, ExecutionResult]:
        self._violations += 1
        if self._violations >= 2:
            return self._end(ExecutionStatus.HARD_ABORT, "ALLOWLIST_VIOLATION",
                             "a second action outside the allowlist was refused")
        return _Next(allowlist_refusal(first=True), call_id, is_error=True)

    def _save(self, status: ExecutionStatus) -> ExecutionResult:
        inputs = ScanInputs(self._run, username_key=self._username_key, extracted=dict(self._outputs))
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
             handoff: Optional[HandoffTelemetry] = None, outputs: Optional[dict[str, str]] = None) -> ExecutionResult:
        if error is None and code is not None:
            error = ErrorDetail(code=code, message=message or code)
        duration_ms = int((time.monotonic() - self._started) * 1000)
        self._logger.execution_ended(status.value, error.code if error else None, error.message if error else None)
        self._logger.summary_metrics(duration_ms, len(self._recorder.steps), self._retries, 0)
        return ExecutionResult(
            capability=self._contract.capability,
            artifact_version=artifact_version,
            mode="DISCOVERY",
            status=status,
            start_time=self._start_time,
            end_time=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            handoff_events=[handoff] if handoff else [],
            evidence_paths=EvidencePaths(log_file=str(self._logger.log_path), screenshots_dir=str(self._screenshots)),
            terminal_outputs=outputs,
            error=error,
        )

    def _on_allowlist(self) -> bool:
        try:
            check_domain(self._page.url)
        except AllowlistViolation:
            return False
        return True

    def _with_dialogs(self, text: str) -> str:
        dialogs = self._session.dialogs[self._dialogs_seen:]
        self._dialogs_seen = len(self._session.dialogs)
        return f"{text} A dialog was dismissed: {'; '.join(dialogs)}." if dialogs else text

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
        return int(settings.discovery_timeout_ms - (time.monotonic() - self._started) * 1000)


def _first_line(error: Exception) -> str:
    text = str(error)
    return text.splitlines()[0] if text else type(error).__name__
