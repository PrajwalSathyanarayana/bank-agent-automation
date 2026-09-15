"""Replay: running a saved artifact with no model. This is the production path.

The capability's latest saved version is loaded and its signature verified before anything
else; an artifact that can't be trusted is refused, never swapped for an older one. The
caller's inputs are checked against the contract. Then each step runs in order: find its
element, pass the safety gate, act, and verify its checks. Only the contract decides what
gets in the way (its known outcomes and interruptions), and an irreversible step runs only
after the payment check. Every run ends in exactly one structured result.

When a person is available, a step that needs one hands them this run's live window
instead of ending the run, and the run carries on or ends by what they choose. Once a
person has had control, replay never makes the irreversible step itself.
"""
import math
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import NoReturn, Optional, Union
from urllib.parse import urlsplit

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from pydantic import SecretStr

from src.config.env import configured_credentials, env
from src.config.settings import settings
from src.handoff.session_manager import Choice, HandoffManager, HandoffOutcome, HandoffRequest, OperatorSetup
from src.locating.checks import element_wording, is_password_box, text_pattern, value_beside, visible_text
from src.locating.values import UnreadableValue, read_output
from src.observability.logger import RunLogger
from src.observability.summary import readable_values, reason_for, summarize
from src.replay.checks import CheckFailed, CheckValues, verify_shown_text, verify_step_checks
from src.replay.locator_resolver import Found, NotFound, find_element
from src.replay.recovery_engine import Recoveries, interruption_showing, missing_option, outcome_showing, recover
from src.safety.allowlist import AllowlistViolation, check_route, enforce_safety
from src.safety.authorization import authorize
from src.safety.classifier import SafetyEscalation, verify_tier
from src.safety.redactor import redact_text, scrub_known_values
from src.safety.secret_typing import typing_refusal
from src.storage.artifacts import latest_saved
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
from src.types.artifact_schema import Artifact, CredentialKind, GlobalAssertionType, KnownOutcome, ParamType
from src.types.placeholders import fill_text
from src.types.result_schema import (
    FAILURE_TEXT_MAX,
    BusinessOutcome,
    ErrorDetail,
    EvidencePaths,
    ExecutionResult,
    ExecutionStatus,
    FailureDetail,
    HandoffResolution,
    HandoffTelemetry,
    RecoveryAttemptLog,
    RecoveryTier,
    StepExecutionTrace,
    StepStatus,
)
from src.types.step_schema import ActionType, SafetyTier, Step
from src.types.versioning import version_text

# A capability names a folder in the store, so only a plain name is looked up.
_CAPABILITY_NAME = re.compile(r"[a-z][a-z0-9_]*")
_ACTIONS = {ActionType.CLICK, ActionType.TYPE, ActionType.SELECT}


@dataclass(frozen=True)
class ReplayRequest:
    """What a caller sends: the capability's name and this run's input values."""

    capability: str
    inputs: Mapping[str, Union[str, int, float, bool]]


async def replay(request: ReplayRequest, logger: RunLogger, *, headless: bool = True,
                 operator: Optional[OperatorSetup] = None, trace: bool = False,
                 slow_mo_ms: Optional[int] = None) -> ExecutionResult:
    """Run the capability's latest trusted artifact with these inputs; one result, never an exception.

    With an operator, a step that needs a person hands them the run's window; without one,
    the run ends HUMAN_ESCALATED there. trace records a Playwright trace to this run's own
    evidence folder, paused around the password's keystroke (off by default: real cost per
    run, so callers opt in). slow_mo_ms paces every action for a person watching a headed
    run; no effect headless.
    """
    return await _Replay(request, logger, headless, operator, trace, slow_mo_ms).execute()


class _Stop(Exception):
    """The run ends with this result."""

    def __init__(self, result: ExecutionResult) -> None:
        super().__init__(result.status.value)
        self.result = result


class _StartOver(Exception):
    """A declared interruption's recovery: run the artifact again from its first step."""


class _Trouble(Exception):
    """A step couldn't do its part; the contract decides what that means."""

    def __init__(self, failed: CheckFailed, code: str) -> None:
        super().__init__(code)
        self.failed = failed
        self.code = code


class _NeedsPerson(Exception):
    """A step needs a person and one is available: the run hands them its window."""

    def __init__(self, code: str, failed: CheckFailed) -> None:
        super().__init__(code)
        self.code = code
        self.failed = failed


@dataclass
class _StepRecord:
    """What happened in one step so far, for its trace."""

    step: Step
    started: float = field(default_factory=time.monotonic)
    attempts: int = 0
    priority: Optional[int] = None
    recoveries: list[RecoveryAttemptLog] = field(default_factory=list)
    # At most one handoff per step: after it, trouble ends the run as it would without a person.
    handed_off: bool = False
    # Something worth saying in the trace of a step that still passed.
    note: Optional[str] = None


class _Replay:
    def __init__(self, request: ReplayRequest, logger: RunLogger, headless: bool,
                 operator: Optional[OperatorSetup], trace: bool = False,
                 slow_mo_ms: Optional[int] = None) -> None:
        self._request = request
        self._logger = logger
        self._headless = headless
        self._operator = operator
        # Not self._trace: _Replay already has a _trace() method (the step-trace recorder).
        self._trace_enabled = trace
        self._slow_mo_ms = slow_mo_ms
        self._handoff: Optional[HandoffManager] = None
        self._handoffs: list[HandoffTelemetry] = []
        # Time spent with a person, which doesn't count against the run's own limit.
        self._paused_ms = 0
        self._people = 0
        self._person_had_control = False
        # How the last person in the run took part, for the summary ("helped", "finished", …).
        self._person_end: Optional[str] = None
        self._started = time.monotonic()
        self._start_time = datetime.now(timezone.utc)
        self._artifact: Optional[Artifact] = None
        self._session: Optional[BrowserSession] = None
        self._traces: list[StepExecutionTrace] = []
        self._outputs: dict[str, Union[str, int, float]] = {}
        self._recoveries = Recoveries()
        self._irreversible_done = False
        # Set once the irreversible step's own checks passed after it: what it led to was seen.
        self._irreversible_confirmed = False
        self._screenshots = logger.screenshots_dir
        self._goal = f"Replay {request.capability}"
        self._secrets: list[str] = []

    # --- the run ---

    async def execute(self) -> ExecutionResult:
        refused = self._load()
        self._logger.execution_started(self._goal)
        if refused is not None:
            return self._end(ExecutionStatus.HARD_ABORT, error=refused)
        try:
            # A visible run is watched by a person, so its page follows their window.
            async with BrowserSession(self._logger, headless=self._headless, fit_window=not self._headless,
                                      trace=self._trace_enabled, slow_mo_ms=self._slow_mo_ms) as session:
                self._session = session
                if self._operator is not None:
                    self._handoff = HandoffManager(
                        session, self._logger, capability=self._request.capability, goal=self._goal,
                        screenshots_dir=self._screenshots, announcer=self._operator.announcer,
                        timeout_ms=self._operator.timeout_ms)
                return await self._run()
        except PlaywrightError as error:
            return self._end(ExecutionStatus.TECHNICAL_FAIL,
                             error=ErrorDetail(code="BROWSER_FAILED", message=_first_line(error)))

    async def _run(self) -> ExecutionResult:
        while True:
            try:
                for position in range(len(self._artifact.steps)):
                    await self._run_step(position)
                return await self._finish()
            except _StartOver:
                continue  # the recovery engine allows this once, and never after the irreversible step
            except _Stop as stop:
                return stop.result

    async def _run_step(self, position: int) -> None:
        steps = self._artifact.steps
        step = steps[position]
        next_step = steps[position + 1] if position + 1 < len(steps) else None
        record = _StepRecord(step)
        self._logger.step_fetched(step.step_id, step.sequence_index, step.action.value)
        if self._time_left_ms() <= 0:
            raise _Stop(await self._failure(record, ExecutionStatus.TECHNICAL_FAIL, "TIMEOUT", CheckFailed(
                f"the run finished within {settings.replay_total_timeout_ms // 1000} s",
                f"step {step.sequence_index} not reached in time")))
        try:
            await self._step_body(record, next_step)
        except _NeedsPerson as need:
            await self._with_person(record, position, next_step, need)
        self._trace(record, StepStatus.RECOVERED if record.recoveries else StepStatus.PASSED, error_message=record.note)
        if step.safety_tier == SafetyTier.IRREVERSIBLE:
            self._irreversible_confirmed = True

    async def _step_body(self, record: _StepRecord, next_step: Optional[Step]) -> None:
        if record.step.action == ActionType.NAVIGATE:
            record.attempts += 1
            await self._open_start(record)
        else:
            await self._clear_interruptions(record)
            await self._act(record)
        await self._verify(record, next_step)

    # --- before the run: the artifact, the inputs, the credentials ---

    def _load(self) -> Optional[ErrorDetail]:
        capability = self._request.capability
        if not _CAPABILITY_NAME.fullmatch(capability):
            return ErrorDetail(code="UNKNOWN_CAPABILITY", message="a capability is named in lower case, e.g. bill_pay")
        saved = latest_saved(capability)
        if saved is None:
            return ErrorDetail(code="NO_ARTIFACT", message=f"no saved artifact for {capability}; discover it first")
        if not saved.trusted:
            return ErrorDetail(code="INTEGRITY_CHECK_FAILED", message=(
                f"the latest saved version ({version_text(saved.version)}) fails its signature or doesn't match "
                "its file; replay never falls back to an older version"))
        self._artifact = saved.artifact
        problem = self._check_inputs()
        if problem is not None:
            return ErrorDetail(code="INPUT_INVALID", message=problem)
        configured = configured_credentials()
        missing = [credential.key for credential in self._artifact.credentials if credential.key not in configured]
        if missing:
            return ErrorDetail(code="CREDENTIAL_MISSING", message=f"no configured value for {', '.join(missing)}")

        declared = {parameter.key: parameter.type for parameter in self._artifact.input_parameters}
        given = self._request.inputs
        numbers = {key: float(value) for key, value in given.items() if declared[key] == ParamType.NUMBER}
        texts = {key: str(value) for key, value in given.items() if declared[key] != ParamType.NUMBER}
        credentials = {credential.key: configured[credential.key] for credential in self._artifact.credentials}
        self._secret_names = {c.key for c in self._artifact.credentials if c.kind == CredentialKind.SECRET}
        self._secrets = [value.get_secret_value() for value in credentials.values() if isinstance(value, SecretStr)]
        self._typing = placeholder_values(texts, numbers, credentials)
        self._values = CheckValues(text=texts, numbers=numbers)
        self._goal = fill_text(self._artifact.metadata.description,
                               {**texts, **{key: number_text(number) for key, number in numbers.items()}})
        return None

    def _check_inputs(self) -> Optional[str]:
        # Typed as the contract declares, all present, nothing extra: never coerced or guessed.
        declared = {parameter.key: parameter for parameter in self._artifact.input_parameters}
        given = self._request.inputs
        unknown = sorted(set(given) - set(declared))
        if unknown:
            return f"not inputs of this capability: {', '.join(unknown)}"
        missing = sorted(key for key, parameter in declared.items() if parameter.required and key not in given)
        if missing:
            return f"missing inputs: {', '.join(missing)}"
        for key, value in given.items():
            kind = declared[key].type
            if kind == ParamType.NUMBER:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    return f"{key} must be a number"
            elif kind == ParamType.BOOLEAN:
                if not isinstance(value, bool):
                    return f"{key} must be true or false"
            elif not isinstance(value, str) or not value.strip():
                return f"{key} must be non-empty text"
        return None

    # --- each step ---

    async def _open_start(self, record: _StepRecord) -> None:
        metadata = self._artifact.metadata
        start = metadata.tenant_override_url or metadata.target_url
        timeout_ms = action_timeout_ms(self._time_left_ms())
        try:
            check_route(start, self._artifact.allowed_paths)
            await self._session.open(start, timeout_ms=timeout_ms)
            # The bank may redirect the start page elsewhere: where it landed counts.
            check_route(self._page.url, self._artifact.allowed_paths)
        except AllowlistViolation as violation:
            raise _Stop(await self._failure(record, ExecutionStatus.HARD_ABORT, "ALLOWLIST_VIOLATION",
                                            CheckFailed("a page this capability may visit", str(violation))))
        except PlaywrightTimeoutError:
            # A bank too slow to answer at all: a clear failure, not a crash.
            raise _Stop(await self._failure(record, ExecutionStatus.TECHNICAL_FAIL, "PAGE_TIMEOUT", CheckFailed(
                f"the start page within {timeout_ms / 1000:g} s", "it didn't arrive in time")))

    async def _clear_interruptions(self, record: _StepRecord) -> None:
        # Looked for before acting, so a popup is closed before it can block a click
        # instead of the click waiting out its timeout.
        while True:
            interruption = await interruption_showing(self._page, self._artifact.known_interruptions, self._values)
            if interruption is None:
                return
            await self._recover(record, interruption, CheckFailed(f"step {record.step.sequence_index} unobstructed",
                                                                  f"{interruption.code} showing"))

    async def _act(self, record: _StepRecord) -> None:
        step = record.step
        while True:
            found = await find_element(self._page, step, self._values.text, self._logger)
            record.attempts += found.attempts
            try:
                if isinstance(found, NotFound):
                    raise _Trouble(CheckFailed(f"the element for step {step.sequence_index}", found.observed()),
                                   "LOCATOR_NOT_FOUND")
                record.priority = found.priority
                await self._perform(record, found)
                return
            except _Trouble as trouble:
                # A RISKY or IRREVERSIBLE action is never done twice: it could submit twice.
                await self._after_trouble(record, trouble, retry=step.safety_tier == SafetyTier.SAFE)

    async def _perform(self, record: _StepRecord, found: Found) -> None:
        step = record.step
        page = self._page
        element = found.element
        try:
            enforce_safety(page.url, step.action, allowed_paths=self._artifact.allowed_paths)
            verify_tier(step, page.url, element_wording=await element_wording(element))
        except AllowlistViolation as violation:
            raise _Stop(await self._failure(record, ExecutionStatus.HARD_ABORT, "ALLOWLIST_VIOLATION",
                                            CheckFailed("a page this capability may act on", str(violation))))
        except SafetyEscalation:
            await self._escalate(record, ExecutionStatus.HUMAN_ESCALATED, "UNDECLARED_RISK", CheckFailed(
                f"a {step.safety_tier.value} step", "the element here is riskier than the artifact declares"))

        if step.action == ActionType.ASSERT_TEXT:
            failed = await verify_shown_text(element, step.input_value or "", self._values,
                                             settings.replay_checkpoint_timeout_ms, self._artifact.known_outcomes)
            if failed is not None:
                raise _Trouble(failed, "CHECK_FAILED")
            return
        if step.action == ActionType.EXTRACT_TEXT:
            await self._read(record, element)
            return
        if step.action not in _ACTIONS:
            raise _Stop(await self._failure(record, ExecutionStatus.HARD_ABORT, "UNSUPPORTED_ACTION", CheckFailed(
                "a click, typing, a choice, a reading or a check", f"a {step.action.value} step")))
        if step.action == ActionType.SELECT:
            outcome = await missing_option(step, element, self._artifact.known_outcomes, self._values)
            if outcome is not None:
                raise _Stop(await self._outcome(record, outcome))
        if step.safety_tier == SafetyTier.IRREVERSIBLE:
            if self._person_had_control:
                # A person had this run's window earlier and may already have made it: replay
                # never makes it after that, so it can't happen twice.
                await self._escalate(record, ExecutionStatus.HUMAN_ESCALATED, "PERSON_HAD_CONTROL", CheckFailed(
                    "no person had control earlier in this run", "a person had control earlier in this run"))
            await self._authorize(record)
        await self._do(record, element)
        if not self._allowed_here():
            raise _Stop(await self._failure(record, ExecutionStatus.HARD_ABORT, "ALLOWLIST_VIOLATION", CheckFailed(
                "a page this capability may visit", f"the action led to {_page_path(page)}")))

    async def _do(self, record: _StepRecord, element) -> None:
        step = record.step
        timeout_ms = action_timeout_ms(self._time_left_ms())
        handle = await element.element_handle(timeout=timeout_ms)
        irreversible = step.safety_tier == SafetyTier.IRREVERSIBLE
        try:
            if step.action == ActionType.TYPE:
                # The typing rule again at the keystroke: a secret only into a password box.
                refusal = typing_refusal(step.input_value or "", into_password_box=await is_password_box(handle),
                                         secret_names=self._secret_names)
                if refusal is not None:
                    raise _Stop(await self._failure(record, ExecutionStatus.HARD_ABORT, "TYPING_REFUSED",
                                                    CheckFailed("a value this field may take", refusal)))
                await type_text(handle, step.input_value or "", self._typing, timeout_ms=timeout_ms,
                                tracing=self._session.tracing, page=self._session.pointer_page)
            elif step.action == ActionType.SELECT:
                await select_option(handle, step.input_value or "", self._typing, timeout_ms=timeout_ms,
                                    page=self._session.pointer_page)
            else:
                if irreversible:
                    # Counted as done from here: even a click that seems to fail may have gone through.
                    self._irreversible_done = True
                    self._session.accepting_dialogs = True
                await click(handle, timeout_ms=timeout_ms, page=self._session.pointer_page)
                await self._page.wait_for_load_state("load", timeout=timeout_ms)
        except (ActionFailed, PlaywrightTimeoutError) as failure:
            raise _Trouble(CheckFailed(f"step {step.sequence_index} to {step.action.value}", _first_line(failure)),
                           "ACTION_FAILED") from None
        finally:
            self._session.accepting_dialogs = False
            await handle.dispose()

    async def _read(self, record: _StepRecord, element) -> None:
        key = record.step.output_key or ""
        definition = next(output for output in self._artifact.output_definitions if output.key == key)
        shown = await visible_text(element)
        try:
            self._outputs[key] = read_output(shown, definition)
        except UnreadableValue as unreadable:
            raise _Trouble(CheckFailed(f"{key} as {definition.type.value}", str(unreadable)), "OUTPUT_UNREADABLE")

    async def _authorize(self, record: _StepRecord) -> None:
        # The payment check: the screen must show exactly what was asked, within the limit.
        checks = self._artifact.confirmation_checks
        readings = {check.label: await value_beside(self._page, check.label) for check in checks}
        result = authorize(checks, readings, self._request.inputs, env.auto_execute_limit, env.auto_execute_currency)
        self._logger.authorization_checked(result.code, [asdict(problem) for problem in result.problems])
        if result.authorized:
            return
        if result.problems:
            first = result.problems[0]
            failed = CheckFailed(f"{first.label} {first.expected}", f"{first.label} {first.seen}: {first.reason}")
        else:
            failed = CheckFailed("confirmation checks declared for this step", "none are declared")
        await self._escalate(record, ExecutionStatus.HUMAN_ESCALATED, result.code, failed)

    async def _verify(self, record: _StepRecord, next_step: Optional[Step]) -> None:
        while True:
            failed = await verify_step_checks(self._page, record.step, next_step, self._values,
                                              self._artifact.known_outcomes)
            if failed is None:
                return
            if failed.next_step and self._handoff is not None:
                # Only the next step's element is missing. This step's own checks come first and
                # held, so it did its part: the person is shown the step that needs them, where
                # that step's own lookup fails. A known outcome on the page still answers first.
                outcome = await outcome_showing(self._page, self._artifact.known_outcomes)
                if outcome is not None:
                    raise _Stop(await self._outcome(record, outcome))
                record.note = f"step {next_step.sequence_index}'s element wasn't found here; left to that step"
                return
            # The action already happened: after a recovery, only the checks are looked at again.
            await self._after_trouble(record, _Trouble(failed, "CHECK_FAILED"), retry=True)

    async def _after_trouble(self, record: _StepRecord, trouble: _Trouble, *, retry: bool) -> None:
        """Returns only when a declared interruption was cleared and the step may try again;
        otherwise the run ends with the declared outcome, a start over, or a failure."""
        outcome = await outcome_showing(self._page, self._artifact.known_outcomes)
        if outcome is not None:
            raise _Stop(await self._outcome(record, outcome))
        if retry:
            interruption = await interruption_showing(self._page, self._artifact.known_interruptions, self._values)
            if interruption is not None:
                await self._recover(record, interruption, trouble.failed)
                return
        await self._escalate(record, ExecutionStatus.TECHNICAL_FAIL, trouble.code, trouble.failed)

    async def _recover(self, record: _StepRecord, interruption, failed: CheckFailed) -> None:
        recovery = await recover(self._page, interruption, self._values, self._artifact.allowed_paths, self._logger,
                                 self._recoveries, irreversible_done=self._irreversible_done)
        record.recoveries.append(recovery.log)
        if recovery.start_over:
            self._trace(record, StepStatus.RECOVERED, error_message=f"started over: {interruption.code}")
            raise _StartOver()
        if not recovery.log.resolved:
            await self._escalate(record, ExecutionStatus.TECHNICAL_FAIL, "INTERRUPTION_NOT_CLEARED",
                                 CheckFailed(failed.expected, f"{interruption.code} couldn't be cleared: "
                                                              f"{recovery.log.details}"))

    # --- a person in the loop ---

    async def _escalate(self, record: _StepRecord, status: ExecutionStatus, code: str,
                        failed: CheckFailed) -> NoReturn:
        """The step needs a person. With one available and none yet at this step, the run
        hands them its window; otherwise it ends here with this status."""
        if self._handoff is not None and not record.handed_off:
            raise _NeedsPerson(code, failed)
        raise _Stop(await self._failure(record, status, code, failed))

    async def _with_person(self, record: _StepRecord, position: int, next_step: Optional[Step],
                           need: _NeedsPerson) -> None:
        """Hand the window to a person; returns only when they handed it back and the step is
        done. Every other ending stops the run."""
        step = record.step
        record.handed_off = True
        outcome = await self._handoff.request(HandoffRequest(
            need.code, reason_for(need.code), self._choices(position), step_index=step.sequence_index,
            step_description=self._clean(step.description)))
        telemetry = outcome.telemetry
        self._handoffs.append(telemetry)
        self._paused_ms += telemetry.duration_ms or 0
        if telemetry.person_actions is not None:
            self._people += 1
            self._person_had_control = True
        if outcome.choice == Choice.HAND_BACK:
            self._person_end = "helped"
            await self._resume(record, next_step, outcome)
            return
        finished = outcome.choice == Choice.FINISHED
        await self._judge_irreversible(outcome, finished=finished)
        if finished:
            self._person_end = "finished"
            raise _Stop(await self._finished_by_person(record, position, outcome))
        if outcome.window_closed:
            self._person_end = "window_closed"
        elif telemetry.resolution == HandoffResolution.OPERATOR_TIMED_OUT:
            self._person_end = "timed_out"
        else:
            self._person_end = "stopped"
        raise _Stop(await self._failure(record, ExecutionStatus.HUMAN_ESCALATED, need.code, need.failed))

    def _choices(self, position: int) -> tuple[Choice, ...]:
        # Handing back only makes sense with something left for replay to do: not at or
        # after the irreversible step, and not at the last action.
        steps = self._artifact.steps
        last_action = max((i for i, step in enumerate(steps) if step.action in _ACTIONS), default=-1)
        irreversible = self._irreversible_index()
        if position >= last_action or self._irreversible_done or (irreversible is not None and position >= irreversible):
            return (Choice.FINISHED, Choice.STOP)
        return (Choice.HAND_BACK, Choice.FINISHED, Choice.STOP)

    async def _resume(self, record: _StepRecord, next_step: Optional[Step], outcome: HandoffOutcome) -> None:
        """Carry on from wherever the person left the page. A click whose checks now hold was
        done by the person; any other step, or a click whose checks don't hold, replay does
        once more itself — filling, choosing, reading and checking are safe to repeat. A
        failure now ends the run as usual, since each step gets one handoff."""
        step = record.step
        await self._clear_interruptions(record)
        by_person = step.action == ActionType.CLICK and await verify_step_checks(
            self._page, step, next_step, self._values, self._artifact.known_outcomes) is None
        log = RecoveryAttemptLog(timestamp=datetime.now(timezone.utc), tier=RecoveryTier.TIER_3_HANDOFF,
                                 resolved=by_person, screenshot_path=outcome.back_screenshot,
                                 details="done by a person" if by_person else "handed back; replay did the step again")
        record.recoveries.append(log)
        if not by_person:
            await self._step_body(record, next_step)
            record.recoveries[record.recoveries.index(log)] = log.model_copy(update={"resolved": True})

    async def _finished_by_person(self, record: _StepRecord, position: int, outcome: HandoffOutcome) -> ExecutionResult:
        """The person finished the task: replay clicks nothing more and reads what the page
        shows of the outputs still unread. One it can't read is left out; a run a person
        finished never fails for it."""
        record.recoveries.append(RecoveryAttemptLog(
            timestamp=datetime.now(timezone.utc), tier=RecoveryTier.TIER_3_HANDOFF, resolved=True,
            screenshot_path=outcome.back_screenshot, details="finished by a person"))
        self._trace(record, StepStatus.RECOVERED)
        for step in self._artifact.steps[position:]:
            if step.action == ActionType.EXTRACT_TEXT and step.output_key not in self._outputs:
                await self._read_if_shown(step)
        return self._end(ExecutionStatus.HUMAN_ESCALATED)

    async def _read_if_shown(self, step: Step) -> None:
        found = await find_element(self._page, step, self._values.text, self._logger)
        if isinstance(found, NotFound):
            return
        definition = next(output for output in self._artifact.output_definitions if output.key == step.output_key)
        try:
            self._outputs[step.output_key] = read_output(await visible_text(found.element), definition)
        except (UnreadableValue, PlaywrightError):
            return

    async def _judge_irreversible(self, outcome: HandoffOutcome, *, finished: bool) -> None:
        """After a handoff that ends the run: whether the person made the irreversible step,
        judged from the page, never from anyone's word. Its own checks hold → completed. They
        don't, after the person said they finished or clicked anything → unknown."""
        index = self._irreversible_index()
        if index is None or self._irreversible_confirmed or outcome.telemetry.person_actions is None:
            return
        clicked = any(action.kind == "click" for action in outcome.actions)
        if not (finished or clicked):
            return
        self._irreversible_done = True
        if outcome.window_closed:
            return
        steps = self._artifact.steps
        after = steps[index + 1] if index + 1 < len(steps) else None
        self._irreversible_confirmed = await verify_step_checks(
            self._page, _briefly(steps[index]), after, self._values, self._artifact.known_outcomes) is None

    def _irreversible_index(self) -> Optional[int]:
        return next((i for i, step in enumerate(self._artifact.steps) if step.safety_tier == SafetyTier.IRREVERSIBLE),
                    None)

    # --- the end of the run ---

    async def _finish(self) -> ExecutionResult:
        last = _StepRecord(self._artifact.steps[-1])
        for assertion in self._artifact.global_assertions:
            failed = await self._final_check(assertion.type, assertion.value)
            if failed is not None:
                raise _Stop(await self._failure(last, ExecutionStatus.TECHNICAL_FAIL, "CHECK_FAILED", failed))
        unread = [output.key for output in self._artifact.output_definitions if output.key not in self._outputs]
        if unread:
            return await self._failure(last, ExecutionStatus.TECHNICAL_FAIL, "OUTPUT_MISSING", CheckFailed(
                "every declared output read", f"not read: {', '.join(unread)}"))
        # The page the run finished on: the one screenshot a successful run gets, so the
        # report's gallery always shows how it ended, not just how it failed.
        await self._screenshot(f"success_step{last.step.sequence_index:02d}")
        return self._end(ExecutionStatus.SUCCESS)

    async def _final_check(self, kind: GlobalAssertionType, value: str) -> Optional[CheckFailed]:
        if kind == GlobalAssertionType.FINAL_URL_MATCH:
            wanted = fill_text(value, self._values.text)
            seen = _page_path(self._page)
            return None if seen == wanted else CheckFailed(f"final page {wanted}", f"final page {seen}")
        if kind == GlobalAssertionType.SUCCESS_BANNER_TEXT:
            pattern = text_pattern(value, self._values.text, self._values.numbers)
            body = await self._page.inner_text("body")
            if pattern is not None and pattern.search(body):
                return None
            return CheckFailed(f'"{value}" shown at the end', "not shown")
        return CheckFailed(f"a final check replay supports", f"a {kind.value} check")

    async def _outcome(self, record: _StepRecord, outcome: KnownOutcome) -> ExecutionResult:
        # An answer, not a failure: the step's checks didn't hold because the bank answered.
        screenshot = await self._screenshot(f"outcome_step{record.step.sequence_index:02d}_{outcome.code}")
        self._trace(record, StepStatus.FAILED, error_message=f"the page shows the known outcome {outcome.code}",
                   screenshot=screenshot)
        return self._end(ExecutionStatus.BUSINESS_OUTCOME,
                         outcome=BusinessOutcome(code=outcome.code, description=outcome.description,
                                                 screenshot_path=screenshot))

    async def _failure(self, record: _StepRecord, status: ExecutionStatus, code: str, failed: CheckFailed) -> ExecutionResult:
        step = record.step
        screenshot = await self._screenshot(f"failure_step{step.sequence_index:02d}")
        detail = FailureDetail(step_index=step.sequence_index, step_description=self._clean(step.description),
                               expected=self._clean(failed.expected)[:FAILURE_TEXT_MAX] or "(nothing)",
                               observed=self._clean(failed.observed)[:FAILURE_TEXT_MAX] or "(nothing)",
                               screenshot_path=screenshot)
        self._trace(record, StepStatus.FAILED, error_message=f"{code}: {detail.observed}", screenshot=screenshot)
        handoff = list(self._handoffs)
        if status == ExecutionStatus.HUMAN_ESCALATED and not handoff:
            # No person is available to this run: the need is recorded and the run stops here.
            handoff = [HandoffTelemetry(triggered_timestamp=datetime.now(timezone.utc), trigger_reason=code,
                                        step_index=step.sequence_index, resolution=HandoffResolution.ABORTED)]
        error = ErrorDetail(code=code, message=f"step {step.sequence_index}: expected {detail.expected}; "
                                               f"observed {detail.observed}")
        return self._end(status, error=error, failure=detail, handoff=handoff)

    def _end(self, status: ExecutionStatus, *, error: Optional[ErrorDetail] = None,
             failure: Optional[FailureDetail] = None, outcome: Optional[BusinessOutcome] = None,
             handoff: Optional[list[HandoffTelemetry]] = None) -> ExecutionResult:
        duration_ms = int((time.monotonic() - self._started) * 1000)
        code = error.code if error else (outcome.code if outcome else None)
        self._logger.execution_ended(status.value, code, error.message if error else None)
        retries = sum(max(0, trace.attempt_count - 1) for trace in self._traces)
        self._logger.summary_metrics(duration_ms, len(self._traces), retries, self._people)
        state = self._irreversible_state()
        inputs, outputs = readable_values(self._request.inputs, self._outputs,
                                          self._artifact.confirmation_checks if self._artifact else [],
                                          self._artifact.output_definitions if self._artifact else [])
        summary = self._clean(summarize(self._request.capability, "REPLAY", status, goal=self._goal, inputs=inputs,
                                        outputs=outputs, irreversible_step=state, outcome=outcome, failure=failure,
                                        error=error, person=self._person_end))
        result = ExecutionResult(
            run_id=self._logger.trace_id,
            capability=self._request.capability,
            artifact_version=self._artifact.metadata.version if self._artifact else None,
            mode="REPLAY",
            status=status,
            summary=summary,
            irreversible_step=state,
            start_time=self._start_time,
            end_time=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            integrity_verified=self._artifact is not None,
            step_traces=list(self._traces),
            handoff_events=handoff if handoff is not None else list(self._handoffs),
            evidence_paths=EvidencePaths(
                log_file=str(self._logger.log_path), screenshots_dir=str(self._screenshots),
                playwright_trace_zip=self._expected_trace_path()),
            terminal_outputs=dict(self._outputs) or None,
            outcome=outcome,
            failure=failure,
            error=error,
        )
        self._logger.write_result(result.to_json())
        return result

    # --- helpers ---

    @property
    def _page(self) -> Page:
        return self._session.page

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

    def _irreversible_state(self) -> Optional[str]:
        """Whether the irreversible step happened: not reached, completed (clicked and what it
        led to was checked), or unknown (clicked, then the run stopped before seeing what
        followed). None when the capability has no such step."""
        if self._artifact is None or not any(s.safety_tier == SafetyTier.IRREVERSIBLE for s in self._artifact.steps):
            return None
        if not self._irreversible_done:
            return "not_reached"
        return "completed" if self._irreversible_confirmed else "unknown"


    def _trace(self, record: _StepRecord, status: StepStatus, *, error_message: Optional[str] = None,
               screenshot: Optional[str] = None) -> None:
        step = record.step
        self._traces.append(StepExecutionTrace(
            step_id=step.step_id, sequence_index=step.sequence_index, description=self._clean(step.description),
            status=status, safety_tier=step.safety_tier,
            attempt_count=max(1, record.attempts), duration_ms=int((time.monotonic() - record.started) * 1000),
            locator_priority=record.priority, recovery_logs=list(record.recoveries),
            failure_screenshot_path=screenshot, error_message=error_message))
        self._logger.step_executed(step.step_id, status.value)

    def _allowed_here(self) -> bool:
        try:
            enforce_safety(self._page.url, ActionType.CLICK, allowed_paths=self._artifact.allowed_paths)
        except AllowlistViolation:
            return False
        return True

    async def _screenshot(self, name: str) -> Optional[str]:
        if self._session is None or self._session.page is None:
            return None
        path = self._screenshots / f"{self._logger.trace_id}_{name}.png"
        try:
            self._screenshots.mkdir(parents=True, exist_ok=True)
            await self._page.screenshot(path=str(path))
        except (PlaywrightError, OSError):
            return None
        return str(path)

    def _clean(self, text: str) -> str:
        # Text for a result or log: sensitive-looking patterns redacted and any secret scrubbed.
        return scrub_known_values(redact_text(text), self._secrets)

    def _time_left_ms(self) -> int:
        return int(settings.replay_total_timeout_ms - (time.monotonic() - self._started) * 1000 + self._paused_ms)


def _page_path(page: Page) -> str:
    return urlsplit(page.url).path or "/"


def _briefly(step: Step) -> Step:
    # Judging a page a person has already left settled: a short look is enough.
    return step.model_copy(update={"checkpoints": [
        checkpoint.model_copy(update={"timeout_ms": min(checkpoint.timeout_ms, 2_000)})
        for checkpoint in step.checkpoints]})


def _first_line(error: Exception) -> str:
    text = str(error)
    return text.splitlines()[0] if text else type(error).__name__
