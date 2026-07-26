"""``ServerController``: the one place a start, stop or restart can be requested from.

This class is the constraint that keeps Discord free of business logic. ``mcmanager start`` and
``/start`` call the same method here; there is nowhere for a rule to hide inside a slash-command
handler, because a slash-command handler has nothing to do but call this and render the result.
Discord becomes the *second* client of an interface the CLI already proved.

**Commands are direct calls, not bus messages.** The plan is explicit about the asymmetry: the bus
carries facts, direct calls carry commands. Routing commands through the bus too would produce a
system where "who stopped the server" is unanswerable, and where a dropped message is a silently
ignored ``/stop``. The honest reading of "no module directly depends on another" is that
``ContainerRuntime`` arrives here as a **constructor-injected interface**, which is what makes
every test in this file run against ``FakeRuntime`` with no Docker in sight.

**Every attempt publishes a** :class:`~mcmanager.core.events.CommandIssued` **first, including the
rejected ones.** That is the audit trail: a session summary can say "stopped by @kunal", and a stop
that was issued and never completed is still visible afterwards, which is exactly the case where
you most want to know somebody asked.

**Restart is stop-then-start, here, in this class.** ``ContainerRuntime`` deliberately has no
``restart()``: a runtime-level restart produces a ``die`` and a ``start`` that the lifecycle
reducer would have to guess about, and the guess would be wrong in the one case that matters (was
the stop ours?). Doing it here means the daemon owns the transition, arms the stop intent before
the stop and the start intent before the start, and therefore emits ``ServerStopping ->
ServerStopped -> ServerStarting -> ServerReady`` in that order with correct attribution on all
four.

Ordering inside a mutating call is fixed and load-bearing:

1. Check the state and the runtime. A rejected command still publishes ``CommandIssued``, with
   ``accepted=False`` and a human-readable ``rejection``.
2. Publish ``CommandIssued(accepted=True)``.
3. Feed the :class:`~mcmanager.services.lifecycle.IntentSignal` into lifecycle - *before* touching
   the runtime, so that if the container dies during the call the exit is already classified as
   intentional.
4. Call the runtime.

A single :class:`asyncio.Lock` serialises the whole sequence, so two ``/stop`` clicks a hundred
milliseconds apart cannot interleave a stop with a start.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import structlog

from mcmanager.containers.errors import RuntimeUnavailableError
from mcmanager.core.events import CommandIssued
from mcmanager.core.types import ControlAction, LifecycleState, Source
from mcmanager.services.lifecycle import Intent, IntentSignal

if TYPE_CHECKING:
    from mcmanager.clock import Clock
    from mcmanager.containers.base import ContainerRuntime
    from mcmanager.core.types import ContainerName, ServerId
    from mcmanager.services.lifecycle import EventSink, LifecycleService

__all__ = ["ControlOutcome", "ServerController"]

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.controller")

_ALREADY_RUNNING: frozenset[LifecycleState] = frozenset(
    {LifecycleState.STARTING, LifecycleState.READY, LifecycleState.DEGRADED}
)
"""States in which a ``start`` is redundant."""

_ALREADY_DOWN: frozenset[LifecycleState] = frozenset(
    {LifecycleState.STOPPED, LifecycleState.CRASHED, LifecycleState.ABSENT}
)
"""States in which a ``stop`` has nothing to do."""


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ControlOutcome:
    """What happened when somebody asked for a mutating action.

    Returned rather than raised, because every caller - the CLI, a slash command, the idle manager
    - wants to *render* the refusal, not catch it. The one thing that does propagate is a
    programming error; a rejected command and a failed runtime call are both outcomes.

    Attributes:
        action: What was asked for.
        actor: Who asked.
        accepted: False when the command was refused before the runtime was touched.
        rejection: Why, in a sentence a human can act on. ``None`` when accepted.
        dry_run: The command was recorded and deliberately not performed.
        error: The runtime failed *after* the command was accepted. The daemon keeps running; a
            :class:`~mcmanager.containers.errors.RuntimeUnavailableError` here has already driven
            lifecycle into ``BLIND``.
        state_before: The lifecycle state when the command arrived. Handy in the reply text.
    """

    action: ControlAction
    actor: str
    accepted: bool
    rejection: str | None = None
    dry_run: bool = False
    error: str | None = None
    state_before: LifecycleState = LifecycleState.UNKNOWN

    @property
    def ok(self) -> bool:
        """True when the command was accepted and the runtime did not fail."""
        return self.accepted and self.error is None

    def __str__(self) -> str:
        if not self.accepted:
            return f"{self.action.value} rejected: {self.rejection}"
        if self.error is not None:
            return f"{self.action.value} failed: {self.error}"
        if self.dry_run:
            return f"{self.action.value} recorded (dry run; nothing was done)"
        return f"{self.action.value} ok"


@final
class ServerController:
    """Start, stop and restart, with an audit trail and no business logic anywhere else.

    Construct one per managed server and hand the same instance to the CLI's control commands, the
    control server's ``POST /control/*`` routes and the Discord cog. There is deliberately no
    second path.
    """

    __slots__ = (
        "_clock",
        "_container",
        "_dry_run",
        "_lifecycle",
        "_lock",
        "_runtime",
        "_server_id",
        "_sink",
        "_stop_timeout",
    )

    def __init__(
        self,
        *,
        runtime: ContainerRuntime,
        lifecycle: LifecycleService,
        sink: EventSink,
        clock: Clock,
        server_id: ServerId,
        container: ContainerName,
        stop_timeout_seconds: int,
        dry_run: bool = False,
    ) -> None:
        """Build a controller.

        Args:
            runtime: The container platform, as an interface. Never ``docker`` directly.
            lifecycle: Where intents are declared before the runtime is touched.
            sink: Where ``CommandIssued`` goes. ``EventBus`` satisfies this structurally.
            clock: Used only to timestamp ``CommandIssued``.
            server_id: Stamped onto every event.
            container: The container **name**. Always by name: a cached id survives a
                ``compose down/up`` and then addresses a container that no longer exists.
            stop_timeout_seconds: Grace period handed to Docker before SIGKILL. Ninety on this
                deployment, because that is how long a world save can take - it used to be a bare
                ``-t 90`` in a bash script with no explanation attached. Required rather than
                defaulted so it can never be inherited by accident.
            dry_run: Record every command and perform none of them. The cutover soak runs like
                this for days.
        """
        self._runtime = runtime
        self._lifecycle = lifecycle
        self._sink = sink
        self._clock = clock
        self._server_id = server_id
        self._container = container
        self._stop_timeout = stop_timeout_seconds
        self._dry_run = dry_run
        self._lock = asyncio.Lock()

    @property
    def state(self) -> LifecycleState:
        """The lifecycle state this controller is about to act against."""
        return self._lifecycle.state

    @property
    def stop_timeout_seconds(self) -> int:
        """The grace period a stop will use. Surfaced so ``/status`` can show it."""
        return self._stop_timeout

    @property
    def dry_run(self) -> bool:
        """True while every command is recorded and none performed."""
        return self._dry_run

    # -- commands -------------------------------------------------------------------------------

    async def start(
        self,
        *,
        actor: str,
        via: Source = Source.INTERNAL,
        reason: str | None = None,
    ) -> ControlOutcome:
        """Start the server.

        Rejected when the server is already up, when the container does not exist (compose owns
        creation, not us), and while the runtime is unreachable.
        """
        async with self._lock:
            return await self._start_locked(actor=actor, via=via, reason=reason)

    async def stop(
        self,
        *,
        actor: str,
        via: Source = Source.INTERNAL,
        reason: str | None = None,
    ) -> ControlOutcome:
        """Stop the server, giving the JVM :attr:`stop_timeout_seconds` to save the world."""
        async with self._lock:
            return await self._stop_locked(actor=actor, via=via, reason=reason)

    async def restart(
        self,
        *,
        actor: str,
        via: Source = Source.INTERNAL,
        reason: str | None = None,
    ) -> ControlOutcome:
        """Stop then start, holding the lock across both halves.

        One ``CommandIssued(RESTART)`` is published, not three: the audit trail records what a
        human asked for. The two intents are still declared to lifecycle separately, because that
        is what makes the exit classify as intentional and the subsequent start attribute to the
        same person.

        A restart against an already-stopped server degrades to a plain start rather than being
        rejected - which is what everybody means by ``/restart`` when the thing is down.
        """
        async with self._lock:
            state = self._lifecycle.state
            rejection = self._reject_reason_for_restart(state)
            issued = self._issue(
                ControlAction.RESTART,
                actor=actor,
                via=via,
                accepted=rejection is None,
                rejection=rejection,
            )
            if rejection is not None:
                return issued
            if self._dry_run:
                _log.info("controller.dry_run", action="restart", actor=actor)
                return issued

            if state not in _ALREADY_DOWN:
                stop_failure = await self._perform_stop(
                    actor=actor,
                    reason=reason or f"restart requested by {actor}",
                    state_before=state,
                    action=ControlAction.RESTART,
                )
                if stop_failure is not None:
                    return stop_failure
            start_failure = await self._perform_start(
                actor=actor,
                reason=reason,
                state_before=state,
                action=ControlAction.RESTART,
            )
            return start_failure if start_failure is not None else issued

    # -- locked bodies --------------------------------------------------------------------------

    async def _start_locked(
        self,
        *,
        actor: str,
        via: Source,
        reason: str | None,
    ) -> ControlOutcome:
        state = self._lifecycle.state
        rejection = self._reject_reason_for_start(state)
        issued = self._issue(
            ControlAction.START,
            actor=actor,
            via=via,
            accepted=rejection is None,
            rejection=rejection,
        )
        if rejection is not None:
            return issued
        if self._dry_run:
            _log.info("controller.dry_run", action="start", actor=actor)
            return issued
        failure = await self._perform_start(
            actor=actor,
            reason=reason,
            state_before=state,
            action=ControlAction.START,
        )
        return failure if failure is not None else issued

    async def _stop_locked(
        self,
        *,
        actor: str,
        via: Source,
        reason: str | None,
    ) -> ControlOutcome:
        state = self._lifecycle.state
        rejection = self._reject_reason_for_stop(state)
        issued = self._issue(
            ControlAction.STOP,
            actor=actor,
            via=via,
            accepted=rejection is None,
            rejection=rejection,
        )
        if rejection is not None:
            return issued
        if self._dry_run:
            # The intent is still declared, marked dry, so the idle soak's logs show exactly what
            # would have happened - but the reducer does not enter STOPPING and nothing is stopped.
            self._lifecycle.on_intent(
                IntentSignal(
                    intent=Intent.STOP,
                    requested_by=actor,
                    reason=reason,
                    timeout_seconds=float(self._stop_timeout),
                    dry_run=True,
                )
            )
            _log.info("controller.dry_run", action="stop", actor=actor, reason=reason)
            return issued
        failure = await self._perform_stop(
            actor=actor,
            reason=reason,
            state_before=state,
            action=ControlAction.STOP,
        )
        return failure if failure is not None else issued

    # -- runtime calls --------------------------------------------------------------------------

    async def _perform_start(
        self,
        *,
        actor: str,
        reason: str | None,
        state_before: LifecycleState,
        action: ControlAction,
    ) -> ControlOutcome | None:
        """Declare the intent, call ``start``. Returns a failure outcome, or ``None`` on success."""
        self._lifecycle.on_intent(
            IntentSignal(intent=Intent.START, requested_by=actor, reason=reason)
        )
        try:
            await self._runtime.start(self._container)
        except RuntimeUnavailableError as exc:
            return self._runtime_failed(action, actor, exc, state_before, blind=True)
        except Exception as exc:
            return self._runtime_failed(action, actor, exc, state_before, blind=False)
        _log.info("controller.started", server_id=self._server_id, actor=actor)
        return None

    async def _perform_stop(
        self,
        *,
        actor: str,
        reason: str | None,
        state_before: LifecycleState,
        action: ControlAction,
    ) -> ControlOutcome | None:
        """Declare the intent, call ``stop``. Returns a failure outcome, or ``None`` on success."""
        # Before, not after: if the container dies during the call, the exit is already classified
        # as intentional and exit 137 reads as "the save ran long", not "something killed us".
        self._lifecycle.on_intent(
            IntentSignal(
                intent=Intent.STOP,
                requested_by=actor,
                reason=reason,
                timeout_seconds=float(self._stop_timeout),
            )
        )
        try:
            await self._runtime.stop(self._container, timeout=self._stop_timeout)
        except RuntimeUnavailableError as exc:
            return self._runtime_failed(action, actor, exc, state_before, blind=True)
        except Exception as exc:
            return self._runtime_failed(action, actor, exc, state_before, blind=False)
        _log.info(
            "controller.stopped",
            server_id=self._server_id,
            actor=actor,
            timeout_seconds=self._stop_timeout,
        )
        return None

    def _runtime_failed(
        self,
        action: ControlAction,
        actor: str,
        exc: BaseException,
        state_before: LifecycleState,
        *,
        blind: bool,
    ) -> ControlOutcome:
        if blind:
            # The platform, not the container. Lifecycle must go BLIND rather than infer anything
            # about the server, and the caller gets a message rather than a traceback.
            self._lifecycle.on_availability(available=False, error=str(exc))
        _log.error(
            "controller.runtime_failed",
            server_id=self._server_id,
            action=action.value,
            actor=actor,
            error=str(exc),
            runtime_unavailable=blind,
        )
        return ControlOutcome(
            action=action,
            actor=actor,
            accepted=True,
            error=str(exc),
            state_before=state_before,
        )

    # -- policy ---------------------------------------------------------------------------------

    def _reject_reason_for_start(self, state: LifecycleState) -> str | None:
        if state is LifecycleState.BLIND:
            return "the container runtime is unreachable"
        if state in _ALREADY_RUNNING:
            return f"the server is already {state.value}"
        if state is LifecycleState.STOPPING:
            return "a stop is still in flight; wait for it to finish"
        if state is LifecycleState.ABSENT:
            return (
                f"there is no container named {self._container!r}. "
                "mcmanager manages a container's lifecycle, not its existence - "
                "run `docker compose up -d` for the minecraft project first"
            )
        return None

    def _reject_reason_for_stop(self, state: LifecycleState) -> str | None:
        if state is LifecycleState.BLIND:
            return "the container runtime is unreachable"
        if state in _ALREADY_DOWN:
            return f"the server is already {state.value}"
        if state is LifecycleState.STOPPING:
            return "a stop is already in flight"
        if state is LifecycleState.UNKNOWN:
            return "the server state is not known yet; try again in a moment"
        return None

    def _reject_reason_for_restart(self, state: LifecycleState) -> str | None:
        if state is LifecycleState.BLIND:
            return "the container runtime is unreachable"
        if state is LifecycleState.STOPPING:
            return "a stop is already in flight"
        if state is LifecycleState.ABSENT:
            return f"there is no container named {self._container!r}"
        if state is LifecycleState.UNKNOWN:
            return "the server state is not known yet; try again in a moment"
        return None

    # -- audit ----------------------------------------------------------------------------------

    def _issue(
        self,
        action: ControlAction,
        *,
        actor: str,
        via: Source,
        accepted: bool,
        rejection: str | None,
    ) -> ControlOutcome:
        """Publish ``CommandIssued`` before anything is attempted, and build the outcome."""
        state = self._lifecycle.state
        self._sink.publish(
            CommandIssued(
                ts=self._clock.now(),
                server_id=self._server_id,
                source=via,
                raw=f"{action.value} by {actor}",
                action=action,
                actor=actor,
                via=via,
                dry_run=self._dry_run,
                accepted=accepted,
                rejection=rejection,
            )
        )
        if accepted:
            _log.info(
                "controller.command",
                server_id=self._server_id,
                action=action.value,
                actor=actor,
                via=via.value,
                state=state.value,
                dry_run=self._dry_run,
            )
        else:
            _log.warning(
                "controller.command_rejected",
                server_id=self._server_id,
                action=action.value,
                actor=actor,
                via=via.value,
                state=state.value,
                rejection=rejection,
            )
        return ControlOutcome(
            action=action,
            actor=actor,
            accepted=accepted,
            rejection=rejection,
            dry_run=self._dry_run,
            state_before=state,
        )
