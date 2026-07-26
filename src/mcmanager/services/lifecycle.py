"""The lifecycle state machine.

States: ``UNKNOWN, ABSENT, STOPPED, CRASHED, STARTING, READY, DEGRADED, STOPPING, BLIND``
(:class:`~mcmanager.core.types.LifecycleState`).

Implemented as a **pure reducer** - :meth:`LifecycleReducer.handle` takes one signal and returns
the events that fact implies - with a thin wrapper, :class:`LifecycleService`, that feeds signals
in and publishes what comes back. That split is the whole reason the tests for this module are a
table over signals with zero I/O, zero Docker and zero real time: the reducer touches nothing but
its own fields and an injected :class:`~mcmanager.clock.Clock`, which it only ever *reads*.

Six signal kinds, and no seventh:

===========================  =========================================================
:class:`RuntimeEvent`        A Docker daemon event, straight off ``DockerManager``.
:class:`SnapshotSignal`      One container inspect. The reconcile safety net.
:class:`LineSignal`          One event the log parser produced, e.g. ``Done (32.521s)!``.
:class:`ProbeSignal`         One Server List Ping result.
:class:`IntentSignal`        We are about to start or stop the server ourselves.
:class:`AvailabilitySignal`  The container platform went away, or came back.
===========================  =========================================================

**Timing is read from the container, never hardcoded.** The snapshot carries the healthcheck's
``StartPeriod``/``Interval``/``Retries``, so::

    guard = start_period + interval * (retries + 1)  # 120 + 30*3 = 210s on this container
    start_deadline = guard + start_deadline_extra  # 270s here

Any ``unhealthy`` inside the guard window is suppressed - during ``start_period`` the healthcheck
is *expected* to fail, and reporting that as a degraded server would make the daemon cry wolf on
every single start. Past the deadline, ``STARTING`` becomes ``DEGRADED``. If the container declares
no healthcheck at all there is no guard, and this module then has **no opinion**: it will not
invent a deadline, because declaring a slow-but-healthy server degraded on a made-up number is
worse than never declaring it degraded. Pass ``fallback_guard_seconds`` if you want one anyway.

**Readiness is first-of-three** - the ``Done (Ns)!`` line, ``health_status: healthy``, or a
successful SLP probe - and :attr:`~mcmanager.core.events.ServerReady.detected_by` records which
fired. All three assert the identical fact (``mc-health`` *is* an SLP query), but the healthcheck
cannot say healthy before ~120-150s while the sampled logs say ``Done (32.521s)``. Telling Discord
the server is up ninety seconds late is bad UX for zero correctness gain. This is a deliberate,
flagged deviation from the original spec; reverting to strict health-only is
``ready_signals=(ReadySignal.HEALTHCHECK,)``, which is a config value. ``health_status: healthy``
remains authoritative for the *negative* case: a later ``unhealthy`` past the guard still drives
``DEGRADED``, whatever the log said.

**A start that overruns its deadline still gets to report ready.** ``STARTING -> DEGRADED`` on the
deadline is a statement about how long the start is taking, not a decision that it failed, so the
first readiness signal to arrive afterwards emits ``ServerReady`` exactly as it would have from
``STARTING``. A freshly generated world, a chunk pre-generation pass or a post-update migration
routinely crosses 270 seconds and then logs ``Done (298.4s)!``; swallowing that would leave
Discord, ``mcmanager events`` and the session manager believing the server never came up, and
``/status`` rendering ``ready`` with a null ``ready_at`` forever. Recovery from a run that *has*
already announced itself stays silent, because that is a different fact with no event for it.

**Crash versus clean stop is one decision function**, :func:`classify_exit`. Verified on this
container: a graceful stop exits **0**, because ``mc-server-runner`` traps SIGTERM and writes
``stop``. 137 *with* a stop intent means the JVM did not finish saving inside the grace period - a
warning, not a crash. 137 *without* one is the host OOM killer.

Three properties this module exists to guarantee, each of which is a live bug in the scripts it
replaces:

1. **A boot reconcile emits nothing.** Coming up against an already-running server *discovers*
   state; it does not witness a transition into it. ``UNKNOWN -> anything`` via a snapshot is
   silent, so redeploying the daemon never announces a server that was up the whole time.
2. **BLIND retains last-known state.** A socket hiccup makes ``inspect`` fail, which is
   indistinguishable from "the server stopped" if you squint - and squinting there produces a
   phantom ``ServerStopped`` every time the daemon reconnects. Instead the reducer parks in
   ``BLIND``, remembers what it last knew, and on restore re-reconciles and emits only the genuine
   delta.
3. **A failed probe is never a state change.** Unreachable means unknown: not stopped, and
   downstream not "zero players" either. That single rule is what keeps the idle manager from
   stopping a populated server.

``ServerCrashed.tail`` comes from the log pipeline's ring buffer via an injected callable, captured
*before* the stream closed. A post-hoc ``logs_tail()`` races the container going away; this does
not, and it is free.

**On events this module deliberately does not emit.** There is no ``ServerDegraded`` and no
``ServerRecovered`` in the vocabulary, so ``READY -> DEGRADED -> READY`` is a state change with no
event attached. It is visible through :attr:`LifecycleReducer.state`, which is what ``/status`` and
``mcmanager status`` render, and it is logged at WARNING. Adding an event for it is a change to
``core/events.py``, not to this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, assert_never, final

import structlog

from mcmanager.containers.dto import ContainerSnapshot, HealthState, RuntimeEvent
from mcmanager.core.events import (
    Event,
    RuntimeRestored,
    RuntimeStatusEvent,
    RuntimeUnavailable,
    ServerCrashed,
    ServerReady,
    ServerStarting,
    ServerStopped,
    ServerStopping,
)
from mcmanager.core.types import LifecycleState, ReadySignal, Source
from mcmanager.games.base import ProbeResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import datetime

    from mcmanager.clock import Clock, TimerHandle
    from mcmanager.core.types import ServerId

__all__ = [
    "DEFAULT_READY_SIGNALS",
    "DEFAULT_START_DEADLINE_EXTRA",
    "DEFAULT_TAIL_LINES",
    "SIGNAL_KINDS",
    "AvailabilitySignal",
    "EventSink",
    "ExitVerdict",
    "Intent",
    "IntentSignal",
    "LifecycleReducer",
    "LifecycleService",
    "LineSignal",
    "ProbeSignal",
    "ReadinessOracle",
    "Signal",
    "SnapshotSignal",
    "classify_exit",
]

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.lifecycle")

DEFAULT_READY_SIGNALS: tuple[ReadySignal, ...] = (
    ReadySignal.LOG_DONE,
    ReadySignal.HEALTHCHECK,
    ReadySignal.PROBE,
)
"""First-of-three. Narrow to ``(ReadySignal.HEALTHCHECK,)`` for the spec-literal behaviour."""

DEFAULT_START_DEADLINE_EXTRA = 60.0
"""Seconds past the container's own guard window before ``STARTING`` becomes ``DEGRADED``."""

DEFAULT_TAIL_LINES = 40
"""How many buffered console lines ``ServerCrashed`` carries."""

_CLEAN_EXIT_CODES: frozenset[int] = frozenset({0, 143})
"""0 because ``mc-server-runner`` traps SIGTERM and exits cleanly; 143 is 128+SIGTERM."""

_SIGKILL_EXIT_CODE = 137
"""128+SIGKILL. Docker sent it because the grace period ran out, or the host OOM killer did."""

_GRACEFUL_KILL_SIGNALS: frozenset[str] = frozenset({"15", "SIGTERM", "TERM"})
"""``kill`` event signals meaning somebody asked politely, i.e. a ``docker stop``."""

_DOWN_STATES: frozenset[LifecycleState] = frozenset(
    {LifecycleState.ABSENT, LifecycleState.STOPPED, LifecycleState.CRASHED}
)
"""States in which the game server is definitely not running."""

_UP_STATES: frozenset[LifecycleState] = frozenset(
    {
        LifecycleState.STARTING,
        LifecycleState.READY,
        LifecycleState.DEGRADED,
        LifecycleState.STOPPING,
    }
)
"""States in which the container is running, whatever the server is doing inside it."""

_HEALTH_RELEVANT_STATES: frozenset[LifecycleState] = frozenset(
    {LifecycleState.STARTING, LifecycleState.READY, LifecycleState.DEGRADED}
)
"""The only states in which a ``health_status`` event means anything.

Outside them the container is stopped, absent, being stopped, or we are blind - and Docker is
perfectly happy to report a health status for a container that exited an hour ago (verified on this
host: ``State.Health.Status == "unhealthy"`` on an *exited* container whose last five probes exited
0). So anything arriving outside this set is dropped with a DEBUG line rather than acted on.
"""


# --------------------------------------------------------------------------------- signals


class Intent(StrEnum):
    """Something *we* are about to do to the container.

    Only two members, deliberately. ``restart`` is not a lifecycle intent: the controller
    decomposes it into a stop followed by a start precisely so the daemon owns the transition and
    emits the right events in the right order, which is also why
    :class:`~mcmanager.containers.base.ContainerRuntime` has no ``restart()``.
    """

    START = "start"
    STOP = "stop"


@dataclass(frozen=True, slots=True, kw_only=True)
class SnapshotSignal:
    """One container inspect.

    Arrives both from ``DockerManager``'s reconnect resolution and from its 60-second reconcile
    poll, which is what makes this the safety net for a Docker event we never saw.
    """

    snapshot: ContainerSnapshot


@dataclass(frozen=True, slots=True, kw_only=True)
class LineSignal:
    """One event the log parser produced from one line.

    Carries the parsed :class:`~mcmanager.core.events.Event` rather than the raw text, so this
    module stays game-agnostic: what a readiness line *looks like* is the adapter's problem, and
    :class:`ReadinessOracle` is the two-question interface used to ask about it.
    """

    event: Event


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbeSignal:
    """One out-of-band status query.

    Consuming a :class:`~mcmanager.games.base.ProbeResult` rather than calling ``mcstatus`` is what
    keeps ``games/`` swappable: this module never learns what a Server List Ping is.
    """

    result: ProbeResult


@dataclass(frozen=True, slots=True, kw_only=True)
class IntentSignal:
    """We are about to start or stop the server ourselves.

    Fed by :class:`~mcmanager.services.controller.ServerController` immediately before it calls the
    runtime, for two reasons that both matter:

    - it is what makes ``exit 137`` mean "the JVM did not finish saving" instead of "the host OOM
      killer struck", via :func:`classify_exit`;
    - it is what attributes the resulting ``ServerStopping``/``ServerStarting`` to a human.

    Attributes:
        intent: Start or stop.
        requested_by: Who asked. Ends up on the emitted event and in the session summary.
        reason: Free text, e.g. ``"idle timeout"``.
        timeout_seconds: The grace period the JVM is being given, for a stop.
        dry_run: The intent was recorded but nothing will actually happen. The idle soak runs like
            this for days, so the machine must **not** enter ``STOPPING`` and must **not** arm a
            stop intent: doing either would classify a still-running server's disconnects as
            shutdown casualties and would make a later unrelated ``exit 137`` look clean.
    """

    intent: Intent
    requested_by: str | None = None
    reason: str | None = None
    timeout_seconds: float | None = None
    dry_run: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class AvailabilitySignal:
    """The container platform became unreachable, or reachable again.

    Edge-triggered: one per outage, not one per failed poll. The reducer enforces that anyway, so a
    caller that gets it wrong still produces no duplicate events.
    """

    available: bool
    error: str | None = None
    endpoint: str | None = None


type Signal = (
    RuntimeEvent | SnapshotSignal | LineSignal | ProbeSignal | IntentSignal | AvailabilitySignal
)
"""Everything :meth:`LifecycleReducer.handle` accepts.

:meth:`LifecycleReducer.handle` ends its ``match`` with :func:`typing.assert_never`, so adding a
seventh member here without handling it is a **type error**, not a silently ignored signal.
"""

SIGNAL_KINDS: tuple[type, ...] = (
    RuntimeEvent,
    SnapshotSignal,
    LineSignal,
    ProbeSignal,
    IntentSignal,
    AvailabilitySignal,
)
"""The six signal classes, in the order this module documents them. Tests iterate this."""


# ------------------------------------------------------------------------------ exit verdict


@dataclass(frozen=True, slots=True, kw_only=True)
class ExitVerdict:
    """What :func:`classify_exit` decided about a container that stopped.

    Attributes:
        clean: The world had time to save, as far as we can honestly tell.
        forced: Docker had to SIGKILL. With a stop intent that is a warning worth surfacing - the
            grace period was too short; without one it is the host OOM killer.
    """

    clean: bool
    forced: bool

    @property
    def crashed(self) -> bool:
        """True when this exit should be reported as a crash rather than a stop."""
        return not self.clean


def classify_exit(
    exit_code: int | None,
    *,
    oom_killed: bool,
    had_stop_intent: bool,
) -> ExitVerdict:
    """Decide whether a container that stopped did so on purpose. The only place this is decided.

    The table, in order, each row backed by an observation rather than a guess:

    1. **OOM is never clean.** ``State.OOMKilled`` is decisive on its own and outranks the exit
       code, which for an OOM kill is also 137 and would otherwise read as "our own SIGKILL".
    2. **137 is clean if and only if we asked for the stop.** 137 is 128+SIGKILL: Docker's grace
       period ran out. With a stop intent that means the JVM did not finish saving inside
       ``stop_timeout_seconds`` - a warning, and ``forced=True`` is how it gets surfaced. Without
       one, nobody asked, so something else killed the process.
    3. **0 and 143 are clean.** Verified on this container: ``mc-server-runner`` traps SIGTERM,
       writes ``stop`` to the console and exits **0**. 143 is 128+SIGTERM, which is what a process
       that does *not* trap it exits with. Both are clean whether or not we asked, because an
       operator running ``docker stop`` by hand, or a player typing ``/stop``, is still a graceful
       shutdown - and reporting those as crashes is how people learn to ignore crash alerts.
    4. **Everything else is a crash.** Including ``exit_code is None``: an exit we have no evidence
       about is not an exit we get to call clean.

    Args:
        exit_code: ``State.ExitCode``, or the ``die`` event's ``exitCode`` attribute parsed to an
            int. Docker reports that attribute as a **string**, which is why nothing here compares
            it raw.
        oom_killed: ``State.OOMKilled``, or an ``oom`` Docker event seen for this container.
        had_stop_intent: Did we, or an operator via ``docker stop``, ask for this?

    Returns:
        The verdict. :attr:`ExitVerdict.clean` selects ``ServerStopped`` over ``ServerCrashed``.
    """
    if oom_killed:
        return ExitVerdict(clean=False, forced=False)
    if exit_code == _SIGKILL_EXIT_CODE:
        return ExitVerdict(clean=had_stop_intent, forced=True)
    if exit_code is not None and exit_code in _CLEAN_EXIT_CODES:
        return ExitVerdict(clean=True, forced=False)
    return ExitVerdict(clean=False, forced=False)


# --------------------------------------------------------------------------------- protocols


class ReadinessOracle(Protocol):
    """The two questions this module asks the game adapter, and nothing else.

    A narrowed structural view of :class:`~mcmanager.games.base.GameAdapter` - which satisfies it -
    for the same reason ``containers/manager.py`` narrows the bus to ``publish``: depending on the
    whole adapter would overstate the coupling and would make every test here construct one.
    """

    def ready_signal(self, event: Event) -> ReadySignal | None:
        """Does this event prove the server is up, and by which of the three signals?"""
        ...

    def stop_signal(self, event: Event) -> bool:
        """Does this event mean a shutdown has begun?"""
        ...


class EventSink(Protocol):
    """The half of :class:`~mcmanager.core.bus.EventBus` :class:`LifecycleService` needs."""

    def publish(self, event: Event) -> None: ...


@final
class _CoreEventOracle:
    """The game-agnostic default oracle: reads the core event types and nothing else.

    Identical in behaviour to ``MinecraftAdapter``'s implementation, because there is nothing
    Minecraft-shaped about "a ``ServerReady`` means the server is ready". Injecting a real adapter
    is still the production wiring; this exists so a reducer can be built with two arguments.
    """

    __slots__ = ()

    def ready_signal(self, event: Event) -> ReadySignal | None:
        return event.detected_by if isinstance(event, ServerReady) else None

    def stop_signal(self, event: Event) -> bool:
        return isinstance(event, ServerStopping)


# ----------------------------------------------------------------------------------- reducer


@final
class LifecycleReducer:
    """The state machine. Pure: reads its own fields and the clock, returns events, does no I/O.

    It never publishes, never awaits and never schedules. :class:`LifecycleService` does all three
    on its behalf, which is why every transition in ``tests/services/test_lifecycle.py`` is one
    synchronous call and one list comparison.
    """

    __slots__ = (
        "_clock",
        "_container_id",
        "_extra_deadline",
        "_fallback_guard",
        "_last_known",
        "_oom_seen",
        "_oracle",
        "_ready_at",
        "_ready_detected_by",
        "_ready_expected",
        "_ready_signals",
        "_report_crash",
        "_server_id",
        "_snapshot",
        "_start_requested_by",
        "_started_at",
        "_starting_since",
        "_state",
        "_stop_intent",
        "_stop_reason",
        "_stop_requested_by",
        "_stop_timeout",
        "_tail_lines",
        "_tail_provider",
        "_transition_at",
        "_unavailable_since",
        "_version",
    )

    def __init__(
        self,
        *,
        server_id: ServerId,
        clock: Clock,
        oracle: ReadinessOracle | None = None,
        ready_signals: Sequence[ReadySignal] = DEFAULT_READY_SIGNALS,
        start_deadline_extra_seconds: float = DEFAULT_START_DEADLINE_EXTRA,
        fallback_guard_seconds: float | None = None,
        tail_provider: Callable[[], Sequence[str]] | None = None,
        tail_lines: int = DEFAULT_TAIL_LINES,
        report_unexpected_exit_as_crash: bool = True,
    ) -> None:
        """Build a reducer.

        Args:
            server_id: Stamped onto every event this reducer emits.
            clock: Read for ``now()`` and ``monotonic()``. Never slept on.
            oracle: How to recognise a readiness or shutdown line. Defaults to the core-event
                oracle, which is what a real ``GameAdapter`` does anyway.
            ready_signals: Which of the three signals may promote ``STARTING -> READY``, first one
                wins. Mirrors ``server.lifecycle.ready_signals``.
            start_deadline_extra_seconds: Added to the container's own guard window to get the
                deadline past which ``STARTING`` becomes ``DEGRADED``. Mirrors
                ``server.lifecycle.start_deadline_extra_seconds``.
            fallback_guard_seconds: Guard window to assume when the container declares no
                healthcheck. ``None`` - the default - means no deadline at all in that case,
                because a made-up number is a made-up alert.
            tail_provider: Returns the log pipeline's ring buffer. Called only when building a
                ``ServerCrashed``, and only ever synchronously.
            tail_lines: How many of those lines to carry.
            report_unexpected_exit_as_crash: When false, an unexplained exit becomes
                ``ServerStopped(clean=False)`` and state ``STOPPED`` instead of ``ServerCrashed``
                and ``CRASHED``. Mirrors ``server.lifecycle.treat_unexpected_exit_as_crash``.
        """
        self._server_id = server_id
        self._clock = clock
        self._oracle: ReadinessOracle = oracle if oracle is not None else _CoreEventOracle()
        self._ready_signals = frozenset(ready_signals)
        self._extra_deadline = start_deadline_extra_seconds
        self._fallback_guard = fallback_guard_seconds
        self._tail_provider = tail_provider
        self._tail_lines = tail_lines
        self._report_crash = report_unexpected_exit_as_crash

        self._state = LifecycleState.UNKNOWN
        self._last_known = LifecycleState.UNKNOWN
        self._transition_at: datetime | None = None

        self._snapshot: ContainerSnapshot | None = None
        self._container_id: str | None = None
        self._started_at: datetime | None = None
        self._starting_since: float | None = None
        self._unavailable_since: float | None = None

        self._version: str | None = None
        self._ready_at: datetime | None = None
        self._ready_detected_by: ReadySignal | None = None
        self._ready_expected = False

        self._start_requested_by: str | None = None
        self._stop_intent = False
        self._stop_reason: str | None = None
        self._stop_requested_by: str | None = None
        self._stop_timeout: float | None = None
        self._oom_seen = False

    # -- introspection --------------------------------------------------------------------------

    @property
    def state(self) -> LifecycleState:
        """Where the server is now. What ``/status`` and ``mcmanager status`` render."""
        return self._state

    @property
    def last_known_state(self) -> LifecycleState:
        """What the state was before we went ``BLIND``. Equal to :attr:`state` when not blind."""
        return self._last_known if self._state is LifecycleState.BLIND else self._state

    @property
    def snapshot(self) -> ContainerSnapshot | None:
        """The most recent inspect, or ``None`` before the first one."""
        return self._snapshot

    @property
    def container_id(self) -> str | None:
        """The current container id. Changes on a ``compose down/up``; half of session identity."""
        return self._container_id

    @property
    def started_at(self) -> datetime | None:
        """``State.StartedAt`` for the running container. The other half of session identity."""
        return self._started_at

    @property
    def ready_at(self) -> datetime | None:
        """When the server first answered this run."""
        return self._ready_at

    @property
    def ready_detected_by(self) -> ReadySignal | None:
        """Which of the three signals promoted this run to ``READY``."""
        return self._ready_detected_by

    @property
    def version(self) -> str | None:
        """Server version, if a ``Starting ... version X`` line has been seen this run."""
        return self._version

    @property
    def has_stop_intent(self) -> bool:
        """True once a stop has been asked for and before the next start.

        Drives :func:`classify_exit`, and is what marks a ``lost connection: Server closed`` during
        the window as a shutdown casualty rather than a voluntary leave.
        """
        return self._stop_intent

    @property
    def transition_at(self) -> datetime | None:
        """When the current state was entered."""
        return self._transition_at

    @property
    def guard_seconds(self) -> float | None:
        """``StartPeriod + Interval * (Retries + 1)`` from the container, or the fallback.

        **210** on the homelab container, and read from it rather than written here. ``None`` means
        the container declares no healthcheck and no fallback was configured, which this module
        treats as "no opinion", never as zero.
        """
        snapshot = self._snapshot
        guard = snapshot.guard_window if snapshot is not None else None
        return guard.total_seconds() if guard is not None else self._fallback_guard

    @property
    def start_deadline_seconds(self) -> float | None:
        """Seconds a start may take before ``DEGRADED``, or ``None`` for "no opinion".

        :attr:`guard_seconds` plus ``start_deadline_extra_seconds``: 210 + 60 = **270** here.
        """
        guard = self.guard_seconds
        return None if guard is None else guard + self._extra_deadline

    @property
    def elapsed_since_start(self) -> float | None:
        """Seconds since the container started, preferring its own ``StartedAt``.

        Falls back to the monotonic instant we first believed it was starting, which is what covers
        the window between a ``start`` Docker event and the inspect that follows it.
        """
        if self._started_at is not None:
            return max((self._clock.now() - self._started_at).total_seconds(), 0.0)
        if self._starting_since is not None:
            return max(self._clock.monotonic() - self._starting_since, 0.0)
        return None

    def describe(self) -> dict[str, object]:
        """A flat dict of the current state. For log context and assertion messages."""
        return {
            "state": self._state.value,
            "last_known": self._last_known.value,
            "container_id": self._container_id,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "ready_detected_by": (
                self._ready_detected_by.value if self._ready_detected_by else None
            ),
            "version": self._version,
            "stop_intent": self._stop_intent,
            "guard_seconds": self.guard_seconds,
            "start_deadline_seconds": self.start_deadline_seconds,
        }

    # -- the entry point ------------------------------------------------------------------------

    def handle(self, signal: Signal) -> list[Event]:
        """Reduce one signal. Returns the events it implies, in the order they must be published.

        Never raises for an unexpected signal in an unexpected state: an illegal combination is a
        no-op plus a DEBUG line. A state machine that throws at 3am because Docker sent a
        ``health_status`` for a container that stopped an hour ago is not a state machine, it is a
        crash.
        """
        events = self.check_deadlines()
        match signal:
            case RuntimeEvent():
                events.extend(self._on_runtime_event(signal))
            case SnapshotSignal():
                events.extend(self._on_snapshot(signal))
            case LineSignal():
                events.extend(self._on_line(signal))
            case ProbeSignal():
                events.extend(self._on_probe(signal))
            case IntentSignal():
                events.extend(self._on_intent(signal))
            case AvailabilitySignal():
                events.extend(self._on_availability(signal))
            case _ as unreachable:  # pragma: no cover - pyright proves this unreachable
                assert_never(unreachable)
        return events

    def check_deadlines(self) -> list[Event]:
        """Apply any time-based transition that has come due. Safe to call at any moment.

        Only one exists: a start that has taken longer than the container's own guard window plus
        ``start_deadline_extra``. It is checked at the top of every :meth:`handle`, so the
        60-second reconcile alone is enough to catch it; :class:`LifecycleService` additionally
        arms a timer so the transition is prompt rather than merely eventual.

        Returns an empty list today, because the event vocabulary has no ``ServerDegraded``. The
        signature is a list so adding one later is not a call-site change.
        """
        if self._state is not LifecycleState.STARTING:
            return []
        elapsed = self.elapsed_since_start
        deadline = self.start_deadline_seconds
        if elapsed is None or deadline is None:
            return []
        if elapsed < deadline:
            return []
        _log.warning(
            "lifecycle.start_deadline_exceeded",
            server_id=self._server_id,
            elapsed_seconds=round(elapsed, 1),
            deadline_seconds=deadline,
            guard_seconds=self.guard_seconds,
            hint="the server never answered; the container is still running",
        )
        self._enter(LifecycleState.DEGRADED, reason="start deadline exceeded")
        return []

    # -- availability ---------------------------------------------------------------------------

    def _on_availability(self, signal: AvailabilitySignal) -> list[Event]:
        if not signal.available:
            if self._state is LifecycleState.BLIND:
                _log.debug("lifecycle.already_blind", server_id=self._server_id)
                return []
            return self._go_blind(
                error=signal.error or "runtime unavailable",
                endpoint=signal.endpoint,
                ts=self._now(),
            )
        if self._state is not LifecycleState.BLIND:
            _log.debug("lifecycle.already_available", state=self._state.value)
            return []
        return self._restore(ts=self._now())

    def _go_blind(self, *, error: str, endpoint: str | None, ts: datetime) -> list[Event]:
        self._last_known = self._state
        self._unavailable_since = self._clock.monotonic()
        self._state = LifecycleState.BLIND
        self._transition_at = ts
        _log.error(
            "lifecycle.blind",
            server_id=self._server_id,
            last_known=self._last_known.value,
            error=error,
            endpoint=endpoint,
            hint="last-known state retained; no ServerStopped will be invented",
        )
        return [
            RuntimeUnavailable(
                ts=ts,
                server_id=self._server_id,
                source=Source.INTERNAL,
                raw=f"runtime unavailable: {error}",
                error=error,
                endpoint=endpoint,
            )
        ]

    def _restore(self, *, ts: datetime) -> list[Event]:
        since = self._unavailable_since
        downtime = None if since is None else self._clock.monotonic() - since
        self._unavailable_since = None
        self._state = self._last_known
        self._transition_at = ts
        _log.info(
            "lifecycle.restored",
            server_id=self._server_id,
            state=self._state.value,
            downtime_seconds=None if downtime is None else round(downtime, 3),
        )
        return [
            RuntimeRestored(
                ts=ts,
                server_id=self._server_id,
                source=Source.INTERNAL,
                raw="runtime restored",
                downtime_seconds=downtime,
            )
        ]

    def _dropped_while_blind(self, what: str) -> list[Event]:
        _log.debug(
            "lifecycle.signal_dropped_while_blind",
            signal=what,
            last_known=self._last_known.value,
        )
        return []

    # -- snapshots ------------------------------------------------------------------------------

    def _on_snapshot(self, signal: SnapshotSignal) -> list[Event]:
        snapshot = signal.snapshot
        ts = snapshot.observed_at
        events: list[Event] = []

        # A successful inspect is itself proof the platform answered. Restoring here as well as on
        # AvailabilitySignal makes the two orderings equivalent, and keeps RuntimeRestored at
        # exactly one per outage either way.
        if self._state is LifecycleState.BLIND:
            events.extend(self._restore(ts=ts))

        previous = self._state
        recreated = (
            self._container_id is not None
            and snapshot.id is not None
            and snapshot.id != self._container_id
        )
        self._snapshot = snapshot
        observed = self._observe(snapshot)

        if previous is LifecycleState.UNKNOWN:
            # Boot reconcile. We discovered this state; we did not witness the transition into it,
            # and announcing it would mean a "server started!" every time the daemon is redeployed
            # against a server that never went anywhere.
            self._adopt(snapshot, observed)
            _log.info(
                "lifecycle.reconciled_at_boot",
                server_id=self._server_id,
                state=observed.value,
                container_id=snapshot.id,
                emitted="nothing (discovered, not witnessed)",
            )
            return events

        if recreated and previous in _UP_STATES and observed in _UP_STATES:
            # compose down/up under us: the server we were watching is gone and a different one is
            # running. Two facts, two events - the session manager needs the close as much as the
            # open, and a cached container id is exactly how that gets missed.
            events.extend(
                self._down_events(
                    exit_code=None,
                    oom_killed=False,
                    ts=ts,
                    source=Source.RUNTIME,
                    raw="reconcile: container was replaced (new id)",
                    force_stopped=True,
                )
            )
            self._begin_run(started_at=snapshot.started_at)
            self._adopt(snapshot, LifecycleState.STARTING)
            events.append(
                self._starting_event(
                    ts=ts,
                    source=Source.RUNTIME,
                    raw="reconcile: container was replaced (new id)",
                )
            )
            return events

        events.extend(self._delta(previous, observed, snapshot=snapshot, ts=ts))
        self._adopt(snapshot, observed)
        return events

    def _observe(self, snapshot: ContainerSnapshot) -> LifecycleState:
        """What this snapshot implies, refined by what we already believe."""
        if snapshot.absent:
            return LifecycleState.ABSENT
        if snapshot.running:
            return self._refine_running(snapshot)
        verdict = classify_exit(
            snapshot.exit_code,
            oom_killed=snapshot.oom_killed or self._oom_seen,
            had_stop_intent=self._stop_intent,
        )
        if verdict.crashed and self._report_crash:
            return LifecycleState.CRASHED
        return LifecycleState.STOPPED

    def _refine_running(self, snapshot: ContainerSnapshot) -> LifecycleState:
        current = self._state
        if current is LifecycleState.STOPPING:
            # A stop is in flight and the container has not gone away yet. Not a fresh start.
            return LifecycleState.STOPPING
        health = snapshot.health
        if health is HealthState.HEALTHY:
            if current is LifecycleState.READY:
                return LifecycleState.READY
            if ReadySignal.HEALTHCHECK in self._ready_signals:
                return LifecycleState.READY
            return current if current in _UP_STATES else LifecycleState.STARTING
        if health is HealthState.UNHEALTHY and not self._within_guard_window(snapshot):
            return LifecycleState.DEGRADED
        if current in _UP_STATES:
            return current
        return LifecycleState.STARTING

    def _delta(
        self,
        previous: LifecycleState,
        observed: LifecycleState,
        *,
        snapshot: ContainerSnapshot,
        ts: datetime,
    ) -> list[Event]:
        """The genuine difference between two *known* states, as events. Never a discovery."""
        if previous == observed:
            return []
        if previous in _DOWN_STATES and observed in _UP_STATES:
            self._begin_run(started_at=snapshot.started_at)
            _log.info(
                "lifecycle.start_detected_by_reconcile",
                server_id=self._server_id,
                container_id=snapshot.id,
            )
            self._container_id = snapshot.id if snapshot.id is not None else self._container_id
            return [self._starting_event(ts=ts, source=Source.RUNTIME, raw="reconcile: running")]
        if previous in _UP_STATES and observed is LifecycleState.ABSENT:
            # The container was removed out from under us. Not a crash - nothing crashed - but not
            # a clean stop either, because the world had no say in it.
            return self._down_events(
                exit_code=None,
                oom_killed=False,
                ts=ts,
                source=Source.RUNTIME,
                raw="reconcile: container is gone",
                force_stopped=True,
            )
        if previous in _UP_STATES and observed in (
            LifecycleState.STOPPED,
            LifecycleState.CRASHED,
        ):
            _log.info(
                "lifecycle.stop_detected_by_reconcile",
                server_id=self._server_id,
                exit_code=snapshot.exit_code,
            )
            return self._down_events(
                exit_code=snapshot.exit_code,
                oom_killed=snapshot.oom_killed or self._oom_seen,
                ts=ts,
                source=Source.RUNTIME,
                raw="reconcile: container exited",
            )
        if (
            previous in (LifecycleState.STARTING, LifecycleState.DEGRADED)
            and observed is LifecycleState.READY
        ):
            # DEGRADED is included because a start that overran its deadline is still a start that
            # has never reported ready. `_promote_ready` is the one place that decides whether
            # that is a first announcement or a recovery from an already-announced run.
            return self._promote_ready(
                detected_by=ReadySignal.HEALTHCHECK,
                ts=ts,
                source=Source.RUNTIME,
                raw="reconcile: container is healthy",
                startup_seconds=None,
            )
        # Everything else - STOPPED <-> CRASHED <-> ABSENT, READY -> DEGRADED - is either a
        # reclassification of a state already reported, or a state change the event vocabulary
        # has no word for.
        _log.debug(
            "lifecycle.silent_transition",
            previous=previous.value,
            observed=observed.value,
            reason="no event exists for this delta",
        )
        return []

    def _adopt(self, snapshot: ContainerSnapshot, state: LifecycleState) -> None:
        """Take identity and timing from a snapshot, then settle into ``state``."""
        if snapshot.id is not None:
            self._container_id = snapshot.id
        if snapshot.running and snapshot.started_at is not None:
            self._started_at = snapshot.started_at
        self._enter(state, reason="reconcile")
        if state is LifecycleState.STARTING and self._starting_since is None:
            self._starting_since = self._clock.monotonic()

    # -- docker events --------------------------------------------------------------------------

    def _on_runtime_event(self, event: RuntimeEvent) -> list[Event]:
        action = event.action
        if action.startswith("health_status: "):
            return self._on_health_event(event)
        if action == "oom":
            self._oom_seen = True
            _log.warning("lifecycle.oom_event", server_id=self._server_id)
            return []
        if action == "start":
            return self._on_start_event(event)
        if action == "die":
            return self._on_die_event(event)
        if action == "kill":
            return self._on_kill_event(event)
        if action == "destroy":
            return self._on_destroy_event(event)
        if action == "create":
            if self._state is LifecycleState.ABSENT:
                # The container exists again but has not been started. That is not a start.
                self._enter(LifecycleState.STOPPED, reason="docker create")
            return []
        _log.debug(
            "lifecycle.runtime_event_ignored",
            action=action,
            state=self._state.value,
        )
        return []

    def _on_start_event(self, event: RuntimeEvent) -> list[Event]:
        if self._state is LifecycleState.BLIND:
            return self._dropped_while_blind("docker start")
        if self._state in _UP_STATES:
            _log.debug("lifecycle.duplicate_start_event", state=self._state.value)
            return []
        if event.container_id is not None:
            self._container_id = event.container_id
        self._begin_run(started_at=event.ts)
        self._enter(LifecycleState.STARTING, reason="docker start")
        return [self._starting_event(ts=event.ts, source=Source.RUNTIME, raw="docker: start")]

    def _on_die_event(self, event: RuntimeEvent) -> list[Event]:
        if self._state is LifecycleState.BLIND:
            return self._dropped_while_blind("docker die")
        if self._state in _DOWN_STATES:
            _log.debug("lifecycle.duplicate_die_event", state=self._state.value)
            return []
        return self._down_events(
            exit_code=event.exit_code,
            oom_killed=self._oom_seen or self._snapshot_oom(),
            ts=event.ts,
            source=Source.RUNTIME,
            raw=f"docker: die exitCode={event.attributes.get('exitCode', '?')}",
        )

    def _on_kill_event(self, event: RuntimeEvent) -> list[Event]:
        """``kill`` is the earliest evidence of a ``docker stop`` nobody told us about.

        Docker's ordering for ``docker stop`` is ``kill`` (signal 15) -> ``die`` -> ``stop``, so
        this is where an externally initiated shutdown first becomes visible. Recording the intent
        here is what makes the subsequent ``die`` classify correctly instead of being reported as a
        crash, and it is why an operator running ``docker stop minecraft`` by hand reconciles
        cleanly without the daemon having requested anything.
        """
        signal_name = event.attributes.get("signal", "")
        if signal_name not in _GRACEFUL_KILL_SIGNALS:
            # SIGKILL and friends: somebody is not asking politely, so there is no stop intent to
            # record and `die` will correctly classify the result as a crash.
            _log.debug("lifecycle.kill_event", signal=signal_name, state=self._state.value)
            return []
        if self._state not in _HEALTH_RELEVANT_STATES:
            _log.debug("lifecycle.kill_event_ignored", state=self._state.value)
            return []
        if not self._stop_intent:
            self._arm_stop_intent(
                reason="external docker stop",
                requested_by=None,
                timeout_seconds=None,
            )
        self._enter(LifecycleState.STOPPING, reason="docker kill")
        return [
            ServerStopping(
                ts=event.ts,
                server_id=self._server_id,
                source=Source.RUNTIME,
                raw=f"docker: kill signal={signal_name}",
                reason=self._stop_reason,
                requested_by=self._stop_requested_by,
                timeout_seconds=self._stop_timeout,
            )
        ]

    def _on_destroy_event(self, event: RuntimeEvent) -> list[Event]:
        if self._state is LifecycleState.BLIND:
            return self._dropped_while_blind("docker destroy")
        events: list[Event] = []
        if self._state in _UP_STATES:
            events.extend(
                self._down_events(
                    exit_code=None,
                    oom_killed=False,
                    ts=event.ts,
                    source=Source.RUNTIME,
                    raw="docker: destroy while running",
                    force_stopped=True,
                )
            )
        self._enter(LifecycleState.ABSENT, reason="docker destroy")
        self._container_id = None
        return events

    def _on_health_event(self, event: RuntimeEvent) -> list[Event]:
        status = event.health_status
        if self._state not in _HEALTH_RELEVANT_STATES:
            _log.debug(
                "lifecycle.health_event_dropped",
                state=self._state.value,
                health=status.value if status is not None else None,
                reason="state is not STARTING, READY or DEGRADED",
            )
            return []
        if status is HealthState.HEALTHY:
            return self._promote_ready(
                detected_by=ReadySignal.HEALTHCHECK,
                ts=event.ts,
                source=Source.RUNTIME,
                raw="docker: health_status: healthy",
                startup_seconds=None,
            )
        if status is HealthState.UNHEALTHY:
            return self._on_unhealthy(event)
        _log.debug(
            "lifecycle.health_event_noop",
            health=status.value if status is not None else None,
            state=self._state.value,
        )
        return []

    def _on_unhealthy(self, event: RuntimeEvent) -> list[Event]:
        if self._within_guard_window():
            # start_period is 120s on this container and the healthcheck is *expected* to fail
            # inside it. Acting here would mean crying wolf on every single start.
            _log.debug(
                "lifecycle.unhealthy_suppressed",
                reason="inside the start-period guard window",
                guard_seconds=self.guard_seconds,
                elapsed_seconds=self.elapsed_since_start,
            )
            return []
        if self._state is LifecycleState.DEGRADED:
            return []
        _log.warning(
            "lifecycle.unhealthy",
            server_id=self._server_id,
            previous=self._state.value,
            failing_streak=self._snapshot.health_failing_streak if self._snapshot else None,
            ts=event.ts.isoformat(),
        )
        self._enter(LifecycleState.DEGRADED, reason="health_status: unhealthy")
        return []

    # -- log lines ------------------------------------------------------------------------------

    def _on_line(self, signal: LineSignal) -> list[Event]:
        event = signal.event
        if isinstance(event, ServerStarting) and event.version:
            # Recorded whatever the state: this is the only place the version is ever available,
            # and a ServerReady emitted later should carry it.
            self._version = event.version

        if self._state is LifecycleState.BLIND:
            return self._dropped_while_blind("log line")
        if self._oracle.stop_signal(event):
            return self._on_stop_line(event)
        detected_by = self._oracle.ready_signal(event)
        if detected_by is not None:
            return self._on_ready_line(event, detected_by=detected_by)
        return []

    def _on_stop_line(self, event: Event) -> list[Event]:
        if self._state is LifecycleState.STOPPING:
            _log.debug("lifecycle.duplicate_stop_line")
            return []
        if not self._line_is_current(event):
            _log.debug("lifecycle.stop_line_ignored", reason="backfilled, or server not running")
            return []
        reason = event.reason if isinstance(event, ServerStopping) else None
        if not self._stop_intent:
            self._arm_stop_intent(
                reason=reason or "shutdown observed in the log",
                requested_by=None,
                timeout_seconds=None,
            )
        self._enter(LifecycleState.STOPPING, reason="stop line")
        return [
            ServerStopping(
                ts=event.ts,
                server_id=self._server_id,
                source=Source.LOG,
                raw=event.raw,
                reason=self._stop_reason,
                requested_by=self._stop_requested_by,
                timeout_seconds=self._stop_timeout,
            )
        ]

    def _on_ready_line(self, event: Event, *, detected_by: ReadySignal) -> list[Event]:
        if not self._line_is_current(event):
            # The json-file driver keeps a 30MB ring and docker's `since` is second-granularity, so
            # every reconnect replays history. A `Done (32.521s)!` from a previous run is a routine
            # input, and announcing it would mean "server is up!" on every daemon restart.
            _log.debug(
                "lifecycle.ready_line_ignored",
                reason="backfilled from a previous run, or the container is not running",
                line_ts=event.ts.isoformat(),
                started_at=self._started_at.isoformat() if self._started_at else None,
            )
            return []
        startup_seconds = event.startup_seconds if isinstance(event, ServerReady) else None
        return self._promote_ready(
            detected_by=detected_by,
            ts=event.ts,
            source=Source.LOG,
            raw=event.raw,
            startup_seconds=startup_seconds,
        )

    def _line_is_current(self, event: Event) -> bool:
        """Does this parsed line describe the run we are watching, or a replayed old one?"""
        if self._started_at is not None and event.ts < self._started_at:
            return False
        if self._state in _UP_STATES:
            return True
        snapshot = self._snapshot
        return snapshot is not None and snapshot.running

    # -- probes ---------------------------------------------------------------------------------

    def _on_probe(self, signal: ProbeSignal) -> list[Event]:
        result = signal.result
        if not result.reachable:
            # The single most important no-op in the project. Unreachable means unknown: it is not
            # "stopped", and downstream it must not become "zero players" either. Treating a failed
            # probe as evidence is how you stop a populated server.
            _log.debug(
                "lifecycle.probe_failed",
                state=self._state.value,
                error=result.error,
                note="unreachable means unknown, never stopped and never empty",
            )
            return []
        if self._state is LifecycleState.BLIND:
            return self._dropped_while_blind("probe")
        return self._promote_ready(
            detected_by=ReadySignal.PROBE,
            ts=result.probed_at,
            source=Source.PROBE,
            raw=f"probe: {result.players_online}/{result.players_max} online",
            startup_seconds=None,
        )

    # -- intents --------------------------------------------------------------------------------

    def _on_intent(self, signal: IntentSignal) -> list[Event]:
        if signal.intent is Intent.START:
            # Deliberately emits nothing. The docker `start` event - or the reconcile that notices
            # the container running - is what emits ServerStarting, and it picks the attribution up
            # from here. So a start that never actually happens produces no phantom event.
            self._start_requested_by = signal.requested_by
            _log.debug("lifecycle.start_intent", requested_by=signal.requested_by)
            return []

        if signal.dry_run:
            # Nothing is going to stop, so the machine must not enter STOPPING and must not arm a
            # stop intent: either would misclassify a still-running server's disconnects as
            # shutdown casualties, and would make an unrelated later exit 137 look clean.
            _log.info(
                "lifecycle.stop_intent_dry_run",
                requested_by=signal.requested_by,
                reason=signal.reason,
                state=self._state.value,
            )
            return []

        self._arm_stop_intent(
            reason=signal.reason,
            requested_by=signal.requested_by,
            timeout_seconds=signal.timeout_seconds,
        )
        if self._state is LifecycleState.STOPPING:
            _log.debug("lifecycle.stop_intent_refreshed", requested_by=signal.requested_by)
            return []
        if self._state not in _HEALTH_RELEVANT_STATES:
            _log.debug("lifecycle.stop_intent_in_down_state", state=self._state.value)
            return []
        ts = self._now()
        self._enter(LifecycleState.STOPPING, reason="stop intent")
        return [
            ServerStopping(
                ts=ts,
                server_id=self._server_id,
                source=Source.INTERNAL,
                raw=f"stop requested by {signal.requested_by or 'unknown'}",
                reason=signal.reason,
                requested_by=signal.requested_by,
                timeout_seconds=signal.timeout_seconds,
            )
        ]

    # -- shared transitions ---------------------------------------------------------------------

    def _promote_ready(
        self,
        *,
        detected_by: ReadySignal,
        ts: datetime,
        source: Source,
        raw: str | None,
        startup_seconds: float | None,
    ) -> list[Event]:
        """First-of-three readiness. Emits ``ServerReady`` at most once per run.

        Two ways into ``READY`` from here, and the difference is :attr:`_ready_expected` - "this
        run was witnessed starting and has not announced itself yet" - not the state alone:

        - **A start that has not yet reported ready.** ``STARTING``, or ``DEGRADED`` because the
          start overran its deadline or the healthcheck failed past the guard window. The first
          readiness signal to arrive emits ``ServerReady`` whichever of the two it finds, because a
          slow start is still a start: a fresh world generating for five minutes crosses the
          270-second deadline into ``DEGRADED`` and then logs ``Done (298.4s)!``, and dropping that
          would leave Discord, ``mcmanager events`` and ``/status`` believing the server never came
          up. First-of-three must not become none-of-three just because the clock won a race.
        - **A run that already announced itself and later went unhealthy.** Recovery. There is no
          ``ServerRecovered`` in the vocabulary, so this is a state change only, and the recovered
          state is what ``/status`` renders.
        """
        if detected_by not in self._ready_signals:
            _log.debug(
                "lifecycle.ready_signal_disabled",
                detected_by=detected_by.value,
                enabled=sorted(s.value for s in self._ready_signals),
            )
            return []
        if self._state is LifecycleState.DEGRADED and not self._ready_expected:
            _log.info(
                "lifecycle.recovered",
                server_id=self._server_id,
                detected_by=detected_by.value,
                emitted="nothing (no recovery event exists)",
            )
            self._enter(LifecycleState.READY, reason=f"recovered via {detected_by.value}")
            return []
        if self._state not in (LifecycleState.STARTING, LifecycleState.DEGRADED):
            _log.debug(
                "lifecycle.ready_ignored",
                state=self._state.value,
                detected_by=detected_by.value,
            )
            return []
        measured = startup_seconds if startup_seconds is not None else self.elapsed_since_start
        overran = self._state is LifecycleState.DEGRADED
        self._ready_at = ts
        self._ready_detected_by = detected_by
        self._ready_expected = False
        self._enter(LifecycleState.READY, reason=f"ready via {detected_by.value}")
        _log.info(
            "lifecycle.ready",
            server_id=self._server_id,
            detected_by=detected_by.value,
            startup_seconds=measured,
            version=self._version,
            overran_start_deadline=overran,
        )
        return [
            ServerReady(
                ts=ts,
                server_id=self._server_id,
                source=source,
                raw=raw,
                startup_seconds=measured,
                version=self._version,
                detected_by=detected_by,
            )
        ]

    def _down_events(
        self,
        *,
        exit_code: int | None,
        oom_killed: bool,
        ts: datetime,
        source: Source,
        raw: str | None,
        force_stopped: bool = False,
    ) -> list[Event]:
        """Build the one event a container going away implies, and move to STOPPED or CRASHED."""
        verdict = classify_exit(
            exit_code,
            oom_killed=oom_killed,
            had_stop_intent=self._stop_intent,
        )
        uptime = self._uptime_seconds(ts)
        if verdict.crashed and self._report_crash and not force_stopped:
            _log.error(
                "lifecycle.crashed",
                server_id=self._server_id,
                exit_code=exit_code,
                oom_killed=oom_killed,
                uptime_seconds=uptime,
            )
            self._end_run(LifecycleState.CRASHED, reason="crashed")
            return [
                ServerCrashed(
                    ts=ts,
                    server_id=self._server_id,
                    source=source,
                    raw=raw,
                    exit_code=exit_code,
                    oom_killed=oom_killed,
                    tail=self._tail(),
                )
            ]
        _log.info(
            "lifecycle.stopped",
            server_id=self._server_id,
            exit_code=exit_code,
            clean=verdict.clean,
            forced=verdict.forced,
            uptime_seconds=uptime,
        )
        self._end_run(LifecycleState.STOPPED, reason="stopped")
        return [
            ServerStopped(
                ts=ts,
                server_id=self._server_id,
                source=source,
                raw=raw,
                exit_code=exit_code,
                clean=verdict.clean,
                forced=verdict.forced,
                uptime_seconds=uptime,
            )
        ]

    def _starting_event(self, *, ts: datetime, source: Source, raw: str | None) -> ServerStarting:
        requested_by = self._start_requested_by
        self._start_requested_by = None
        return ServerStarting(
            ts=ts,
            server_id=self._server_id,
            source=source,
            raw=raw,
            version=self._version,
            container_id=self._container_id,
            requested_by=requested_by,
        )

    # -- state bookkeeping ----------------------------------------------------------------------

    def _enter(self, state: LifecycleState, *, reason: str) -> None:
        if state is self._state:
            return
        previous = self._state
        self._state = state
        self._last_known = state
        self._transition_at = self._now()
        if state is LifecycleState.STARTING:
            self._starting_since = self._clock.monotonic()
            # This run is now on the hook for a ServerReady, and stays on it across a subsequent
            # STARTING -> DEGRADED deadline transition. Only emitting one, or the run ending,
            # clears it - which is what keeps a boot reconcile that *discovers* a degraded server
            # from announcing a readiness nobody witnessed.
            self._ready_expected = True
        elif state is not LifecycleState.BLIND:
            self._starting_since = None
        _log.info(
            "lifecycle.transition",
            server_id=self._server_id,
            previous=previous.value,
            state=state.value,
            reason=reason,
        )

    def _begin_run(self, *, started_at: datetime | None) -> None:
        """A new server process. Clear everything that was about the previous one."""
        self._started_at = started_at
        self._starting_since = self._clock.monotonic()
        self._ready_at = None
        self._ready_detected_by = None
        self._ready_expected = True
        self._version = None
        self._oom_seen = False
        self._stop_intent = False
        self._stop_reason = None
        self._stop_requested_by = None
        self._stop_timeout = None

    def _end_run(self, state: LifecycleState, *, reason: str) -> None:
        self._enter(state, reason=reason)
        self._starting_since = None
        self._ready_at = None
        self._ready_detected_by = None
        self._ready_expected = False

    def _arm_stop_intent(
        self,
        *,
        reason: str | None,
        requested_by: str | None,
        timeout_seconds: float | None,
    ) -> None:
        self._stop_intent = True
        self._stop_reason = reason
        self._stop_requested_by = requested_by
        self._stop_timeout = timeout_seconds

    # -- derivations ----------------------------------------------------------------------------

    def _within_guard_window(self, snapshot: ContainerSnapshot | None = None) -> bool:
        """Is the container still inside ``start_period + interval * (retries + 1)``?

        A ``None`` guard means the container declares no healthcheck, in which case there is no
        window and nothing to suppress. A ``None`` elapsed means we do not know when the container
        started, and refusing to suppress on no evidence is the safer of the two mistakes: the
        alternative is silently ignoring a genuinely unhealthy server forever.
        """
        source = snapshot if snapshot is not None else self._snapshot
        guard = source.guard_window if source is not None else None
        seconds = guard.total_seconds() if guard is not None else self._fallback_guard
        if seconds is None:
            return False
        elapsed = self._elapsed_for_guard(source)
        return elapsed is not None and elapsed < seconds

    def _elapsed_for_guard(self, snapshot: ContainerSnapshot | None) -> float | None:
        """Container uptime, preferring our own reading and falling back to the snapshot's.

        The fallback exists for exactly one moment: the boot reconcile, where ``_observe`` runs
        *before* ``_adopt`` has taken ``StartedAt`` off the snapshot. Without it, a daemon starting
        up thirty seconds into a container's 120-second ``start_period`` would read the expected
        ``unhealthy`` as a genuinely degraded server.
        """
        own = self.elapsed_since_start
        if own is not None:
            return own
        if snapshot is None or not snapshot.running:
            return None
        uptime = snapshot.uptime(snapshot.observed_at)
        return None if uptime is None else max(uptime.total_seconds(), 0.0)

    def _uptime_seconds(self, ts: datetime) -> float | None:
        if self._started_at is None:
            return None
        return max((ts - self._started_at).total_seconds(), 0.0)

    def _snapshot_oom(self) -> bool:
        return self._snapshot is not None and self._snapshot.oom_killed

    def _tail(self) -> tuple[str, ...]:
        provider = self._tail_provider
        if provider is None:
            return ()
        try:
            lines = tuple(provider())
        except Exception:
            # A ring buffer that throws must not turn a crash report into a second crash.
            _log.exception("lifecycle.tail_provider_failed")
            return ()
        return lines[-self._tail_lines :]

    def _now(self) -> datetime:
        return self._clock.now()


# ----------------------------------------------------------------------------------- service


@final
class LifecycleService:
    """The wrapper: feeds signals into the reducer and publishes what comes back.

    Every ``on_*`` method is synchronous, because :meth:`~mcmanager.core.bus.EventBus.publish` is
    synchronous and pretending otherwise would be a lie about the coupling - ``DockerManager``'s
    callbacks are ``def`` too, and so is the log pipeline's line handler. The one ``async`` method,
    :meth:`on_runtime_status`, exists only because bus subscribers must be coroutines.

    The single thing this class adds beyond publishing is a timer: it arms
    :meth:`~mcmanager.clock.Clock.call_later` on the start deadline so ``STARTING -> DEGRADED`` is
    prompt rather than waiting for the next 60-second reconcile. The reducer re-checks the deadline
    on every signal regardless, so losing the timer degrades promptness, never correctness.
    """

    __slots__ = ("_clock", "_deadline_handle", "_reducer", "_sink")

    def __init__(self, *, reducer: LifecycleReducer, sink: EventSink, clock: Clock) -> None:
        """Wire a reducer to a bus.

        Args:
            reducer: The state machine.
            sink: Where events go. ``EventBus`` satisfies this structurally.
            clock: Used only for :meth:`~mcmanager.clock.Clock.call_later`.
        """
        self._reducer = reducer
        self._sink = sink
        self._clock = clock
        self._deadline_handle: TimerHandle | None = None

    @property
    def reducer(self) -> LifecycleReducer:
        """The state machine, for status rendering and for tests."""
        return self._reducer

    @property
    def state(self) -> LifecycleState:
        """Shorthand for ``service.reducer.state``."""
        return self._reducer.state

    @property
    def ready_at(self) -> datetime | None:
        """Shorthand for ``service.reducer.ready_at``.

        Exposed because the idle manager's ``min_uptime`` gate needs the current run's age, and
        reaching it through ``.reducer`` would hand a subsystem the whole state machine to read
        one timestamp.
        """
        return self._reducer.ready_at

    @property
    def started_at(self) -> datetime | None:
        """Shorthand for ``service.reducer.started_at``.

        The container's ``State.StartedAt``. Unlike :attr:`ready_at` this survives a daemon
        restart, because a boot reconcile reads it straight off the snapshot - which is exactly
        why the idle gate falls back to it.
        """
        return self._reducer.started_at

    # -- feeds ----------------------------------------------------------------------------------

    def feed(self, signal: Signal) -> list[Event]:
        """Reduce one signal, publish every event it implied, and return them.

        Returning the list as well as publishing it is what lets a test assert on a transition
        without a running dispatch loop.
        """
        events = self._reducer.handle(signal)
        self._publish(events)
        self._rearm_deadline()
        return events

    def on_snapshot(self, snapshot: ContainerSnapshot) -> None:
        """``DockerManager``'s ``on_snapshot`` callback."""
        self.feed(SnapshotSignal(snapshot=snapshot))

    def on_runtime_event(self, event: RuntimeEvent) -> None:
        """``DockerManager``'s ``on_event`` callback."""
        self.feed(event)

    def on_line_event(self, event: Event) -> None:
        """The log pipeline's hand-off for parsed, lifecycle-shaped events.

        The pipeline routes ``ServerStarting``/``ServerReady``/``ServerStopping`` here instead of
        publishing them itself, so readiness stays first-of-three and ``ServerReady`` is emitted
        exactly once per run rather than once per matching line.
        """
        self.feed(LineSignal(event=event))

    def on_probe(self, result: ProbeResult) -> None:
        """The status poller's hand-off. Failed probes are fed too, and are a no-op by design."""
        self.feed(ProbeSignal(result=result))

    def on_intent(self, signal: IntentSignal) -> None:
        """``ServerController``'s hand-off, immediately before it calls the runtime."""
        self.feed(signal)

    def on_availability(
        self,
        *,
        available: bool,
        error: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        """Direct availability feed, for callers not going through the bus."""
        self.feed(AvailabilitySignal(available=available, error=error, endpoint=endpoint))

    async def on_runtime_status(self, event: RuntimeStatusEvent) -> None:
        """Bus subscriber for ``RuntimeUnavailable`` / ``RuntimeRestored``.

        ``DockerManager`` publishes those two itself, edge-triggered, so this translates them into
        the reducer's :class:`AvailabilitySignal` rather than duplicating the outage detection.
        Wire it with ``bus.subscribe(RuntimeStatusEvent, service.on_runtime_status,
        name="lifecycle")``.
        """
        if isinstance(event, RuntimeUnavailable):
            self.on_availability(available=False, error=event.error, endpoint=event.endpoint)
        elif isinstance(event, RuntimeRestored):
            self.on_availability(available=True)

    # -- shutdown -------------------------------------------------------------------------------

    async def aclose(self) -> None:
        """Cancel the deadline timer. Safe to call twice, never raises."""
        self._cancel_deadline()

    # -- internals ------------------------------------------------------------------------------

    def _publish(self, events: list[Event]) -> None:
        for event in events:
            self._sink.publish(event)

    def _rearm_deadline(self) -> None:
        self._cancel_deadline()
        if self._reducer.state is not LifecycleState.STARTING:
            return
        deadline = self._reducer.start_deadline_seconds
        if deadline is None:
            return
        elapsed = self._reducer.elapsed_since_start or 0.0
        self._deadline_handle = self._clock.call_later(
            max(deadline - elapsed, 0.0),
            self._fire_deadline,
        )

    def _fire_deadline(self) -> None:
        self._deadline_handle = None
        self._publish(self._reducer.check_deadlines())

    def _cancel_deadline(self) -> None:
        handle = self._deadline_handle
        if handle is not None:
            handle.cancel()
            self._deadline_handle = None
