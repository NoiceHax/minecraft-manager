"""Composition root: builds every object, wires the bus, runs, shuts down. **Phase 9.**

Every file in the tree exists from day one specifically so this wiring is exercised from day one
and M4-M6 are body-fills rather than re-architecture.

Construction order (dependencies before dependents): clock -> config -> logging -> bus ->
supervisor -> runtime -> game adapter -> lifecycle -> log pipeline -> roster -> poller ->
controller -> session -> idle -> control server -> Discord.

**Shutdown order is the reverse, with the bus closed late** so subsystems can emit their final
events on the way out::

    Discord
    -> control surface
    -> IdleManager (cancel timer, publish IdleCancelled(reason="daemon_shutdown"),
                    and explicitly DO NOT trigger a stop)
    -> poller
    -> DockerManager
    -> SessionManager checkpoint
    -> state flush
    -> bus drain
    -> supervised tasks cancelled, in reverse spawn order
    -> runtime close

The idle step is called out because an idle timer firing during teardown - and stopping the server
because the *manager* was restarting - would be the single most dangerous bug in this design. It is
first among the subsystems for that reason, it sets its closing flag before anything else can fire,
and ``tests/integration/test_app.py`` asserts that no stop is issued.

Two orderings deserve their justification written down, because both look wrong at a glance:

- **The bus drains before the supervised tasks are cancelled.** The bus's dispatch loop *is* one of
  those tasks, so cancelling first would throw away exactly the final events - the last
  ``IdleCancelled``, the last session counters - that closing the bus late was supposed to deliver.
- **The runtime is closed last, after task cancellation.** A log pump still parked in a blocking
  read would otherwise wake up holding a closed docker client.

Docker's default stop grace is 10 seconds and our shutdown does a Discord round trip, so compose
sets ``stop_grace_period: 30s`` and this sequence targets under 5. That target is enforced rather
than hoped for: :class:`_ShutdownBudget` gives every step a slice of one wall-clock budget, and a
step that overruns is logged and abandoned rather than allowed to hold the whole teardown.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import signal
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, cast, final

import structlog

from mcmanager.clock import Clock, SystemClock
from mcmanager.config import (
    RuntimeUnreachableError,
    Settings,
    runtime_unreachable_message,
    startup_banner,
)
from mcmanager.containers.base import ContainerRuntime
from mcmanager.containers.factory import build_runtime
from mcmanager.containers.manager import BackoffPolicy, DockerManager
from mcmanager.core.bus import EventBus
from mcmanager.core.events import (
    Event,
    RuntimeStatusEvent,
    ServerEvent,
    ServerReady,
    ServerStarting,
    ServerStopping,
)
from mcmanager.core.supervisor import Supervisor, TaskPolicy
from mcmanager.discordbot.module import DiscordModule
from mcmanager.errors import EXIT_OK, McManagerError
from mcmanager.games.registry import get_adapter
from mcmanager.logging_setup import configure_from_settings
from mcmanager.persistence.session_log import MountedLogSource, SessionLog
from mcmanager.persistence.state_store import StateStore
from mcmanager.services.controller import ServerController
from mcmanager.services.idle import IdleManager
from mcmanager.services.lifecycle import LifecycleReducer, LifecycleService
from mcmanager.services.log_pipeline import LogPipeline
from mcmanager.services.players import PlayerRoster
from mcmanager.services.session import SessionManager
from mcmanager.services.status_poller import StatusPoller

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from types import FrameType

    from mcmanager.containers.dto import ContainerSnapshot
    from mcmanager.control.views import StatusView

__all__ = ["AppServices", "Application", "ControlSurface", "SignalRelay", "run", "run_app"]

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.app")

DEFAULT_SHUTDOWN_TIMEOUT = 4.5
"""Total wall-clock budget for the whole shutdown sequence, in seconds. Compose allows 30; the
design targets under 5, and this is what makes that a number rather than an aspiration."""

_LIFECYCLE_LINE_EVENTS: Final = (ServerStarting, ServerReady, ServerStopping)
"""Parsed events that go to the lifecycle machine **instead of** straight to the bus.

Lifecycle republishes whichever of them survive its own state machine, which is what keeps
readiness first-of-three and ``ServerReady`` emitted once per run rather than once per matching
line. Publishing them here as well would double every one of them."""


# ------------------------------------------------------------------------------ control seam


class ControlSurface(Protocol):
    """The aiohttp control server, as much of it as ``app.py`` needs to know about.

    ``control/`` is owned elsewhere, so this is a structural seam rather than an import: whatever
    that package builds needs only these two methods. The daemon is fully functional without one -
    it degrades to "no ``/healthz``, no CLI-over-HTTP" and logs the fact - which is what lets the
    two halves land independently.
    """

    async def start(self) -> None:
        """Bind and begin serving. aiohttp's runner keeps itself on the loop; no task needed."""
        ...

    async def aclose(self) -> None:
        """Stop serving and release the socket. Safe to call twice."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class AppServices:
    """Everything the daemon built, handed to whatever needs to read it.

    This is the argument a control-surface factory receives and the object ``/status`` renders
    from. A frozen container of already-constructed objects rather than a service locator: nothing
    here creates anything, and passing it around cannot change the wiring.
    """

    settings: Settings
    clock: Clock
    bus: EventBus
    supervisor: Supervisor
    runtime: ContainerRuntime
    manager: DockerManager
    lifecycle: LifecycleService
    controller: ServerController
    pipeline: LogPipeline
    roster: PlayerRoster
    poller: StatusPoller
    store: StateStore
    session: SessionManager
    session_log: SessionLog
    idle: IdleManager
    discord: DiscordModule


def _load_control_surface(services: AppServices) -> ControlSurface | None:
    """Build the control surface if ``mcmanager.control.server`` provides a factory.

    Looked up by name at runtime rather than imported, for one reason: this module must import
    cleanly whether or not the control package has landed, and a ``from ... import`` of a name that
    does not exist yet is a hard failure at import time rather than a degraded feature at runtime.

    The contract is one function::

        def build_control_surface(services: AppServices) -> ControlSurface

    Anything else - a missing module, a missing name, a factory that raises - is logged and the
    daemon runs without a control surface.
    """
    try:
        module = importlib.import_module("mcmanager.control.server")
    except ImportError as exc:  # pragma: no cover - the package is part of this repo
        _log.warning("control.import_failed", error=str(exc))
        return None
    factory: object = getattr(module, "build_control_surface", None)
    if not callable(factory):
        _log.warning(
            "control.not_wired",
            hint="mcmanager.control.server.build_control_surface(services) is not defined yet; "
            "the daemon runs without /healthz and without the CLI-over-HTTP surface",
        )
        return None
    try:
        surface = factory(services)
    except Exception:
        _log.exception("control.build_failed")
        return None
    return cast("ControlSurface", surface)


# ----------------------------------------------------------------------------- internal glue


@final
class _PipelineSink:
    """Where the log pipeline's events go: the bus, except the lifecycle-shaped ones.

    ``ServerStarting`` / ``ServerReady`` / ``ServerStopping`` parsed out of a log line are handed to
    :class:`~mcmanager.services.lifecycle.LifecycleService` **instead of** being published, because
    lifecycle is what decides whether they are true: readiness is first-of-three signals, and a
    server that logs ``Done (32.5s)!`` twice in one run has still only become ready once. Lifecycle
    publishes the ones that survive.
    """

    __slots__ = ("_bus", "_lifecycle")

    def __init__(self, *, bus: EventBus, lifecycle: LifecycleService) -> None:
        self._bus = bus
        self._lifecycle = lifecycle

    def publish(self, event: Event) -> None:
        """Route one event. Synchronous, like every other publish path in the daemon."""
        if isinstance(event, _LIFECYCLE_LINE_EVENTS):
            self._lifecycle.on_line_event(event)
            return
        self._bus.publish(event)


@final
class _ShutdownBudget:
    """One wall-clock budget shared by every shutdown step.

    Without this, "target under five seconds" is a comment. With it, a subsystem that hangs costs
    its own slice and no more, and the daemon still exits inside compose's grace period.
    """

    __slots__ = ("_clock", "_deadline", "_total")

    def __init__(self, *, clock: Clock, total: float) -> None:
        self._clock = clock
        self._total = total
        self._deadline = clock.monotonic() + total

    def slice_for(self, want: float) -> float:
        """The smaller of what a step asked for and what is left. Never zero, never negative."""
        remaining = self._deadline - self._clock.monotonic()
        return max(min(want, remaining), 0.05)

    @property
    def remaining(self) -> float:
        """Seconds left in the budget; negative once it is blown."""
        return self._deadline - self._clock.monotonic()


@final
class SignalRelay:
    """SIGINT/SIGTERM to a shutdown request, and a second one to an immediate hard exit.

    ``loop.add_signal_handler`` is the correct mechanism and is Unix-only; on Windows it raises
    ``NotImplementedError`` and the fallback is ``signal.signal`` plus ``call_soon_threadsafe``,
    because a C-level handler runs between bytecodes on the main thread and must not touch loop
    state directly.

    The escalation is deliberate. The first signal starts an orderly teardown that flushes the
    session record; a second one means the operator has decided the orderly path is not working,
    and continuing to wait politely at that point is how a container gets SIGKILLed mid-write
    anyway - only later and with less information.

    :class:`~mcmanager.core.supervisor.Supervisor` can install its own handlers; this relay is used
    instead of that, not as well, because the escalation has to live in one place and two
    installers would fight over ``signal.signal``.
    """

    __slots__ = ("_hard_exit", "_installed_loop", "_on_shutdown", "_previous", "_seen")

    def __init__(
        self,
        *,
        on_shutdown: Callable[[str], None],
        hard_exit: Callable[[int], None] | None = None,
    ) -> None:
        """Take the shutdown callback, and an injectable exit so the escalation is testable.

        The default really is ``os._exit``: this path must skip atexit handlers, buffered writers
        and any other code that might itself be the thing that is stuck.
        """
        self._on_shutdown = on_shutdown
        self._hard_exit = hard_exit if hard_exit is not None else _hard_exit
        self._previous: list[tuple[signal.Signals, _RawHandler]] = []
        self._installed_loop: list[signal.Signals] = []
        self._seen = 0

    @property
    def signals_seen(self) -> int:
        """How many signals have arrived. Two or more means the next one already hard-exited."""
        return self._seen

    def install(self, *signals: signal.Signals) -> None:
        """Install handlers for SIGINT and SIGTERM (or the given signals)."""
        wanted = signals if signals else _default_signals()
        loop = asyncio.get_running_loop()
        for sig in wanted:
            try:
                loop.add_signal_handler(sig, self.handle, sig)
            except (NotImplementedError, RuntimeError, ValueError):
                previous = signal.getsignal(sig)
                signal.signal(sig, self._raw_handler(loop, sig))
                self._previous.append((sig, previous))
                _log.debug("app.signal_installed", signal=sig.name, via="signal.signal")
            else:
                self._installed_loop.append(sig)
                _log.debug("app.signal_installed", signal=sig.name, via="loop")

    def _raw_handler(
        self,
        loop: asyncio.AbstractEventLoop,
        sig: signal.Signals,
    ) -> Callable[[int, FrameType | None], None]:
        def _handler(signum: int, frame: FrameType | None) -> None:  # noqa: ARG001
            # Runs between bytecodes on the main thread. It hands the fact to the loop and does
            # nothing else, because anything else here risks re-entering half-updated state.
            loop.call_soon_threadsafe(self.handle, sig)

        return _handler

    def handle(self, sig: signal.Signals) -> None:
        """First signal: request shutdown. Second: hard exit, immediately."""
        self._seen += 1
        if self._seen == 1:
            _log.warning("app.signal", signal=sig.name, action="graceful shutdown")
            self._on_shutdown(f"signal:{sig.name}")
            return
        _log.critical(
            "app.signal_escalated",
            signal=sig.name,
            count=self._seen,
            action="hard exit; the session record may be incomplete",
        )
        _flush_streams()
        self._hard_exit(128 + int(sig))

    def remove(self) -> None:
        """Put every handler back exactly as it was found. Idempotent, and never raises."""
        if self._installed_loop:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                for sig in self._installed_loop:
                    try:
                        loop.remove_signal_handler(sig)
                    except (NotImplementedError, RuntimeError, ValueError):
                        _log.debug("app.signal_removal_failed", signal=sig.name)
            self._installed_loop.clear()
        for sig, previous in self._previous:
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError, TypeError):
                _log.debug("app.signal_restore_failed", signal=sig.name)
        self._previous.clear()


type _RawHandler = Callable[[int, FrameType | None], object] | int | signal.Handlers | None
"""What ``signal.getsignal`` returns and ``signal.signal`` accepts."""


def _default_signals() -> tuple[signal.Signals, ...]:
    """SIGINT and SIGTERM. Both exist on Windows; only their delivery differs."""
    return (signal.SIGINT, signal.SIGTERM)


def _hard_exit(code: int) -> None:
    """Leave now, skipping atexit and every buffered writer.

    ``os._exit`` is exactly the right tool here and the only one: the whole point of the second
    signal is that the orderly path is not working, so any exit that runs more of our own code
    could be stuck on the same thing.
    """
    os._exit(code)


def _flush_streams() -> None:
    """Best-effort flush before a hard exit, so the last log line is not lost."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(ValueError, OSError):
            stream.flush()


def _docker_endpoint(settings: Settings) -> str:
    """What to name in the "cannot reach Docker" message. The single most useful thing to print."""
    if settings.docker.host:
        return settings.docker.host
    return os.environ.get("DOCKER_HOST") or "unix:///var/run/docker.sock"


# ------------------------------------------------------------------------------- application


@final
class Application:
    """The whole daemon: constructed, started, awaited, and shut down in one place.

    Construction does no I/O and touches no loop, so a test can build the entire object graph,
    assert on the wiring, and never start anything. Everything that can fail - the Docker ping, the
    state file, the control socket - happens in :meth:`start`.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        clock: Clock | None = None,
        runtime: ContainerRuntime | None = None,
        control_surface: ControlSurface | None = None,
        shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
    ) -> None:
        """Build the object graph.

        Args:
            settings: The resolved configuration. Already validated; nothing here re-checks it.
            clock: Injected time. Defaults to :class:`~mcmanager.clock.SystemClock`. Everything
                downstream takes it from here, which is what makes a full-daemon test run in
                virtual time.
            runtime: A pre-built container runtime, for tests. Otherwise built from
                ``settings.runtime`` by the factory, which is the only place that chooses.
            control_surface: A pre-built control server. Otherwise looked up from
                ``mcmanager.control.server`` after the services exist, since it needs them.
            shutdown_timeout: Total budget for the shutdown sequence.
        """
        self._settings = settings
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._shutdown_timeout = shutdown_timeout
        self._started = False
        self._closed = False
        self._shutdown_steps: list[str] = []

        server = settings.server
        events = settings.events

        self._bus = EventBus(
            maxsize=events.queue_max,
            handler_timeout=events.handler_timeout_seconds,
            max_consecutive_failures=events.max_consecutive_failures,
        )
        self._supervisor = Supervisor(
            self._clock,
            on_critical_failure=self._on_critical_failure,
        )
        self._runtime: ContainerRuntime = (
            runtime
            if runtime is not None
            else build_runtime(
                settings.runtime,
                clock=self._clock,
                docker_host=settings.docker.host,
            )
        )
        self._adapter = get_adapter(server.game, clock=self._clock)
        self._store = StateStore(path=settings.state.dir, clock=self._clock)
        self._roster = PlayerRoster(clock=self._clock)

        reducer = LifecycleReducer(
            server_id=server.id,
            clock=self._clock,
            oracle=self._adapter,
            ready_signals=server.lifecycle.ready_signals,
            start_deadline_extra_seconds=server.lifecycle.start_deadline_extra_seconds,
            tail_provider=self._tail,
            tail_lines=events.crash_tail_lines,
            report_unexpected_exit_as_crash=server.lifecycle.treat_unexpected_exit_as_crash,
        )
        self._lifecycle = LifecycleService(reducer=reducer, sink=self._bus, clock=self._clock)

        self._pipeline = LogPipeline(
            adapter=self._adapter,
            sink=_PipelineSink(bus=self._bus, lifecycle=self._lifecycle),
            clock=self._clock,
            server_id=server.id,
            roster=self._roster,
            tail_lines=events.crash_tail_lines,
            rate_per_second=float(events.rate_limit_per_second),
            rate_burst=float(events.rate_limit_burst),
        )
        self._manager = DockerManager(
            runtime=self._runtime,
            container=server.container,
            server_id=server.id,
            clock=self._clock,
            sink=self._bus,
            on_line=self._pipeline.handle_line,
            on_event=self._lifecycle.on_runtime_event,
            on_snapshot=self._on_snapshot,
            on_log_eof=self._pipeline.on_eof,
            backoff=BackoffPolicy(
                base_seconds=settings.docker.reconnect_initial_seconds,
                max_seconds=settings.docker.reconnect_max_seconds,
            ),
            reconcile_interval=settings.docker.reconcile_interval_seconds,
            dedupe_capacity=settings.docker.dedupe_ring_size,
        )
        self._poller = StatusPoller(
            adapter=self._adapter,
            host=server.host,
            port=server.port,
            clock=self._clock,
            sink=self._bus,
            roster=self._roster,
            server_id=server.id,
            interval=settings.probe.interval_seconds,
            timeout=settings.probe.timeout_seconds,
            trust_partial_sample=settings.probe.trust_partial_sample,
            on_probe=self._lifecycle.on_probe,
            enabled=settings.probe.enabled,
        )
        self._controller = ServerController(
            runtime=self._runtime,
            lifecycle=self._lifecycle,
            sink=self._bus,
            clock=self._clock,
            server_id=server.id,
            container=server.container,
            stop_timeout_seconds=server.lifecycle.stop_timeout_seconds,
        )
        self._session_log = SessionLog(
            archive_dir=settings.logs.archive_dir,
            clock=self._clock,
            source=MountedLogSource(settings.logs.server_log_dir),
            retain_archives=settings.logs.retain_archives,
            archive_on_stop=settings.logs.archive_on_stop,
        )
        self._session = SessionManager(
            sink=self._bus,
            clock=self._clock,
            server_id=server.id,
            roster=self._roster,
            store=self._store,
            session_log=self._session_log,
        )
        self._idle = IdleManager(
            sink=self._bus,
            clock=self._clock,
            server_id=server.id,
            roster=self._roster,
            controller=self._controller,
            store=self._store,
            enabled=settings.idle.enabled,
            dry_run=settings.idle.dry_run,
            timeout_seconds=settings.idle.timeout_seconds,
            warn_seconds=settings.idle.warn_minutes * 60.0,
            poll_interval_seconds=settings.idle.poll_interval_seconds,
            min_uptime_seconds=settings.idle.min_uptime_minutes * 60.0,
            treat_probe_failure_as_empty=settings.idle.treat_probe_failure_as_empty,
        )
        self._discord = DiscordModule(
            clock=self._clock,
            server_id=server.id,
            controller=self._controller,
            roster=self._roster,
            # Providers, not values: `self._services` does not exist yet at this point in the
            # constructor, and `/status` must render the *same* view the HTTP surface renders
            # rather than a second Discord-shaped one that can drift from it.
            status=self._status_view,
            console_tail=self._pipeline.tail,
            idle_describe=self._idle.describe,
            mode=settings.discord.mode,
            enabled=settings.discord.enabled,
            token=settings.discord.token,
            guild_id=settings.discord.guild_id,
            channel_id=settings.discord.channel_id,
            console_channel_id=settings.discord.console_channel_id,
            admin_role_id=settings.discord.admin_role_id,
            register_commands_globally=settings.discord.register_commands_globally,
        )

        self._services = AppServices(
            settings=settings,
            clock=self._clock,
            bus=self._bus,
            supervisor=self._supervisor,
            runtime=self._runtime,
            manager=self._manager,
            lifecycle=self._lifecycle,
            controller=self._controller,
            pipeline=self._pipeline,
            roster=self._roster,
            poller=self._poller,
            store=self._store,
            session=self._session,
            session_log=self._session_log,
            idle=self._idle,
            discord=self._discord,
        )
        self._control: ControlSurface | None = control_surface
        self._control_injected = control_surface is not None
        self._signals = SignalRelay(on_shutdown=self._supervisor.request_shutdown)
        self._subscribe()

    # -------------------------------------------------------------------------- introspection

    @property
    def services(self) -> AppServices:
        """Every constructed object, for the control surface and for the tests."""
        return self._services

    @property
    def shutdown_steps(self) -> tuple[str, ...]:
        """The teardown steps that have completed, in order.

        Recorded rather than merely logged because the *order* is the requirement, and an order is
        only guaranteed by something that asserts on it.
        """
        return tuple(self._shutdown_steps)

    # --------------------------------------------------------------------------------- wiring

    def _subscribe(self) -> None:
        """Register every bus subscription. Called once, from the constructor."""
        self._bus.subscribe(
            RuntimeStatusEvent,
            self._lifecycle.on_runtime_status,
            name="lifecycle.runtime_status",
        )
        self._bus.subscribe(
            ServerEvent,
            self._on_server_event,
            name="pipeline.stopping_flag",
        )
        self._session.subscribe(self._bus)
        self._idle.subscribe(self._bus)
        self._discord.subscribe(self._bus)

    async def _on_server_event(self, event: ServerEvent) -> None:
        """Keep the log pipeline's shutdown flag in step with the lifecycle machine.

        Between "we asked Docker to stop" and "the JVM logged ``Stopping the server``" there is a
        real window, and a disconnect inside it is a shutdown casualty rather than a voluntary
        quit. The pipeline sets the flag itself from the log line; this covers the earlier window,
        because ``ServerStopping`` is published the moment the controller declares its intent -
        before the runtime call.

        Done here rather than inside ``ServerController`` so the controller stays a pure command
        object with no knowledge of the parsing layer.
        """
        if isinstance(event, ServerStopping):
            self._pipeline.set_stopping(True)
        elif isinstance(event, ServerStarting | ServerReady):
            self._pipeline.set_stopping(False)

    def _on_snapshot(self, snapshot: ContainerSnapshot) -> None:
        """``DockerManager``'s single snapshot callback, fanned out to its two consumers."""
        self._lifecycle.on_snapshot(snapshot)
        self._session.on_snapshot(snapshot)

    def _tail(self) -> tuple[str, ...]:
        """The log pipeline's ring buffer, for ``ServerCrashed.tail``. Called synchronously."""
        return self._pipeline.tail()

    def _status_view(self) -> StatusView:
        """The aggregated status view, for Discord's ``/status``.

        Imported here rather than at module scope for the same reason
        :func:`_load_control_surface` does it: this module must import cleanly whether or not the
        control package is present. Unlike the surface, a missing one here is not degradable - the
        command has nothing to render - so it propagates and the command handler reports it.
        """
        from mcmanager.control.server import build_status_view

        return build_status_view(self._services)

    def _on_critical_failure(self, task: str, error: BaseException | None) -> None:
        """A critical task died. The supervisor has already set ``shutdown_requested``."""
        _log.critical(
            "app.critical_task_died",
            task=task,
            error=None if error is None else str(error),
        )

    # -------------------------------------------------------------------------------- startup

    async def start(self) -> None:
        """Load state, prove the runtime is reachable, then spawn every task.

        Raises:
            RuntimeUnreachableError: The container runtime does not answer. Exit 69, with the
                socket **and** this process's uid/gid/groups named, because that failure is a
                docker-group membership problem the overwhelming majority of the time and printing
                ``id`` saves twenty minutes.
        """
        if self._started:
            return
        self._started = True

        _log.info("app.starting", banner=startup_banner(self._settings))
        for warning in self._settings.warnings:
            _log.warning("app.config_warning", detail=warning)

        self._ensure_writable_dirs()
        state = self._store.load()
        for name in state.known_players:
            self._roster.note_known(name)
        if state.last_log_ts is not None:
            _log.info(
                "app.resume_marker",
                last_log_ts=state.last_log_ts.isoformat(),
                backfilled=False,
                hint="DockerManager has no way to be seeded with a starting `since` yet, so this "
                "boot streams from now; see the notes in deploy/README.md",
            )

        if not await self._runtime.ping():
            endpoint = _docker_endpoint(self._settings)
            raise RuntimeUnreachableError(
                runtime_unreachable_message(endpoint=endpoint, error="ping() returned False")
            )

        await self._session.start()
        await self._idle.start()
        await self._discord.start()
        await self._start_control()

        self._spawn_tasks()
        self._signals.install()
        _log.info(
            "app.started",
            runtime=self._settings.runtime,
            container=self._settings.server.container,
            tasks=list(self._supervisor.task_names),
        )

    def _ensure_writable_dirs(self) -> None:
        """Create the two directories this daemon owns, if the volume did not already have them.

        Deliberately narrow. ``logs.server_log_dir`` is the game server's, mounted read-only, and a
        missing one must stay a warning rather than being papered over with an empty directory that
        then reads as "no archives exist". ``logs.daemon_log_dir`` is likewise left alone by
        ``logging_setup``, on the grounds that a missing one means a missing volume.

        These two are different: a named volume arrives empty and ``state.dir`` and
        ``archive_dir`` are subdirectories inside it, so without this the archive directory
        silently never appears and ``mcmanager sessions`` reads an empty roster from a path that
        does not exist.

        This runs *after* ``__init__`` has already built the :class:`StateStore`, so it is
        explicitly not what makes the state path come out right. The store decides file-versus-
        directory from the path's suffix rather than from whether it exists, precisely so the two
        cannot get out of order again.
        """
        for label, directory in (
            ("state", self._settings.state.dir),
            ("archives", self._settings.logs.archive_dir),
        ):
            if directory.is_dir():
                continue
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                _log.error(
                    "app.directory_create_failed", kind=label, path=str(directory), error=str(exc)
                )
            else:
                _log.info("app.directory_created", kind=label, path=str(directory))

    async def _start_control(self) -> None:
        """Bring up the control surface, if there is one. Its absence is a warning, not a failure.

        ``web.enabled = false`` is a legitimate configuration, and so - for now - is a control
        package that has not landed yet. The daemon manages the server either way; what is lost is
        ``/healthz`` and the CLI's HTTP path.
        """
        if not self._settings.web.enabled:
            _log.info("control.disabled")
            return
        if self._control is None and not self._control_injected:
            self._control = _load_control_surface(self._services)
        if self._control is None:
            return
        try:
            await self._control.start()
        except Exception:
            _log.exception("control.start_failed")
            self._control = None
            return
        _log.info("control.started", host=self._settings.web.host, port=self._settings.web.port)

    def _spawn_tasks(self) -> None:
        """Spawn every long-lived task, dependencies first.

        Order matters twice over: the supervisor cancels in reverse, and the bus must be dispatching
        before anything can publish into it.
        """
        self._supervisor.spawn(
            "bus",
            self._bus.run,
            policy=TaskPolicy.ONCE,
            critical=True,
        )
        self._supervisor.spawn("state-checkpoint", self._run_checkpoints)
        self._supervisor.spawn("docker-logs", self._manager.run_log_stream)
        self._supervisor.spawn("docker-events", self._manager.run_event_watcher)
        self._supervisor.spawn("docker-reconcile", self._manager.run_reconcile)
        if self._settings.probe.enabled:
            self._supervisor.spawn("status-poller", self._poller.run)
        else:
            # Not spawned rather than spawned-and-immediately-returning: a disabled poller's run()
            # returns at once, and a RESTART policy would then log a restart warning every backoff
            # interval forever, for a subsystem that is off on purpose.
            _log.info("app.status_poller_disabled", hint="probe.enabled is false")
        self._supervisor.spawn("idle", self._idle.run)
        if self._discord.live and self._settings.discord.console_channel_id > 0:
            # Only when there is a relay to run: `run_relay` returns immediately otherwise, and a
            # restart policy would then log a restart every backoff interval forever.
            self._supervisor.spawn("discord-console", self._discord.run_relay)

    async def _run_checkpoints(self) -> None:
        """Fold state into the store and write it, every ``state.checkpoint_interval_seconds``."""
        interval = self._settings.state.checkpoint_interval_seconds
        while not self._closed:
            await self._clock.sleep(interval)
            if self._closed:
                return
            await self._session.checkpoint()
            self._flush_state()

    def _flush_state(self, *, force: bool = False) -> None:
        """Persist the state file. An I/O failure is logged, never raised into the caller.

        The store itself propagates ``OSError`` on purpose - silently not persisting is discovered
        a week later - but a checkpoint tick is not the place to take the daemon down over it.
        """
        self._store.record_log_ts(self._manager.last_line_ts)
        try:
            self._store.save(force=force)
        except OSError as exc:
            _log.error("app.state_save_failed", path=str(self._store.path), error=str(exc))

    # ------------------------------------------------------------------------------- running

    async def wait(self) -> None:
        """Block until a signal arrives or a critical task dies."""
        await self._supervisor.shutdown_requested.wait()

    async def run(self) -> int:
        """Start, wait, shut down. Returns the process exit code.

        A failed :meth:`start` is torn down too. It raises before anything is spawned, but the
        runtime client is already open by then, and leaking a docker connection pool on the way out
        of a startup failure is exactly the sort of thing that only shows up as a mystery on the
        tenth restart.
        """
        try:
            await self.start()
        except BaseException:
            await self.aclose()
            raise
        try:
            await self.wait()
        finally:
            await self.aclose()
        return EXIT_OK

    # ------------------------------------------------------------------------------ shutdown

    async def aclose(self) -> None:
        """The shutdown sequence, in the order the plan fixes it. Idempotent, and never raises.

        Every step is bounded by a slice of one shared budget, so a subsystem that hangs costs its
        own slice and the daemon still exits inside compose's grace period.

        **The signal handlers stay installed for the whole of this method**, and are restored only
        at the very end. That is the entire point of :class:`SignalRelay`'s escalation: a second
        SIGINT or SIGTERM arrives precisely when the first shutdown is wedged, which is exactly
        when it must reach :meth:`SignalRelay.handle` and hard-exit deterministically. Removing
        them first - which is what this used to do - restored the default handlers before the only
        code that can run after them, so a second Ctrl-C raised ``KeyboardInterrupt`` somewhere
        inside the sequence instead, abandoning teardown at an arbitrary point with no log line and
        possibly before the state flush at step 7.
        """
        if self._closed:
            return
        self._closed = True
        budget = _ShutdownBudget(clock=self._clock, total=self._shutdown_timeout)
        _log.info("app.shutting_down", budget_seconds=self._shutdown_timeout)

        # 1. Discord first: the only subsystem that does a network round trip on the way out, and
        #    everything after it may still publish a final event a live bot would have announced.
        await self._step("discord", self._discord.aclose(), budget, 1.5)
        # 2. The control surface, for the same reason: stop accepting outside callers early.
        if self._control is not None:
            await self._step("control", self._control.aclose(), budget, 0.5)
        # 3. Idle. THE dangerous one: cancel the timer, publish IdleCancelled("daemon_shutdown"),
        #    and issue no stop. An idle timer firing during teardown and killing the Minecraft
        #    server because the *manager* restarted would be the worst bug this design could have.
        await self._step("idle", self._idle.aclose(), budget, 0.5)
        # 4. The status poller: stop generating probe-derived roster changes.
        await self._step("poller", self._poller.aclose(), budget, 0.5)
        # 5. The docker attachments: stop generating log lines and container events.
        await self._step("docker", self._manager.aclose(), budget, 0.5)
        self._pipeline.flush()
        # 6. Session checkpoint, with the session left OPEN: the server outlives the daemon.
        await self._step("session", self._session.aclose(), budget, 0.5)
        # 7. State flush: force, because "nothing changed" and "we never got that far" are worth
        #    distinguishing on disk.
        self._flush_state(force=True)
        self._record("state")
        # 8. Bus drain, LATE and before task cancellation, because the bus's dispatch loop is
        #    itself a supervised task and cancelling it first would discard exactly the final
        #    events that closing the bus late exists to deliver.
        await self._step(
            "bus",
            self._bus.aclose(drain_timeout=budget.slice_for(self._drain_timeout())),
            budget,
            self._drain_timeout(),
        )
        # 9. Everything still running, cancelled in reverse spawn order.
        await self._step(
            "tasks",
            self._supervisor.shutdown(task_timeout=budget.slice_for(1.0)),
            budget,
            1.5,
        )
        await self._step("lifecycle", self._lifecycle.aclose(), budget, 0.2)
        # 10. The runtime last, so no pump wakes up holding a closed docker client.
        await self._step("runtime", self._runtime.aclose(), budget, 1.0)
        # 11. Only now hand the signals back. Until this line a second SIGINT/SIGTERM still
        #     reaches the relay and hard-exits with a log line, which is the documented and
        #     deployed contract (deploy/README.md) and is unreachable if this runs any earlier.
        self._signals.remove()
        self._record("signals")

        _log.info(
            "app.stopped",
            steps=list(self._shutdown_steps),
            budget_remaining=round(budget.remaining, 3),
            bus=dict(self._bus.stats),
        )

    def _drain_timeout(self) -> float:
        return self._settings.events.drain_timeout_seconds

    async def _step(
        self,
        name: str,
        coro: Coroutine[object, object, object],
        budget: _ShutdownBudget,
        want: float,
    ) -> None:
        """Run one teardown step inside its slice of the budget. Records it either way.

        A step that times out or raises is logged and the sequence continues: a shutdown that gives
        up halfway leaves the state file unwritten, which is strictly worse than a subsystem that
        did not close tidily.
        """
        allowance = budget.slice_for(want)
        try:
            async with asyncio.timeout(allowance):
                await coro
        except TimeoutError:
            _log.error("app.shutdown_step_timeout", step=name, allowance_seconds=allowance)
        except Exception:
            _log.exception("app.shutdown_step_failed", step=name)
        finally:
            self._record(name)

    def _record(self, name: str) -> None:
        self._shutdown_steps.append(name)
        _log.debug("app.shutdown_step", step=name)


# ------------------------------------------------------------------------------- entry points


async def run_app(
    settings: Settings,
    *,
    clock: Clock | None = None,
    runtime: ContainerRuntime | None = None,
    control_surface: ControlSurface | None = None,
) -> int:
    """Build and run the daemon on the current loop. Returns the process exit code."""
    app = Application(
        settings,
        clock=clock,
        runtime=runtime,
        control_surface=control_surface,
    )
    return await app.run()


def run(settings: Settings, *, clock: Clock | None = None) -> int:
    """``mcmanager run``: configure logging, run the daemon, translate errors to exit codes.

    The one place ``asyncio.run`` is called. Expected failures - a bad config, an unreachable
    Docker socket - come back as their ``sysexits.h`` code with a message already printed by the
    logger; anything else is a bug and keeps its traceback.
    """
    configure_from_settings(settings.logs, clock=clock)
    try:
        return asyncio.run(run_app(settings, clock=clock))
    except McManagerError as exc:
        _log.critical("app.fatal", error=str(exc), exit_code=exc.exit_code)
        return exc.exit_code
    except KeyboardInterrupt:
        _log.warning("app.interrupted")
        return 130
