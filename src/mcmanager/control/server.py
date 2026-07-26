"""The aiohttp application and its runner.

Bound to ``127.0.0.1:8787`` and ``expose``d rather than published, because the documented access
path on the homelab is ``docker exec mcmanager mcmanager status``: no published port, no ufw
involvement, no auth surface on the LAN.

Pulling this forward from M6 to M2 is a simplification, not scope creep. The daemon needs
``/healthz`` for its own compose healthcheck regardless, and having ``/status`` and ``/events``
early means the CLI, Discord and any eventual web UI are three clients of one surface rather than
three parallel implementations that disagree in three different ways.

The runner is deliberately thin. It owns an ``AppRunner`` and a ``TCPSite``, it starts and stops
them, and it holds no state of its own - :class:`~mcmanager.control.routes.ControlContext` is the
whole interface between the daemon and the HTTP layer. That is what lets ``build_app`` be called
by a test with four lambdas.

Binding is a **hard failure**. A daemon whose control port did not come up looks healthy from the
outside and is undebuggable from the inside, which is the exact combination this endpoint exists
to prevent, so :meth:`ControlServer.start` propagates the ``OSError`` rather than logging it.

:func:`build_control_surface` is the seam ``app.py`` looks up by name. It is also where the
read-model is assembled: the four provider callables on
:class:`~mcmanager.control.routes.ControlContext` are closures over the daemon's live objects, so
``/status`` reads state rather than computing it, and this module is the only place that knows how
a lifecycle reducer, a roster, a poller and an idle manager add up to one status view.

The type-only import of ``AppServices`` is the one reference this package makes to the composition
root. It costs nothing at runtime - ``app.py`` imports *this* module dynamically, and the import
here lives under ``TYPE_CHECKING`` - and the alternative, a fifteen-field Protocol restating
``AppServices``, would be a second definition to keep in step with the first.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, final

import structlog
from aiohttp import web

from mcmanager.containers.dto import HealthState
from mcmanager.control.routes import ControlContext, error_middleware, register_routes
from mcmanager.control.sse import SseChannel
from mcmanager.control.views import (
    IdleView,
    LivenessView,
    PlayerView,
    ProbeView,
    ReadinessView,
    SessionView,
    StatusView,
    evaluate_readiness,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from mcmanager.app import AppServices
    from mcmanager.clock import Clock
    from mcmanager.core.events import Event
    from mcmanager.games.base import ProbeResult

__all__ = [
    "ControlServer",
    "build_app",
    "build_control_surface",
    "build_status_view",
    "disabled_surface",
]

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.control.server")


def build_app(context: ControlContext) -> web.Application:
    """Build the aiohttp application for ``context``.

    ``client_max_size`` is cut to 64KB from aiohttp's 1MB default: every request this API accepts
    is either empty or a two-field JSON object, and an endpoint that will buffer a megabyte for
    you is an endpoint somebody will eventually point a file at.
    """
    app = web.Application(middlewares=[error_middleware], client_max_size=64 * 1024)
    register_routes(app, context)
    return app


@final
class ControlServer:
    """Runs :func:`build_app` on a TCP site, and stops it again.

    Not a supervised task: aiohttp's runner is already a set of background tasks, so this is
    started once during wiring and stopped once during teardown. It goes down **early** in the
    shutdown sequence - before the bus drains - so that no client is mid-stream while subsystems
    are emitting their final events.
    """

    __slots__ = (
        "_app",
        "_channel",
        "_context",
        "_host",
        "_keepalive",
        "_port",
        "_runner",
        "_site",
    )

    def __init__(
        self,
        *,
        context: ControlContext,
        host: str = "127.0.0.1",
        port: int = 8787,
    ) -> None:
        """Prepare a server. Nothing is bound until :meth:`start`.

        Args:
            context: The provider bag the handlers read.
            host: Bind address. Loopback by default and on the homelab; the port is ``expose``d,
                never published, so there is no LAN surface to authenticate.
            port: Bind port.
        """
        self._context = context
        self._host = host
        self._port = port
        self._app = build_app(context)
        self._channel: SseChannel = context.channel
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._keepalive: asyncio.Task[None] | None = None

    @property
    def app(self) -> web.Application:
        """The application, for tests that want to mount it on their own server."""
        return self._app

    @property
    def url(self) -> str:
        """The base URL a client should dial. Matches ``web.url``'s default exactly."""
        host = "127.0.0.1" if self._host in ("0.0.0.0", "") else self._host  # noqa: S104
        return f"http://{host}:{self._port}"

    @property
    def running(self) -> bool:
        return self._site is not None

    async def start(self) -> None:
        """Bind and serve. Raises ``OSError`` if the port is taken - deliberately fatal."""
        if self._runner is not None:
            return
        runner = web.AppRunner(self._app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, host=self._host, port=self._port)
        try:
            await site.start()
        except OSError:
            await runner.cleanup()
            _log.error("control.bind_failed", host=self._host, port=self._port)
            raise
        self._runner = runner
        self._site = site
        # The SSE keepalive belongs to this object, not to the app's supervisor: it exists only
        # while there is something to keep alive, and bundling it here is what lets `app.py` treat
        # the whole control surface as one start()/aclose() pair.
        self._keepalive = asyncio.create_task(self._channel.run_keepalive())
        _log.info("control.listening", url=self.url, server_id=self._context.server_id)

    async def aclose(self) -> None:
        """Stop serving and end every SSE stream. Safe to call twice, never raises."""
        keepalive, self._keepalive = self._keepalive, None
        if keepalive is not None:
            keepalive.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await keepalive
        await self._channel.aclose()
        site, self._site = self._site, None
        runner, self._runner = self._runner, None
        if site is not None:
            await site.stop()
        if runner is not None:
            await runner.cleanup()
        if runner is not None or site is not None:
            _log.info("control.stopped", url=self.url)

    async def __aenter__(self) -> ControlServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


# --------------------------------------------------------------------------- the app.py seam


@final
class _DisabledSurface:
    """What :func:`build_control_surface` returns when ``web.enabled`` is false.

    A no-op object rather than ``None`` so the caller has one shape to hold. ``app.py`` checks the
    setting itself and normally never gets here; this exists so that a future caller which does
    not check cannot end up calling ``start()`` on ``None``.
    """

    __slots__ = ()

    async def start(self) -> None:
        _log.info("control.disabled", reason="web.enabled is false")

    async def aclose(self) -> None:
        return


def disabled_surface() -> _DisabledSurface:
    """A control surface that does nothing, for ``web.enabled = false``."""
    return _DisabledSurface()


def build_control_surface(services: AppServices) -> ControlServer | _DisabledSurface:
    """The factory ``app.py`` looks up by name. Wires the read-model to the live daemon.

    Every provider on the context is a closure over already-constructed objects, so a request
    *reads* state and never computes it: ``/status`` must not block on a Docker inspect, or the
    endpoint becomes as unreliable as the thing it describes.

    The SSE channel is attached to the bus here, as a ``CONCURRENT`` wildcard subscriber, and its
    bounded replay ring doubles as the source of ``StatusView.last_event`` - which is why nothing
    else has to keep a "most recent event" variable in step.
    """
    settings = services.settings
    if not settings.web.enabled:
        return disabled_surface()

    channel = SseChannel(clock=services.clock, queue_max=settings.web.sse_queue_max)
    channel.attach(services.bus)

    token = settings.web.token.get_secret_value() if settings.web.token is not None else None
    context = ControlContext(
        server_id=settings.server.id,
        clock=services.clock,
        channel=channel,
        status=lambda: build_status_view(services, channel=channel),
        players=lambda: build_player_views(services),
        sessions=lambda limit: build_session_views(services, limit=limit),
        readiness=lambda: build_readiness_view(services),
        liveness=lambda: build_liveness_view(services),
        controller=services.controller,
        token=token,
    )
    return ControlServer(context=context, host=settings.web.host, port=settings.web.port)


# ------------------------------------------------------------------------------- the read model


def build_status_view(services: AppServices, *, channel: SseChannel | None = None) -> StatusView:
    """Assemble ``/status`` from the daemon's live objects.

    This is the "status aggregator" the CLI and Discord both render from, and it lives here rather
    than in ``services/`` for one reason: it is a *view*, it belongs to the surface that serves it,
    and putting it in the services layer would give every service a reason to import a presenter
    type.

    Nothing here reaches into a private attribute or triggers I/O. Where a subsystem cannot supply
    a fact - a peak player count that only the session manager can track - the field stays at its
    default rather than being invented.
    """
    reducer = services.lifecycle.reducer
    snapshot = reducer.snapshot
    probe = services.poller.last_result
    now = services.clock.now()

    uptime = None if snapshot is None else snapshot.uptime(now)
    last_event: Event | None = None
    if channel is not None:
        history = channel.history()
        last_event = history[-1] if history else None

    return StatusView(
        server_id=services.settings.server.id,
        container=services.settings.server.container,
        state=reducer.state,
        daemon_online=True,
        observed_at=now,
        exists=snapshot is None or snapshot.exists,
        running=snapshot is not None and snapshot.running,
        health=HealthState.UNKNOWN if snapshot is None else snapshot.health,
        health_reported_raw=None if snapshot is None else snapshot.health_reported_raw,
        container_id=reducer.container_id,
        image=None if snapshot is None else snapshot.image,
        started_at=reducer.started_at,
        uptime_seconds=None if uptime is None else uptime.total_seconds(),
        exit_code=None if snapshot is None else snapshot.exit_code,
        oom_killed=snapshot is not None and snapshot.oom_killed,
        restart_count=0 if snapshot is None else snapshot.restart_count,
        version=reducer.version,
        ready_at=reducer.ready_at,
        ready_detected_by=(
            None if reducer.ready_detected_by is None else reducer.ready_detected_by.value
        ),
        players_online=services.roster.count,
        players_max=None if probe is None else probe.players_max,
        roster=build_player_views(services),
        probe=None if probe is None else _probe_view(probe),
        idle=_idle_view(services),
        session=_session_view(services),
        last_event=last_event,
        stop_timeout_seconds=services.controller.stop_timeout_seconds,
        dry_run=services.controller.dry_run,
    )


def build_player_views(services: AppServices) -> tuple[PlayerView, ...]:
    """The roster, with durations measured monotonically rather than from wall clocks."""
    monotonic = services.clock.monotonic()
    return tuple(
        PlayerView(
            name=session.name,
            uuid=session.uuid,
            online_since=session.joined_at,
            session_seconds=session.seconds_at(monotonic),
            first_seen=session.first_seen,
            source=session.source,
        )
        for session in services.roster.sessions.values()
    )


def build_session_views(services: AppServices, *, limit: int) -> Sequence[SessionView]:
    """The session in flight, if there is one.

    Archived records are **not** served here. They are files under ``state.dir`` that
    ``mcmanager sessions`` reads directly, precisely so that history is available when the daemon
    is not - which is exactly when somebody wants to know what the last session looked like.
    """
    current = _session_view(services)
    return [] if current is None or limit < 1 else [current]


def build_readiness_view(services: AppServices) -> ReadinessView:
    """Apply the readiness policy to the daemon's live subsystems."""
    snapshot = services.lifecycle.reducer.snapshot
    running = snapshot is not None and snapshot.running
    manager = services.manager
    attaches = manager.stats.get("log_attaches", 0)
    return evaluate_readiness(
        runtime_available=manager.available,
        discord_enabled=services.settings.discord.enabled,
        discord_connected=services.discord.active,
        # A proxy, and flagged as one: DockerManager counts attaches but does not expose whether a
        # stream is attached *right now*. Combined with `expected`, it is right in the case that
        # matters - a running container whose stream never came up.
        log_stream_attached=manager.available and attaches > 0,
        log_stream_expected=running,
        dropped_critical=services.bus.stats.dropped_critical,
    )


def build_liveness_view(services: AppServices) -> LivenessView:
    """Loop and supervisor only. **Never Docker, never Discord.**

    The HTTP response itself is the proof the loop is turning; what this adds is whether the
    supervisor has given up, which is the one condition where restarting the process is the
    correct response.
    """
    supervisor = services.supervisor
    stopping = supervisor.shutdown_requested.is_set()
    return LivenessView(
        alive=True,
        supervisor_ok=not stopping,
        tasks=len(supervisor.task_names),
        detail="shutdown requested" if stopping else None,
    )


def _probe_view(probe: ProbeResult) -> ProbeView:
    """Flatten the last status probe.

    ``games.base`` is a Protocol plus a DTO with no game-specific content, and
    ``services/lifecycle.py`` already consumes ``ProbeResult`` for the same reason: it is the
    game-agnostic shape of "the server answered". Nothing here imports ``mcstatus``.
    """
    return ProbeView(
        reachable=probe.reachable,
        players_online=probe.players_online,
        players_max=probe.players_max,
        sample=probe.sample,
        sample_is_complete=probe.sample_is_complete,
        version=probe.version,
        motd=probe.motd,
        latency_ms=probe.latency_ms,
        error=probe.error,
        probed_at=probe.probed_at,
    )


def _idle_view(services: AppServices) -> IdleView:
    idle = services.idle
    return IdleView(
        enabled=idle.enabled,
        dry_run=idle.dry_run,
        armed=idle.armed,
        deadline=idle.deadline,
        seconds_remaining=_remaining(idle.deadline, services.clock),
        timeout_seconds=services.settings.idle.timeout_seconds,
        empty_since=idle.empty_since,
    )


def _session_view(services: AppServices) -> SessionView | None:
    session = services.session
    identity = session.identity
    if identity is None:
        return None
    counters = session.counters
    return SessionView(
        id=_session_id(identity.container_id, identity.started_at),
        server_id=services.settings.server.id,
        container_id=identity.container_id,
        started_at=identity.started_at,
        open=session.is_open,
        partial=session.partial,
        duration_seconds=session.uptime_seconds,
        players=tuple(services.roster.online),
        # peak_online stays 0 until the session manager tracks it: a "peak" that is really the
        # current count would be a fabricated number in a summary somebody quotes.
        joins=counters.get("joins", 0),
        leaves=counters.get("leaves", 0),
        deaths=counters.get("deaths", 0),
        advancements=counters.get("advancements", 0),
        chat_messages=counters.get("chat_lines", 0),
    )


def _session_id(container_id: str | None, started_at: datetime) -> str:
    """``(container_id, started_at)`` rendered as one stable string.

    The same identity the session manager uses, so a record on disk and a live ``/status`` name the
    session identically. A ``compose down/up`` changes the container id and therefore the id, which
    is correct: that is a different session.
    """
    short = (container_id or "unknown")[:12]
    return f"{short}-{started_at.strftime('%Y%m%dT%H%M%SZ')}"


def _remaining(deadline: datetime | None, clock: Clock) -> float | None:
    if deadline is None:
        return None
    return max((deadline - clock.now()).total_seconds(), 0.0)
