"""``mcmanager status`` and ``mcmanager players``.

Prefers the daemon; falls back to a standalone view - one container inspect plus one status probe -
behind an explicit ``daemon: offline`` banner, so nobody mistakes standalone numbers for live ones.

Both paths produce the same :class:`~mcmanager.control.views.StatusView` that ``/status``
serialises and that Discord's ``/status`` embed will be built from. If those three ever disagree,
the view is the bug and there is one place to fix it.

**What the standalone view cannot know, it reports as ``None``, never as zero.** No session, no
idle countdown, no per-player durations, and a lifecycle state that is *inferred* from the
container rather than observed by the reducer. Every one of those is called out in
:attr:`~mcmanager.control.views.StatusView.notes`, because a status view that silently downgrades
its own confidence is how somebody ends up acting on a number that was never measured.

The fallback is only ever for the *reads*. ``events`` and the control commands refuse to degrade
at all - see :mod:`mcmanager.cli.cmd_events` and :mod:`mcmanager.cli.cmd_control`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcmanager.cli import render
from mcmanager.containers.dto import ContainerState
from mcmanager.containers.errors import RuntimeUnavailableError
from mcmanager.containers.factory import build_runtime
from mcmanager.control.views import PlayerView, ProbeView, StatusView
from mcmanager.core.types import LifecycleState
from mcmanager.errors import EXIT_OK, EXIT_UNAVAILABLE, DaemonUnreachableError

if TYPE_CHECKING:
    from datetime import datetime

    from mcmanager.cli.main import CliContext
    from mcmanager.containers.dto import ContainerSnapshot
    from mcmanager.games.base import ProbeResult

__all__ = ["run", "standalone_status"]

_CLEAN_EXITS = (0, 143)
"""Verified on this container: ``mc-server-runner`` traps SIGTERM, writes ``stop`` and exits 0."""


async def run(ctx: CliContext, *, players_only: bool = False, standalone: bool = False) -> int:
    """Print the status block, or just the roster when ``players_only``."""
    view: StatusView | None = None
    payload: dict[str, object] | None = None

    if not standalone:
        view, payload = await _from_daemon(ctx, players_only=players_only)

    if view is None:
        try:
            view = await standalone_status(ctx)
        except RuntimeUnavailableError as exc:
            from mcmanager.config import runtime_unreachable_message

            render.emit_error(runtime_unreachable_message(endpoint=None, error=str(exc)))
            return EXIT_UNAVAILABLE
        payload = None

    if ctx.json_output:
        if players_only:
            body: dict[str, object] = {
                "online": len(view.roster),
                "daemon_online": view.daemon_online,
                "players": [player.to_dict() for player in view.roster],
            }
            render.emit(json.dumps(body, indent=2, sort_keys=True))
        else:
            render.emit(json.dumps(payload or view.to_dict(), indent=2, sort_keys=True))
        return EXIT_OK

    if players_only:
        render.emit(
            render.render_players(
                view.roster,
                palette=ctx.palette,
                daemon_online=view.daemon_online,
            )
        )
    else:
        render.emit(render.render_status(view, palette=ctx.palette))
    return EXIT_OK


async def _from_daemon(
    ctx: CliContext,
    *,
    players_only: bool,
) -> tuple[StatusView | None, dict[str, object] | None]:
    """Ask the daemon. Returns ``(None, None)`` when it is not there, which means fall back.

    The ``--json`` path returns the daemon's **own bytes**, decoded but not rebuilt, so that
    ``mcmanager status --json`` and ``curl /status`` cannot produce different documents.
    """
    from mcmanager.cli.client import DaemonClient

    async with DaemonClient(ctx.url, token=ctx.token) as client:
        try:
            payload = await client.get_json("/status")
        except DaemonUnreachableError:
            render.emit_error(f"daemon: offline at {ctx.url} - falling back to a standalone view")
            return None, None
        view = StatusView.from_dict(payload)
        if players_only and not view.roster:
            # /status carries the roster, but /players is the endpoint that owns it; ask there
            # too so a daemon that populates one and not the other is visibly wrong rather than
            # quietly empty.
            view = _replace_roster(view, await client.players())
        return view, dict(payload)


def _replace_roster(view: StatusView, roster: tuple[PlayerView, ...]) -> StatusView:
    from dataclasses import replace

    return replace(view, roster=roster)


async def standalone_status(ctx: CliContext) -> StatusView:
    """Build a status view from one container inspect plus one status probe.

    No daemon, no bus, no supervisor. This is the view that answers "is the thing up" from a
    laptop with ``DOCKER_HOST`` pointed at the homelab, and it is deliberately honest about being
    a snapshot rather than an observation.
    """
    from mcmanager.games.registry import get_adapter

    settings = ctx.settings
    runtime = build_runtime(settings.runtime, clock=ctx.clock, docker_host=settings.docker.host)
    try:
        snapshot = await runtime.inspect(settings.server.container)
    finally:
        await runtime.aclose()

    probe: ProbeResult | None = None
    if settings.probe.enabled and snapshot.running:
        adapter = get_adapter(settings.server.game, clock=ctx.clock)
        probe = await adapter.probe(
            settings.server.host,
            settings.server.port,
            timeout=settings.probe.timeout_seconds,
        )
    return build_standalone_view(
        snapshot,
        probe,
        server_id=settings.server.id,
        now=ctx.clock.now(),
        stop_timeout_seconds=settings.server.lifecycle.stop_timeout_seconds,
    )


def build_standalone_view(
    snapshot: ContainerSnapshot,
    probe: ProbeResult | None,
    *,
    server_id: str,
    now: datetime,
    stop_timeout_seconds: int | None = None,
) -> StatusView:
    """Assemble the standalone view. Pure: everything it needs is an argument.

    Separated from :func:`standalone_status` so the mapping - which is the part with judgement in
    it - is testable without a runtime, a probe or a clock.
    """
    state = _infer_state(snapshot, probe)
    uptime = snapshot.uptime(now)
    notes = [
        "state is inferred from the container and one probe, not observed by the lifecycle "
        "reducer; start the daemon for the real state machine",
        "session, idle countdown and per-player durations are unavailable without the daemon",
    ]
    roster: tuple[PlayerView, ...] = ()
    probe_view: ProbeView | None = None
    if probe is not None:
        probe_view = ProbeView(
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
        # A sample is names only: no join time, no duration, and saying "seen via probe" is the
        # honest label for that. A truncated sample is worse still, so it is called out.
        roster = tuple(PlayerView(name=name, source="probe") for name in probe.sample)
        if probe.reachable and not probe.sample_is_complete:
            notes.append(
                "the status probe's player sample is truncated or randomised, so this roster is "
                "incomplete - the count is authoritative, the names are not"
            )
        if not probe.reachable:
            notes.append("the status probe failed: the player count is UNKNOWN, not zero")

    return StatusView(
        server_id=server_id,
        container=snapshot.name,
        state=state,
        daemon_online=False,
        observed_at=snapshot.observed_at,
        exists=snapshot.exists,
        running=snapshot.running,
        health=snapshot.health,
        health_reported_raw=snapshot.health_reported_raw,
        container_id=snapshot.id,
        image=snapshot.image,
        started_at=snapshot.started_at,
        uptime_seconds=None if uptime is None else uptime.total_seconds(),
        exit_code=snapshot.exit_code,
        oom_killed=snapshot.oom_killed,
        restart_count=snapshot.restart_count,
        version=None if probe is None else probe.version,
        players_online=None if probe is None else probe.players_online,
        players_max=None if probe is None else probe.players_max,
        roster=roster,
        probe=probe_view,
        idle=None,
        session=None,
        last_event=None,
        stop_timeout_seconds=stop_timeout_seconds,
        notes=tuple(notes),
    )


def _infer_state(snapshot: ContainerSnapshot, probe: ProbeResult | None) -> LifecycleState:
    """Map a snapshot plus a probe onto a lifecycle state, conservatively.

    Mirrors the reducer's ``_observe`` without importing it: the reducer's judgement depends on
    history this command does not have, and calling into it with a single snapshot would produce a
    state that looks authoritative and is not. When in doubt this returns ``STARTING`` rather than
    ``READY`` - claiming a server is up when it is still loading chunks is the worse error.
    """
    if snapshot.absent:
        return LifecycleState.ABSENT
    if snapshot.running:
        from mcmanager.containers.dto import HealthState

        if probe is not None and probe.reachable:
            return LifecycleState.READY
        if snapshot.health is HealthState.HEALTHY:
            return LifecycleState.READY
        if snapshot.health is HealthState.UNHEALTHY:
            return LifecycleState.DEGRADED
        return LifecycleState.STARTING
    if snapshot.state is ContainerState.CREATED:
        return LifecycleState.STOPPED
    if snapshot.oom_killed:
        return LifecycleState.CRASHED
    if snapshot.exit_code in _CLEAN_EXITS:
        return LifecycleState.STOPPED
    return LifecycleState.CRASHED
