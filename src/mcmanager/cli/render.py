r"""Terminal rendering. **The only module in the project permitted to ``print``.**

Two rules, both enforced rather than described:

- **Rendering is pure and separately tested.** Every ``render_*`` function takes DTOs and returns a
  string. The ``--json`` path skips this module entirely and emits :mod:`mcmanager.core.serde` /
  :mod:`mcmanager.control.views` output, so the CLI and the web API cannot drift. The golden tests
  in ``tests/cli/test_render.py`` compare whole blocks byte for byte.
- **Colour only when the stream is a tty and ``NO_COLOR`` is unset.** This project exists partly
  because ANSI escape codes corrupted player names in a Discord relay; a CLI that pipes escape
  sequences into ``grep`` would be a poor joke. :func:`supports_color` is the single decision
  point, ``Palette.plain()`` is what everything else gets, and there is a test asserting that
  non-tty output contains no ``\x1b`` byte anywhere.

``T20`` (no ``print``) is enforced project-wide, so this file carries the only per-file ignore for
it and nothing in ``core``, ``services``, ``games`` or ``containers`` can print. Two functions do
the writing - :func:`emit` and :func:`emit_error` - and every command goes through them.

Formatting choices worth stating once:

- **All times render as UTC with a trailing ``Z``.** The homelab runs ``Asia/Kolkata`` and the
  server's own log prefix is time-only in that zone; rendering local time here would produce three
  different clocks in one debugging session.
- **Durations are human** (``3h 12m``), because the question is always "how long has it been" and
  never "how many seconds exactly". The ``--json`` path keeps the raw float.
- **Unknown prints as ``unknown``, never as ``0`` or ``-``.** A status view built without a daemon
  is missing real information, and the most dangerous thing this CLI could do is make an absence
  look like a measurement.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import UTC, timedelta
from typing import TYPE_CHECKING, Final, final

from mcmanager.containers.dto import ContainerState, HealthState
from mcmanager.core.events import (
    ChatMessage,
    CommandIssued,
    ConsoleLog,
    PlayerAdvancement,
    PlayerDeath,
    PlayerEvent,
    PlayerJoined,
    PlayerLeft,
    ServerCrashed,
    ServerReady,
    ServerStarting,
    ServerStopped,
    ServerStopping,
)
from mcmanager.core.types import LifecycleState

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from datetime import datetime
    from typing import TextIO

    from mcmanager.containers.dto import ContainerSnapshot
    from mcmanager.control.views import (
        ControlResultView,
        LivenessView,
        PlayerView,
        ReadinessView,
        SessionView,
        StatusView,
    )
    from mcmanager.core.events import Event
    from mcmanager.games.minecraft.parser import ParseStats

__all__ = [
    "NO_COLOR_ENV",
    "Palette",
    "emit",
    "emit_error",
    "format_clock",
    "format_duration",
    "format_timestamp",
    "render_control_result",
    "render_event",
    "render_inspect",
    "render_liveness",
    "render_log_line",
    "render_players",
    "render_readiness",
    "render_replay_summary",
    "render_session",
    "render_sessions",
    "render_status",
    "render_table",
    "supports_color",
]

NO_COLOR_ENV: Final = "NO_COLOR"
"""https://no-color.org - set to anything at all, including the empty string, means no colour."""

_RESET: Final = "\x1b[0m"
_CODES: Final[Mapping[str, str]] = {
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
    "grey": "\x1b[90m",
}

_STATE_COLOURS: Final[Mapping[LifecycleState, str]] = {
    LifecycleState.READY: "green",
    LifecycleState.STARTING: "cyan",
    LifecycleState.STOPPING: "yellow",
    LifecycleState.DEGRADED: "yellow",
    LifecycleState.STOPPED: "grey",
    LifecycleState.ABSENT: "grey",
    LifecycleState.UNKNOWN: "grey",
    LifecycleState.CRASHED: "red",
    LifecycleState.BLIND: "red",
}

_LABEL_WIDTH: Final = 16
_INDENT: Final = "  "
_PAD: Final = " " * (_LABEL_WIDTH + len(_INDENT))
_UNKNOWN: Final = "unknown"


# ------------------------------------------------------------------------------------ colour


def supports_color(
    stream: TextIO | None = None,
    *,
    no_color: bool = False,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Should this stream get ANSI escapes?

    Three ways to say no, and any one of them wins: the ``--no-color`` flag, the ``NO_COLOR``
    environment variable, or the stream not being a terminal. There is deliberately no
    ``FORCE_COLOR`` override: the only reason to force colour into a pipe is a demo, and the cost
    of getting it wrong is escape bytes in a file somebody later parses.
    """
    if no_color:
        return False
    environ = env if env is not None else os.environ
    if NO_COLOR_ENV in environ:
        return False
    target = stream if stream is not None else sys.stdout
    try:
        return bool(target.isatty())
    except (AttributeError, ValueError):
        # A closed or exotic stream. Assume not a terminal, which is the safe direction.
        return False


@final
@dataclass(frozen=True, slots=True)
class Palette:
    """Colour, or the complete absence of it.

    ``Palette.plain()`` is not "colour with the codes blanked out": every method returns its input
    unchanged, so a plain palette cannot emit an escape byte even by accident.
    """

    enabled: bool = False

    @classmethod
    def plain(cls) -> Palette:
        """A palette that never emits an escape sequence."""
        return cls(enabled=False)

    @classmethod
    def for_stream(
        cls,
        stream: TextIO | None = None,
        *,
        no_color: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> Palette:
        """The palette this stream deserves, via :func:`supports_color`."""
        return cls(enabled=supports_color(stream, no_color=no_color, env=env))

    def paint(self, text: str, colour: str) -> str:
        """Wrap ``text`` in ``colour``, or return it untouched when colour is off."""
        if not self.enabled or not text:
            return text
        code = _CODES.get(colour)
        return text if code is None else f"{code}{text}{_RESET}"

    def bold(self, text: str) -> str:
        return self.paint(text, "bold")

    def dim(self, text: str) -> str:
        return self.paint(text, "dim")

    def red(self, text: str) -> str:
        return self.paint(text, "red")

    def green(self, text: str) -> str:
        return self.paint(text, "green")

    def yellow(self, text: str) -> str:
        return self.paint(text, "yellow")

    def cyan(self, text: str) -> str:
        return self.paint(text, "cyan")

    def grey(self, text: str) -> str:
        return self.paint(text, "grey")

    def state(self, state: LifecycleState) -> str:
        """Colour a lifecycle state by severity. ``CRASHED`` and ``BLIND`` are red for a reason."""
        return self.paint(state.value.upper(), _STATE_COLOURS.get(state, "grey"))

    def verdict(self, value: bool) -> str:  # a two-state renderer legitimately takes a bool
        """``ok`` or ``FAIL``, for the health endpoints."""
        return self.green("ok") if value else self.red("FAIL")


# ------------------------------------------------------------------------------------ output


def emit(text: str = "", *, stream: TextIO | None = None, flush: bool = False) -> None:
    """Write one line. The project's only ``print``.

    ``flush`` matters for the two following commands: piped stdout is block-buffered, so
    ``mcmanager events --follow | tee`` would otherwise emit nothing for 4KB, which reads exactly
    like a daemon that has stopped publishing.
    """
    print(text, file=stream if stream is not None else sys.stdout, flush=flush)


def emit_error(text: str) -> None:
    """Write one line to stderr, so ``mcmanager status --json | jq`` is never polluted."""
    print(text, file=sys.stderr)


# --------------------------------------------------------------------------------- formatting


def format_timestamp(value: datetime | None, *, seconds: bool = True) -> str:
    """``2026-07-25 22:58:20Z``. Always UTC, never the local zone."""
    if value is None:
        return _UNKNOWN
    stamped = value.astimezone(UTC)
    pattern = "%Y-%m-%d %H:%M:%S" if seconds else "%Y-%m-%d %H:%M"
    return stamped.strftime(pattern) + "Z"


def format_clock(value: datetime | None) -> str:
    """``22:58:20`` - the time-only form used in event and log listings."""
    if value is None:
        return "--:--:--"
    return value.astimezone(UTC).strftime("%H:%M:%S")


def format_duration(seconds: float | None) -> str:
    """``45s`` / ``12m 30s`` / ``3h 12m`` / ``2d 04h``. ``None`` renders as ``unknown``."""
    if seconds is None:
        return _UNKNOWN
    total = int(max(seconds, 0.0))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def _row(label: str, value: str) -> str:
    return f"{_INDENT}{label.ljust(_LABEL_WIDTH)}{value}".rstrip()


def _or_unknown(value: object) -> str:
    return _UNKNOWN if value is None else str(value)


def _seconds(value: timedelta | None) -> str:
    return _UNKNOWN if value is None else f"{value.total_seconds():.0f}s"


def _health_text(health: HealthState, reported_raw: str | None, pal: Palette) -> str:
    """Render health, and say so out loud when Docker's raw field disagrees.

    This is the user-visible half of the stale-health fix. On the real homelab container an
    *exited* server still reported ``State.Health.Status == "unhealthy"`` with a failing streak of
    zero and five successful probes behind it. Our derived value is ``unknown``, because a
    container that is not running has no health; printing the raw string next to it is what stops
    the next person rediscovering the same surprise from scratch.
    """
    if health is HealthState.HEALTHY:
        text = pal.green(health.value)
    elif health is HealthState.UNHEALTHY:
        text = pal.red(health.value)
    elif health is HealthState.STARTING:
        text = pal.cyan(health.value)
    else:
        text = pal.grey(health.value)
    if reported_raw is not None and reported_raw != health.value:
        text += pal.dim(f"  (docker still reports {reported_raw!r}; that value is stale)")
    return text


# ------------------------------------------------------------------------------------ status


def render_status(view: StatusView, *, palette: Palette | None = None) -> str:
    """The ``mcmanager status`` block.

    A view built standalone - no daemon - carries ``daemon_online=False`` and a set of ``None``
    fields, and both are rendered explicitly. The banner is the caveat on everything under it.
    """
    pal = palette if palette is not None else Palette.plain()
    banner = (
        pal.green("daemon: online")
        if view.daemon_online
        else pal.yellow("daemon: offline (standalone: container inspect + status probe only)")
    )
    lines: list[str] = [f"{pal.bold(view.server_id)}  {pal.state(view.state)}    {banner}"]

    container = view.container
    if view.container_id:
        container += pal.dim(f" ({view.container_id[:12]})")
    if not view.exists:
        container += pal.yellow("  [no such container]")
    lines.append(_row("container", container))
    if view.image:
        lines.append(_row("image", view.image))
    lines.append(_row("health", _health_text(view.health, view.health_reported_raw, pal)))

    uptime = format_duration(view.uptime_seconds) if view.running else "not running"
    since = pal.dim(f"  since {format_timestamp(view.started_at)}") if view.started_at else ""
    lines.append(_row("uptime", f"{uptime}{since}"))

    if not view.running and (view.exit_code is not None or view.oom_killed):
        exit_text = _or_unknown(view.exit_code)
        if view.oom_killed:
            exit_text += pal.red("  OOM-killed")
        lines.append(_row("exit", exit_text))

    if view.version or view.ready_detected_by or view.startup_seconds is not None:
        ready = view.version or _UNKNOWN
        if view.startup_seconds is not None:
            ready += f"   ready in {view.startup_seconds:.1f}s"
        if view.ready_detected_by:
            ready += pal.dim(f" (via {view.ready_detected_by})")
        lines.append(_row("version", ready))

    lines.extend(_status_players(view, pal))
    lines.extend(_status_probe(view, pal))
    lines.extend(_status_idle(view, pal))
    lines.extend(_status_session(view, pal))

    if view.last_event is not None:
        lines.append(_row("last event", render_event(view.last_event, palette=pal, timestamp=True)))
    if view.dry_run:
        lines.append(_row("dry run", pal.yellow("commands are recorded and never performed")))
    lines.extend(_row("note", pal.yellow(note)) for note in view.notes)
    return "\n".join(lines)


def _status_players(view: StatusView, pal: Palette) -> list[str]:
    if view.players_online is None and not view.roster:
        return [_row("players", pal.grey(_UNKNOWN))]
    count = view.players_online if view.players_online is not None else len(view.roster)
    capacity = f"/{view.players_max}" if view.players_max is not None else ""
    lines = [_row("players", f"{count}{capacity}")]
    for player in view.roster:
        detail = format_duration(player.session_seconds) if player.session_seconds else ""
        marks: list[str] = []
        if player.first_seen:
            marks.append("first visit")
        if player.source != "log":
            marks.append(f"seen via {player.source}")
        suffix = pal.dim("  (" + ", ".join(marks) + ")") if marks else ""
        lines.append(f"{_PAD}{player.name.ljust(18)}{detail}{suffix}".rstrip())
    return lines


def _status_probe(view: StatusView, pal: Palette) -> list[str]:
    probe = view.probe
    if probe is None:
        return []
    if not probe.reachable:
        # "unreachable" and "empty" are different facts, and conflating them is precisely the bug
        # that stops a populated server. The word "unknown" is doing real work here.
        detail = pal.yellow("unreachable") + pal.dim("  - player count unknown, not zero")
        if probe.error:
            detail += pal.dim(f": {probe.error}")
        return [_row("probe", detail)]
    parts = [pal.green("reachable")]
    if probe.latency_ms is not None:
        parts.append(f"{probe.latency_ms:.0f}ms")
    if probe.motd:
        parts.append(f'"{probe.motd}"')
    if probe.players_online and not probe.sample_is_complete:
        parts.append(pal.dim("(sample truncated; roster not reconciled from it)"))
    return [_row("probe", "  ".join(parts))]


def _status_idle(view: StatusView, pal: Palette) -> list[str]:
    idle = view.idle
    if idle is None:
        return []
    if not idle.enabled:
        return [_row("idle", pal.grey("disabled"))]
    if not idle.armed:
        window = format_duration(idle.timeout_seconds)
        return [_row("idle", f"enabled, not armed ({window} window)")]
    text = f"stops in {format_duration(idle.seconds_remaining)}"
    text += pal.dim(f" at {format_timestamp(idle.deadline)}")
    if idle.dry_run:
        text += pal.yellow("  [dry run: nothing will be stopped]")
    return [_row("idle", text)]


def _status_session(view: StatusView, pal: Palette) -> list[str]:
    session = view.session
    if session is None:
        return []
    state = pal.green("open") if session.open else "closed"
    if session.partial:
        state += pal.yellow(" partial")
    counts = (
        f"joins {session.joins}  deaths {session.deaths}  "
        f"chat {session.chat_messages}  peak {session.peak_online}"
    )
    return [
        _row("session", f"{format_timestamp(session.started_at, seconds=False)}  {state}"),
        _row("", pal.dim(counts)),
    ]


# ----------------------------------------------------------------------------------- players


def render_players(
    players: Sequence[PlayerView],
    *,
    palette: Palette | None = None,
    daemon_online: bool = True,
) -> str:
    """The ``mcmanager players`` table."""
    pal = palette if palette is not None else Palette.plain()
    lines: list[str] = []
    if not daemon_online:
        lines.append(pal.yellow("daemon: offline - names come from the status probe's sample only"))
    if not players:
        lines.append("nobody is online")
        return "\n".join(lines)

    lines.append(pal.bold(f"{'player'.ljust(18)}{'online for'.ljust(12)}{'since'.ljust(24)}source"))
    for player in players:
        first = "  first visit" if player.first_seen else ""
        lines.append(
            f"{player.name.ljust(18)}"
            f"{format_duration(player.session_seconds).ljust(12)}"
            f"{format_timestamp(player.online_since).ljust(24)}"
            f"{player.source}{first}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------------- sessions


def render_sessions(
    sessions: Sequence[SessionView],
    *,
    now: datetime,
    palette: Palette | None = None,
) -> str:
    """The ``mcmanager sessions`` list, newest first."""
    pal = palette if palette is not None else Palette.plain()
    if not sessions:
        return "no session records yet"
    lines = [
        pal.bold(
            f"{'started'.ljust(20)}{'duration'.ljust(12)}{'players'.ljust(9)}"
            f"{'deaths'.ljust(8)}{'chat'.ljust(7)}id"
        )
    ]
    for session in sessions:
        flags: list[str] = []
        if session.open:
            flags.append("open")
        if session.partial:
            flags.append("partial")
        marker = pal.yellow("  " + " ".join(flags)) if flags else ""
        lines.append(
            f"{format_timestamp(session.started_at, seconds=False).ljust(20)}"
            f"{format_duration(session.elapsed_seconds(now)).ljust(12)}"
            f"{str(len(session.players)).ljust(9)}"
            f"{str(session.deaths).ljust(8)}"
            f"{str(session.chat_messages).ljust(7)}"
            f"{session.id}{marker}"
        )
    return "\n".join(lines)


def render_session(
    session: SessionView,
    *,
    now: datetime,
    palette: Palette | None = None,
) -> str:
    """One session in full, for ``mcmanager sessions --id <id>``."""
    pal = palette if palette is not None else Palette.plain()
    lines = [pal.bold(f"session {session.id}")]
    lines.append(_row("server", session.server_id or _UNKNOWN))
    lines.append(_row("container", _or_unknown(session.container_id)))
    lines.append(_row("started", format_timestamp(session.started_at)))
    lines.append(_row("ended", format_timestamp(session.ended_at) if session.ended_at else "open"))
    lines.append(_row("duration", format_duration(session.elapsed_seconds(now))))
    if session.partial:
        lines.append(_row("partial", pal.yellow("yes - the daemon missed part of this session,")))
        lines.append(_row("", pal.yellow("so these counts are incomplete and say so")))
    lines.append(_row("players", ", ".join(session.players) if session.players else "nobody"))
    lines.append(_row("peak online", str(session.peak_online)))
    lines.append(
        _row(
            "counts",
            f"joins {session.joins}  leaves {session.leaves}  deaths {session.deaths}  "
            f"advancements {session.advancements}  chat {session.chat_messages}",
        )
    )
    if session.stop_reason or session.stopped_by:
        who = f" by {session.stopped_by}" if session.stopped_by else ""
        lines.append(_row("stopped", f"{session.stop_reason or _UNKNOWN}{who}"))
    if session.exit_code is not None or session.clean is not None:
        clean = ""
        if session.clean is not None:
            clean = "  clean" if session.clean else pal.red("  unclean")
        lines.append(_row("exit", f"{_or_unknown(session.exit_code)}{clean}"))
    if session.archive:
        lines.append(_row("archive", session.archive))
    return "\n".join(lines)


# ----------------------------------------------------------------------------------- inspect


def render_inspect(
    snapshot: ContainerSnapshot,
    *,
    palette: Palette | None = None,
    extra: Mapping[str, str] | None = None,
) -> str:
    """``mcmanager inspect``: the raw truth from Docker, with health **derived**.

    This block is the user-visible demonstration of the stale-health fix. Run it against the real
    homelab container while the server is stopped and it prints ``health  unknown  (docker still
    reports 'unhealthy'; that value is stale)`` - the finding that made
    :attr:`~mcmanager.containers.dto.ContainerSnapshot.health` a derived property in the first
    place.

    ``extra`` carries runtime-specific diagnostics, notably which log-stream code path is live,
    which this function passes through without knowing what they mean.
    """
    pal = palette if palette is not None else Palette.plain()
    lines = [f"{pal.bold(snapshot.name)}  {_state_text(snapshot.state, pal)}"]

    if snapshot.absent:
        lines.append(_row("absent", "no container with this name exists. That is a legitimate"))
        lines.append(_row("", "state: mcmanager owns a container's lifecycle, not its existence."))
        lines.append(_row("observed", format_timestamp(snapshot.observed_at)))
        return "\n".join(lines)

    lines.append(_row("id", _or_unknown(snapshot.id)))
    lines.append(_row("image", _or_unknown(snapshot.image)))
    lines.append(_row("status", _or_unknown(snapshot.status_text)))
    lines.append(_row("running", "yes" if snapshot.running else "no"))
    lines.append(_row("health", _health_text(snapshot.health, snapshot.health_reported_raw, pal)))
    if snapshot.health_failing_streak is not None:
        lines.append(_row("  streak", str(snapshot.health_failing_streak)))

    timing = snapshot.health_timing
    if timing.interval is not None or timing.start_period is not None or timing.retries is not None:
        lines.append(
            _row(
                "healthcheck",
                f"interval {_seconds(timing.interval)}  timeout {_seconds(timing.timeout)}  "
                f"start_period {_seconds(timing.start_period)}  "
                f"retries {_or_unknown(timing.retries)}",
            )
        )
        guard = pal.dim("  = start_period + interval * (retries + 1), read from the container")
        lines.append(_row("  guard", f"{_seconds(snapshot.guard_window)}{guard}"))

    lines.append(_row("exit code", _or_unknown(snapshot.exit_code)))
    lines.append(_row("oom killed", pal.red("yes") if snapshot.oom_killed else "no"))
    tty_note = "" if snapshot.tty else pal.dim("  (the log stream is multiplexed frames, not raw)")
    lines.append(_row("tty", f"{snapshot.tty}{tty_note}"))
    lines.append(_row("restarts", str(snapshot.restart_count)))
    lines.append(_row("restart policy", _or_unknown(snapshot.restart_policy)))
    lines.append(_row("stop signal", _or_unknown(snapshot.stop_signal)))
    lines.append(_row("created", format_timestamp(snapshot.created_at)))
    lines.append(_row("started", format_timestamp(snapshot.started_at)))
    lines.append(_row("finished", format_timestamp(snapshot.finished_at)))

    driver = _or_unknown(snapshot.log_driver)
    if snapshot.log_options:
        caps = " ".join(f"{key}={value}" for key, value in sorted(snapshot.log_options.items()))
        driver += f"  {caps}"
    if snapshot.log_driver == "json-file":
        driver += pal.dim("  (a ring, not an archive)")
    lines.append(_row("log driver", driver))

    if snapshot.networks:
        lines.append(_row("networks", ""))
        for network in snapshot.networks:
            aliases = ", ".join(network.aliases) if network.aliases else "-"
            lines.append(
                f"{_PAD}{network.name.ljust(20)}"
                f"{(network.ip_address or '-').ljust(18)}aliases: {aliases}"
            )
    else:
        lines.append(_row("networks", pal.yellow("none - SLP and RCON by DNS name cannot work")))

    if snapshot.mounts:
        lines.append(_row("mounts", ""))
        for mount in snapshot.mounts:
            mode = "rw" if mount.rw else pal.yellow("ro")
            lines.append(f"{_PAD}{mount.source} -> {mount.destination}  {mount.kind} {mode}")

    compose = {k: v for k, v in snapshot.labels.items() if k.startswith("com.docker.compose")}
    if compose:
        lines.append(_row("compose", ""))
        lines.extend(f"{_PAD}{key} = {compose[key]}" for key in sorted(compose))

    lines.extend(_row(key, value) for key, value in sorted((extra or {}).items()))
    lines.append(_row("observed", format_timestamp(snapshot.observed_at)))
    return "\n".join(lines)


def _state_text(state: ContainerState, pal: Palette) -> str:
    if state is ContainerState.RUNNING:
        return pal.green(state.value.upper())
    if state in (ContainerState.DEAD, ContainerState.ABSENT):
        return pal.red(state.value.upper())
    return pal.grey(state.value.upper())


# ------------------------------------------------------------------------------------ events


def render_event(event: Event, *, palette: Palette | None = None, timestamp: bool = False) -> str:
    """One event as one line.

    Deliberately compact and deliberately lossy: ``--json`` exists for the whole record. What this
    has to get right is that somebody tailing ``mcmanager events --follow`` sees a join, a death
    and a crash go past without reading JSON.

    **Player names print as-is and are already sanitised.** The parser strips ANSI before a name is
    ever captured, and the palette wraps colour *around* the rendered field rather than inside it.
    That ordering is the entire point: the reverse is the defect that produced this project.
    """
    pal = palette if palette is not None else Palette.plain()
    head = f"{format_clock(event.ts)}  " if timestamp else ""
    return f"{head}{pal.dim(event.name.ljust(18))} {_event_body(event, pal)}"


def _event_body(event: Event, pal: Palette) -> str:
    match event:
        case ChatMessage():
            return f"<{event.player.name}> {event.message}"
        case PlayerDeath():
            suffix = pal.dim("  (unmatched template)") if event.template is None else ""
            return pal.red(event.message) + suffix
        case PlayerAdvancement():
            return pal.cyan(f"{event.player.name} earned {event.kind.value} [{event.title}]")
        case PlayerJoined():
            first = pal.dim("  (first visit)") if event.first_seen else ""
            # PlayerJoined.address is a client IP. It is never rendered, here or anywhere.
            return (
                pal.green(f"{event.player.name} joined") + f"  online {event.online_count}" + first
            )
        case PlayerLeft():
            took = (
                f" after {format_duration(event.session_seconds)}" if event.session_seconds else ""
            )
            return (
                pal.yellow(f"{event.player.name} left")
                + f" ({event.reason.value}){took}  online {event.online_count}"
            )
        case PlayerEvent():
            return event.player.name
        case ServerReady():
            took = f" in {event.startup_seconds:.1f}s" if event.startup_seconds else ""
            return pal.green(f"server ready{took} (via {event.detected_by.value})")
        case ServerStarting():
            return pal.cyan(f"server starting {event.version or ''}".rstrip())
        case ServerStopping():
            return pal.yellow(f"server stopping: {event.reason or 'no reason given'}")
        case ServerStopped():
            state = "clean" if event.clean else "unclean"
            forced = ", SIGKILLed" if event.forced else ""
            return f"server stopped ({state}, exit {_or_unknown(event.exit_code)}{forced})"
        case ServerCrashed():
            oom = ", OOM-killed" if event.oom_killed else ""
            return pal.red(f"server crashed (exit {_or_unknown(event.exit_code)}{oom})")
        case CommandIssued():
            verdict = "accepted" if event.accepted else f"rejected: {event.rejection}"
            return f"{event.action.value} by {event.actor} via {event.via.value} - {verdict}"
        case ConsoleLog():
            return event.message
        case _:
            return event.raw or ""


def render_log_line(
    *,
    ts: datetime | None,
    message: str,
    level: str | None = None,
    thread: str | None = None,
    palette: Palette | None = None,
    show_time: bool = True,
) -> str:
    """One console line, for ``mcmanager logs``.

    ``--raw`` bypasses this function entirely and writes the original text, which is what makes it
    useful for diagnosing the parser itself.
    """
    pal = palette if palette is not None else Palette.plain()
    prefix = f"{format_clock(ts)}  " if show_time else ""
    tag = pal.dim(f"[{thread or '?'}/{level or '?'}] ") if (thread or level) else ""
    body = message
    if level in ("ERROR", "FATAL"):
        body = pal.red(message)
    elif level == "WARN":
        body = pal.yellow(message)
    return f"{prefix}{tag}{body}"


# ------------------------------------------------------------------------------------- health


def render_readiness(view: ReadinessView, *, palette: Palette | None = None) -> str:
    """The ``/readyz`` report, one line per subsystem.

    Per-subsystem output is the point of the endpoint: "not ready" without a name means somebody
    has to go and check four things by hand.
    """
    pal = palette if palette is not None else Palette.plain()
    header = pal.green("ready") if view.ready else pal.red("NOT ready")
    lines = [f"readiness: {header}"]
    for check in view.checks:
        note = "" if check.required else pal.dim(" (advisory)")
        detail = f"  {check.detail}" if check.detail else ""
        lines.append(f"  {check.name.ljust(12)}{pal.verdict(check.ok)}{note}{detail}".rstrip())
    return "\n".join(lines)


def render_liveness(view: LivenessView, *, palette: Palette | None = None) -> str:
    """The ``/healthz`` report."""
    pal = palette if palette is not None else Palette.plain()
    lines = [f"liveness: {pal.verdict(view.alive and view.supervisor_ok)}"]
    lines.append(_row("supervisor", pal.verdict(view.supervisor_ok)))
    lines.append(_row("tasks", str(view.tasks)))
    if view.loop_lag_seconds is not None:
        lines.append(_row("loop lag", f"{view.loop_lag_seconds * 1000:.1f}ms"))
    if view.detail:
        lines.append(_row("detail", view.detail))
    return "\n".join(lines)


def render_control_result(result: ControlResultView, *, palette: Palette | None = None) -> str:
    """The one-line answer to ``mcmanager start`` / ``stop`` / ``restart``."""
    pal = palette if palette is not None else Palette.plain()
    if not result.accepted:
        return pal.yellow(f"{result.action} refused: {result.rejection or 'no reason given'}")
    if result.error is not None:
        return pal.red(f"{result.action} failed: {result.error}")
    if result.dry_run:
        return pal.yellow(f"{result.action} recorded (dry run; nothing was done)")
    return pal.green(result.message or f"{result.action} ok")


# ------------------------------------------------------------------------------------ replay


def render_replay_summary(
    stats: ParseStats,
    *,
    source: str,
    palette: Palette | None = None,
    threshold: float = 0.01,
) -> str:
    """The report ``mcmanager replay`` prints after the event stream.

    The unrecognised ratio is the canary for a Paper upgrade quietly breaking the patterns. A
    parser that stops matching does not crash - it goes silent - so this number, and the samples
    beneath it, are the only warning anybody gets.
    """
    pal = palette if palette is not None else Palette.plain()
    ratio = stats.unrecognised_ratio
    verdict = pal.green("ok") if ratio <= threshold else pal.red("OVER THRESHOLD")
    context = pal.dim("  (context only; about 0.9 is normal)")
    lines = [
        pal.bold(f"replay summary: {source}"),
        _row("lines", str(stats.total)),
        _row("server lines", str(stats.server_lines)),
        _row("players", str(len(stats.known_players))),
        _row("player lines", str(stats.player_lines)),
        _row("unrecognised", f"{stats.unrecognised_player_lines}  ratio {ratio:.4f}  {verdict}"),
        _row("console ratio", f"{stats.console_ratio:.4f}{context}"),
    ]
    if stats.by_event:
        lines.append(_row("events", ""))
        ordered = sorted(stats.by_event, key=lambda key: (-stats.by_event[key], key))
        lines.extend(f"{_PAD}{name.ljust(20)}{stats.by_event[name]}" for name in ordered)
    if stats.samples:
        lines.append(_row("samples", pal.yellow("named a known player and matched nothing:")))
        lines.extend(f"{_PAD}{sample}" for sample in stats.samples)
    return "\n".join(lines)


def render_table(rows: Iterable[tuple[str, str]]) -> str:
    """A two-column block, for anything with plain key/value output."""
    return "\n".join(_row(key, value) for key, value in rows)
