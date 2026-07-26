"""Slash commands.

``/status`` ``/players`` ``/logs`` are read-only; ``/start`` ``/stop`` ``/restart`` mutate and are
gated by :mod:`mcmanager.discordbot.permissions`.

Every one of them is a thin call into ``ServerController`` or the status surface - the same objects
``mcmanager status`` and ``mcmanager stop`` use. If a handler in this file ever grows a branch on
server state, the logic is in the wrong place.

Each returns a rendered string (or a ``ControlOutcome``) rather than touching the gateway, which is
what makes them testable without discord.py and what stops "what the bot says" and "what the CLI
says" drifting apart.

**A note on the read-only signatures.** The stubs originally took only a ``ServerController``. They
take providers instead: ``/status`` needs the aggregated
:class:`~mcmanager.control.views.StatusView` that ``control/server.py`` already assembles for the
HTTP surface, and rendering a second, Discord-shaped status from raw pieces is precisely the drift
this package exists to avoid. The mutating signatures are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from mcmanager.core.types import ControlAction, LifecycleState
from mcmanager.discordbot.presenters import (
    MAX_MESSAGE_LENGTH,
    TRUNCATION_MARKER,
    neutralise_mentions,
    sanitize,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mcmanager.control.views import StatusView
    from mcmanager.core.types import Source
    from mcmanager.services.controller import ControlOutcome, ServerController
    from mcmanager.services.players import PlayerRoster

__all__ = [
    "COMMAND_NAMES",
    "MUTATING_COMMANDS",
    "READ_ONLY_COMMANDS",
    "cmd_logs",
    "cmd_players",
    "cmd_restart",
    "cmd_start",
    "cmd_status",
    "cmd_stop",
    "render_outcome",
]

COMMAND_NAMES: tuple[str, ...] = (
    "status",
    "players",
    "logs",
    "start",
    "stop",
    "restart",
)
"""Every command, in registration order. Phase 2 of the cutover registers only the first three."""

MUTATING_COMMANDS: frozenset[str] = frozenset({"start", "stop", "restart"})
"""The ones :mod:`mcmanager.discordbot.permissions` gates. Phase 3 of the cutover."""

READ_ONLY_COMMANDS: frozenset[str] = frozenset(COMMAND_NAMES) - MUTATING_COMMANDS
"""What phase 2 registers. Derived rather than written twice, so the two cannot disagree."""

_STATE_LABEL: Final[dict[LifecycleState, str]] = {
    LifecycleState.UNKNOWN: ":grey_question: unknown",
    LifecycleState.ABSENT: ":ghost: container missing",
    LifecycleState.STOPPED: ":black_circle: stopped",
    LifecycleState.CRASHED: ":boom: crashed",
    LifecycleState.STARTING: ":hourglass: starting",
    LifecycleState.READY: ":green_circle: ready",
    LifecycleState.DEGRADED: ":yellow_circle: degraded",
    LifecycleState.STOPPING: ":octagonal_sign: stopping",
    LifecycleState.BLIND: ":satellite: docker unreachable",
}

_MAX_ROSTER_LINES: Final = 25
_MAX_LOG_LINES: Final = 50


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# ------------------------------------------------------------------------------- read-only


async def cmd_status(
    status: Callable[[], StatusView],
    *,
    idle: Mapping[str, object] | None = None,
) -> str:
    """``/status``: state, uptime, players online, current session, idle countdown.

    Renders the same :class:`~mcmanager.control.views.StatusView` that ``GET /status`` and
    ``mcmanager status`` render, so the three cannot disagree about what the server is doing.
    """
    view = status()
    label = _STATE_LABEL.get(view.state, view.state.value)
    lines = [f"**{sanitize(view.container)}** {label}"]

    if view.running:
        lines.append(f"up {_duration(view.uptime_seconds)}")
        if view.version:
            lines.append(f"version `{sanitize(view.version)}`")
    elif view.exit_code is not None:
        killed = " (out of memory)" if view.oom_killed else ""
        lines.append(f"last exit {view.exit_code}{killed}")

    if view.players_online is not None:
        cap = f"/{view.players_max}" if view.players_max is not None else ""
        lines.append(f"players {view.players_online}{cap}")

    if idle:
        lines.append(_render_idle_line(idle))

    if not view.daemon_online:
        lines.append(":warning: the manager is reporting from a degraded view")

    return "\n".join(lines)


def _render_idle_line(idle: Mapping[str, object]) -> str:
    if not idle.get("enabled"):
        return "idle shutdown: off"
    if not idle.get("armed"):
        return "idle shutdown: on, not counting down"
    suffix = " *(dry run)*" if idle.get("dry_run") else ""
    deadline = idle.get("deadline")
    when = f" at {sanitize(str(deadline))}" if isinstance(deadline, str) else ""
    return f"idle shutdown: counting down{when}{suffix}"


async def cmd_players(roster: PlayerRoster) -> str:
    """``/players``: the roster with session durations.

    Renders names only. ``PlayerJoined.address`` exists on the event because idle logic and abuse
    investigation want it; relaying a player's IP into a chat channel is not acceptable, so the
    roster's ``address`` field is never read here.
    """
    if roster.count == 0:
        return "Nobody is online."

    names = sorted(roster.online)
    shown = names[:_MAX_ROSTER_LINES]
    lines = [f"**{roster.count} online**"]
    for name in shown:
        seconds = roster.session_seconds(name)
        played = f" - {_duration(seconds)}" if seconds is not None else ""
        lines.append(f"- {neutralise_mentions(sanitize(name))}{played}")
    if len(names) > len(shown):
        lines.append(f"...and {len(names) - len(shown)} more")
    return "\n".join(lines)


async def cmd_logs(
    tail: Callable[[int], Sequence[str]],
    *,
    lines: int = 20,
) -> str:
    """``/logs``: the recent console tail, escaped and truncated to Discord's message limit.

    Reads the pipeline's ring buffer rather than re-fetching from Docker: the buffer is already
    populated, already sanitised, and does not cost a round trip on a chat command.
    """
    count = max(1, min(lines, _MAX_LOG_LINES))
    recent = list(tail(count))
    if not recent:
        return "No console output yet."

    # Fences are the only markdown that matters inside a code block, and a line that closes the
    # fence early would let console output style the rest of the message.
    body = "\n".join(neutralise_mentions(sanitize(line)).replace("```", "`​``") for line in recent)
    budget = MAX_MESSAGE_LENGTH - len("```\n\n```")
    if len(body) > budget:
        # Keep the *newest* output and drop the oldest: on a log tail the last lines are the ones
        # somebody asked the question about.
        marker = TRUNCATION_MARKER.strip() + "\n"
        body = marker + body[-(budget - len(marker)) :]
    return f"```\n{body}\n```"


# -------------------------------------------------------------------------------- mutating


async def cmd_start(controller: ServerController, *, actor: str, via: Source) -> ControlOutcome:
    """``/start``: exactly ``controller.start(...)``, with the Discord user as the actor."""
    return await controller.start(actor=actor, via=via, reason="discord /start")


async def cmd_stop(controller: ServerController, *, actor: str, via: Source) -> ControlOutcome:
    """``/stop``: exactly ``controller.stop(...)``, which uses the configured 90-second timeout."""
    return await controller.stop(actor=actor, via=via, reason="discord /stop")


async def cmd_restart(controller: ServerController, *, actor: str, via: Source) -> ControlOutcome:
    """``/restart``: exactly ``controller.restart(...)``, one audit record, two intents."""
    return await controller.restart(actor=actor, via=via, reason="discord /restart")


def render_outcome(outcome: ControlOutcome) -> str:
    """Turn a controller result into something worth reading in a chat channel.

    A refusal is rendered as plainly as an acceptance. "Nothing happened" with no explanation is
    the single most annoying way for a bot to fail.
    """
    verb = {
        ControlAction.START: "Start",
        ControlAction.STOP: "Stop",
        ControlAction.RESTART: "Restart",
    }[outcome.action]

    if outcome.ok:
        note = " *(dry run: nothing was actually done)*" if outcome.dry_run else ""
        return f":white_check_mark: {verb} accepted{note}."
    if outcome.error:
        return f":x: {verb} failed: {neutralise_mentions(sanitize(outcome.error))}"
    reason = outcome.rejection or f"the server is {outcome.state_before.value}"
    return f":no_entry: {verb} refused: {neutralise_mentions(sanitize(reason))}"
