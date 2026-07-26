"""Slash commands. **File and signatures only; body in M6.**

``/status`` ``/players`` ``/logs`` land first (read-only), then ``/start`` ``/stop`` ``/restart``.

Every one of them is a thin call into ``ServerController`` or the status surface - the same objects
``mcmanager status`` and ``mcmanager stop`` use. If a handler in this file ever grows a branch on
server state, the logic is in the wrong place.

Each returns a rendered string rather than touching the gateway, which is what makes them testable
without discord.py and what stops "what the bot says" and "what the CLI says" drifting apart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcmanager.core.types import Source
    from mcmanager.services.controller import ControlOutcome, ServerController
    from mcmanager.services.players import PlayerRoster

__all__ = [
    "COMMAND_NAMES",
    "MUTATING_COMMANDS",
    "cmd_logs",
    "cmd_players",
    "cmd_restart",
    "cmd_start",
    "cmd_status",
    "cmd_stop",
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


async def cmd_status(controller: ServerController) -> str:
    """``/status``: state, uptime, players online, current session, idle countdown."""
    raise NotImplementedError


async def cmd_players(roster: PlayerRoster) -> str:
    """``/players``: the roster with session durations.

    Renders names only. ``PlayerJoined.address`` exists on the event because idle logic and abuse
    investigation want it; relaying a player's IP into a chat channel is not acceptable.
    """
    raise NotImplementedError


async def cmd_logs(*, lines: int = 20) -> str:
    """``/logs``: the recent console tail, escaped and truncated to Discord's message limit."""
    raise NotImplementedError


async def cmd_start(controller: ServerController, *, actor: str, via: Source) -> ControlOutcome:
    """``/start``: exactly ``controller.start(...)``, with the Discord user as the actor."""
    raise NotImplementedError


async def cmd_stop(controller: ServerController, *, actor: str, via: Source) -> ControlOutcome:
    """``/stop``: exactly ``controller.stop(...)``, which uses the configured 90-second timeout."""
    raise NotImplementedError


async def cmd_restart(controller: ServerController, *, actor: str, via: Source) -> ControlOutcome:
    """``/restart``: exactly ``controller.restart(...)``, one audit record, two intents."""
    raise NotImplementedError
