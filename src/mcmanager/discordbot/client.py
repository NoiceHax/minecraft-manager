"""The gateway client wrapper. **File and signatures only; body in M6.**

The only place ``import discord`` appears alongside ``commands.py``. Wraps connection, reconnect
and the guild-scoped command registration - guild-scoped because it is instant, while global
registration takes about an hour and makes every iteration painful.

The import is **inside** :meth:`DiscordGateway.connect`, not at module scope, so importing this
module costs nothing and ``discord.enabled = false`` really is free. Nothing outside this package
imports it at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcmanager.clock import Clock

__all__ = ["DiscordGateway", "GatewayFactory"]


class GatewayFactory(Protocol):
    """How :class:`~mcmanager.discordbot.module.DiscordModule` obtains a gateway.

    A factory rather than a direct constructor call, so M6's tests can substitute a fake without
    discord.py being installed, and so this file stays the only one that knows the library exists.
    """

    def __call__(self, *, token: str, clock: Clock, guild_id: int) -> DiscordGateway: ...


class DiscordGateway:
    """Connection, reconnect and command registration. Every body lands in M6."""

    def __init__(self, *, token: str, clock: Clock, guild_id: int) -> None:
        """Hold the token (never logged), the clock, and the guild to register against."""
        raise NotImplementedError

    async def connect(self) -> None:
        """Open the gateway, importing discord.py here and nowhere else.

        Reconnection is discord.py's own; the supervisor restart around this is the outer net for
        the case where the library gives up entirely.
        """
        raise NotImplementedError

    async def register_commands(self, names: Sequence[str]) -> None:
        """Sync the command tree to the configured guild. Guild-scoped, therefore instant."""
        raise NotImplementedError

    async def send(self, channel_id: int, content: str) -> bool:
        """Send one already-rendered, already-escaped message. Returns whether it was delivered.

        Escaping is the presenter's job and has already happened before content reaches here. This
        method must never be where a mention is neutralised: then only the messages that happen to
        go through it would be safe.
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        """Close the gateway. Called first in the daemon's shutdown sequence."""
        raise NotImplementedError
