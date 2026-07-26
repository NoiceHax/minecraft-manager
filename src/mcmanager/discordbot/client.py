"""The gateway client wrapper.

The only place ``import discord`` appears. Wraps connection, reconnect and the guild-scoped command
registration - guild-scoped because it is instant, while global registration takes about an hour
and makes every iteration painful.

The import is **inside** :meth:`DiscordGateway.connect`, not at module scope, so importing this
module costs nothing and ``discord.enabled = false`` really is free. Nothing outside this package
imports it at all.

**Intents are deliberately minimal: ``guilds`` and nothing else.** Slash-command interactions
carry the invoking member and their roles in the payload, so gating ``/stop`` needs neither the
Guild Members nor the Message Content privileged intent. Asking for privileged intents we do not
need would be a permission escalation nobody reviewed, and it would make the bot fail to start
once it reaches a hundred guilds without verification.

``guilds`` itself is **not** privileged and is switched on deliberately: it is what populates the
channel cache. With ``Intents.none()`` discord.py logs *"Guilds intent seems to be disabled"* and
every message has to resolve its channel over HTTP first.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, Final, Protocol, final

import structlog

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    # Type-only, so the runtime import stays inside connect() where it belongs. discord.py ships
    # py.typed, so this buys real checking of the gateway surface rather than a wall of `Any`.
    import discord
    from discord import app_commands

    from mcmanager.clock import Clock

__all__ = [
    "CommandHandler",
    "CommandSpec",
    "DiscordGateway",
    "Gateway",
    "GatewayFactory",
    "InvocationContext",
    "build_gateway",
]

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.discord.gateway")

_CONNECT_TIMEOUT: Final = 60.0
"""How long :meth:`DiscordGateway.connect` waits for the gateway to report ready before giving up
and letting the supervisor retry. Without a bound, a network black hole would hang startup."""


@final
class InvocationContext:
    """What a command handler is told about its invocation.

    Plain integers and strings rather than a discord.py ``Interaction``, so
    :mod:`mcmanager.discordbot.permissions` and the command handlers stay free of the library and
    their tests stay a table.
    """

    __slots__ = ("channel_id", "command", "role_ids", "user_id", "user_name")

    def __init__(
        self,
        *,
        command: str,
        user_id: int,
        user_name: str,
        role_ids: tuple[int, ...],
        channel_id: int,
    ) -> None:
        self.command = command
        self.user_id = user_id
        self.user_name = user_name
        self.role_ids = role_ids
        self.channel_id = channel_id


type CommandHandler = Callable[[InvocationContext], Awaitable[str]]
"""Takes an invocation, returns the message to reply with. Never touches the gateway."""


@final
class CommandSpec:
    """One slash command: its name, its help text, and what to run."""

    __slots__ = ("description", "ephemeral", "handler", "name")

    def __init__(
        self,
        *,
        name: str,
        description: str,
        handler: CommandHandler,
        ephemeral: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.handler = handler
        self.ephemeral = ephemeral


class Gateway(Protocol):
    """What :class:`~mcmanager.discordbot.module.DiscordModule` is allowed to ask of a gateway.

    A Protocol rather than the concrete class so the module depends on the surface rather than on
    discord.py, and so a test fake needs neither the library nor a subclass of a ``@final`` class.
    """

    @property
    def connected(self) -> bool: ...

    def add_command(self, spec: CommandSpec) -> None: ...

    async def connect(self) -> None: ...

    async def register_commands(self, names: Sequence[str]) -> None: ...

    async def send(self, channel_id: int, content: str) -> bool: ...

    async def aclose(self) -> None: ...


class GatewayFactory(Protocol):
    """How :class:`~mcmanager.discordbot.module.DiscordModule` obtains a gateway.

    A factory rather than a direct constructor call, so the tests can substitute a fake without
    discord.py being installed, and so this file stays the only one that knows the library exists.
    """

    def __call__(self, *, token: str, clock: Clock, guild_id: int) -> Gateway: ...


@final
class DiscordGateway:
    """Connection, reconnect and command registration."""

    def __init__(self, *, token: str, clock: Clock, guild_id: int) -> None:
        """Hold the token (never logged), the clock, and the guild to register against."""
        self._token = token
        self._clock = clock
        self._guild_id = guild_id
        self._client: discord.Client | None = None
        self._tree: app_commands.CommandTree[discord.Client] | None = None
        self._runner: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._closing = False
        self._commands: dict[str, CommandSpec] = {}
        self._stats: dict[str, int] = {"sent": 0, "send_failures": 0, "invocations": 0}

    # ------------------------------------------------------------------------ introspection

    @property
    def connected(self) -> bool:
        """True once the gateway has reported ready and has not been closed."""
        return self._ready.is_set() and not self._closing

    @property
    def stats(self) -> dict[str, int]:
        """``sent`` / ``send_failures`` / ``invocations``."""
        return dict(self._stats)

    # ------------------------------------------------------------------------------ commands

    def add_command(self, spec: CommandSpec) -> None:
        """Register a command locally. :meth:`register_commands` syncs it to Discord."""
        self._commands[spec.name] = spec

    # ------------------------------------------------------------------------------ lifecycle

    async def connect(self) -> None:
        """Open the gateway, importing discord.py here and nowhere else.

        Reconnection is discord.py's own; the supervisor restart around this is the outer net for
        the case where the library gives up entirely.
        """
        import discord
        from discord import app_commands

        # Guilds and nothing else. It is NOT a privileged intent, and without it the client keeps
        # no channel cache, so every send falls through get_channel() to an HTTP fetch_channel() -
        # an extra API call per event, and needless rate-limit pressure on a busy server. The two
        # privileged intents stay off: see the module docstring.
        intents = discord.Intents.none()
        intents.guilds = True
        client = discord.Client(intents=intents)
        tree = app_commands.CommandTree(client)
        self._client = client
        self._tree = tree

        @client.event
        async def on_ready() -> None:  # pyright: ignore[reportUnusedFunction]
            # Registered by @client.event, so nothing references it by name. pragma: no cover -
            # it needs a live gateway to fire.
            user = client.user
            _log.info(
                "discord.ready",
                bot=None if user is None else str(user),
                guild_id=self._guild_id,
            )
            self._ready.set()

        for spec in self._commands.values():
            self._attach(tree, spec)

        self._runner = asyncio.create_task(self._run_client(client), name="discord-gateway")
        try:
            async with asyncio.timeout(_CONNECT_TIMEOUT):
                await self._ready.wait()
        except TimeoutError:
            _log.error("discord.connect_timeout", seconds=_CONNECT_TIMEOUT)
            raise

    def _attach(self, tree: app_commands.CommandTree[discord.Client], spec: CommandSpec) -> None:
        """Bind one CommandSpec into discord.py's command tree."""
        import discord

        async def callback(interaction: discord.Interaction) -> None:  # pragma: no cover
            self._stats["invocations"] += 1
            member = interaction.user
            role_ids: tuple[int, ...] = ()
            roles = getattr(member, "roles", None)
            if roles is not None:
                role_ids = tuple(int(r.id) for r in roles)
            ctx = InvocationContext(
                command=spec.name,
                user_id=int(member.id),
                user_name=member.name,
                role_ids=role_ids,
                channel_id=int(interaction.channel_id or 0),
            )
            # Defer first: a controller start can take well over Discord's three-second budget,
            # and a timed-out interaction shows the user "the application did not respond" even
            # though the server is coming up perfectly well.
            await interaction.response.defer(ephemeral=spec.ephemeral, thinking=True)
            try:
                message = await spec.handler(ctx)
            except Exception:
                _log.exception("discord.command_failed", command=spec.name)
                message = ":x: That command failed. The daemon log has the details."
            await interaction.followup.send(message, ephemeral=spec.ephemeral)

        # GroupT is bound to Cog and this command belongs to no cog, so Any is the honest
        # annotation here rather than a lie that happens to type-check.
        command: app_commands.Command[Any, ..., None] = discord.app_commands.Command(
            name=spec.name,
            description=spec.description,
            callback=callback,
        )
        tree.add_command(command, guild=discord.Object(id=self._guild_id))

    async def _run_client(self, client: discord.Client) -> None:
        """discord.py's own connect loop, supervised from outside."""
        try:
            await client.start(self._token)
        except asyncio.CancelledError:
            raise
        except Exception:
            # The token itself is never in the message, but be explicit that we are not logging it.
            _log.exception("discord.gateway_stopped", hint="token is never logged")
            raise

    async def register_commands(self, names: Sequence[str]) -> None:
        """Sync the command tree to the configured guild. Guild-scoped, therefore instant."""
        if self._tree is None:
            raise RuntimeError("connect() must run before register_commands()")
        import discord

        guild = discord.Object(id=self._guild_id)
        synced = await self._tree.sync(guild=guild)
        _log.info(
            "discord.commands_registered",
            guild_id=self._guild_id,
            requested=list(names),
            synced=[c.name for c in synced],
        )

    async def send(self, channel_id: int, content: str) -> bool:
        """Send one already-rendered, already-escaped message. Returns whether it was delivered.

        Escaping is the presenter's job and has already happened before content reaches here. This
        method must never be where a mention is neutralised: then only the messages that happen to
        go through it would be safe.
        """
        if self._client is None or self._closing or not content:
            return False
        import discord

        try:
            channel = self._client.get_channel(channel_id)
            if channel is None:
                channel = await self._client.fetch_channel(channel_id)
            if not isinstance(channel, discord.abc.Messageable):
                # A category or forum id pasted into `discord.channel_id` resolves fine and then
                # has no `send`. Refusing with a named log beats an AttributeError every event.
                self._stats["send_failures"] += 1
                _log.error(
                    "discord.channel_not_messageable",
                    channel_id=channel_id,
                    kind=type(channel).__name__,
                    hint="discord.channel_id must be a text channel, not a category or forum",
                )
                return False
            await channel.send(content)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats["send_failures"] += 1
            _log.warning("discord.send_failed", channel_id=channel_id, error=str(exc))
            return False
        self._stats["sent"] += 1
        return True

    async def aclose(self) -> None:
        """Close the gateway. Called first in the daemon's shutdown sequence."""
        if self._closing:
            return
        self._closing = True
        self._ready.clear()
        client = self._client
        if client is not None:
            try:
                await client.close()
            except Exception as exc:  # pragma: no cover - teardown must never raise
                _log.warning("discord.close_failed", error=str(exc))
        runner = self._runner
        if runner is not None and not runner.done():
            runner.cancel()
            # Both are expected here: CancelledError because we just cancelled it, and anything
            # else because discord.py's connect loop can fail on the way down. Teardown swallows
            # both by design; the send/close failures above are already logged.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await runner
        self._runner = None
        _log.info("discord.gateway_closed", **self._stats)


def build_gateway(*, token: str, clock: Clock, guild_id: int) -> Gateway:
    """The default :class:`GatewayFactory`. Named so tests can point the module elsewhere."""
    return DiscordGateway(token=token, clock=clock, guild_id=guild_id)
