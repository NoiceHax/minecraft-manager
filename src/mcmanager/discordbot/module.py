"""The Discord subsystem's lifecycle.

Three modes: ``live``, ``dryrun`` (logs what it would send, sends nothing - the shadow-run setting)
and ``disabled`` (the gateway is never opened). A cross-field config validator refuses
``runtime = "fake"`` together with ``mode = "live"``, so fake events cannot reach a real channel.

Subscriptions are ``CONCURRENT``: a Discord round trip is network I/O and must never sit on the
bus's sequential path.

**No ``import discord`` at module scope, here or anywhere in this package.** The gateway import
lives inside :meth:`DiscordGateway.connect`, reached only from the ``live`` branch of
:meth:`DiscordModule.start`. That is what keeps ``mcmanager replay``, ``mcmanager inspect`` and
every service test free of discord.py, and it is also what makes ``discord.enabled = false``
genuinely cost nothing.

Discord is a **client of** :class:`~mcmanager.services.controller.ServerController` and the status
surface, never a peer of them. If a handler in this package ever grows a branch on server state,
the logic is in the wrong file.

**``dryrun`` is a real mode, not a debug flag.** It renders every message through the same
presenters the live path uses and logs the result, so the shadow run compares rendered output
against ``latest.log`` rather than against intentions. The only difference between ``dryrun`` and
``live`` is the final ``gateway.send``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, final

import structlog

from mcmanager.core.events import Event
from mcmanager.core.types import ControlAction, DispatchMode, Source
from mcmanager.discordbot import commands as cmds
from mcmanager.discordbot.client import CommandSpec, InvocationContext, build_gateway
from mcmanager.discordbot.console_relay import ConsoleRelay
from mcmanager.discordbot.permissions import RateLimiter, check_admin
from mcmanager.discordbot.presenters import render_event

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mcmanager.clock import Clock
    from mcmanager.control.views import StatusView
    from mcmanager.core.bus import EventBus, Subscription
    from mcmanager.core.types import ServerId
    from mcmanager.discordbot.client import CommandHandler, Gateway, GatewayFactory
    from mcmanager.services.controller import ControlOutcome, ServerController
    from mcmanager.services.players import PlayerRoster

__all__ = ["DiscordMode", "DiscordModule"]

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.discord")

type DiscordMode = str
"""``"live"``, ``"dryrun"`` or ``"disabled"``. A plain string alias rather than an enum because it
comes straight from ``discord.mode`` in the TOML and crosses no other boundary."""

MODE_LIVE = "live"
MODE_DRYRUN = "dryrun"
MODE_DISABLED = "disabled"

_DESCRIPTIONS: Mapping[str, str] = {
    "status": "Show the server's state, uptime and player count",
    "players": "List who is online right now",
    "logs": "Show the recent server console output",
    "start": "Start the Minecraft server (admin only)",
    "stop": "Stop the Minecraft server gracefully (admin only)",
    "restart": "Restart the Minecraft server (admin only)",
}

_ACTIONS: Mapping[str, ControlAction] = {
    "start": ControlAction.START,
    "stop": ControlAction.STOP,
    "restart": ControlAction.RESTART,
}


class TokenProvider(Protocol):
    """Anything that can hand over the bot token without it having been a ``str`` on the way.

    ``pydantic.SecretStr`` satisfies this. Typed structurally so this module needs no pydantic
    import and no plain-string field that a stray ``log.info("config", cfg=...)`` could leak.
    """

    def get_secret_value(self) -> str: ...


@final
class DiscordModule:
    """Owns the gateway connection, the slash commands and the relays."""

    def __init__(
        self,
        *,
        clock: Clock,
        server_id: ServerId,
        controller: ServerController,
        roster: PlayerRoster,
        status: Callable[[], StatusView] | None = None,
        console_tail: Callable[[int], Sequence[str]] | None = None,
        idle_describe: Callable[[], Mapping[str, object]] | None = None,
        mode: DiscordMode = MODE_DISABLED,
        enabled: bool = False,
        token: TokenProvider | None = None,
        guild_id: int = 0,
        channel_id: int = 0,
        console_channel_id: int = 0,
        admin_role_id: int = 0,
        register_commands_globally: bool = False,
        rate_limit_per_minute: int = 5,
        gateway_factory: GatewayFactory | None = None,
    ) -> None:
        """Wire the module.

        Args:
            clock: Injected time, for rate limiting and batching.
            server_id: Stamped onto the ``CommandIssued`` events slash commands raise.
            controller: The **only** way this package may change the server's state. The same
                object ``mcmanager start`` uses, which is the structural reason no business logic
                can hide in a command handler.
            roster: Read for ``/players``.
            status: Provider for the aggregated status view ``/status`` renders. The same one the
                HTTP surface uses, so the two cannot drift.
            console_tail: Provider for ``/logs``, normally ``LogPipeline.tail``.
            idle_describe: Provider for the idle countdown line in ``/status``.
            mode: ``live`` / ``dryrun`` / ``disabled``.
            enabled: ``discord.enabled``. Both this and a non-disabled mode are required before
                anything connects.
            token: The bot token, still wrapped. Never a plain ``str`` in this process.
            guild_id: Commands are registered guild-scoped because that is instant, while global
                registration takes about an hour and makes every iteration painful.
            channel_id: Where events are announced.
            console_channel_id: Optional raw console relay channel. Zero disables it.
            admin_role_id: Who may run the mutating commands.
            register_commands_globally: Escape hatch; off by default, for the reason above.
            rate_limit_per_minute: Per-user command budget.
            gateway_factory: Injected for the tests, which must not need discord.py.
        """
        self._clock = clock
        self._server_id = server_id
        self._controller = controller
        self._roster = roster
        self._status = status
        self._console_tail = console_tail
        self._idle_describe = idle_describe
        self._mode = mode
        self._enabled = enabled
        self._token = token
        self._guild_id = guild_id
        self._channel_id = channel_id
        self._console_channel_id = console_channel_id
        self._admin_role_id = admin_role_id
        self._register_globally = register_commands_globally
        self._gateway_factory: GatewayFactory = gateway_factory or build_gateway

        self._gateway: Gateway | None = None
        self._relay: ConsoleRelay | None = None
        self._limiter = RateLimiter(clock=clock, per_minute=rate_limit_per_minute)
        self._subscriptions: tuple[Subscription, ...] = ()
        self._started = False
        self._closing = False
        self._stats: dict[str, int] = {
            "events_seen": 0,
            "messages_sent": 0,
            "messages_suppressed": 0,
            "commands_handled": 0,
            "commands_refused": 0,
            "errors": 0,
        }

    # -------------------------------------------------------------------------- introspection

    @property
    def mode(self) -> DiscordMode:
        """``live`` / ``dryrun`` / ``disabled``."""
        return self._mode

    @property
    def live(self) -> bool:
        """True only when the gateway should actually be opened and messages actually sent."""
        return self._enabled and self._mode == MODE_LIVE

    @property
    def active(self) -> bool:
        """True when this module wants events at all, which includes ``dryrun``."""
        return self._enabled and self._mode != MODE_DISABLED

    @property
    def connected(self) -> bool:
        """True when a gateway exists and reports itself ready. Surfaced by ``/readyz``."""
        return self._gateway is not None and self._gateway.connected

    @property
    def stats(self) -> Mapping[str, int]:
        """Counters. ``messages_sent`` must stay at zero for the whole shadow-run phase."""
        merged = dict(self._stats)
        if self._relay is not None:
            for key, value in self._relay.stats.items():
                merged[f"console_{key}"] = value
        return merged

    # --------------------------------------------------------------------------------- wiring

    def subscribe(self, bus: EventBus) -> tuple[Subscription, ...]:
        """Subscribe to every event, ``CONCURRENT``, and return the subscription for the tests.

        ``Event`` rather than ``ConsoleLog``: recognised lines deliberately do not also emit a
        ``ConsoleLog``, so a relay watching only ``ConsoleLog`` would show gaps exactly where the
        joins, the chat and the deaths were. Every event carrying its ``raw`` line is what makes a
        console channel possible at all.

        ``CONCURRENT`` because a Discord round trip is network I/O, and a ``SEQUENTIAL`` handler
        that awaits the network stalls the whole bus.
        """
        if not self.active:
            _log.info("discord.not_subscribed", mode=self._mode, enabled=self._enabled)
            return ()
        self._subscriptions = (
            bus.subscribe(
                Event,
                self.on_event,
                name="discord.relay",
                mode=DispatchMode.CONCURRENT,
            ),
        )
        return self._subscriptions

    async def start(self) -> None:
        """Open the gateway when live; otherwise log the decision and connect nothing."""
        self._started = True
        if not self.active:
            _log.info("discord.disabled", mode=self._mode, enabled=self._enabled)
            return
        if not self.live:
            _log.info(
                "discord.dry_run",
                mode=self._mode,
                channel_id=self._channel_id,
                hint="every message is rendered and logged, none is sent",
            )
            return
        if self._token is None:
            # Belt and braces: config already refuses live-without-token at load time.
            _log.error("discord.no_token", mode=self._mode)
            return

        gateway = self._gateway_factory(
            token=self._token.get_secret_value(),
            clock=self._clock,
            guild_id=self._guild_id,
        )
        self._gateway = gateway
        for name in cmds.COMMAND_NAMES:
            gateway.add_command(
                CommandSpec(
                    name=name,
                    description=_DESCRIPTIONS[name],
                    handler=self._make_handler(name),
                    ephemeral=name in cmds.MUTATING_COMMANDS,
                )
            )
        await gateway.connect()
        await gateway.register_commands(cmds.COMMAND_NAMES)

        if self._console_channel_id > 0:
            self._relay = ConsoleRelay(
                gateway=gateway,
                clock=self._clock,
                channel_id=self._console_channel_id,
            )
        _log.info(
            "discord.started",
            guild_id=self._guild_id,
            channel_id=self._channel_id,
            console_channel_id=self._console_channel_id,
            commands=list(cmds.COMMAND_NAMES),
        )

    async def run_relay(self) -> None:
        """The console relay's batching loop, spawned as a supervised task when configured."""
        if self._relay is not None:
            await self._relay.run()

    async def aclose(self) -> None:
        """Close the gateway and unsubscribe. **First step of the daemon's shutdown sequence.**

        Discord goes first because it is the only subsystem that does a network round trip on the
        way out, and because everything after it may still want to publish a final event that a
        live bot would have announced.

        Safe to call twice, and never raises.
        """
        if self._closing:
            return
        self._closing = True
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions = ()
        if self._relay is not None:
            await self._relay.aclose()
        if self._gateway is not None:
            await self._gateway.aclose()
        _log.info("discord.closed", started=self._started, **dict(self.stats))

    # ------------------------------------------------------------------------------- handlers

    async def on_event(self, event: Event) -> None:
        """Bus handler: render through the presenters and send, or log in dry run."""
        if self._closing or not self.active:
            return
        self._stats["events_seen"] += 1

        if self._relay is not None:
            await self._relay.on_event(event)

        try:
            message = render_event(event)  # pyright: ignore[reportArgumentType]
        except Exception:
            self._stats["errors"] += 1
            _log.exception("discord.render_failed", event_type=event.name, seq=event.seq)
            return
        if message is None:
            return

        if not self.live or self._gateway is None:
            self._stats["messages_suppressed"] += 1
            _log.info(
                "discord.would_send",
                event_type=event.name,
                seq=event.seq,
                mode=self._mode,
                message=message,
            )
            return

        if await self._gateway.send(self._channel_id, message):
            self._stats["messages_sent"] += 1
        else:
            self._stats["errors"] += 1

    # ------------------------------------------------------------------------------- commands

    def _make_handler(self, name: str) -> CommandHandler:
        """Build the handler for one command, closing over this module's collaborators."""

        async def handler(ctx: InvocationContext) -> str:
            self._stats["commands_handled"] += 1
            try:
                return await self._dispatch(name, ctx)
            except Exception:
                self._stats["errors"] += 1
                _log.exception("discord.command_error", command=name, user=ctx.user_name)
                return ":x: That command failed. The daemon log has the details."

        return handler

    async def _dispatch(self, name: str, ctx: InvocationContext) -> str:
        """Route one invocation, applying the admin gate to the mutating commands."""
        if name in cmds.MUTATING_COMMANDS:
            refusal = self._refuse(name, ctx)
            if refusal is not None:
                return refusal
            outcome = await self._mutate(name, ctx)
            return cmds.render_outcome(outcome)

        if name == "status":
            if self._status is None:
                return ":grey_question: Status is not available in this configuration."
            idle = None if self._idle_describe is None else self._idle_describe()
            return await cmds.cmd_status(self._status, idle=idle)
        if name == "players":
            return await cmds.cmd_players(self._roster)
        if name == "logs":
            if self._console_tail is None:
                return ":grey_question: Console output is not available in this configuration."
            return await cmds.cmd_logs(self._console_tail)
        return f":grey_question: Unknown command `{name}`."

    def _refuse(self, name: str, ctx: InvocationContext) -> str | None:
        """Apply the admin gate and the rate limit; audit and render any refusal.

        A refusal is published as ``CommandIssued(accepted=False, ...)`` by the controller path for
        accepted commands, but a *gate* refusal never reaches the controller, so it is logged here
        explicitly. A refused stop should be as visible as an accepted one.
        """
        verdict = check_admin(ctx.role_ids, admin_role_id=self._admin_role_id)
        if not verdict:
            self._stats["commands_refused"] += 1
            _log.warning(
                "discord.command_refused",
                command=name,
                user=ctx.user_name,
                user_id=ctx.user_id,
                reason=verdict.reason,
            )
            return f":no_entry: {verdict.reason}"

        limited = self._limiter.allow(ctx.user_id)
        if not limited:
            self._stats["commands_refused"] += 1
            _log.warning(
                "discord.command_rate_limited",
                command=name,
                user=ctx.user_name,
                reason=limited.reason,
            )
            return f":hourglass: {limited.reason}"
        return None

    async def _mutate(self, name: str, ctx: InvocationContext) -> ControlOutcome:
        """Call the controller. The actor is the Discord user, so the audit trail names them."""
        actor = f"{ctx.user_name}#{ctx.user_id}"
        action = _ACTIONS[name]
        if action is ControlAction.START:
            return await cmds.cmd_start(self._controller, actor=actor, via=Source.DISCORD)
        if action is ControlAction.STOP:
            return await cmds.cmd_stop(self._controller, actor=actor, via=Source.DISCORD)
        return await cmds.cmd_restart(self._controller, actor=actor, via=Source.DISCORD)
