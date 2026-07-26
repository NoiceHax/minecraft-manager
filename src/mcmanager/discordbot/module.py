"""The Discord subsystem's lifecycle. **Stub: real interface and wiring now, body in M6.**

``start()``, ``aclose()`` and its bus subscriptions exist from day one so ``app.py``'s wiring is
exercised before there is a bot at all, and so the shutdown ordering is proven with nothing to
break.

Three modes: ``live``, ``dryrun`` (logs what it would send, sends nothing - the 48-hour shadow-run
setting) and ``disabled`` (the gateway is never opened). A cross-field config validator refuses
``runtime = "fake"`` together with ``mode = "live"``, so fake events cannot reach a real channel.

Subscriptions are ``CONCURRENT``: a Discord round trip is network I/O and must never sit on the
bus's sequential path.

**No ``import discord`` at module scope, here or anywhere in this package.** M6's gateway import
lives inside :meth:`DiscordModule.start`, behind the ``live`` branch. That is what keeps
``mcmanager replay``, ``mcmanager inspect`` and every service test free of discord.py, and it is
also what makes ``discord.enabled = false`` genuinely cost nothing.

Discord is a **client of** :class:`~mcmanager.services.controller.ServerController` and the status
surface, never a peer of them. If a handler in this package ever grows a branch on server state,
the logic is in the wrong file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, final

import structlog

from mcmanager.core.events import Event
from mcmanager.core.types import DispatchMode

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mcmanager.clock import Clock
    from mcmanager.core.bus import EventBus, Subscription
    from mcmanager.core.types import ServerId
    from mcmanager.services.controller import ServerController
    from mcmanager.services.players import PlayerRoster

__all__ = ["DiscordMode", "DiscordModule"]

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.discord")

type DiscordMode = str
"""``"live"``, ``"dryrun"`` or ``"disabled"``. A plain string alias rather than an enum because it
comes straight from ``discord.mode`` in the TOML and crosses no other boundary."""

MODE_LIVE = "live"
MODE_DRYRUN = "dryrun"
MODE_DISABLED = "disabled"


class TokenProvider(Protocol):
    """Anything that can hand over the bot token without it having been a ``str`` on the way.

    ``pydantic.SecretStr`` satisfies this. Typed structurally so this module needs no pydantic
    import and no plain-string field that a stray ``log.info("config", cfg=...)`` could leak.
    """

    def get_secret_value(self) -> str: ...


@final
class DiscordModule:
    """Owns the gateway connection, the slash commands and the relays. **Body in M6.**

    What is real today: construction, the bus subscriptions, ``start``/``aclose``, and the mode
    gate. What M6 adds: the gateway client, the command tree, the presenters and the console relay.
    Every handler here logs what it *would* have sent, which is exactly what the 48-hour shadow run
    needs anyway.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        server_id: ServerId,
        controller: ServerController,
        roster: PlayerRoster,
        mode: DiscordMode = MODE_DISABLED,
        enabled: bool = False,
        token: TokenProvider | None = None,
        guild_id: int = 0,
        channel_id: int = 0,
        console_channel_id: int = 0,
        admin_role_id: int = 0,
        register_commands_globally: bool = False,
    ) -> None:
        """Wire the module.

        Args:
            clock: Injected time, for rate limiting and batching in M6.
            server_id: Stamped onto the ``CommandIssued`` events slash commands will raise.
            controller: The **only** way this package may change the server's state. The same
                object ``mcmanager start`` uses, which is the structural reason no business logic
                can hide in a command handler.
            roster: Read for ``/players``.
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
        """
        self._clock = clock
        self._server_id = server_id
        self._controller = controller
        self._roster = roster
        self._mode = mode
        self._enabled = enabled
        self._token = token
        self._guild_id = guild_id
        self._channel_id = channel_id
        self._console_channel_id = console_channel_id
        self._admin_role_id = admin_role_id
        self._register_globally = register_commands_globally

        self._subscriptions: tuple[Subscription, ...] = ()
        self._started = False
        self._closing = False
        self._stats: dict[str, int] = {
            "events_seen": 0,
            "messages_sent": 0,
            "messages_suppressed": 0,
            "commands_handled": 0,
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
    def stats(self) -> Mapping[str, int]:
        """Counters. ``messages_sent`` must stay at zero for the whole shadow-run phase."""
        return dict(self._stats)

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
        """Open the gateway. **M6 body**; today it logs the decision and connects nothing.

        M6 imports discord.py *here*, inside the ``live`` branch, so nothing outside that branch
        ever pays for the dependency.
        """
        self._started = True
        if not self.active:
            _log.info("discord.disabled", mode=self._mode, enabled=self._enabled)
            return
        if self._token is None and self.live:
            # Belt and braces: config already refuses live-without-token at load time.
            _log.error("discord.no_token", mode=self._mode)
            return
        _log.warning(
            "discord.gateway_not_implemented",
            mode=self._mode,
            live=self.live,
            guild_id=self._guild_id,
            channel_id=self._channel_id,
            console_channel_id=self._console_channel_id,
            admin_role_id=self._admin_role_id,
            register_globally=self._register_globally,
            hint="M6 opens the gateway here; nothing is connected today",
        )

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
        _log.info("discord.closed", started=self._started, **dict(self._stats))

    # ------------------------------------------------------------------------------- handlers

    async def on_event(self, event: Event) -> None:
        """Bus handler. **M6 body**: render through the presenters and send, or log in dry run.

        Counting happens today so the shadow run has something to compare against ``latest.log``.
        """
        if self._closing or not self.active:
            return
        self._stats["events_seen"] += 1
        self._stats["messages_suppressed"] += 1
        _log.debug(
            "discord.would_send",
            event_type=event.name,
            seq=event.seq,
            mode=self._mode,
            hint="M6 renders this through discordbot/presenters.py",
        )
