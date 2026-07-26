"""Optional raw console channel. **File and signatures only; body in M6.**

Subscribes to ``Event`` - not to ``ConsoleLog`` - and renders ``Event.raw``. Recognised lines
deliberately do not also emit a ``ConsoleLog``, so a relay watching only ``ConsoleLog`` would show
gaps exactly where the joins, chat and deaths were. Every event carrying its ``raw`` line is what
makes this work, and it is worth settling here rather than rediscovering it in M6.

Batches lines and respects Discord's rate limits; drops with a counter rather than queueing without
bound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mcmanager.clock import Clock
    from mcmanager.core.events import Event
    from mcmanager.discordbot.client import DiscordGateway

__all__ = ["ConsoleRelay"]


class ConsoleRelay:
    """Batches console lines into one channel. Every body lands in M6."""

    def __init__(
        self,
        *,
        gateway: DiscordGateway,
        clock: Clock,
        channel_id: int,
        batch_seconds: float = 2.0,
        queue_max: int = 500,
    ) -> None:
        """Hold the gateway, the batching window, and a bounded queue.

        Bounded and dropping, never blocking: back-pressuring the bus because Discord is slow would
        make a chatty console able to stall log processing.
        """
        raise NotImplementedError

    @property
    def stats(self) -> Mapping[str, int]:
        """``queued`` / ``sent`` / ``dropped``. ``dropped`` being non-zero is expected under load
        and is exactly why it is counted rather than hidden."""
        raise NotImplementedError

    async def on_event(self, event: Event) -> None:
        """Bus handler. Enqueues ``event.raw``; never awaits the network on the bus's path."""
        raise NotImplementedError

    async def run(self) -> None:
        """The batching loop: drain the queue every ``batch_seconds`` and send one message."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Flush what fits, drop the rest, and stop. Never raises."""
        raise NotImplementedError
