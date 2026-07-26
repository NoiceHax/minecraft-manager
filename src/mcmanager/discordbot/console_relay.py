"""Optional raw console channel.

Subscribes to ``Event`` - not to ``ConsoleLog`` - and renders ``Event.raw``. Recognised lines
deliberately do not also emit a ``ConsoleLog``, so a relay watching only ``ConsoleLog`` would show
gaps exactly where the joins, chat and deaths were. Every event carrying its ``raw`` line is what
makes this work.

Batches lines and respects Discord's rate limits; drops with a counter rather than queueing without
bound.

**Bounded and dropping, never blocking.** ``on_event`` runs on the bus. If it awaited the network,
or waited for room in a full queue, a chatty console would stall log processing behind Discord's
rate limiter - which is exactly backwards, since the console channel is the least important
consumer in the system.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Final, final

import structlog

from mcmanager.discordbot.presenters import MAX_MESSAGE_LENGTH, render_raw

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mcmanager.clock import Clock
    from mcmanager.core.events import Event
    from mcmanager.discordbot.client import Gateway

__all__ = ["ConsoleRelay"]

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.discord.console")

_FENCE: Final = "```"
_BUDGET: Final = MAX_MESSAGE_LENGTH - (2 * len(_FENCE)) - 2
"""Room for the batch itself once the code fence and its two newlines are accounted for."""


@final
class ConsoleRelay:
    """Batches console lines into one channel."""

    def __init__(
        self,
        *,
        gateway: Gateway,
        clock: Clock,
        channel_id: int,
        batch_seconds: float = 2.0,
        queue_max: int = 500,
    ) -> None:
        """Hold the gateway, the batching window, and a bounded queue.

        Bounded and dropping, never blocking: back-pressuring the bus because Discord is slow would
        make a chatty console able to stall log processing.
        """
        self._gateway = gateway
        self._clock = clock
        self._channel_id = channel_id
        self._batch_seconds = batch_seconds
        self._queue_max = queue_max
        self._queue: deque[str] = deque()
        self._closing = False
        self._stats: dict[str, int] = {"queued": 0, "sent": 0, "dropped": 0, "batches": 0}

    @property
    def enabled(self) -> bool:
        """A channel id of zero disables the relay entirely."""
        return self._channel_id > 0

    @property
    def stats(self) -> Mapping[str, int]:
        """``queued`` / ``sent`` / ``dropped``. ``dropped`` being non-zero is expected under load
        and is exactly why it is counted rather than hidden."""
        return dict(self._stats)

    async def on_event(self, event: Event) -> None:
        """Bus handler. Enqueues ``event.raw``; never awaits the network on the bus's path."""
        if self._closing or not self.enabled:
            return
        line = render_raw(event)
        if line is None:
            return
        if len(self._queue) >= self._queue_max:
            # Drop the oldest: on a console tail the newest lines are the ones somebody is
            # watching for, and a crash dump's first thousand frames are not the interesting part.
            self._queue.popleft()
            self._stats["dropped"] += 1
        self._queue.append(line)
        self._stats["queued"] += 1

    async def run(self) -> None:
        """The batching loop: drain the queue every ``batch_seconds`` and send one message."""
        if not self.enabled:
            _log.info("console_relay.disabled", channel_id=self._channel_id)
            return
        while not self._closing:
            await self._clock.sleep(self._batch_seconds)
            if self._closing:
                return
            await self._flush()

    async def _flush(self) -> None:
        """Send at most one message, taking as many queued lines as fit in the budget."""
        if not self._queue:
            return
        lines: list[str] = []
        size = 0
        while self._queue:
            candidate = self._queue[0]
            # +1 for the newline this line will contribute.
            if lines and size + len(candidate) + 1 > _BUDGET:
                break
            if not lines and len(candidate) > _BUDGET:
                # A single line larger than a whole message: truncate rather than spin forever
                # refusing to send it.
                candidate = candidate[:_BUDGET]
                self._queue.popleft()
                lines.append(candidate)
                size = len(candidate)
                break
            self._queue.popleft()
            lines.append(candidate)
            size += len(candidate) + 1

        if not lines:
            return
        body = "\n".join(lines)
        delivered = await self._gateway.send(self._channel_id, f"{_FENCE}\n{body}\n{_FENCE}")
        self._stats["batches"] += 1
        if delivered:
            self._stats["sent"] += len(lines)
        else:
            self._stats["dropped"] += len(lines)

    async def aclose(self) -> None:
        """Flush what fits, drop the rest, and stop. Never raises."""
        if self._closing:
            return
        self._closing = True
        try:
            # One last batch, so the final few lines before a restart are not simply lost. Bounded
            # to a single message: teardown has a budget and Discord is not entitled to all of it.
            await self._flush()
        except Exception as exc:  # pragma: no cover - teardown must never raise
            _log.warning("console_relay.final_flush_failed", error=str(exc))
        remaining = len(self._queue)
        if remaining:
            self._stats["dropped"] += remaining
            self._queue.clear()
        _log.info("console_relay.closed", remaining_dropped=remaining, **self._stats)
