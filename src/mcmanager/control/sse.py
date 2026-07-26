"""Server-sent events for ``/events`` and ``/logs``.

Fed by a ``CONCURRENT`` wildcard bus subscriber with a **bounded per-client queue that drops and
counts** rather than applying backpressure to the bus. A slow ``curl`` must never stall the daemon.
That is the single design constraint here and everything else follows from it:

- :meth:`SseClient.offer` never awaits and never raises. It appends to a bounded deque and, when
  that deque is full, discards the **oldest** frame and increments a counter.
- The counter is not silent. The next frame the client receives is preceded by a
  ``: dropped N`` comment, so a human watching ``curl -N`` sees the gap rather than wondering why
  the numbers do not add up. ``dropped`` is also in :attr:`SseChannel.stats`.
- Dropping the oldest rather than the newest is deliberate for a *live tail*: the reader who fell
  behind wants to catch up to now, not to replay what they already missed.

``/logs`` streams ``ConsoleLog`` **plus the ``raw`` field of every other event**. Recognised lines
deliberately do not also emit a ``ConsoleLog`` - that is the spec - so a relay watching only
``ConsoleLog`` would show gaps exactly where the joins and the chat were, which is the most
confusing possible failure for a console view.

``since`` is served from a bounded in-memory ring of recent events, so a client that reconnects
within the ring's depth resumes without a hole. Beyond that depth the stream simply starts at now:
this is a live tail, not a durable log, and pretending otherwise would require a persistence layer
that ``mcmanager events`` does not need.

Keepalives go out on the injected :class:`~mcmanager.clock.Clock`, which is what lets a test assert
that a 15-second keepalive fired without waiting 15 seconds.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Literal, final

import structlog

from mcmanager.core.events import (
    EVENT_BY_NAME,
    ConsoleLog,
    Event,
    IdleEvent,
    PlayerEvent,
    RuntimeStatusEvent,
    ServerEvent,
)
from mcmanager.core.serde import event_to_dict
from mcmanager.core.types import DispatchMode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Mapping

    from mcmanager.clock import Clock
    from mcmanager.core.bus import EventBus, Subscription

__all__ = [
    "DEFAULT_HISTORY",
    "DEFAULT_KEEPALIVE_SECONDS",
    "DEFAULT_QUEUE_MAX",
    "SSE_CONTENT_TYPE",
    "EventFilter",
    "SseChannel",
    "SseClient",
    "StreamMode",
    "known_type_names",
    "parse_since",
    "render_frame",
    "resolve_types",
]

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.control.sse")

SSE_CONTENT_TYPE: Final = "text/event-stream"
DEFAULT_QUEUE_MAX: Final = 500
DEFAULT_HISTORY: Final = 512
DEFAULT_KEEPALIVE_SECONDS: Final = 15.0

type StreamMode = Literal["events", "logs"]

_CATEGORIES: Final[Mapping[str, type[Event]]] = {
    "Event": Event,
    "ServerEvent": ServerEvent,
    "PlayerEvent": PlayerEvent,
    "IdleEvent": IdleEvent,
    "RuntimeStatusEvent": RuntimeStatusEvent,
}
"""The category bases, which ``--type PlayerEvent`` resolves against.

They are absent from ``EVENT_BY_NAME`` on purpose - that table is for serde, which only ever sees
concrete types - so the filter adds them back here. This is the same subclass matching the bus
does, applied client-side.
"""


_DURATION_UNITS: Final[Mapping[str, float]] = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_since(text: str, *, now: datetime) -> datetime:
    """Parse a ``since`` value: an RFC3339 timestamp, or a relative age like ``15m``.

    The relative form exists because ``--since 15m`` is what anybody actually types, and making
    them compute a UTC timestamp first is the kind of friction that stops a debugging tool from
    being used. ``now`` is injected; nothing here reads a clock.

    Raises:
        ValueError: on anything else, naming both accepted forms.
    """
    raw = text.strip()
    if not raw:
        msg = "since is empty; expected an RFC3339 timestamp or a relative age like '15m'"
        raise ValueError(msg)

    unit = _DURATION_UNITS.get(raw[-1].lower())
    if unit is not None:
        amount = _finite(raw[:-1])
        if amount is not None and amount >= 0:
            return now - timedelta(seconds=amount * unit)

    candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        msg = (
            f"could not read since={text!r}: expected an RFC3339 timestamp "
            "(2026-07-25T22:58:20Z) or a relative age (30s, 15m, 2h, 1d)"
        )
        raise ValueError(msg) from exc
    if parsed.tzinfo is None:
        msg = f"since={text!r} is naive; every timestamp in this project is tz-aware UTC"
        raise ValueError(msg)
    return parsed.astimezone(UTC)


def _finite(text: str) -> float | None:
    """``float(text)`` when it is a real number, else ``None``. ``inf`` and ``nan`` are not."""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def known_type_names() -> tuple[str, ...]:
    """Every name ``--type`` accepts, sorted. Printed in the error when one does not match."""
    return tuple(sorted(set(EVENT_BY_NAME) | set(_CATEGORIES)))


def resolve_types(names: Iterable[str]) -> frozenset[type[Event]]:
    """Resolve ``--type`` / ``?type=`` names to event classes.

    Raises:
        ValueError: naming every accepted value. A silently ignored filter is worse than an error:
            the user sees an empty stream and concludes the daemon is broken.
    """
    resolved: set[type[Event]] = set()
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        found = EVENT_BY_NAME.get(name) or _CATEGORIES.get(name)
        if found is None:
            known = ", ".join(known_type_names())
            msg = f"unknown event type {name!r}; expected one of: {known}"
            raise ValueError(msg)
        resolved.add(found)
    return frozenset(resolved)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class EventFilter:
    """What one subscriber asked to see.

    Attributes:
        types: Match by class, **with subclass semantics** - ``PlayerEvent`` gets all four. An
            empty/``None`` set means everything.
        since: Only events at or after this instant. Compared against ``Event.ts``, which is
            Docker's timestamp for log-derived events, so a reconnecting client asking for
            ``since=<last ts seen>`` gets exactly the gap.
        since_seq: Only events with a strictly greater ``seq``. More precise than ``since``
            whenever the client has one, because ``ts`` has duplicates within a second and ``seq``
            is a total order.
    """

    types: frozenset[type[Event]] | None = None
    since: datetime | None = None
    since_seq: int | None = None

    def matches(self, event: Event) -> bool:
        """Does ``event`` pass this filter?"""
        if self.types is not None and not any(isinstance(event, cls) for cls in self.types):
            return False
        if self.since is not None and event.ts < self.since:
            return False
        return not (self.since_seq is not None and event.seq <= self.since_seq)


# --------------------------------------------------------------------------------- framing


def _frame(*, event_name: str, data: str, seq: int | None = None) -> str:
    """One SSE frame. ``data`` must already be a single line - JSON always is."""
    lines: list[str] = []
    if seq is not None:
        lines.append(f"id: {seq}")
    lines.append(f"event: {event_name}")
    lines.append(f"data: {data}")
    return "\n".join(lines) + "\n\n"


def _comment(text: str) -> str:
    """An SSE comment. Keeps proxies from closing an idle connection, and is ignored by clients."""
    return f": {text}\n\n"


def _log_payload(event: Event) -> dict[str, Any] | None:
    """The ``/logs`` projection of one event, or ``None`` if it carries no text.

    A ``ConsoleLog`` contributes its parsed message; everything else contributes ``Event.raw``,
    which is why every event carries one.
    """
    if isinstance(event, ConsoleLog):
        return {
            "ts": event.ts.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "seq": event.seq,
            "type": event.name,
            "level": event.level,
            "thread": event.thread,
            "origin": event.origin.value,
            "stream": event.stream.value,
            "message": event.message,
        }
    if event.raw is None:
        return None
    return {
        "ts": event.ts.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "seq": event.seq,
        "type": event.name,
        "level": None,
        "thread": None,
        "origin": None,
        "stream": None,
        "message": event.raw,
    }


def render_frame(event: Event, mode: StreamMode) -> str | None:
    """Format one event for one stream mode, or ``None`` when it has nothing to contribute.

    ``events`` mode emits :func:`~mcmanager.core.serde.event_to_dict` verbatim, which is the same
    JSON ``mcmanager events --json`` prints and the same JSON a session record holds. There is one
    wire format for an event and this is it.
    """
    if mode == "events":
        payload = event_to_dict(event)
        return _frame(
            event_name=event.name,
            data=json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
            seq=event.seq,
        )
    projected = _log_payload(event)
    if projected is None:
        return None
    return _frame(
        event_name="log",
        data=json.dumps(projected, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
        seq=event.seq,
    )


# ---------------------------------------------------------------------------------- client


@final
class SseClient:
    """One connected reader. Bounded, lossy, and honest about it."""

    __slots__ = (
        "_closed",
        "_dropped",
        "_filter",
        "_maxsize",
        "_mode",
        "_pending",
        "_queue",
        "_wake",
    )

    def __init__(
        self,
        *,
        wake: asyncio.Event,
        event_filter: EventFilter,
        mode: StreamMode = "events",
        maxsize: int = DEFAULT_QUEUE_MAX,
    ) -> None:
        """Create a client.

        Args:
            wake: The event the reader waits on. Injected rather than created here so the channel
                owns loop affinity: an ``asyncio.Event`` constructed off-loop is a subtle bug.
            event_filter: What this reader asked for.
            mode: ``events`` for the full event stream, ``logs`` for the console projection.
            maxsize: Frames buffered before the oldest starts being discarded.
        """
        if maxsize < 1:
            msg = "SseClient maxsize must be at least 1"
            raise ValueError(msg)
        self._wake = wake
        self._filter = event_filter
        self._mode: StreamMode = mode
        self._maxsize = maxsize
        self._queue: deque[str] = deque()
        self._dropped = 0
        self._pending = 0
        self._closed = False

    @property
    def dropped(self) -> int:
        """Frames discarded because this reader could not keep up. Non-zero is visible in-band."""
        return self._dropped

    @property
    def queued(self) -> int:
        """Frames waiting to be written."""
        return len(self._queue)

    @property
    def mode(self) -> StreamMode:
        return self._mode

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def event_filter(self) -> EventFilter:
        return self._filter

    def offer(self, event: Event) -> None:
        """Enqueue ``event`` if it passes the filter. Never awaits, never raises, may drop."""
        if self._closed or not self._filter.matches(event):
            return
        frame = render_frame(event, self._mode)
        if frame is None:
            return
        self.push(frame)

    def push(self, frame: str) -> None:
        """Enqueue an already-formatted frame - a keepalive, or a replayed history entry."""
        if self._closed:
            return
        if len(self._queue) >= self._maxsize:
            self._queue.popleft()
            self._dropped += 1
            self._pending += 1
        self._queue.append(frame)
        self._wake.set()

    def close(self) -> None:
        """Stop the stream. Idempotent; the reader's iterator finishes on its next pass."""
        self._closed = True
        self._wake.set()

    async def __aiter__(self) -> AsyncIterator[str]:
        """Yield frames until :meth:`close`.

        Drains everything queued before waiting again, so a burst is written as a burst rather
        than one frame per loop iteration.
        """
        while True:
            while self._queue:
                if self._pending:
                    dropped = self._pending
                    self._pending = 0
                    yield _comment(f"dropped {dropped} frame(s): this reader fell behind")
                yield self._queue.popleft()
            if self._closed:
                return
            self._wake.clear()
            # Re-check after clearing: offer() may have run between the drain and the clear, and
            # waiting on an event that was just consumed is how a reader hangs with data queued.
            if self._queue or self._closed:
                continue
            await self._wake.wait()


# --------------------------------------------------------------------------------- channel


@final
class SseChannel:
    """Fans the bus out to every connected reader, and keeps a short replay ring.

    Wire it once at startup: :meth:`attach` subscribes, :meth:`run_keepalive` is spawned as a
    non-critical supervised task, and :meth:`aclose` runs during shutdown before the bus drains.
    """

    __slots__ = (
        "_clients",
        "_clock",
        "_history",
        "_keepalive_seconds",
        "_keepalives",
        "_name",
        "_opened",
        "_queue_max",
        "_running",
        "_subscription",
    )

    def __init__(
        self,
        *,
        clock: Clock,
        queue_max: int = DEFAULT_QUEUE_MAX,
        history: int = DEFAULT_HISTORY,
        keepalive_seconds: float = DEFAULT_KEEPALIVE_SECONDS,
        name: str = "sse",
    ) -> None:
        """Create a channel.

        Args:
            clock: Injected time. Keepalives and their timestamps both come from it.
            queue_max: Per-client buffer depth. From ``web.sse_queue_max``.
            history: How many recent events to keep for ``since``. Bounded on purpose: this is a
                live tail, not a log store.
            keepalive_seconds: Comment interval. Anything longer than a proxy's idle timeout means
                a stream that silently dies after five minutes of a quiet server.
            name: Subscriber name, which appears in every bus log line about this channel.
        """
        self._clock = clock
        self._queue_max = queue_max
        self._keepalive_seconds = keepalive_seconds
        self._name = name
        self._clients: list[SseClient] = []
        self._history: deque[Event] = deque(maxlen=max(history, 0))
        self._subscription: Subscription | None = None
        # True from construction, not from attach(): the keepalive is about connected readers, not
        # about the bus, and a channel serving a `/logs` stream on a daemon whose bus subscription
        # failed should still hold its connections open rather than silently letting proxies drop
        # them. aclose() is the only thing that clears it.
        self._running = True
        self._opened = 0
        self._keepalives = 0

    # -- wiring --------------------------------------------------------------------------------

    def attach(self, bus: EventBus) -> Subscription:
        """Subscribe to every event, CONCURRENT so a write can never block the dispatch loop.

        The handler itself does not await - it appends to bounded deques - but CONCURRENT is still
        the honest declaration: this subscriber exists to serve network clients, and the mode is
        what documents that a slow reader is *its* problem, not the bus's.
        """
        self._subscription = bus.subscribe_all(
            self._on_event,
            name=self._name,
            mode=DispatchMode.CONCURRENT,
        )
        self._running = True
        return self._subscription

    async def _on_event(self, event: Event) -> None:
        """Bus handler. Records history and offers to every client."""
        if self._history.maxlen:
            self._history.append(event)
        for client in tuple(self._clients):
            client.offer(event)

    # -- clients -------------------------------------------------------------------------------

    def open(
        self,
        *,
        event_filter: EventFilter | None = None,
        mode: StreamMode = "events",
        replay: bool = True,
    ) -> SseClient:
        """Register a new reader and pre-load the replay ring into it.

        ``replay`` is honoured only when the filter carries a ``since`` or ``since_seq``: a plain
        ``mcmanager events --follow`` wants what happens next, not a dump of the last five hundred
        events every time it starts.
        """
        chosen = event_filter if event_filter is not None else EventFilter()
        client = SseClient(
            wake=asyncio.Event(),
            event_filter=chosen,
            mode=mode,
            maxsize=self._queue_max,
        )
        wants_history = chosen.since is not None or chosen.since_seq is not None
        if replay and wants_history:
            for event in tuple(self._history):
                client.offer(event)
        self._clients.append(client)
        self._opened += 1
        _log.debug(
            "sse.client_opened",
            channel=self._name,
            mode=mode,
            clients=len(self._clients),
            replayed=client.queued,
        )
        return client

    def close_client(self, client: SseClient) -> None:
        """Deregister a reader. Safe to call twice - a disconnect races the handler."""
        client.close()
        try:
            self._clients.remove(client)
        except ValueError:
            return
        _log.debug(
            "sse.client_closed",
            channel=self._name,
            clients=len(self._clients),
            dropped=client.dropped,
        )

    # -- keepalive -----------------------------------------------------------------------------

    async def run_keepalive(self) -> None:
        """Push a comment frame to every reader on the configured interval, forever.

        Spawned as a non-critical supervised task. Sleeps on the injected clock, so a test proves
        the fifteen-second keepalive with ``await clock.advance(15)`` and no wall time at all.
        """
        while self._running:
            await self._clock.sleep(self._keepalive_seconds)
            if not self._running:
                return
            self.send_keepalive()

    def send_keepalive(self) -> None:
        """Send one keepalive comment now. Separated out so tests need no timing at all."""
        stamp = self._clock.now().astimezone(UTC).isoformat().replace("+00:00", "Z")
        frame = _comment(f"keepalive {stamp}")
        for client in tuple(self._clients):
            client.push(frame)
        self._keepalives += 1

    # -- introspection -------------------------------------------------------------------------

    @property
    def clients(self) -> tuple[SseClient, ...]:
        """Currently connected readers."""
        return tuple(self._clients)

    @property
    def stats(self) -> Mapping[str, int]:
        """Counters, polled rather than pushed - the same rule the bus follows."""
        return {
            "clients": len(self._clients),
            "opened": self._opened,
            "dropped": sum(client.dropped for client in self._clients),
            "queued": sum(client.queued for client in self._clients),
            "history": len(self._history),
            "keepalives": self._keepalives,
        }

    def history(self, event_filter: EventFilter | None = None) -> tuple[Event, ...]:
        """The replay ring, optionally filtered. For ``/events?since=`` and for tests."""
        events = tuple(self._history)
        if event_filter is None:
            return events
        return tuple(event for event in events if event_filter.matches(event))

    # -- shutdown ------------------------------------------------------------------------------

    async def aclose(self) -> None:
        """Unsubscribe and end every stream. Safe to call twice, never raises."""
        self._running = False
        subscription = self._subscription
        self._subscription = None
        if subscription is not None:
            subscription.unsubscribe()
        for client in tuple(self._clients):
            self.close_client(client)
