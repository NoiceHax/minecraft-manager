"""The event bus: fan-out with total ordering, error isolation and a bounded queue.

Design decisions the rest of the daemon is written against:

- **``publish()`` is synchronous and non-blocking.** Producers must never block on a slow
  subscriber, and a sync publish makes accidental re-entrancy unwritable. It also keeps the parser
  path ``def``, not ``async def``.
- **Type matching has subclass semantics**, resolved through a cached MRO walk: subscribing to
  ``PlayerEvent`` gets all four player events; subscribing to ``Event`` gets everything. The cache
  is a ``{concrete event class: matching subscriptions}`` dict, invalidated wholesale on every
  subscribe and unsubscribe, because those happen at wiring time and dispatch happens millions of
  times.
- **One queue, total FIFO order** - a ``deque`` plus an ``asyncio.Event``, not an
  ``asyncio.Queue``, because the eviction policy below cannot be expressed with a Queue. Total
  order is what makes ``PlayerJoined -> IdleCancelled -> PlayerLeft -> IdleStarted`` a checkable
  causal chain rather than four independent observations.
- **``SEQUENTIAL`` is the default.** The stateful consumers are FSMs that are only correct if they
  observe events in order, one at a time. ``CONCURRENT`` is for network I/O. Documented contract:
  *a SEQUENTIAL handler must not await network I/O.*
- **Per-handler timeout** (default 5s). Without it, one hung SEQUENTIAL handler stalls the bus
  permanently. ``CancelledError`` is re-raised; everything else is logged and swallowed; a
  subscriber is paused after ``max_consecutive_failures`` failures in a row.
- **The bus never publishes from its own error path.** A "subscriber failed" event would let a
  broken wildcard subscriber recurse infinitely. Bus health is polled via :attr:`EventBus.stats`,
  never pushed.
- **Backpressure**: on a full queue, drop an incoming ``ConsoleLog``; for anything else, evict the
  oldest ``ConsoleLog`` to make room; if there is none, drop and log CRITICAL and increment
  ``dropped_critical``. That counter being non-zero means the design is wrong, and it surfaces in
  ``/readyz``.
- **``aclose()`` makes publish a counted no-op at DEBUG, not an exception** - teardown paths
  publish and should not have to guard - then drains with a timeout and cancels stragglers. The
  bus is closed late in the shutdown sequence so subsystems can emit final events on the way out.

**On time.** This module takes no :class:`~mcmanager.clock.Clock`. It never sleeps and never reads
a wall clock; its only deadlines are :func:`asyncio.timeout` windows around a handler and around
the drain, which are event-loop deadlines rather than wall-clock ones. That is also the mechanism
that makes the "was this an outer cancellation or my own timeout?" question answerable: 3.11's
``asyncio.timeout`` is uncancel-aware, so a cancellation it did not cause is re-raised as
``CancelledError`` and only its own expiry surfaces as ``TimeoutError``. Rebuilding that on top of
``Clock.call_later`` would mean reimplementing that check, and getting it wrong means a shutdown
that looks like a subscriber fault.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, final

import structlog

from mcmanager.core.events import ConsoleLog, Event
from mcmanager.core.types import DispatchMode

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

__all__ = ["BusStats", "EventBus", "EventHandler", "Subscription"]

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.bus")

type EventHandler[E: Event] = Callable[[E], Awaitable[None]]

DEFAULT_QUEUE_SIZE = 10_000
"""Queue capacity. Large enough that only a pathological producer reaches the eviction policy."""

DEFAULT_HANDLER_TIMEOUT = 5.0
"""Seconds one handler may take before it is cancelled and counted as a failure."""

DEFAULT_MAX_CONSECUTIVE_FAILURES = 5
"""Consecutive failures after which a subscriber is paused rather than retried forever."""


@final
@dataclass(frozen=True, slots=True)
class BusStats(Mapping[str, int]):
    """A snapshot of the bus's counters.

    It is a :class:`~collections.abc.Mapping` as well as a dataclass so that ``stats["queued"]``,
    ``dict(bus.stats)`` and ``bus.stats.dropped_critical`` all work: ``/readyz`` wants to serialise
    it, tests want to read one field, and neither should need a conversion step.

    Attributes:
        queued: Events waiting in the dispatch queue right now.
        published: Events accepted by :meth:`EventBus.publish` since construction.
        dispatched: Events fully fanned out by the dispatch loop.
        dropped_console: ``ConsoleLog`` events lost to the eviction policy. Non-zero means the log
            pipeline's token bucket is not keeping up; unpleasant, but by design survivable.
        dropped_critical: Non-``ConsoleLog`` events lost because the queue was full of events that
            could not be evicted. **Non-zero means the design is wrong**, so it is surfaced in
            ``/readyz`` rather than merely logged.
        handler_errors: Handler invocations that raised, timed out, or whose predicate raised.
        suppressed_after_close: Publishes that arrived after :meth:`EventBus.aclose`. Counted
            rather than raised, because teardown paths publish and should not have to guard.
        subscribers: Live subscriptions.
        paused_subscribers: Subscriptions currently paused after repeated failures.
        concurrent_tasks: In-flight ``CONCURRENT`` handler tasks.
    """

    queued: int = 0
    published: int = 0
    dispatched: int = 0
    dropped_console: int = 0
    dropped_critical: int = 0
    handler_errors: int = 0
    suppressed_after_close: int = 0
    subscribers: int = 0
    paused_subscribers: int = 0
    concurrent_tasks: int = 0

    def __getitem__(self, key: str) -> int:
        if key not in _STAT_FIELDS:
            raise KeyError(key)
        value: object = getattr(self, key)
        if not isinstance(value, int):  # pragma: no cover - defensive, all fields are ints
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(_STAT_FIELDS)

    def __len__(self) -> int:
        return len(_STAT_FIELDS)


_STAT_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(BusStats))


@final
class Subscription:
    """A live subscription. Returned instead of an opaque id so it can carry its own name.

    :attr:`name` appears in every log line the bus writes about this subscriber, which is what
    turns "the bus is wedged" into "``discord-relay`` timed out three times".

    :attr:`handler`, :attr:`predicate`, :meth:`note_success` and :meth:`note_failure` are public
    only because the bus - a different class - drives them. Treat them as bus-internal; the
    supported surface is :meth:`unsubscribe`, :meth:`pause`, :meth:`resume` and the counters.

    Attributes:
        name: Human name, used in every log line about this subscriber.
        event_type: The type subscribed to. Matching is by subclass, so this may be a category
            base such as ``PlayerEvent`` or ``Event`` itself.
        mode: SEQUENTIAL (awaited inline, total order) or CONCURRENT (spawned task).
        handler: The registered coroutine function. Bus-internal.
        predicate: The registered synchronous filter, or ``None``. Bus-internal.
    """

    __slots__ = (
        "_active",
        "_bus",
        "_consecutive_failures",
        "_failures",
        "_paused",
        "event_type",
        "handler",
        "mode",
        "name",
        "predicate",
    )

    def __init__(
        self,
        bus: EventBus,
        *,
        name: str,
        event_type: type[Event],
        handler: EventHandler[Any],
        mode: DispatchMode,
        predicate: Callable[[Any], bool] | None,
    ) -> None:
        self._bus = bus
        self.name = name
        self.event_type = event_type
        self.mode = mode
        self.handler = handler
        self.predicate = predicate
        self._active = True
        self._paused = False
        self._failures = 0
        self._consecutive_failures = 0

    def __repr__(self) -> str:
        state = "active" if self._active else "detached"
        if self._paused:
            state = "paused"
        return (
            f"<Subscription {self.name!r} {self.event_type.__name__} "
            f"{self.mode.value} {state} failures={self._failures}>"
        )

    # -- state ------------------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """False once :meth:`unsubscribe` has been called. Checked immediately before every
        invocation, so unsubscribing from inside a handler takes effect for the event in flight."""
        return self._active

    @property
    def paused(self) -> bool:
        """True once the subscriber has failed ``max_consecutive_failures`` times in a row, or
        after an explicit :meth:`pause`. A paused subscriber stays registered and stops receiving
        events; the bus keeps running."""
        return self._paused

    @property
    def failures(self) -> int:
        """Total failed invocations - raises, timeouts and raising predicates."""
        return self._failures

    @property
    def consecutive_failures(self) -> int:
        """Failures since the last successful invocation. Reset by any success."""
        return self._consecutive_failures

    # -- control ----------------------------------------------------------------------------

    def unsubscribe(self) -> None:
        """Detach. Idempotent, and safe to call from inside a handler."""
        if not self._active:
            return
        self._active = False
        self._bus.detach(self)

    def pause(self) -> None:
        """Stop delivering to this subscriber without detaching it. Idempotent."""
        self._paused = True

    def resume(self) -> None:
        """Undo :meth:`pause` and clear the consecutive-failure count."""
        self._paused = False
        self._consecutive_failures = 0

    # -- bus-internal -----------------------------------------------------------------------

    def note_success(self) -> None:
        """Record a clean invocation. Called by the bus; clears the consecutive-failure count."""
        self._consecutive_failures = 0

    def note_failure(self) -> None:
        """Record a failed invocation. Called by the bus."""
        self._failures += 1
        self._consecutive_failures += 1


class EventBus:
    """Fan-out with total ordering, error isolation and a bounded queue.

    Lifecycle: construct, :meth:`subscribe` during wiring, hand :meth:`run` to the supervisor as a
    ``critical`` task, :meth:`publish` from anywhere on the loop, :meth:`aclose` late in shutdown.
    """

    def __init__(
        self,
        *,
        maxsize: int = DEFAULT_QUEUE_SIZE,
        handler_timeout: float = DEFAULT_HANDLER_TIMEOUT,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    ) -> None:
        """Create a bus.

        Args:
            maxsize: Queue capacity, after which the eviction policy applies.
            handler_timeout: Seconds a single handler invocation may take.
            max_consecutive_failures: Failures in a row before a subscriber is paused.
        """
        if maxsize < 1:
            msg = "EventBus maxsize must be at least 1"
            raise ValueError(msg)
        self._maxsize = maxsize
        self._handler_timeout = handler_timeout
        self._max_consecutive_failures = max_consecutive_failures

        self._queue: deque[Event] = deque()
        self._nonempty = asyncio.Event()
        self._drained = asyncio.Event()
        self._drained.set()

        self._subscriptions: list[Subscription] = []
        self._cache: dict[type[Event], tuple[Subscription, ...]] = {}
        self._tasks: set[asyncio.Task[None]] = set()

        self._seq = 0
        self._published = 0
        self._dispatched = 0
        self._dropped_console = 0
        self._dropped_critical = 0
        self._handler_errors = 0
        self._suppressed = 0

        self._running = False
        self._stopped = False
        self._closing = False
        self._closed = False

    # -- subscription -------------------------------------------------------------------------

    def subscribe[E: Event](
        self,
        event_type: type[E],
        handler: EventHandler[E],
        *,
        name: str,
        mode: DispatchMode = DispatchMode.SEQUENTIAL,
        predicate: Callable[[E], bool] | None = None,
    ) -> Subscription:
        """Register ``handler`` for ``event_type`` and every subclass of it.

        Args:
            event_type: A concrete event or a category base (``PlayerEvent``, ``Event``).
            handler: Async callable taking one event. It must not raise; if it does, the bus logs
                and swallows, and repeated failures pause the subscriber.
            name: Identifies this subscriber in every log line the bus writes.
            mode: SEQUENTIAL (default) keeps total order and must not await network I/O.
                CONCURRENT spawns a tracked task.
            predicate: Optional cheap synchronous filter, evaluated at dispatch time. A raising
                predicate is counted as a failure and the event is skipped for this subscriber
                only.

        Returns:
            The live :class:`Subscription`, which is how you unsubscribe and how you inspect
            failure counts.
        """
        subscription = Subscription(
            self,
            name=name,
            event_type=event_type,
            handler=handler,
            mode=mode,
            predicate=predicate,
        )
        self._subscriptions.append(subscription)
        self._cache.clear()
        _log.debug(
            "bus.subscribed",
            subscriber=name,
            event_type=event_type.__name__,
            mode=mode.value,
        )
        return subscription

    def subscribe_all(
        self,
        handler: EventHandler[Event],
        *,
        name: str,
        mode: DispatchMode = DispatchMode.SEQUENTIAL,
        predicate: Callable[[Event], bool] | None = None,
    ) -> Subscription:
        """Subscribe to every event. Exactly ``subscribe(Event, ...)``, spelled for readers.

        This is what the SSE fan-out and the Discord console relay use, and why every event carries
        :attr:`~mcmanager.core.events.Event.raw`.
        """
        return self.subscribe(Event, handler, name=name, mode=mode, predicate=predicate)

    def detach(self, subscription: Subscription) -> None:
        """Remove a subscription from the registry.

        Prefer :meth:`Subscription.unsubscribe`, which also marks the subscription inactive so an
        in-flight dispatch skips it. This exists because ``Subscription`` is a separate class.
        """
        try:
            self._subscriptions.remove(subscription)
        except ValueError:  # pragma: no cover - unsubscribe() guards against double removal
            return
        self._cache.clear()
        _log.debug("bus.unsubscribed", subscriber=subscription.name)

    def _resolve(self, event_cls: type[Event]) -> tuple[Subscription, ...]:
        """Subscriptions matching ``event_cls``, in registration order, via a cached MRO walk."""
        cached = self._cache.get(event_cls)
        if cached is not None:
            return cached
        mro = set(event_cls.__mro__)
        matched = tuple(s for s in self._subscriptions if s.event_type in mro)
        self._cache[event_cls] = matched
        return matched

    # -- publishing ---------------------------------------------------------------------------

    def publish(self, event: Event) -> None:
        """Enqueue ``event``. Synchronous, non-blocking, never raises, stamps ``Event.seq``.

        The sequence number is consumed even when the event is subsequently dropped by the
        eviction policy, so a gap in ``seq`` is itself the record that something was lost.
        """
        try:
            self._publish(event)
        except Exception:
            # Reaching here is a bug in the bus, not in the caller. Producers must never have to
            # wrap publish() in a try, so we swallow and count instead of propagating.
            self._handler_errors += 1
            _log.exception("bus.publish_failed", event_type=type(event).__name__)

    def _publish(self, event: Event) -> None:
        if self._closing:
            self._suppressed += 1
            _log.debug(
                "bus.publish_after_close",
                event_type=type(event).__name__,
                suppressed=self._suppressed,
            )
            return

        self._seq += 1
        stamped = event.with_seq(self._seq)

        if len(self._queue) >= self._maxsize and not self._make_room(stamped):
            return

        self._queue.append(stamped)
        self._published += 1
        self._drained.clear()
        self._nonempty.set()

    def _make_room(self, incoming: Event) -> bool:
        """Apply the eviction policy. Returns True if ``incoming`` may now be enqueued."""
        if isinstance(incoming, ConsoleLog):
            self._dropped_console += 1
            _log.warning(
                "bus.dropped_console_log",
                reason="queue_full",
                dropped_console=self._dropped_console,
                queued=len(self._queue),
            )
            return False

        for index, queued in enumerate(self._queue):
            if isinstance(queued, ConsoleLog):
                del self._queue[index]
                self._dropped_console += 1
                _log.warning(
                    "bus.evicted_console_log",
                    reason="make_room",
                    for_event=type(incoming).__name__,
                    dropped_console=self._dropped_console,
                )
                return True

        self._dropped_critical += 1
        _log.critical(
            "bus.dropped_event",
            event_type=type(incoming).__name__,
            seq=incoming.seq,
            dropped_critical=self._dropped_critical,
            queued=len(self._queue),
            hint="queue full of non-evictable events; this counter being non-zero is a design bug",
        )
        return False

    # -- dispatch -----------------------------------------------------------------------------

    async def run(self) -> None:
        """The dispatch loop. Spawned by the supervisor as a ``critical`` task.

        Exits cleanly once :meth:`aclose` has drained the queue. Any other exit is a crash, which
        is why it is supervised as critical: a bus that is not dispatching is a daemon that has
        silently stopped working.
        """
        if self._running:
            _log.error("bus.run_called_twice")
            return
        self._running = True
        try:
            while not self._stopped:
                if not self._queue:
                    self._drained.set()
                    self._nonempty.clear()
                    if self._stopped:
                        break
                    await self._nonempty.wait()
                    continue
                event = self._queue.popleft()
                await self._dispatch(event)
                self._dispatched += 1
        finally:
            self._running = False
            self._drained.set()

    async def _dispatch(self, event: Event) -> None:
        for subscription in self._resolve(type(event)):
            # Re-checked per subscriber rather than once per event: a handler is allowed to
            # unsubscribe another subscriber (or itself) and that must take effect immediately.
            if not subscription.active or subscription.paused:
                continue
            if not self._passes(subscription, event):
                continue
            if subscription.mode is DispatchMode.CONCURRENT:
                task = asyncio.create_task(
                    self._invoke(subscription, event),
                    name=f"bus-handler:{subscription.name}",
                )
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            else:
                await self._invoke(subscription, event)

    def _passes(self, subscription: Subscription, event: Event) -> bool:
        predicate = subscription.predicate
        if predicate is None:
            return True
        try:
            return bool(predicate(event))
        except Exception as exc:
            self._record_failure(subscription, kind="predicate", exc=exc)
            return False

    async def _invoke(self, subscription: Subscription, event: Event) -> None:
        """Run one handler under a timeout, absorbing everything except an outer cancellation.

        The two cancellations are told apart by :func:`asyncio.timeout` itself: it uncancels the
        task and compares cancellation counts, so a ``CancelledError`` it did not cause propagates
        untouched and only its own expiry becomes ``TimeoutError``. Getting that wrong would make
        every shutdown look like a subscriber fault.
        """
        try:
            async with asyncio.timeout(self._handler_timeout):
                await subscription.handler(event)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self._record_failure(subscription, kind="timeout", exc=None)
        except Exception as exc:
            self._record_failure(subscription, kind="error", exc=exc)
        else:
            subscription.note_success()

    def _record_failure(
        self,
        subscription: Subscription,
        *,
        kind: str,
        exc: BaseException | None,
    ) -> None:
        self._handler_errors += 1
        subscription.note_failure()
        if exc is None:
            _log.error(
                "bus.handler_timeout",
                subscriber=subscription.name,
                timeout_seconds=self._handler_timeout,
                consecutive_failures=subscription.consecutive_failures,
            )
        else:
            _log.error(
                "bus.handler_failed",
                subscriber=subscription.name,
                kind=kind,
                error=str(exc),
                consecutive_failures=subscription.consecutive_failures,
                exc_info=exc,
            )
        if (
            not subscription.paused
            and subscription.consecutive_failures >= self._max_consecutive_failures
        ):
            subscription.pause()
            _log.error(
                "bus.subscriber_paused",
                subscriber=subscription.name,
                consecutive_failures=subscription.consecutive_failures,
                hint="subscriber will receive no further events until resume()",
            )

    # -- shutdown -----------------------------------------------------------------------------

    async def aclose(self, *, drain_timeout: float = 5.0) -> None:
        """Stop accepting, drain what is queued, then cancel stragglers.

        Idempotent. After this returns, :meth:`publish` is a counted no-op rather than an error,
        and :meth:`run` has exited (or will exit the moment it is next scheduled).
        """
        if self._closed:
            return
        self._closed = True
        self._closing = True
        self._nonempty.set()

        if self._running and self._queue:
            try:
                async with asyncio.timeout(drain_timeout):
                    await self._drained.wait()
            except TimeoutError:
                _log.warning(
                    "bus.drain_timeout",
                    drain_timeout=drain_timeout,
                    undelivered=len(self._queue),
                )

        self._stopped = True
        self._nonempty.set()
        await self._settle_tasks(drain_timeout)
        _log.debug("bus.closed", **dict(self.stats))

    async def _settle_tasks(self, grace: float) -> None:
        """Give CONCURRENT handlers ``grace`` seconds to finish, then cancel them."""
        pending = {task for task in self._tasks if not task.done()}
        if not pending:
            return
        # asyncio.wait is used rather than gather-under-timeout because gather propagates its own
        # cancellation to the children, which would turn "wait for them" into "cancel them" and
        # lose the grace period entirely.
        _, still_running = await asyncio.wait(pending, timeout=grace)
        if not still_running:
            return
        _log.warning(
            "bus.cancelling_handlers",
            count=len(still_running),
            names=sorted(task.get_name() for task in still_running),
        )
        for task in still_running:
            task.cancel()
        await asyncio.wait(still_running, timeout=grace)

    # -- introspection ------------------------------------------------------------------------

    @property
    def stats(self) -> BusStats:
        """A snapshot of every counter. Polled by ``/readyz``; never pushed as an event."""
        return BusStats(
            queued=len(self._queue),
            published=self._published,
            dispatched=self._dispatched,
            dropped_console=self._dropped_console,
            dropped_critical=self._dropped_critical,
            handler_errors=self._handler_errors,
            suppressed_after_close=self._suppressed,
            subscribers=len(self._subscriptions),
            paused_subscribers=sum(1 for s in self._subscriptions if s.paused),
            concurrent_tasks=sum(1 for t in self._tasks if not t.done()),
        )

    @property
    def subscriptions(self) -> tuple[Subscription, ...]:
        """Every live subscription, in registration order. For diagnostics and tests."""
        return tuple(self._subscriptions)

    @property
    def closed(self) -> bool:
        """True once :meth:`aclose` has been called."""
        return self._closed
