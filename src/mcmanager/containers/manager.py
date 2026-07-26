"""Keeps the log stream and the event watcher attached.

The reconnect loop, and the five behaviours that make it survive real life:

- **Exponential backoff with jitter** between attempts.
- **Re-resolve the container by name on every reconnect.** A ``compose down/up`` changes the id,
  and a cached id silently follows a dead container forever - which looks exactly like "the server
  stopped logging".
- **A 256-entry ``(timestamp, hash)`` dedupe ring**, because Docker's ``since`` is
  second-granularity and replays the whole of that second on every reattach.
- **``RuntimeUnavailable`` is edge-triggered, once per outage**, never once per failed attempt.
- **A clean EOF is not an error**: it corroborates the ``die`` event.
- **A stopped container is not attached to at all.** Docker ends a follow on a container that is
  not running *immediately*, so attaching would EOF at once - and since a clean EOF resets the
  backoff, that pair spins at one inspect plus one attach per second for as long as the server
  stays down. After an idle auto-shutdown that is days. The loop backs off instead, and remembers
  to resume from the next run's ``StartedAt`` so waiting can never cost us the startup lines.

Plus a 60-second reconcile inspect as the safety net for docker events we missed entirely - which
is not hypothetical, since ``health_status`` events are edge-triggered and cannot be treated as a
heartbeat.

This module imports no Docker. It drives a :class:`~mcmanager.containers.base.ContainerRuntime`,
so its entire behaviour is exercised against ``FakeRuntime`` in about a millisecond of virtual
time.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Final, Protocol, final

from mcmanager.containers.errors import (
    ContainerNotFoundError,
    ContainerRuntimeError,
    RuntimeUnavailableError,
)
from mcmanager.core.events import RuntimeRestored, RuntimeUnavailable
from mcmanager.core.types import Source

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping
    from datetime import datetime

    from mcmanager.clock import Clock
    from mcmanager.containers.base import ContainerRuntime
    from mcmanager.containers.dto import ContainerSnapshot, LogLine, RuntimeEvent
    from mcmanager.core.events import Event
    from mcmanager.core.types import ServerId

__all__ = ["BackoffPolicy", "DedupeRing", "DockerManager", "EventSink"]

log: Final = logging.getLogger(__name__)

DEFAULT_DEDUPE_CAPACITY: Final = 256
"""Enough to cover the replay of one second at a rate that would already be pathological.

The window that matters is "lines Docker will hand us twice", which is bounded by one second of
output. 256 is roughly ten times what a busy Paper server emits in a second, and the ring is
cleared outright whenever the container id changes, so it never has to span a restart.
"""

DEFAULT_RECONCILE_INTERVAL: Final = 60.0
DEFAULT_EOF_RETRY_SECONDS: Final = 1.0


class _LogAttach(Enum):
    """How one pass of :meth:`DockerManager._pump_logs` ended.

    Three outcomes, not two, because "the stream ended cleanly after being attached to a running
    container" and "there was nothing to attach to" call for opposite retry policies: the first is
    the ``die`` corroboration the plan wants retried promptly, the second is a server that is
    simply off and must be backed off from.
    """

    EOF = auto()
    """Attached, streamed, and the stream ended cleanly. The container stopped under us."""

    NOT_RUNNING = auto()
    """The container exists but is not running, so no stream was opened."""

    FAILED = auto()
    """An exception was raised and handled by the caller."""


def _default_jitter() -> float:
    """Uniform ``[0, 1)``. Not a security primitive: this only decorrelates reconnect storms."""
    return random.random()  # noqa: S311


class EventSink(Protocol):
    """The half of :class:`~mcmanager.core.bus.EventBus` this module needs.

    Narrowed to one method on purpose. The manager publishes two facts and subscribes to nothing,
    so depending on the whole bus would be a lie about the coupling and would make every test here
    need a working dispatch loop. ``EventBus`` satisfies this structurally.
    """

    def publish(self, event: Event) -> None: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class BackoffPolicy:
    """Exponential backoff with proportional jitter.

    Jitter is not decoration. Without it, the log stream and the event watcher fail together (they
    share a daemon), back off together, and retry in the same millisecond forever - so a daemon
    that is briefly overloaded gets hit by a synchronised herd on every cycle.
    """

    base_seconds: float = 1.0
    factor: float = 2.0
    max_seconds: float = 30.0
    jitter: float = 0.2
    """Fraction of the delay to spread by, so 0.2 means the delay lands within +/-20%."""

    def delay_for(self, attempt: int, *, jitter_source: Callable[[], float]) -> float:
        """Delay before attempt number ``attempt`` (1-based). 1, 2, 4, 8, 16, 30, 30, ..."""
        exponent = max(attempt - 1, 0)
        raw = min(self.base_seconds * self.factor**exponent, self.max_seconds)
        spread = raw * self.jitter
        return max(0.0, raw + (jitter_source() * 2.0 - 1.0) * spread)


@final
class DedupeRing:
    """A bounded set of recently seen ``(timestamp, hash)`` pairs.

    Docker's ``since`` parameter has **second** granularity and is inclusive, so every reattach
    replays the whole of the second the last line landed in. Without this, a container that
    flaps once a minute produces a duplicate join for every reconnect, and every consumer
    downstream - roster, session stats, Discord - is quietly wrong.

    Hashing the text rather than storing it keeps the ring small and means a 4KB chat message
    costs the same as a short one.
    """

    __slots__ = ("_capacity", "_order", "_seen")

    def __init__(self, capacity: int = DEFAULT_DEDUPE_CAPACITY) -> None:
        self._capacity = capacity
        self._order: deque[tuple[float | None, int]] = deque()
        self._seen: set[tuple[float | None, int]] = set()

    def check(self, ts: datetime | None, text: str) -> bool:
        """Record ``(ts, text)`` and return True if it had been seen before."""
        key = (ts.timestamp() if ts is not None else None, hash(text))
        if key in self._seen:
            return True
        self._seen.add(key)
        self._order.append(key)
        if len(self._order) > self._capacity:
            self._seen.discard(self._order.popleft())
        return False

    def clear(self) -> None:
        """Forget everything. Called when the container id changes: a new container's log stream
        shares no lines with the old one, and carrying the old hashes over could only suppress a
        genuinely new line that happened to be identical."""
        self._order.clear()
        self._seen.clear()

    def __len__(self) -> int:
        return len(self._order)


async def _aclose(stream: AsyncIterator[object]) -> None:
    """Close an async generator promptly rather than waiting for the collector.

    The runtime's ``finally`` block is what stops the pump thread and closes the socket. Leaving
    that to garbage collection means a reconnect can briefly hold two streams open on the same
    container, and the second one then replays lines the first is still delivering.
    """
    if isinstance(stream, AsyncGenerator):
        await stream.aclose()


@final
class DockerManager:
    """Owns the two long-lived attachments to the container runtime.

    Three coroutines, spawned by the supervisor as separate non-critical tasks:
    :meth:`run_log_stream`, :meth:`run_event_watcher`, :meth:`run_reconcile`. They are separate
    because a broken log stream must not stop the event watcher noticing that the container died,
    and neither must stop the reconcile inspect that is the safety net for both.
    """

    def __init__(
        self,
        *,
        runtime: ContainerRuntime,
        container: str,
        server_id: ServerId,
        clock: Clock,
        sink: EventSink,
        on_line: Callable[[LogLine], None],
        on_event: Callable[[RuntimeEvent], None] | None = None,
        on_snapshot: Callable[[ContainerSnapshot], None] | None = None,
        on_log_eof: Callable[[], None] | None = None,
        backoff: BackoffPolicy | None = None,
        reconcile_interval: float = DEFAULT_RECONCILE_INTERVAL,
        eof_retry_seconds: float = DEFAULT_EOF_RETRY_SECONDS,
        dedupe_capacity: int = DEFAULT_DEDUPE_CAPACITY,
        initial_tail: int = 0,
        jitter_source: Callable[[], float] | None = None,
    ) -> None:
        """Wire the manager.

        Args:
            runtime: The container runtime. Faked in every test.
            container: The container **name**. Never an id - see :meth:`_resolve`.
            server_id: Stamped onto the two events this module publishes.
            clock: Injected time. Every backoff and every interval is measured on it.
            sink: Where ``RuntimeUnavailable`` / ``RuntimeRestored`` go.
            on_line: Called once per de-duplicated log line, in order. Synchronous, because the
                bus's ``publish`` is synchronous and the log pipeline is a pure function of the
                line plus its own state.
            on_event: Called once per de-duplicated Docker event.
            on_snapshot: Called with every inspect this module performs - both the reconnect
                resolution and the reconcile poll - so lifecycle sees them without polling twice.
            on_log_eof: Called on a clean end of the log stream, which corroborates ``die``.
            backoff: Reconnect policy.
            reconcile_interval: Seconds between safety-net inspects.
            eof_retry_seconds: How long to wait before reattaching after a clean EOF. Short,
                because a clean EOF usually means the container stopped and will be started again
                by us in a moment.
            dedupe_capacity: Size of the replay-suppression ring.
            initial_tail: Lines of history to request on the very first attach, when there is no
                ``since`` to work from. Zero means "only what happens from now".
            jitter_source: Uniform ``[0, 1)`` source. Injected so backoff timing is exact in
                tests: pass ``lambda: 0.5`` for no jitter at all.
        """
        self._runtime = runtime
        self._container = container
        self._server_id = server_id
        self._clock = clock
        self._sink = sink
        self._on_line = on_line
        self._on_event = on_event
        self._on_snapshot = on_snapshot
        self._on_log_eof = on_log_eof
        self._backoff = backoff if backoff is not None else BackoffPolicy()
        self._reconcile_interval = reconcile_interval
        self._eof_retry_seconds = eof_retry_seconds
        self._initial_tail = initial_tail
        self._jitter = jitter_source if jitter_source is not None else _default_jitter

        self._line_dedupe = DedupeRing(dedupe_capacity)
        self._event_dedupe = DedupeRing(dedupe_capacity)
        self._last_line_ts: datetime | None = None
        self._last_event_ts: datetime | None = None
        self._container_id: str | None = None
        self._resume_from_run_start = False

        self._available = True
        self._unavailable_at: float | None = None
        self._closing = False

        self._log_attaches = 0
        self._lines_delivered = 0
        self._event_attaches = 0
        self._duplicates_dropped = 0
        self._outages = 0
        self._clean_eofs = 0
        self._not_running_polls = 0

    # ------------------------------------------------------------------------------- inspect

    @property
    def available(self) -> bool:
        """False between a ``RuntimeUnavailable`` and its matching ``RuntimeRestored``."""
        return self._available

    @property
    def container_id(self) -> str | None:
        """The id most recently resolved from the name. Informational only; never cached for use
        as a handle."""
        return self._container_id

    @property
    def last_line_ts(self) -> datetime | None:
        """The ``since`` the next reattach will use."""
        return self._last_line_ts

    @property
    def stats(self) -> Mapping[str, int]:
        """Counters, polled rather than pushed - the same rule the bus follows."""
        return {
            "log_attaches": self._log_attaches,
            "lines_delivered": self._lines_delivered,
            "event_attaches": self._event_attaches,
            "duplicates_dropped": self._duplicates_dropped,
            "outages": self._outages,
            "clean_eofs": self._clean_eofs,
            "not_running_polls": self._not_running_polls,
            "dedupe_size": len(self._line_dedupe),
        }

    # ---------------------------------------------------------------------------- the loops

    async def run_log_stream(self) -> None:
        """Keep a log stream attached forever. Spawned as a supervised task."""
        attempt = 0
        while not self._closing:
            outcome = _LogAttach.FAILED
            delivered_before = self._lines_delivered
            try:
                outcome = await self._pump_logs()
            except ContainerNotFoundError:
                # A legitimate state, not a fault: the container may not have been created yet, or
                # somebody renamed it. Warn, back off, keep trying - never exit.
                log.warning("container %r does not exist; will keep looking", self._container)
            except RuntimeUnavailableError as exc:
                self._mark_unavailable(exc, source="log stream")
            except ContainerRuntimeError as exc:
                log.warning("log stream failed: %s", exc)
            if self._closing:
                return
            if self._lines_delivered > delivered_before:
                # The attach was productive, so whatever ended it starts a fresh backoff ladder
                # rather than inheriting the one that got us here - otherwise a stream that ran for
                # a day after a long outage would reconnect at the 30-second cap. Lines delivered
                # is the marker rather than "we called follow_logs", because a follow that fails on
                # its first read has been *called* without ever having been attached.
                attempt = 0
            if outcome is _LogAttach.EOF:
                # Clean EOF. The container stopped; that is a signal corroborating `die`, not an
                # error, so the retry is prompt. Exactly one of these is produced per stop: the
                # reattach that follows finds the container not running and takes the branch below.
                self._clean_eofs += 1
                if self._on_log_eof is not None:
                    self._on_log_eof()
                await self._clock.sleep(self._eof_retry_seconds)
                continue
            attempt += 1
            if outcome is _LogAttach.NOT_RUNNING:
                # Nothing to attach to, and nothing alarming about that. Backing off is what turns
                # "the server has been off for two days" from ~173,000 inspect-plus-attach round
                # trips into one inspect every thirty seconds.
                self._not_running_polls += 1
                await self._sleep_backoff(attempt, what="log stream", quiet=True)
                continue
            await self._sleep_backoff(attempt, what="log stream")

    async def run_event_watcher(self) -> None:
        """Keep the Docker event stream attached forever."""
        attempt = 0
        while not self._closing:
            attached = False
            try:
                attached = await self._pump_events()
            except RuntimeUnavailableError as exc:
                self._mark_unavailable(exc, source="event watcher")
            except ContainerRuntimeError as exc:
                log.warning("event watcher failed: %s", exc)
            if self._closing:
                return
            if attached:
                attempt = 0
                await self._clock.sleep(self._eof_retry_seconds)
                continue
            attempt += 1
            await self._sleep_backoff(attempt, what="event watcher")

    async def run_reconcile(self) -> None:
        """Inspect on a timer, as the safety net for events we never saw.

        ``health_status`` events are **edge-triggered**: Docker emits one when the status changes
        and never again. So a missed event is permanent, and the only way to notice is to ask.
        This is also what recovers state after an outage without needing the event stream's
        ``since`` to be exactly right.
        """
        while not self._closing:
            await self._clock.sleep(self._reconcile_interval)
            if self._closing:
                return
            try:
                snapshot = await self._runtime.inspect(self._container)
            except RuntimeUnavailableError as exc:
                self._mark_unavailable(exc, source="reconcile")
                continue
            except ContainerRuntimeError as exc:
                log.warning("reconcile inspect failed: %s", exc)
                continue
            self._mark_available()
            self._emit_snapshot(snapshot)

    async def aclose(self) -> None:
        """Ask the loops to stop at their next opportunity. Idempotent, never raises.

        Deliberately does not cancel anything: the supervisor cancels the tasks in reverse spawn
        order with per-task timeouts, and a manager that cancelled itself would race that.
        """
        self._closing = True

    # -------------------------------------------------------------------------------- pumping

    async def _pump_logs(self) -> _LogAttach:
        """Attach and deliver lines until the stream ends. See :class:`_LogAttach`."""
        snapshot = await self._resolve()
        if not snapshot.running:
            # Real Docker ends a follow on a stopped container immediately, and a clean EOF resets
            # the backoff, so attaching here would spin forever at one round trip per second while
            # the server is simply off. Waiting is free as long as it costs no lines - which is
            # what the `_resume_from_run_start` flag below buys.
            self._resume_from_run_start = True
            log.debug("container %r is not running; not attaching a log stream", self._container)
            return _LogAttach.NOT_RUNNING
        since = self._last_line_ts
        if since is None and self._resume_from_run_start:
            # We waited out a stopped container. Resuming from this run's StartedAt means the
            # backoff can never cost us `Starting minecraft server version ...` or `Done (Ns)!`,
            # which is what readiness detection is built on.
            since = snapshot.started_at
        self._resume_from_run_start = False
        # tail=-1 means "all" - required whenever `since` is set, because Docker applies `tail`
        # after `since` and a tail of 0 would return nothing at all, defeating the backfill.
        tail = -1 if since is not None else self._initial_tail
        stream = self._runtime.follow_logs(self._container, since=since, tail=tail)
        self._log_attaches += 1
        try:
            async for line in stream:
                if self._line_dedupe.check(line.ts, line.text):
                    self._duplicates_dropped += 1
                    continue
                if line.ts is not None:
                    self._last_line_ts = line.ts
                self._lines_delivered += 1
                self._on_line(line)
        finally:
            await _aclose(stream)
        return _LogAttach.EOF

    async def _pump_events(self) -> bool:
        """Attach and deliver Docker events until the stream ends."""
        since = self._last_event_ts
        stream = self._runtime.watch_events(self._container, since=since)
        self._event_attaches += 1
        try:
            async for event in stream:
                self._mark_available()
                if self._event_dedupe.check(event.ts, f"{event.action}|{event.container_id}"):
                    self._duplicates_dropped += 1
                    continue
                self._last_event_ts = event.ts
                if self._on_event is not None:
                    self._on_event(event)
        finally:
            await _aclose(stream)
        return True

    async def _resolve(self) -> ContainerSnapshot:
        """Look the container up **by name** and notice if it is a different container now.

        This is the single most important line in the reconnect path. A ``compose down && compose
        up`` produces a container with the same name and a new id; a manager holding the old id
        goes on asking Docker about a container that no longer exists, gets an empty stream, and
        reports a healthy silence forever. Resolving by name every time makes that impossible, and
        comparing the id is what tells us the dedupe ring and the ``since`` cursor now refer to a
        log that does not exist.
        """
        snapshot = await self._runtime.inspect(self._container)
        self._mark_available()
        self._emit_snapshot(snapshot)
        if snapshot.absent:
            raise ContainerNotFoundError(self._container)
        new_id = snapshot.id
        if new_id is not None and new_id != self._container_id:
            if self._container_id is not None:
                log.info(
                    "container %r was recreated (%s -> %s); clearing dedupe ring and log cursor",
                    self._container,
                    self._container_id[:12],
                    new_id[:12],
                )
                self._line_dedupe.clear()
                self._last_line_ts = None
            self._container_id = new_id
        return snapshot

    # ------------------------------------------------------------------------------ availability

    def _mark_unavailable(self, error: BaseException, *, source: str) -> None:
        """Publish ``RuntimeUnavailable`` **once per outage**, not once per failed attempt.

        Level-triggering this would fill Discord with one alert per second for the duration of a
        socket permissions problem. The whole point of the event is to move lifecycle into BLIND
        and keep last-known state, and that only needs saying once.
        """
        log.warning("docker unavailable (%s): %s", source, error)
        if not self._available:
            return
        self._available = False
        self._outages += 1
        self._unavailable_at = self._clock.monotonic()
        endpoint = getattr(error, "endpoint", None)
        self._sink.publish(
            RuntimeUnavailable(
                ts=self._clock.now(),
                server_id=self._server_id,
                source=Source.RUNTIME,
                raw=f"docker unavailable ({source}): {error}",
                error=str(error),
                endpoint=endpoint if isinstance(endpoint, str) else None,
            )
        )

    def _mark_available(self) -> None:
        """Publish ``RuntimeRestored`` on the rising edge, with the outage measured monotonically.

        Monotonic, not wall clock: an outage long enough to matter is long enough for NTP to step
        the clock, and a negative downtime in a session summary is a bug report waiting to happen.
        """
        if self._available:
            return
        self._available = True
        started = self._unavailable_at
        downtime = self._clock.monotonic() - started if started is not None else None
        self._unavailable_at = None
        log.info("docker reachable again after %.1fs", downtime if downtime is not None else 0.0)
        self._sink.publish(
            RuntimeRestored(
                ts=self._clock.now(),
                server_id=self._server_id,
                source=Source.RUNTIME,
                raw="docker reachable again",
                downtime_seconds=downtime,
            )
        )

    def _emit_snapshot(self, snapshot: ContainerSnapshot) -> None:
        if self._on_snapshot is not None:
            self._on_snapshot(snapshot)

    async def _sleep_backoff(self, attempt: int, *, what: str, quiet: bool = False) -> None:
        """Wait out one backoff step.

        ``quiet`` drops the line to DEBUG. It is set for the "container is not running" path,
        which is a normal state rather than a fault and would otherwise write an INFO line every
        thirty seconds for as long as the server is off.
        """
        delay = self._backoff.delay_for(attempt, jitter_source=self._jitter)
        log.log(
            logging.DEBUG if quiet else logging.INFO,
            "reattaching %s in %.1fs (attempt %d)",
            what,
            delay,
            attempt,
        )
        await self._clock.sleep(delay)
