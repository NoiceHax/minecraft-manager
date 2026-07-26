"""Time, injected.

This is the *only* module in the project allowed to call :func:`asyncio.sleep`,
:func:`time.monotonic` or :meth:`datetime.datetime.now`. Everything else takes a :class:`Clock`
through its constructor. The rule is enforced mechanically by the
``flake8-tidy-imports.banned-api`` block in ``pyproject.toml``; this file carries the only
``TID251`` per-file-ignore.

The payoff is that "15-minute idle timeout, 60-second polling, warn at 2 minutes" is a test that
runs in about a millisecond with exact ordering assertions, instead of a test that either sleeps
for fifteen minutes or lies. ``freezegun``/``time-machine`` do not solve this: they patch the wall
clock but not the event loop's, so ``asyncio.sleep(900)`` still takes 900 real seconds.

All wall-clock values are tz-aware UTC. There are no naive datetimes anywhere in this codebase.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol, final

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "Clock",
    "ManualClock",
    "SystemClock",
    "TimerHandle",
]


class TimerHandle(Protocol):
    """What :meth:`Clock.call_later` hands back. :class:`asyncio.TimerHandle` satisfies it."""

    def cancel(self) -> None:
        """Cancel the pending callback. Calling this after it fired is a no-op."""
        ...


class Clock(Protocol):
    """The four time primitives any component may use.

    Implementations: :class:`SystemClock` in production, :class:`ManualClock` in tests.
    """

    def now(self) -> datetime:
        """Current wall-clock time, always tz-aware UTC.

        Use for anything that gets persisted, serialised or shown to a human. Never for measuring
        a duration: wall clocks jump.
        """
        ...

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin, never decreasing.

        Use for every duration, deadline and backoff computation.
        """
        ...

    async def sleep(self, delay: float) -> None:
        """Suspend the calling coroutine for ``delay`` seconds."""
        ...

    def call_later(self, delay: float, callback: Callable[[], None]) -> TimerHandle:
        """Schedule ``callback`` to run ``delay`` seconds from now. Returns a cancellable handle."""
        ...


@final
class SystemClock:
    """Real time. The production :class:`Clock`."""

    __slots__ = ()

    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, delay: float) -> None:
        await asyncio.sleep(delay)

    def call_later(self, delay: float, callback: Callable[[], None]) -> TimerHandle:
        return asyncio.get_running_loop().call_later(delay, callback)


class _Timer:
    """One pending waiter inside a :class:`ManualClock`."""

    __slots__ = ("callback", "cancelled", "deadline", "future", "seq")

    def __init__(
        self,
        *,
        deadline: float,
        seq: int,
        callback: Callable[[], None] | None = None,
        future: asyncio.Future[None] | None = None,
    ) -> None:
        self.deadline = deadline
        self.seq = seq
        self.callback = callback
        self.future = future
        self.cancelled = False

    def __repr__(self) -> str:
        kind = "callback" if self.callback is not None else "sleep"
        return (
            f"<_Timer {kind} deadline={self.deadline!r} seq={self.seq} cancelled={self.cancelled}>"
        )


@final
class _ManualTimerHandle:
    """The :class:`TimerHandle` returned by :meth:`ManualClock.call_later`."""

    __slots__ = ("_timer",)

    def __init__(self, timer: _Timer) -> None:
        self._timer = timer

    def cancel(self) -> None:
        self._timer.cancelled = True
        self._timer.callback = None

    def cancelled(self) -> bool:
        return self._timer.cancelled


# How many event-loop passes to give the loop after firing each waiter. One is enough for a
# coroutine that wakes and immediately re-sleeps (set_result schedules the task wakeup with
# call_soon, and a single `await asyncio.sleep(0)` runs it). Three covers a woken coroutine that
# has to hop through an intermediate await - an asyncio.Event, a queue get, a nested helper -
# before it registers its next timer. Getting this wrong makes a poll loop advance exactly one
# iteration per advance() call, which is the subtly-wrong-test failure mode this whole class
# exists to prevent.
_YIELDS_PER_STEP = 3


@final
class ManualClock:
    """A :class:`Clock` whose time only moves when a test tells it to.

    ``sleep`` and ``call_later`` register waiters against a virtual monotonic timeline;
    :meth:`advance` fires everything due, in deadline order, ties broken by registration order.

    The contract that makes timing tests correct: :meth:`advance` yields to the event loop
    **between firing each waiter**, so a coroutine that wakes and immediately re-sleeps has
    registered its new timer before ``advance`` decides what is due next. Without that, a
    ``while True: await clock.sleep(interval)`` poll loop would run exactly one iteration per
    ``advance()`` call no matter how far you advanced.
    """

    __slots__ = ("_monotonic", "_now", "_seq", "_timers")

    def __init__(
        self,
        *,
        start: datetime | None = None,
        monotonic_start: float = 0.0,
    ) -> None:
        """Create a clock.

        Args:
            start: Initial wall-clock time. Must be tz-aware; defaults to 2026-01-01T00:00:00Z so
                that test output is stable and obviously synthetic.
            monotonic_start: Initial monotonic value.
        """
        if start is None:
            start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        if start.tzinfo is None:
            msg = "ManualClock start must be tz-aware (all datetimes in this project are UTC)"
            raise ValueError(msg)
        self._now: datetime = start.astimezone(UTC)
        self._monotonic: float = monotonic_start
        self._timers: dict[int, _Timer] = {}
        self._seq: int = 0

    # -- Clock ------------------------------------------------------------------------------

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    async def sleep(self, delay: float) -> None:
        """Suspend until virtual time has advanced by ``delay``.

        A non-positive delay yields to the event loop once and returns, matching
        :func:`asyncio.sleep`.
        """
        if delay <= 0:
            await asyncio.sleep(0)
            return
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        timer = self._register(delay, future=future)
        try:
            await future
        finally:
            self._timers.pop(id(timer), None)

    def call_later(self, delay: float, callback: Callable[[], None]) -> TimerHandle:
        timer = self._register(max(delay, 0.0), callback=callback)
        return _ManualTimerHandle(timer)

    # -- test control -----------------------------------------------------------------------

    async def advance(self, seconds: float) -> None:
        """Move virtual time forward by ``seconds``, firing every waiter that comes due.

        Waiters fire in ``(deadline, registration order)`` order. The clock is set to each
        waiter's exact deadline before it fires, so a coroutine woken by a 30s sleep observes
        ``now()`` at ``+30s``, not at the end of the whole advance. The loop is given a few passes
        after each firing so newly registered timers are visible to the rest of this advance.
        """
        if seconds < 0:
            msg = "ManualClock.advance does not go backwards"
            raise ValueError(msg)
        target = self._monotonic + seconds
        while True:
            timer = self._next_due(target)
            if timer is None:
                break
            self._set_monotonic(timer.deadline)
            self._fire(timer)
            await self._drain_loop()
        self._set_monotonic(target)
        await self._drain_loop()

    async def advance_to(self, when: datetime) -> None:
        """Advance until :meth:`now` reaches ``when``. Never moves backwards."""
        if when.tzinfo is None:
            msg = "ManualClock.advance_to needs a tz-aware datetime"
            raise ValueError(msg)
        delta = (when.astimezone(UTC) - self._now).total_seconds()
        await self.advance(max(delta, 0.0))

    async def tick(self) -> None:
        """Yield to the event loop without moving time. Lets pending tasks make progress."""
        await self._drain_loop()

    @property
    def pending_timers(self) -> int:
        """How many waiters are registered. Useful for asserting a timer was cancelled."""
        return sum(1 for t in self._timers.values() if not t.cancelled)

    @property
    def pending_deadlines(self) -> Sequence[float]:
        """Monotonic deadlines of every live waiter, sorted. For assertions in tests."""
        return sorted(t.deadline for t in self._timers.values() if not t.cancelled)

    # -- internals --------------------------------------------------------------------------

    def _register(
        self,
        delay: float,
        *,
        callback: Callable[[], None] | None = None,
        future: asyncio.Future[None] | None = None,
    ) -> _Timer:
        self._seq += 1
        timer = _Timer(
            deadline=self._monotonic + delay,
            seq=self._seq,
            callback=callback,
            future=future,
        )
        self._timers[id(timer)] = timer
        return timer

    def _next_due(self, target: float) -> _Timer | None:
        due = [t for t in self._timers.values() if not t.cancelled and t.deadline <= target]
        if not due:
            return None
        return min(due, key=lambda t: (t.deadline, t.seq))

    def _set_monotonic(self, value: float) -> None:
        if value <= self._monotonic:
            return
        self._now += timedelta(seconds=value - self._monotonic)
        self._monotonic = value

    def _fire(self, timer: _Timer) -> None:
        self._timers.pop(id(timer), None)
        if timer.cancelled:
            return
        future = timer.future
        if future is not None and not future.done():
            future.set_result(None)
        callback = timer.callback
        if callback is not None:
            timer.callback = None
            callback()

    async def _drain_loop(self) -> None:
        for _ in range(_YIELDS_PER_STEP):
            await asyncio.sleep(0)
