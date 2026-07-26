"""ManualClock is the foundation every timing test in this project stands on.

If the ordering guarantees below break, tests elsewhere do not fail - they quietly assert the
wrong thing. Hence the paranoia in this file.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta

import pytest

from mcmanager.clock import ManualClock, SystemClock


def test_starts_tz_aware_utc_and_does_not_move_on_its_own() -> None:
    clock = ManualClock()
    first = clock.now()

    assert first.tzinfo is not None
    assert first.utcoffset() == timedelta(0)
    assert clock.now() == first
    assert clock.monotonic() == 0.0


def test_rejects_a_naive_start() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        ManualClock(start=datetime(2026, 1, 1))  # noqa: DTZ001


async def test_advance_moves_wall_clock_and_monotonic_in_lockstep() -> None:
    clock = ManualClock(start=datetime(2026, 7, 25, 12, 0, tzinfo=UTC))

    await clock.advance(90)

    assert clock.monotonic() == 90.0
    assert clock.now() == datetime(2026, 7, 25, 12, 1, 30, tzinfo=UTC)


async def test_advance_refuses_to_go_backwards() -> None:
    clock = ManualClock()
    with pytest.raises(ValueError, match="backwards"):
        await clock.advance(-1)


async def test_sleep_resumes_only_once_its_deadline_is_reached() -> None:
    clock = ManualClock()
    woke = False

    async def sleeper() -> None:
        nonlocal woke
        await clock.sleep(60)
        woke = True

    task = asyncio.create_task(sleeper())
    await clock.tick()

    await clock.advance(59)
    assert woke is False

    await clock.advance(1)
    assert woke is True
    await task


async def test_each_sleeper_observes_its_own_deadline_not_the_end_of_the_advance() -> None:
    """A coroutine woken by a 10s sleep must see ``now()`` at +10s, not at +100s.

    Otherwise every duration computed inside a woken handler is wrong by however far the test
    happened to advance.
    """
    clock = ManualClock()
    seen: list[tuple[str, float]] = []

    async def sleeper(name: str, delay: float) -> None:
        await clock.sleep(delay)
        seen.append((name, clock.monotonic()))

    later = asyncio.create_task(sleeper("later", 20))
    sooner = asyncio.create_task(sleeper("sooner", 10))
    await clock.tick()

    await clock.advance(100)

    assert seen == [("sooner", 10.0), ("later", 20.0)]
    assert clock.monotonic() == 100.0
    await asyncio.gather(sooner, later)


async def test_ties_break_on_registration_order() -> None:
    clock = ManualClock()
    order: list[str] = []

    async def sleeper(name: str) -> None:
        await clock.sleep(5)
        order.append(name)

    first = asyncio.create_task(sleeper("first"))
    await clock.tick()
    second = asyncio.create_task(sleeper("second"))
    await clock.tick()

    await clock.advance(5)

    assert order == ["first", "second"]
    await asyncio.gather(first, second)


async def test_a_poll_loop_runs_once_per_interval_within_a_single_advance() -> None:
    """The reason ``advance`` yields to the loop between firings.

    A ``while True: await clock.sleep(30)`` loop must tick five times when time moves 150 seconds,
    not once. Without the yield, the coroutine has not re-registered its next timer by the time
    ``advance`` decides what is still due, and every polling test in the project silently becomes
    a single-iteration test.
    """
    clock = ManualClock()
    ticks: list[float] = []

    async def poller() -> None:
        while True:
            await clock.sleep(30)
            ticks.append(clock.monotonic())

    task = asyncio.create_task(poller())
    await clock.tick()

    await clock.advance(150)

    assert ticks == [30.0, 60.0, 90.0, 120.0, 150.0]

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_a_coroutine_that_hops_through_an_event_still_re_registers_in_time() -> None:
    """The same guarantee, one await deeper.

    A handler that wakes, signals something, waits on an ``asyncio.Event`` and only then sleeps
    again is the shape of the idle manager. It must still tick once per interval.
    """
    clock = ManualClock()
    gate = asyncio.Event()
    gate.set()
    ticks: list[float] = []

    async def poller() -> None:
        while True:
            await clock.sleep(10)
            await gate.wait()
            ticks.append(clock.monotonic())

    task = asyncio.create_task(poller())
    await clock.tick()

    await clock.advance(50)

    assert ticks == [10.0, 20.0, 30.0, 40.0, 50.0]

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_sleep_of_zero_yields_without_needing_an_advance() -> None:
    clock = ManualClock()
    done = False

    async def zero_sleeper() -> None:
        nonlocal done
        await clock.sleep(0)
        done = True

    task = asyncio.create_task(zero_sleeper())
    await clock.tick()

    assert done is True
    assert clock.monotonic() == 0.0
    await task


async def test_call_later_fires_at_its_deadline() -> None:
    clock = ManualClock()
    fired_at: list[float] = []

    clock.call_later(45, lambda: fired_at.append(clock.monotonic()))
    assert clock.pending_timers == 1

    await clock.advance(44)
    assert fired_at == []

    await clock.advance(1)
    assert fired_at == [45.0]
    assert clock.pending_timers == 0


async def test_a_cancelled_timer_never_fires() -> None:
    clock = ManualClock()
    fired: list[str] = []

    handle = clock.call_later(10, lambda: fired.append("boom"))
    handle.cancel()

    await clock.advance(60)

    assert fired == []
    assert clock.pending_timers == 0


async def test_cancelling_a_sleeping_task_deregisters_its_timer() -> None:
    clock = ManualClock()

    async def sleeper() -> None:
        await clock.sleep(900)

    task = asyncio.create_task(sleeper())
    await clock.tick()
    assert clock.pending_timers == 1

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert clock.pending_timers == 0


async def test_pending_deadlines_are_visible_for_assertions() -> None:
    clock = ManualClock()
    clock.call_later(120, lambda: None)
    clock.call_later(30, lambda: None)

    assert list(clock.pending_deadlines) == [30.0, 120.0]


async def test_advance_to_a_wall_clock_time() -> None:
    clock = ManualClock(start=datetime(2026, 7, 25, 12, 0, tzinfo=UTC))
    fired: list[str] = []
    clock.call_later(600, lambda: fired.append("idle-stop"))

    await clock.advance_to(datetime(2026, 7, 25, 12, 15, tzinfo=UTC))

    assert fired == ["idle-stop"]
    assert clock.now() == datetime(2026, 7, 25, 12, 15, tzinfo=UTC)


async def test_a_fifteen_minute_timeout_costs_no_wall_clock_time() -> None:
    """The whole point: exact assertions about a 15-minute window, in about a millisecond."""
    clock = ManualClock()
    events: list[tuple[str, float]] = []

    async def idle_manager() -> None:
        await clock.sleep(13 * 60)
        events.append(("warn", clock.monotonic()))
        await clock.sleep(2 * 60)
        events.append(("stop", clock.monotonic()))

    task = asyncio.create_task(idle_manager())
    await clock.tick()

    await clock.advance(15 * 60)

    assert events == [("warn", 780.0), ("stop", 900.0)]
    await task


def test_system_clock_returns_tz_aware_utc() -> None:
    clock = SystemClock()
    now = clock.now()

    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)
    assert isinstance(clock.monotonic(), float)
