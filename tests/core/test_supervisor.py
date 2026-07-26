"""Task supervision: restart, backoff, critical failure, ordered shutdown.

Every wait in :mod:`mcmanager.core.supervisor` goes through the injected clock, so the whole
backoff schedule - 1s, 2s, 4s, ... capped at 30s, reset after 60 healthy seconds - is asserted
exactly by advancing a :class:`~mcmanager.clock.ManualClock`. Nothing here sleeps for real.

``jitter_ratio=0.0`` is used wherever a delay is asserted; the jitter itself gets its own test.
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING

import pytest

from mcmanager.core.supervisor import Supervisor, TaskPolicy

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from mcmanager.clock import ManualClock


def crash_after(
    calls: list[int],
    *,
    error: str = "boom",
) -> Callable[[], Coroutine[None, None, None]]:
    """A factory whose coroutine records the attempt and immediately raises."""

    async def factory() -> None:
        calls.append(len(calls))
        raise RuntimeError(error)

    return factory


def block_forever(
    started: asyncio.Event | None = None,
) -> Callable[[], Coroutine[None, None, None]]:
    async def factory() -> None:
        if started is not None:
            started.set()
        await asyncio.Event().wait()

    return factory


async def stop(supervisor: Supervisor) -> None:
    await supervisor.shutdown(task_timeout=0.2)


# ---------------------------------------------------------------------------- restart policy


async def test_a_crashing_task_is_restarted_after_the_initial_backoff(clock: ManualClock) -> None:
    supervisor = Supervisor(clock, jitter_ratio=0.0)
    calls: list[int] = []
    supervisor.spawn("pump", crash_after(calls), backoff_initial=1.0)

    await clock.tick()
    assert calls == [0], "the first run happens immediately"
    assert clock.pending_deadlines == [1.0], "and then it waits exactly the initial backoff"

    await clock.advance(1.0)
    assert calls == [0, 1]

    await stop(supervisor)


async def test_backoff_doubles_and_is_capped(clock: ManualClock) -> None:
    supervisor = Supervisor(clock, jitter_ratio=0.0)
    calls: list[int] = []
    supervisor.spawn("pump", crash_after(calls), backoff_initial=1.0, backoff_max=8.0)

    await clock.tick()
    waits: list[float] = []
    for _ in range(6):
        pending = clock.pending_deadlines
        assert len(pending) == 1
        wait = pending[0] - clock.monotonic()
        waits.append(wait)
        await clock.advance(wait)

    assert waits == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]
    assert len(calls) == 7

    await stop(supervisor)


async def test_backoff_resets_after_the_task_has_been_healthy(clock: ManualClock) -> None:
    """Otherwise one crash an hour eventually means a 30-second wait from ancient history."""
    supervisor = Supervisor(clock, jitter_ratio=0.0, healthy_after=60.0)
    attempts: list[float] = []

    async def factory() -> None:
        attempts.append(clock.monotonic())
        # Runs healthily for two minutes on the third attempt, then dies.
        lifetime = 120.0 if len(attempts) == 3 else 0.0
        await clock.sleep(lifetime)
        msg = "died"
        raise RuntimeError(msg)

    supervisor.spawn("pump", factory, backoff_initial=1.0, backoff_max=30.0)

    await clock.tick()
    await clock.advance(1.0)  # first restart after 1s
    await clock.advance(2.0)  # second restart after 2s -> third attempt runs 120s
    await clock.advance(120.0)  # the healthy run ends
    assert clock.pending_deadlines == [pytest.approx(clock.monotonic() + 1.0)], (
        "a task that survived longer than healthy_after restarts at the initial backoff"
    )

    await stop(supervisor)


async def test_a_task_that_returns_is_also_restarted_under_the_restart_policy(
    clock: ManualClock,
) -> None:
    """A pump returning is as much a stopped pump as a pump raising."""
    supervisor = Supervisor(clock, jitter_ratio=0.0)
    runs = 0

    async def factory() -> None:
        nonlocal runs
        runs += 1

    supervisor.spawn("pump", factory, backoff_initial=1.0)
    await clock.tick()
    await clock.advance(1.0)

    assert runs == 2
    await stop(supervisor)


async def test_the_once_policy_does_not_restart(clock: ManualClock) -> None:
    supervisor = Supervisor(clock, jitter_ratio=0.0)
    calls: list[int] = []
    supervisor.spawn("one-shot", crash_after(calls), policy=TaskPolicy.ONCE)

    await clock.tick()
    await clock.advance(600.0)

    assert calls == [0]
    assert clock.pending_timers == 0
    assert supervisor.failure_counts["one-shot"] == 1
    await stop(supervisor)


async def test_restart_and_failure_counts_are_tracked(clock: ManualClock) -> None:
    supervisor = Supervisor(clock, jitter_ratio=0.0)
    calls: list[int] = []
    supervisor.spawn("pump", crash_after(calls), backoff_initial=1.0)

    await clock.tick()
    await clock.advance(1.0)
    await clock.advance(2.0)

    assert supervisor.restart_counts["pump"] == 3
    assert supervisor.failure_counts["pump"] == 3
    assert supervisor.task_names == ("pump",)
    await stop(supervisor)


# ----------------------------------------------------------------------------------- jitter


async def test_jitter_stays_within_the_configured_band(clock: ManualClock) -> None:
    """Jitter exists so every pump does not restart in lockstep after a daemon-wide outage."""
    supervisor = Supervisor(clock, jitter_ratio=0.2, rng=random.Random(1234))  # noqa: S311
    calls: list[int] = []
    supervisor.spawn("pump", crash_after(calls), backoff_initial=10.0, backoff_max=10.0)

    await clock.tick()
    observed: list[float] = []
    for _ in range(20):
        wait = clock.pending_deadlines[0] - clock.monotonic()
        observed.append(wait)
        await clock.advance(wait)

    assert all(8.0 <= wait <= 12.0 for wait in observed)
    assert len(set(observed)) > 1, "jitter must actually vary the delay"

    await stop(supervisor)


# -------------------------------------------------------------------------- critical tasks


async def test_a_critical_task_dying_requests_shutdown_and_is_not_restarted(
    clock: ManualClock,
) -> None:
    reported: list[tuple[str, str | None]] = []
    supervisor = Supervisor(
        clock,
        jitter_ratio=0.0,
        on_critical_failure=lambda name, error: reported.append(
            (name, None if error is None else str(error))
        ),
    )
    calls: list[int] = []
    supervisor.spawn("bus", crash_after(calls, error="dispatch loop died"), critical=True)

    await clock.tick()
    await clock.advance(600.0)

    assert calls == [0], "a critical task is never restarted; a second dispatch loop would race"
    assert supervisor.shutdown_requested.is_set()
    assert reported == [("bus", "dispatch loop died")]
    await stop(supervisor)


async def test_a_critical_task_returning_cleanly_also_requests_shutdown(
    clock: ManualClock,
) -> None:
    supervisor = Supervisor(clock, jitter_ratio=0.0)

    async def factory() -> None:
        return

    supervisor.spawn("bus", factory, critical=True)
    await clock.tick()

    assert supervisor.shutdown_requested.is_set()
    await stop(supervisor)


async def test_a_failing_shutdown_callback_does_not_prevent_the_shutdown(
    clock: ManualClock,
) -> None:
    def explode(name: str, error: BaseException | None) -> None:
        msg = "the shutdown callback is broken"
        raise RuntimeError(msg)

    supervisor = Supervisor(clock, jitter_ratio=0.0, on_critical_failure=explode)
    supervisor.spawn("bus", crash_after([]), critical=True)

    await clock.tick()

    assert supervisor.shutdown_requested.is_set()
    await stop(supervisor)


async def test_request_shutdown_is_idempotent_and_only_reports_once(clock: ManualClock) -> None:
    reported: list[str] = []
    supervisor = Supervisor(
        clock,
        on_critical_failure=lambda name, _error: reported.append(name),
    )

    supervisor.request_shutdown("sigterm")
    supervisor.request_shutdown("sigterm again")

    assert reported == ["sigterm"]
    assert supervisor.shutdown_requested.is_set()


# -------------------------------------------------------------------------------- shutdown


async def test_shutdown_cancels_in_reverse_spawn_order(clock: ManualClock) -> None:
    """Producers are spawned after the consumers they feed, so they must stop first."""
    supervisor = Supervisor(clock)
    cancelled: list[str] = []

    def factory_for(name: str) -> Callable[[], Coroutine[None, None, None]]:
        async def factory() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(name)
                raise

        return factory

    for name in ("bus", "docker", "discord"):
        supervisor.spawn(name, factory_for(name))
    await clock.tick()

    await supervisor.shutdown(task_timeout=0.2)

    assert cancelled == ["discord", "docker", "bus"]


async def test_shutdown_cancels_a_task_parked_on_its_backoff_sleep(clock: ManualClock) -> None:
    supervisor = Supervisor(clock, jitter_ratio=0.0)
    calls: list[int] = []
    supervisor.spawn("pump", crash_after(calls), backoff_initial=30.0)

    await clock.tick()
    assert clock.pending_timers == 1

    await supervisor.shutdown(task_timeout=0.2)

    assert clock.pending_timers == 0, "the backoff timer must not outlive the task"
    assert calls == [0]


async def test_shutdown_is_idempotent(clock: ManualClock) -> None:
    supervisor = Supervisor(clock)
    supervisor.spawn("pump", block_forever())
    await clock.tick()

    await supervisor.shutdown(task_timeout=0.2)
    await supervisor.shutdown(task_timeout=0.2)


async def test_a_task_that_swallows_cancellation_is_abandoned_not_awaited_forever(
    clock: ManualClock,
) -> None:
    """A stuck task must not hold up the rest of the 30-second compose grace period."""
    supervisor = Supervisor(clock)
    release = asyncio.Event()
    swallowed = asyncio.Event()
    running: list[asyncio.Task[None]] = []

    async def stubborn() -> None:
        current = asyncio.current_task()
        assert current is not None
        running.append(current)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            swallowed.set()
            await release.wait()

    supervisor.spawn("stubborn", stubborn)
    await clock.tick()

    await supervisor.shutdown(task_timeout=0.05)

    assert swallowed.is_set()
    task = running[0]
    assert not task.done(), "shutdown returned rather than waiting on a task that will not stop"

    release.set()
    await asyncio.wait_for(task, timeout=1.0)


async def test_spawning_after_shutdown_is_refused(clock: ManualClock) -> None:
    supervisor = Supervisor(clock)
    await supervisor.shutdown(task_timeout=0.2)

    supervisor.spawn("late", block_forever())

    assert supervisor.task_names == ()


async def test_a_task_cancelled_mid_run_does_not_restart(clock: ManualClock) -> None:
    """Cancellation is a shutdown instruction, not a crash to recover from."""
    supervisor = Supervisor(clock, jitter_ratio=0.0)
    starts = 0
    started = asyncio.Event()

    async def factory() -> None:
        nonlocal starts
        starts += 1
        started.set()
        await asyncio.Event().wait()

    supervisor.spawn("pump", factory)
    await clock.tick()
    assert started.is_set()

    await supervisor.shutdown(task_timeout=0.2)
    await clock.advance(600.0)

    assert starts == 1


# --------------------------------------------------------------------------------- signals


async def test_signal_handlers_install_and_uninstall_cleanly(clock: ManualClock) -> None:
    """Unix takes ``loop.add_signal_handler``; Windows falls back to ``signal.signal``. Both must
    install, restore, and leave the process's handler table as they found it."""
    import signal

    supervisor = Supervisor(clock)
    before = signal.getsignal(signal.SIGINT)

    supervisor.install_signal_handlers(signal.SIGINT)
    try:
        assert not supervisor.shutdown_requested.is_set()
    finally:
        supervisor.remove_signal_handlers()

    assert signal.getsignal(signal.SIGINT) is before
    supervisor.remove_signal_handlers()


async def test_the_installed_handler_only_requests_shutdown(clock: ManualClock) -> None:
    """Whichever transport is used, the handler must do nothing but hand the fact to the loop.

    On Unix the loop takes the signal directly and there is no Python-level handler to poke, so
    that half of the assertion is skipped rather than faked.
    """
    import signal

    supervisor = Supervisor(clock)
    before = signal.getsignal(signal.SIGINT)
    supervisor.install_signal_handlers(signal.SIGINT)
    installed = signal.getsignal(signal.SIGINT)
    try:
        if installed is before or not callable(installed):
            pytest.skip("this platform routes signals through the event loop, not signal.signal")
        installed(int(signal.SIGINT), None)
        await asyncio.sleep(0)
        assert supervisor.shutdown_requested.is_set()
    finally:
        supervisor.remove_signal_handlers()
