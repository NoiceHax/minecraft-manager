"""``FakeRuntime`` is test infrastructure, so it gets tested harder than most production code.

Every service test in this project asserts against a fake. A fake that is wrong in the same
direction as a wrong assumption produces a green suite and a broken daemon, which is the exact
failure mode plan section 13 forbids by saying *fake the interface, never the library*. That only
holds if the fake reproduces the parts of Docker's behaviour we depend on - and three of those are
deliberately awkward:

- ``since`` is floored to the second and replays that whole second;
- ``State.Health.Status`` is not cleared when the container stops;
- a follow on a container that is not running ends immediately instead of blocking.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from mcmanager.containers.dto import ContainerState, ExecResult, HealthState, HealthTiming
from mcmanager.containers.errors import (
    ContainerNotFoundError,
    LogStreamError,
    RuntimeUnavailableError,
)
from mcmanager.containers.fake import FakeRuntime
from mcmanager.core.types import Stream

if TYPE_CHECKING:
    from mcmanager.clock import ManualClock


@pytest.fixture
def runtime(clock: ManualClock) -> FakeRuntime:
    return FakeRuntime(clock=clock)


# --------------------------------------------------------------------------------- inspect


async def test_inspect_returns_a_snapshot_for_the_named_container(runtime: FakeRuntime) -> None:
    snapshot = await runtime.inspect("minecraft")
    assert snapshot.exists
    assert snapshot.name == "minecraft"
    assert snapshot.id == runtime.container_id
    assert snapshot.observed_at.tzinfo is not None


async def test_inspect_of_another_name_is_absent_not_an_error(runtime: FakeRuntime) -> None:
    """Absence is a legitimate state. Raising here would make a renamed container crash the
    daemon instead of warning."""
    snapshot = await runtime.inspect("valheim")
    assert snapshot.absent
    assert not snapshot.exists
    assert snapshot.state is ContainerState.ABSENT


async def test_set_absent_makes_the_container_disappear(runtime: FakeRuntime) -> None:
    runtime.set_absent()
    snapshot = await runtime.inspect("minecraft")
    assert snapshot.absent


async def test_defaults_mirror_the_homelab_container(runtime: FakeRuntime) -> None:
    """Tty false (so the multiplexed log path applies) and a 210s health guard window, which is
    the number plan section 5's start-period suppression is derived from."""
    snapshot = await runtime.inspect("minecraft")
    assert snapshot.tty is False
    assert snapshot.guard_window == timedelta(seconds=210)
    assert snapshot.restart_policy == "no"
    assert snapshot.stop_signal == "SIGTERM"
    assert snapshot.alias_for("homelab") == ("minecraft",)


async def test_a_stopped_container_reports_unknown_health_despite_a_stale_unhealthy(
    runtime: FakeRuntime,
) -> None:
    """The observed gotcha, reproduced deliberately.

    On the real exited container, ``State.Health.Status`` still read ``"unhealthy"`` with
    ``FailingStreak`` at 0. The fake keeps that string on stop for exactly this reason: any
    readiness logic that trusts it would report a permanently unhealthy server the moment it is
    turned off, and this is the test that proves the derived property covers for it.
    """
    runtime.set_health("unhealthy", failing_streak=0)
    runtime.set_state(ContainerState.EXITED, exit_code=0)

    snapshot = await runtime.inspect("minecraft")
    assert snapshot.health_reported_raw == "unhealthy"
    assert snapshot.health is HealthState.UNKNOWN


async def test_a_running_container_reports_the_health_docker_gave_it(runtime: FakeRuntime) -> None:
    runtime.set_state(ContainerState.RUNNING)
    runtime.set_health("healthy")
    snapshot = await runtime.inspect("minecraft")
    assert snapshot.health is HealthState.HEALTHY


async def test_health_timing_is_overridable_for_a_container_with_no_healthcheck(
    runtime: FakeRuntime,
) -> None:
    runtime.set_health_timing(HealthTiming())
    snapshot = await runtime.inspect("minecraft")
    assert snapshot.guard_window is None


# ----------------------------------------------------------------------------- start / stop


async def test_start_transitions_the_container_and_emits_a_start_event(
    runtime: FakeRuntime,
) -> None:
    await runtime.start("minecraft")
    snapshot = await runtime.inspect("minecraft")
    assert snapshot.running
    assert runtime.start_calls == ["minecraft"]
    assert [event.action for event in runtime.event_history] == ["start"]


async def test_stop_records_its_timeout_and_exits_zero_like_the_real_container(
    runtime: FakeRuntime,
) -> None:
    """Verified on the homelab: ``mc-server-runner`` traps SIGTERM, writes ``stop``, exits **0**.
    Not 143. A fake that said 143 would let a wrong crash-vs-clean-stop rule pass."""
    await runtime.start("minecraft")
    await runtime.stop("minecraft", timeout=90)

    assert runtime.stop_calls == [("minecraft", 90)]
    snapshot = await runtime.inspect("minecraft")
    assert not snapshot.running
    assert snapshot.exit_code == 0
    assert [event.action for event in runtime.event_history] == ["start", "die", "stop"]


async def test_die_event_carries_exit_code_as_a_string(runtime: FakeRuntime) -> None:
    """``Actor.Attributes["exitCode"]`` is a string in real Docker, which is why
    ``attrs["exitCode"] == 0`` is silently always false and ``RuntimeEvent.exit_code`` exists."""
    event = runtime.emit_event("die", exitCode="137")
    assert event.attributes["exitCode"] == "137"
    assert event.exit_code == 137


async def test_assert_stopped_with_timeout_passes_and_fails_honestly(
    runtime: FakeRuntime,
) -> None:
    await runtime.start("minecraft")
    with pytest.raises(AssertionError, match="never called"):
        runtime.assert_stopped_with_timeout(90)

    await runtime.stop("minecraft", timeout=90)
    runtime.assert_stopped_with_timeout(90)
    with pytest.raises(AssertionError, match=r"saw timeouts \[90\]"):
        runtime.assert_stopped_with_timeout(10)


async def test_assert_never_stopped_is_the_idle_dry_run_gate(runtime: FakeRuntime) -> None:
    runtime.assert_never_stopped()
    await runtime.start("minecraft")
    await runtime.stop("minecraft", timeout=90)
    with pytest.raises(AssertionError):
        runtime.assert_never_stopped()


async def test_start_and_stop_are_idempotent(runtime: FakeRuntime) -> None:
    await runtime.start("minecraft")
    await runtime.start("minecraft")
    await runtime.stop("minecraft", timeout=90)
    await runtime.stop("minecraft", timeout=90)
    assert [event.action for event in runtime.event_history] == ["start", "die", "stop"]


async def test_operations_on_a_missing_container_raise_not_found(runtime: FakeRuntime) -> None:
    with pytest.raises(ContainerNotFoundError):
        await runtime.start("valheim")


# ----------------------------------------------------------------------------------- logs


def _running(runtime: FakeRuntime) -> None:
    """Put the fake into ``running``.

    Every follow test needs this now, because ``follow_logs`` reproduces Docker's third
    awkward behaviour: on a container that is **not** running it delivers the history and ends
    immediately rather than waiting for output that can never come. The fake's resting state is
    ``exited``, mirroring the homelab container.
    """
    runtime.set_state(ContainerState.RUNNING, running=True, exit_code=None)


async def test_emitted_lines_reach_an_attached_follower(
    runtime: FakeRuntime, clock: ManualClock
) -> None:
    _running(runtime)
    received: list[str] = []

    async def follow() -> None:
        async for line in runtime.follow_logs("minecraft"):
            received.append(line.text)

    task = asyncio.create_task(follow())
    await clock.tick()
    runtime.emit_line("first")
    runtime.emit_line("second")
    await clock.tick()
    assert received == ["first", "second"]

    runtime.close_log_stream()
    await task
    assert runtime.attached_log_streams == 0


async def test_scripted_lines_fire_on_virtual_time(
    runtime: FakeRuntime, clock: ManualClock
) -> None:
    """A 33-second server startup, tested in about a millisecond, with exact ordering."""
    runtime.script_lines(
        [
            (0.0, "[13:24:04] [Server thread/INFO]: Starting minecraft server version 26.2"),
            (32.5, '[13:24:37] [Server thread/INFO]: Done (32.521s)! For help, type "help"'),
        ]
    )
    assert len(runtime.history) == 1

    await clock.advance(32.4)
    assert len(runtime.history) == 1

    await clock.advance(0.2)
    assert len(runtime.history) == 2
    assert "Done (32.521s)" in runtime.history[1].text


async def test_a_follower_that_attaches_late_still_gets_scripted_history(
    runtime: FakeRuntime, clock: ManualClock
) -> None:
    """Lines are emitted by the clock, not by the presence of a reader, so a reconnect after a
    gap can still backfill what it missed."""
    runtime.script_lines([(1.0, "early"), (2.0, "late")])
    await clock.advance(3.0)

    lines = await runtime.logs_tail("minecraft", lines=10)
    assert [line.text for line in lines] == ["early", "late"]


async def test_tail_zero_means_no_history_at_all(runtime: FakeRuntime) -> None:
    """Docker's semantics, and the reason the reconnect path asks for ``tail=-1`` with a
    ``since``: a tail of 0 would otherwise throw the backfill away."""
    runtime.emit_lines(["a", "b", "c"])
    assert await runtime.logs_tail("minecraft", lines=0) == []


async def test_tail_takes_the_last_n_lines(runtime: FakeRuntime) -> None:
    runtime.emit_lines(["a", "b", "c"])
    assert [line.text for line in await runtime.logs_tail("minecraft", lines=2)] == ["b", "c"]


async def test_since_is_floored_to_the_second_and_replays_that_whole_second(
    runtime: FakeRuntime, clock: ManualClock
) -> None:
    """**The behaviour the dedupe ring exists for.**

    Docker's ``since`` parameter has second granularity and is inclusive, so reattaching with the
    timestamp of the last line you saw hands you every line from that second again. A fake that
    filtered at microsecond precision would let a missing dedupe ring pass every test and then
    duplicate every join in production.
    """
    await clock.advance(0.25)
    runtime.emit_line("a")
    await clock.advance(0.25)
    second = runtime.emit_line("b")
    await clock.advance(1.0)
    third = runtime.emit_line("c")

    assert second.ts is not None
    assert third.ts is not None

    # Reattaching with the timestamp of "b" hands back "a" as well, even though "a" happened
    # 250ms *earlier*, because both landed in second 0. That is the duplicate.
    replayed = await runtime.logs_tail("minecraft", lines=-1, since=second.ts)
    assert [line.text for line in replayed] == ["a", "b", "c"]

    # A line from a strictly earlier second is not replayed, so the window really is one second
    # wide and the dedupe ring never has to span more than that.
    later = await runtime.logs_tail("minecraft", lines=-1, since=third.ts)
    assert [line.text for line in later] == ["c"]


async def test_log_lines_carry_tz_aware_timestamps_and_a_stream(runtime: FakeRuntime) -> None:
    line = runtime.emit_line("boom", stream=Stream.STDERR)
    assert line.stream is Stream.STDERR
    assert line.ts is not None
    assert line.ts.tzinfo is not None
    assert line.event_ts == line.ts


async def test_close_log_stream_ends_the_follower_cleanly(
    runtime: FakeRuntime, clock: ManualClock
) -> None:
    ended = False

    async def follow() -> None:
        nonlocal ended
        async for _ in runtime.follow_logs("minecraft"):
            pass
        ended = True

    task = asyncio.create_task(follow())
    await clock.tick()
    runtime.close_log_stream()
    await task
    assert ended


async def test_following_a_stopped_container_ends_immediately(
    runtime: FakeRuntime, clock: ManualClock
) -> None:
    """Docker's behaviour, and the one that hid a spin in the reconnect loop.

    A follow on a container that is not running delivers the history and returns. If this fake
    blocked instead, ``manager.py`` could attach every second to a stopped container - resetting
    its backoff on each instant EOF - and every test would stay green.
    """
    _running(runtime)
    runtime.emit_line("from the last run")
    runtime.set_state(ContainerState.EXITED, running=False, exit_code=0)

    received = [line.text async for line in runtime.follow_logs("minecraft", tail=-1)]

    assert received == ["from the last run"]
    assert runtime.attached_log_streams == 0


async def test_break_log_stream_faults_the_follower(
    runtime: FakeRuntime, clock: ManualClock
) -> None:
    _running(runtime)

    async def follow() -> None:
        async for _ in runtime.follow_logs("minecraft"):
            pass

    task = asyncio.create_task(follow())
    await clock.tick()
    runtime.break_log_stream()
    with pytest.raises(LogStreamError):
        await task


async def test_two_followers_each_see_every_line(runtime: FakeRuntime, clock: ManualClock) -> None:
    """The daemon and a ``mcmanager logs --follow`` are both attached in normal operation."""
    _running(runtime)
    a: list[str] = []
    b: list[str] = []

    async def follow(sink: list[str]) -> None:
        async for line in runtime.follow_logs("minecraft"):
            sink.append(line.text)

    tasks = [asyncio.create_task(follow(a)), asyncio.create_task(follow(b))]
    await clock.tick()
    assert runtime.attached_log_streams == 2
    runtime.emit_line("shared")
    await clock.tick()
    runtime.close_log_stream()
    for task in tasks:
        await task
    assert a == ["shared"]
    assert b == ["shared"]


# ---------------------------------------------------------------------------------- events


async def test_events_reach_an_attached_watcher(runtime: FakeRuntime, clock: ManualClock) -> None:
    seen: list[str] = []

    async def watch() -> None:
        async for event in runtime.watch_events("minecraft"):
            seen.append(event.action)

    task = asyncio.create_task(watch())
    await clock.tick()
    runtime.emit_event("health_status: healthy")
    runtime.emit_event("die", exitCode="0")
    await clock.tick()
    runtime.close_event_stream()
    await task
    assert seen == ["health_status: healthy", "die"]


async def test_health_status_events_decode_to_a_health_state(runtime: FakeRuntime) -> None:
    event = runtime.emit_event("health_status: healthy")
    assert event.health_status is HealthState.HEALTHY
    assert runtime.emit_event("start").health_status is None


async def test_lifecycle_events_are_flagged(runtime: FakeRuntime) -> None:
    assert runtime.emit_event("die").is_lifecycle
    assert not runtime.emit_event("health_status: unhealthy").is_lifecycle


# ------------------------------------------------------------------------------ scripting


async def test_fail_next_makes_exactly_one_call_fail(runtime: FakeRuntime) -> None:
    runtime.fail_next("inspect", RuntimeUnavailableError("socket gone"))
    with pytest.raises(RuntimeUnavailableError):
        await runtime.inspect("minecraft")
    assert (await runtime.inspect("minecraft")).exists


async def test_fail_next_queues_so_a_backoff_sequence_can_be_scripted(
    runtime: FakeRuntime,
) -> None:
    for _ in range(3):
        runtime.fail_next("start", RuntimeUnavailableError("still down"))
    for _ in range(3):
        with pytest.raises(RuntimeUnavailableError):
            await runtime.start("minecraft")
    await runtime.start("minecraft")


async def test_fail_next_on_follow_logs_surfaces_on_first_iteration(
    runtime: FakeRuntime,
) -> None:
    """``follow_logs`` is defined not to block when called, so its failure cannot appear until
    somebody iterates. The fake has to behave the same way or the reconnect loop's error handling
    would be tested against the wrong shape."""
    runtime.fail_next("follow_logs", LogStreamError("attach refused"))
    stream = runtime.follow_logs("minecraft")
    with pytest.raises(LogStreamError):
        await anext(stream)


async def test_unreachable_runtime_fails_everything_except_ping(runtime: FakeRuntime) -> None:
    runtime.set_reachable(False)
    assert await runtime.ping() is False
    with pytest.raises(RuntimeUnavailableError):
        await runtime.inspect("minecraft")
    with pytest.raises(RuntimeUnavailableError):
        await runtime.start("minecraft")

    runtime.set_reachable(True)
    assert await runtime.ping() is True
    assert (await runtime.inspect("minecraft")).exists


async def test_going_unreachable_faults_attached_streams(
    runtime: FakeRuntime, clock: ManualClock
) -> None:
    _running(runtime)

    async def follow() -> None:
        async for _ in runtime.follow_logs("minecraft"):
            pass

    task = asyncio.create_task(follow())
    await clock.tick()
    runtime.set_reachable(False)
    with pytest.raises(RuntimeUnavailableError):
        await task


async def test_recreate_changes_the_id_and_drops_the_old_log(runtime: FakeRuntime) -> None:
    """``compose down && compose up``: same name, new id, and the old log is gone."""
    runtime.emit_line("from the old container")
    original = runtime.container_id

    runtime.recreate("beef" * 15)

    assert runtime.container_id != original
    snapshot = await runtime.inspect("minecraft")
    assert snapshot.id == runtime.container_id
    assert runtime.history == ()


# ------------------------------------------------------------------------------------ exec


async def test_exec_returns_the_default_result(runtime: FakeRuntime) -> None:
    result = await runtime.exec("minecraft", ["rcon-cli", "list"])
    assert result.ok
    assert runtime.exec_calls == [("minecraft", ("rcon-cli", "list"))]


async def test_queued_exec_results_are_consumed_in_order(runtime: FakeRuntime) -> None:
    runtime.queue_exec_result(ExecResult(exit_code=0, stdout="There are 2 of a max of 5 players"))
    runtime.queue_exec_result(ExecResult(exit_code=1, stderr="nope"))

    first = await runtime.exec("minecraft", ["rcon-cli", "list"])
    second = await runtime.exec("minecraft", ["rcon-cli", "list"])
    third = await runtime.exec("minecraft", ["rcon-cli", "list"])

    assert first.output.startswith("There are 2")
    assert not second.ok
    assert second.output == "nope"
    assert third == runtime.default_exec_result


async def test_an_exec_handler_can_answer_by_argv(runtime: FakeRuntime) -> None:
    def handler(argv: tuple[str, ...]) -> ExecResult:
        if argv[-1] == "list":
            return ExecResult(exit_code=0, stdout="There are 0 of a max of 5 players online:")
        return ExecResult(exit_code=1, stderr=f"unknown command {argv[-1]}")

    runtime.set_exec_handler(handler)
    assert "0 of a max of 5" in (await runtime.exec("minecraft", ["rcon-cli", "list"])).stdout
    assert not (await runtime.exec("minecraft", ["rcon-cli", "banana"])).ok


# --------------------------------------------------------------------------------- teardown


async def test_aclose_detaches_everything_and_is_safe_twice(
    runtime: FakeRuntime, clock: ManualClock
) -> None:
    _running(runtime)

    async def follow() -> None:
        async for _ in runtime.follow_logs("minecraft"):
            pass

    async def watch() -> None:
        async for _ in runtime.watch_events("minecraft"):
            pass

    tasks = [asyncio.create_task(follow()), asyncio.create_task(watch())]
    await clock.tick()
    assert runtime.attached_log_streams == 1
    assert runtime.attached_event_streams == 1

    await runtime.aclose()
    await runtime.aclose()
    for task in tasks:
        await task
    assert runtime.attached_log_streams == 0
    assert runtime.attached_event_streams == 0


async def test_aclose_cancels_pending_scripted_lines(
    runtime: FakeRuntime, clock: ManualClock
) -> None:
    """A scheduled line firing after teardown would emit into a torn-down daemon."""
    runtime.script_lines([(10.0, "too late")])
    await runtime.aclose()
    await clock.advance(20.0)
    assert runtime.history == ()


async def test_calls_are_recorded_in_order(runtime: FakeRuntime) -> None:
    await runtime.ping()
    await runtime.inspect("minecraft")
    await runtime.start("minecraft")
    assert runtime.calls == ["ping", "inspect", "start"]


async def test_describe_reports_the_scripted_state(runtime: FakeRuntime) -> None:
    runtime.emit_line("one")
    described = runtime.describe()
    assert described["name"] == "minecraft"
    assert described["lines"] == 1
    assert described["reachable"] is True


async def test_manual_clock_start_is_the_synthetic_epoch(runtime: FakeRuntime) -> None:
    line = runtime.emit_line("stamped")
    assert line.ts == datetime(2026, 1, 1, tzinfo=UTC)
