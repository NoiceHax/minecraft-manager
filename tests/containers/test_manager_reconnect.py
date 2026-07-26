"""The reconnect loop, which is where a container manager quietly goes wrong.

Every behaviour asserted here corresponds to a failure that is invisible from the outside: the
daemon keeps running, publishes nothing alarming, and is simply wrong. In order of how long each
would take to diagnose in production:

1. **A cached container id.** After ``compose down/up`` the id changes; a manager holding the old
   one attaches to nothing and reports a healthy silence forever.
2. **No dedupe.** Docker's ``since`` is second-granularity, so every reattach replays that second
   and duplicates whatever was in it.
3. **Level-triggered ``RuntimeUnavailable``.** One alert per retry instead of one per outage.
4. **Treating a clean EOF as a fault.** The container stopping is not an error, and backing off
   from it delays the reattach when it starts again.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import pytest

from mcmanager.containers.dto import ContainerState
from mcmanager.containers.errors import LogStreamError, RuntimeUnavailableError
from mcmanager.containers.fake import FakeRuntime
from mcmanager.containers.manager import BackoffPolicy, DedupeRing, DockerManager
from mcmanager.core.events import RuntimeRestored, RuntimeUnavailable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine
    from datetime import datetime

    from mcmanager.clock import ManualClock
    from mcmanager.containers.dto import ContainerSnapshot, LogLine, RuntimeEvent
    from mcmanager.core.events import Event


class RecordingSink:
    """The slice of ``EventBus`` the manager needs: one synchronous ``publish``."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)

    def of_type[E: Event](self, event_type: type[E]) -> list[E]:
        return [event for event in self.events if isinstance(event, event_type)]


class Harness:
    """A manager wired to a fake, plus the three callback sinks, plus task bookkeeping."""

    def __init__(self, runtime: FakeRuntime, clock: ManualClock) -> None:
        self.runtime = runtime
        self.clock = clock
        self.sink = RecordingSink()
        self.lines: list[LogLine] = []
        self.events: list[RuntimeEvent] = []
        self.snapshots: list[ContainerSnapshot] = []
        self.eofs = 0
        self.tasks: list[asyncio.Task[None]] = []

        def on_log_eof() -> None:
            self.eofs += 1

        self.manager = DockerManager(
            runtime=runtime,
            container="minecraft",
            server_id="mc",
            clock=clock,
            sink=self.sink,
            on_line=self.lines.append,
            on_event=self.events.append,
            on_snapshot=self.snapshots.append,
            on_log_eof=on_log_eof,
            # No jitter: 0.5 is the midpoint of the [0, 1) source, so the delay is exactly the
            # unjittered one and every assertion below can name a number.
            jitter_source=lambda: 0.5,
        )

    def spawn(self, loop_fn: Callable[[], Coroutine[Any, Any, None]]) -> None:
        self.tasks.append(asyncio.create_task(loop_fn()))

    @property
    def texts(self) -> list[str]:
        return [line.text for line in self.lines]

    async def shutdown(self) -> None:
        await self.manager.aclose()
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.fixture
def runtime(clock: ManualClock) -> FakeRuntime:
    built = FakeRuntime(clock=clock)
    # Running, because that is what a manager attaching a log stream is looking at. The fake
    # defaults to `exited` to mirror the homelab container's resting state, and `follow_logs`
    # faithfully EOFs immediately on a stopped container - so a test that wants a live stream has
    # to say so.
    built.set_state(ContainerState.RUNNING, running=True, exit_code=None)
    return built


@pytest.fixture
async def harness(runtime: FakeRuntime, clock: ManualClock) -> AsyncIterator[Harness]:
    built = Harness(runtime, clock)
    try:
        yield built
    finally:
        await built.shutdown()


# ------------------------------------------------------------------------------- backoff


def test_backoff_doubles_and_then_caps() -> None:
    policy = BackoffPolicy()
    delays = [policy.delay_for(n, jitter_source=lambda: 0.5) for n in range(1, 8)]
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_backoff_jitter_stays_within_the_configured_spread() -> None:
    """Without jitter the log stream and the event watcher fail together, back off together, and
    retry in the same millisecond forever."""
    policy = BackoffPolicy()
    assert policy.delay_for(3, jitter_source=lambda: 0.0) == pytest.approx(3.2)
    assert policy.delay_for(3, jitter_source=lambda: 1.0) == pytest.approx(4.8)


def test_backoff_never_returns_a_negative_delay() -> None:
    policy = BackoffPolicy(base_seconds=0.1, jitter=2.0)
    assert policy.delay_for(1, jitter_source=lambda: 0.0) >= 0.0


# ------------------------------------------------------------------------------ dedupe ring


def test_dedupe_ring_reports_repeats(clock: ManualClock) -> None:
    ring = DedupeRing(4)
    now = clock.now()
    assert ring.check(now, "Steve joined the game") is False
    assert ring.check(now, "Steve joined the game") is True


def test_dedupe_ring_distinguishes_the_same_text_at_a_different_time(clock: ManualClock) -> None:
    """A player can join twice. Only the same line at the same timestamp is a replay."""
    ring = DedupeRing(4)
    first = clock.now()
    later = first.replace(second=30)
    assert ring.check(first, "Steve joined the game") is False
    assert ring.check(later, "Steve joined the game") is False


def test_dedupe_ring_evicts_oldest_beyond_capacity(clock: ManualClock) -> None:
    ring = DedupeRing(2)
    now = clock.now()
    ring.check(now, "a")
    ring.check(now, "b")
    ring.check(now, "c")
    assert len(ring) == 2
    assert ring.check(now, "a") is False  # evicted, so it looks new again


def test_dedupe_ring_clear_forgets_everything(clock: ManualClock) -> None:
    ring = DedupeRing(8)
    now = clock.now()
    ring.check(now, "a")
    ring.clear()
    assert len(ring) == 0
    assert ring.check(now, "a") is False


# ------------------------------------------------------------------------- happy path


async def test_lines_flow_to_the_callback_in_order(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()

    runtime.emit_line("[13:24:04] Starting minecraft server version 26.2")
    runtime.emit_line("[13:24:37] Done (32.521s)!")
    await clock.tick()

    assert harness.texts == [
        "[13:24:04] Starting minecraft server version 26.2",
        "[13:24:37] Done (32.521s)!",
    ]
    assert harness.manager.available


async def test_the_first_attach_asks_for_no_history(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    """``tail=0, since=None``: on a cold start we want what happens from now, not a replay of the
    30MB ring that lifecycle would then interpret as live events."""
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()
    assert runtime.follow_calls == [(None, 0)]


async def test_the_container_id_is_resolved_from_the_name_before_attaching(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()
    assert harness.manager.container_id == runtime.container_id
    assert len(harness.snapshots) == 1


# ------------------------------------------------------------------------------ reconnect


async def test_a_broken_stream_reattaches_after_the_backoff(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()
    runtime.emit_line("before")
    await clock.tick()

    runtime.break_log_stream()
    await clock.tick()
    assert len(runtime.follow_calls) == 1, "must not reattach before the backoff elapses"

    await clock.advance(1.0)
    assert len(runtime.follow_calls) == 2

    runtime.emit_line("after")
    await clock.tick()
    assert harness.texts == ["before", "after"]


async def test_the_reattach_asks_for_everything_since_the_last_line(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    """``tail=-1`` is not a detail. Docker applies ``tail`` *after* ``since``, so reattaching with
    ``tail=0`` would ask for "everything since then, of which show me none"."""
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()
    line = runtime.emit_line("marker")
    await clock.tick()

    runtime.break_log_stream()
    await clock.tick()
    await clock.advance(1.0)

    assert runtime.follow_calls[-1] == (line.ts, -1)


async def test_the_replayed_second_is_deduplicated(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    """**The dedupe ring, doing the only job it has.**

    Three lines land in the same second. The stream breaks. Docker replays that whole second on
    reattach. Without the ring, every consumer downstream - roster, session stats, Discord - sees
    three phantom joins.
    """
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()
    runtime.emit_lines(["Steve joined the game", "Alex joined the game", "<Steve> hi"])
    await clock.tick()
    assert len(harness.lines) == 3

    runtime.break_log_stream()
    await clock.tick()
    await clock.advance(1.0)

    assert harness.texts == ["Steve joined the game", "Alex joined the game", "<Steve> hi"]
    assert harness.manager.stats["duplicates_dropped"] == 3


async def test_backoff_escalates_across_repeated_failures(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    for _ in range(3):
        runtime.fail_next("follow_logs", LogStreamError("attach refused"))
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()

    assert len(runtime.follow_calls) == 1
    await clock.advance(1.0)
    assert len(runtime.follow_calls) == 2
    await clock.advance(1.9)
    assert len(runtime.follow_calls) == 2, "second retry waits 2s, not 1s"
    await clock.advance(0.2)
    assert len(runtime.follow_calls) == 3


async def test_a_successful_reattach_resets_the_backoff(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    for _ in range(3):
        runtime.fail_next("follow_logs", LogStreamError("attach refused"))
    harness.spawn(harness.manager.run_log_stream)
    await clock.advance(10.0)
    attaches_before = len(runtime.follow_calls)

    runtime.close_log_stream()
    await clock.tick()
    await clock.advance(1.0)
    assert len(runtime.follow_calls) == attaches_before + 1


# ------------------------------------------------------------- re-resolution by name


async def test_a_recreated_container_is_re_resolved_by_name(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    """**The bug a cached container id causes, made impossible.**

    ``compose down && compose up`` produces the same name with a new id. A manager holding the old
    id goes on asking Docker about a container that does not exist, receives an empty stream, and
    reports a healthy silence forever - which is indistinguishable from "nobody is playing".
    """
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()
    runtime.emit_line("from the old container")
    await clock.tick()
    original_id = harness.manager.container_id
    assert original_id is not None

    runtime.recreate("beef" * 15)
    await clock.tick()
    await clock.advance(1.0)

    assert harness.manager.container_id == runtime.container_id
    assert harness.manager.container_id != original_id


async def test_recreation_resets_the_log_cursor_and_the_dedupe_ring(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    """A new container's log shares no lines with the old one, so carrying the cursor over would
    ask for a ``since`` that means nothing, and carrying the hashes over could suppress a
    genuinely new line that happened to read the same."""
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()
    runtime.emit_line("Steve joined the game")
    await clock.tick()

    runtime.recreate("beef" * 15)
    await clock.tick()
    await clock.advance(1.0)

    assert runtime.follow_calls[-1] == (None, 0), "cursor must be reset, so no `since`"

    runtime.emit_line("Steve joined the game")
    await clock.tick()
    assert harness.texts == ["Steve joined the game", "Steve joined the game"]


# --------------------------------------------------------------------------- availability


async def test_runtime_unavailable_is_published_once_per_outage_not_once_per_retry(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    """Level-triggering this would post one Discord alert per second for the whole of a socket
    permissions problem. Lifecycle only needs telling once to go BLIND and hold last-known
    state."""
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()

    runtime.set_reachable(False)
    await clock.tick()
    await clock.advance(120.0)

    unavailable = harness.sink.of_type(RuntimeUnavailable)
    assert len(unavailable) == 1
    assert not harness.manager.available
    assert harness.manager.stats["outages"] == 1
    # It kept trying the whole time - it just never got past the by-name resolution, because that
    # inspect is the first thing that touches the daemon.
    assert runtime.calls.count("inspect") > 5


async def test_runtime_restored_carries_a_monotonic_downtime(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()

    runtime.set_reachable(False)
    await clock.tick()
    await clock.advance(60.0)
    runtime.set_reachable(True)
    await clock.advance(60.0)

    restored = harness.sink.of_type(RuntimeRestored)
    assert len(restored) == 1
    downtime = restored[0].downtime_seconds
    assert downtime is not None
    assert downtime > 0
    assert harness.manager.available


async def test_the_unavailable_event_names_the_endpoint(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    """That error is a socket-permissions problem most of the time, and the endpoint is the single
    most useful thing to print."""
    runtime.fail_next("inspect", RuntimeUnavailableError("permission denied", endpoint="unix:///x"))
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()

    unavailable = harness.sink.of_type(RuntimeUnavailable)
    assert len(unavailable) == 1
    assert unavailable[0].endpoint == "unix:///x"
    assert "permission denied" in unavailable[0].error


async def test_an_absent_container_warns_and_keeps_looking(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    """A legitimate state, per the ABC. Exiting here would mean a renamed container takes the
    daemon down; publishing ``RuntimeUnavailable`` would mean lifecycle goes BLIND for something
    it can actually see."""
    runtime.set_absent()
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()
    await clock.advance(30.0)

    assert harness.sink.of_type(RuntimeUnavailable) == []
    assert harness.manager.available
    assert len(runtime.calls) > 2, "it is still polling"


# ---------------------------------------------------------------------------- clean EOF


async def test_a_clean_eof_is_a_signal_not_a_fault(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    """The container stopped. That corroborates the ``die`` event; it is not an outage, does not
    trip the backoff, and does not deserve an alert."""
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()
    runtime.emit_line("stopping")
    await clock.tick()

    runtime.close_log_stream()
    await clock.tick()

    assert harness.eofs == 1
    assert harness.sink.of_type(RuntimeUnavailable) == []
    assert harness.manager.stats["clean_eofs"] == 1

    await clock.advance(1.0)
    assert len(runtime.follow_calls) == 2


async def test_a_stop_produces_a_clean_eof_rather_than_an_error(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()
    await runtime.start("minecraft")
    await runtime.stop("minecraft", timeout=90)
    await clock.tick()

    assert harness.eofs == 1
    assert harness.sink.of_type(RuntimeUnavailable) == []


# ------------------------------------------------------------------- a server that is simply off


async def test_a_stopped_container_is_not_attached_to_once_per_second_forever(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    """**The idle-auto-shutdown spin.**

    Docker ends a follow on a stopped container immediately, and a clean EOF resets the backoff.
    Together that used to mean an inspect plus a log attach every second, plus a ``pipeline.on_eof``
    and a ``SnapshotSignal`` into lifecycle every second, for as long as the server stayed down -
    two days of nobody playing being ~173,000 round trips against the Docker socket.
    """
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()
    runtime.emit_line("[13:31:00] [Server thread/INFO]: [Rcon: Stopping the server]")
    await clock.tick()

    runtime.set_state(ContainerState.EXITED, running=False, exit_code=0)
    runtime.close_log_stream()
    await clock.tick()
    assert harness.eofs == 1, "the stop itself is still one clean EOF"

    attaches_after_stop = len(runtime.follow_calls)
    await clock.advance(3600.0)

    # One hour off. The exponential backoff caps at 30s, so this is bounded by ~120 inspects and
    # zero further attaches - not 3,600 of each.
    assert len(runtime.follow_calls) == attaches_after_stop, "nothing was attached while stopped"
    assert harness.eofs == 1, "on_eof fires once per stop, not once per second"
    assert runtime.calls.count("inspect") < 200
    assert harness.manager.stats["not_running_polls"] > 0


async def test_the_reattach_after_a_stop_resumes_from_the_new_runs_started_at(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    """Backing off must not cost us the startup lines readiness is detected from.

    ``Starting minecraft server version ...`` and ``Done (Ns)!`` can both land inside the backoff
    window, and with no ``since`` cursor from a previous run a ``tail=0`` attach would simply miss
    them - which is ``ServerReady`` never firing.
    """
    runtime.set_state(ContainerState.EXITED, running=False, exit_code=0)
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()
    assert runtime.follow_calls == [], "nothing to attach to"

    await clock.advance(120.0)
    started_at = clock.now()
    runtime.set_state(ContainerState.RUNNING, running=True, exit_code=None, started_at=started_at)
    await clock.advance(60.0)

    assert runtime.follow_calls, "it reattached once the container came back"
    since, tail = runtime.follow_calls[-1]
    assert since == started_at
    assert tail == -1, "tail is applied after since, so anything but -1 discards the backfill"


# ------------------------------------------------------------------------- event watcher


async def test_docker_events_reach_the_callback(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    harness.spawn(harness.manager.run_event_watcher)
    await clock.tick()

    runtime.emit_event("start")
    runtime.emit_event("health_status: healthy")
    runtime.emit_event("die", exitCode="137")
    await clock.tick()

    assert [event.action for event in harness.events] == [
        "start",
        "health_status: healthy",
        "die",
    ]
    assert harness.events[-1].exit_code == 137


async def test_replayed_events_are_deduplicated_on_reattach(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    """``since`` on the event stream is second-granularity too."""
    harness.spawn(harness.manager.run_event_watcher)
    await clock.tick()
    runtime.emit_event("start")
    await clock.tick()

    runtime.break_event_stream()
    await clock.tick()
    await clock.advance(1.0)

    assert [event.action for event in harness.events] == ["start"]


async def test_the_event_watcher_reattaches_after_a_clean_end(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    harness.spawn(harness.manager.run_event_watcher)
    await clock.tick()
    runtime.close_event_stream()
    await clock.tick()
    await clock.advance(1.0)
    await clock.tick()
    assert len(runtime.watch_calls) == 2


# ---------------------------------------------------------------------------- reconcile


async def test_reconcile_inspects_on_a_timer(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    """The safety net for events we never saw. ``health_status`` is edge-triggered: Docker emits
    one when the status changes and never again, so a missed event is permanent and the only way
    to notice is to ask."""
    harness.spawn(harness.manager.run_reconcile)
    await clock.tick()
    assert harness.snapshots == []

    await clock.advance(60.0)
    assert len(harness.snapshots) == 1

    await clock.advance(60.0)
    assert len(harness.snapshots) == 2


async def test_reconcile_reports_an_outage_and_recovers(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    harness.spawn(harness.manager.run_reconcile)
    await clock.tick()
    runtime.set_reachable(False)
    await clock.advance(180.0)
    assert len(harness.sink.of_type(RuntimeUnavailable)) == 1

    runtime.set_reachable(True)
    await clock.advance(60.0)
    assert len(harness.sink.of_type(RuntimeRestored)) == 1


async def test_reconcile_sees_a_stale_health_string_as_unknown(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    """End to end through the manager: the snapshot lifecycle receives has already had the
    stale-health rule applied, because it is a derived property and not a field anyone sets."""
    runtime.set_state(ContainerState.EXITED, running=False, exit_code=0)
    runtime.set_health("unhealthy", failing_streak=0)
    harness.spawn(harness.manager.run_reconcile)
    await clock.tick()
    await clock.advance(60.0)

    snapshot = harness.snapshots[-1]
    assert snapshot.health_reported_raw == "unhealthy"
    assert snapshot.health.value == "unknown"


# ----------------------------------------------------------------------------- shutdown


async def test_aclose_stops_the_loops_without_cancelling_them(
    runtime: FakeRuntime, clock: ManualClock
) -> None:
    """The supervisor owns cancellation, in reverse spawn order with per-task timeouts. A manager
    that cancelled itself would race that."""
    harness = Harness(runtime, clock)
    task: asyncio.Task[None] = asyncio.create_task(harness.manager.run_log_stream())
    await clock.tick()

    await harness.manager.aclose()
    runtime.close_log_stream()
    await clock.advance(2.0)

    assert task.done()
    assert not task.cancelled()
    await task


async def test_no_streams_are_left_attached_after_shutdown(
    runtime: FakeRuntime, clock: ManualClock
) -> None:
    """A leaked stream is a leaked pump thread against the real runtime."""
    harness = Harness(runtime, clock)
    harness.spawn(harness.manager.run_log_stream)
    harness.spawn(harness.manager.run_event_watcher)
    await clock.tick()
    assert runtime.attached_log_streams == 1
    assert runtime.attached_event_streams == 1

    await harness.shutdown()
    assert runtime.attached_log_streams == 0
    assert runtime.attached_event_streams == 0


# ------------------------------------------------------------------------------ timestamps


async def test_the_log_cursor_tracks_the_last_line_timestamp(
    harness: Harness, runtime: FakeRuntime, clock: ManualClock
) -> None:
    harness.spawn(harness.manager.run_log_stream)
    await clock.tick()
    assert harness.manager.last_line_ts is None

    await clock.advance(5.0)
    line = runtime.emit_line("stamped")
    await clock.tick()

    seen: datetime | None = harness.manager.last_line_ts
    assert seen == line.ts
