"""The lifecycle reducer, as a table over the full transition matrix.

No Docker, no network, no real time. Every timing assertion runs on
:class:`~mcmanager.clock.ManualClock`, so the 270-second start deadline is tested in microseconds,
and the 210-second health guard window comes off :class:`~mcmanager.containers.fake.FakeRuntime`'s
snapshot rather than being restated here - if the container's healthcheck config changes, these
tests change with it, which is the whole point of reading the timing from the container.

Four named regressions the transitions here exist to prevent:

- a phantom ``ServerStopped`` every time the Docker socket hiccups (``BLIND``);
- a "server started!" announcement every time the *daemon* restarts (boot reconcile);
- a "server is up!" announcement from a ``Done (32.521s)!`` line replayed out of the 30MB log ring;
- a populated server reported stopped because one SLP probe timed out.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from builders import (  # sibling test helper module, see builders.py's docstring
    CONTAINER,
    GUARD_SECONDS,
    SERVER_ID,
    START_DEADLINE_SECONDS,
    RecordingSink,
    absent,
    docker_event,
    done_line,
    exited,
    probe_failed,
    probe_ok,
    running,
    snapshot_of,
    stopping_line,
    version_line,
)

from mcmanager.containers.dto import ContainerState
from mcmanager.containers.fake import FakeRuntime
from mcmanager.core.events import (
    RuntimeRestored,
    RuntimeUnavailable,
    ServerCrashed,
    ServerReady,
    ServerStarting,
    ServerStopped,
    ServerStopping,
)
from mcmanager.core.types import LifecycleState, ReadySignal, Source
from mcmanager.services.lifecycle import (
    SIGNAL_KINDS,
    AvailabilitySignal,
    ExitVerdict,
    Intent,
    IntentSignal,
    LifecycleReducer,
    LifecycleService,
    LineSignal,
    ProbeSignal,
    SnapshotSignal,
    classify_exit,
)

if TYPE_CHECKING:
    from mcmanager.clock import ManualClock
    from mcmanager.core.events import Event
    from mcmanager.services.lifecycle import Signal


# ------------------------------------------------------------------------------------- fixtures
#
# Declared here rather than imported from builders.py: a fixture imported into a test module reads
# to ruff as an unused import and is deleted by ``--fix``, which produces a hundred
# "fixture not found" errors and no obvious cause.


@pytest.fixture
def runtime(clock: ManualClock) -> FakeRuntime:
    """A fake with the homelab container's healthcheck timings: a real 210-second guard window.

    ``auto_transition=False`` so each test narrates Docker's events itself and nothing transitions
    behind its back.
    """
    return FakeRuntime(clock=clock, name=CONTAINER, auto_transition=False)


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


@pytest.fixture
def reducer(clock: ManualClock) -> LifecycleReducer:
    """Production defaults: first-of-three readiness, 60 seconds of deadline slack."""
    return LifecycleReducer(server_id=SERVER_ID, clock=clock)


@pytest.fixture
def service(
    reducer: LifecycleReducer,
    sink: RecordingSink,
    clock: ManualClock,
) -> LifecycleService:
    return LifecycleService(reducer=reducer, sink=sink, clock=clock)


# ==================================================================== classify_exit, the table


@pytest.mark.parametrize(
    ("exit_code", "oom", "intent", "expected", "why"),
    [
        (0, False, True, ExitVerdict(clean=True, forced=False), "we asked; runner exits 0"),
        (0, False, False, ExitVerdict(clean=True, forced=False), "operator ran docker stop"),
        (143, False, True, ExitVerdict(clean=True, forced=False), "128+SIGTERM, untrapped"),
        (143, False, False, ExitVerdict(clean=True, forced=False), "still graceful"),
        (137, False, True, ExitVerdict(clean=True, forced=True), "save ran past the grace period"),
        (137, False, False, ExitVerdict(clean=False, forced=True), "host OOM killer / docker kill"),
        (1, False, True, ExitVerdict(clean=False, forced=False), "JVM died mid-shutdown"),
        (1, False, False, ExitVerdict(clean=False, forced=False), "plain crash"),
        (None, False, True, ExitVerdict(clean=False, forced=False), "no evidence is not clean"),
        (None, False, False, ExitVerdict(clean=False, forced=False), "no evidence is not clean"),
        (137, True, True, ExitVerdict(clean=False, forced=False), "OOM outranks the exit code"),
        (0, True, True, ExitVerdict(clean=False, forced=False), "OOM is never clean"),
    ],
)
def test_classify_exit_table(
    exit_code: int | None,
    oom: bool,
    intent: bool,
    expected: ExitVerdict,
    why: str,
) -> None:
    assert classify_exit(exit_code, oom_killed=oom, had_stop_intent=intent) == expected, why


def test_137_with_intent_is_the_grace_period_warning_not_a_crash() -> None:
    """The distinction the whole function exists for.

    ``docker stop -t 90`` that ran out is a world that may not have finished saving - a warning.
    The same exit code with nobody having asked is something else killing the process.
    """
    ours = classify_exit(137, oom_killed=False, had_stop_intent=True)
    theirs = classify_exit(137, oom_killed=False, had_stop_intent=False)
    assert ours.clean
    assert ours.forced
    assert not theirs.clean
    assert theirs.forced
    assert ours.crashed is False
    assert theirs.crashed is True


# ============================================================================= timing is read


async def test_guard_window_and_deadline_come_from_the_container(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
) -> None:
    """210s and 270s are derived, not written down here."""
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))
    assert reducer.guard_seconds == GUARD_SECONDS
    assert reducer.start_deadline_seconds == START_DEADLINE_SECONDS

    snapshot = reducer.snapshot
    assert snapshot is not None
    timing = snapshot.health_timing
    assert timing.start_period == timedelta(seconds=120)
    assert timing.interval == timedelta(seconds=30)
    assert timing.retries == 2
    assert snapshot.guard_window == timedelta(seconds=GUARD_SECONDS)


async def test_no_healthcheck_means_no_opinion_about_the_deadline(
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    """A container that declares no healthcheck gets no invented deadline."""
    from mcmanager.containers.dto import HealthTiming

    runtime.set_health_timing(HealthTiming())
    reducer = LifecycleReducer(server_id=SERVER_ID, clock=clock)
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health=None)))

    assert reducer.guard_seconds is None
    assert reducer.start_deadline_seconds is None

    reducer.handle(docker_event("start", ts=clock.now()))
    await clock.advance(60 * 60)
    assert reducer.check_deadlines() == []
    assert reducer.state is LifecycleState.STARTING


async def test_fallback_guard_is_opt_in(clock: ManualClock, runtime: FakeRuntime) -> None:
    from mcmanager.containers.dto import HealthTiming

    runtime.set_health_timing(HealthTiming())
    reducer = LifecycleReducer(
        server_id=SERVER_ID,
        clock=clock,
        fallback_guard_seconds=100.0,
        start_deadline_extra_seconds=10.0,
    )
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health=None)))
    assert reducer.start_deadline_seconds == 110.0


# ============================================================== start deadline -> DEGRADED


async def test_start_deadline_drives_degraded_with_no_real_waiting(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """270 virtual seconds, roughly zero real ones."""
    reducer.handle(SnapshotSignal(snapshot=await exited(runtime)))
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="starting")))
    assert reducer.state is LifecycleState.STARTING

    await clock.advance(START_DEADLINE_SECONDS - 1)
    assert reducer.check_deadlines() == []
    assert reducer.state is LifecycleState.STARTING, "one second early is still STARTING"

    await clock.advance(2)
    assert reducer.check_deadlines() == []
    assert reducer.state is LifecycleState.DEGRADED


async def test_the_deadline_is_also_caught_by_the_reconcile_alone(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """Every ``handle`` checks deadlines, so the 60s reconcile is sufficient on its own."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))
    await clock.advance(START_DEADLINE_SECONDS + 5)
    reducer.handle(SnapshotSignal(snapshot=await snapshot_of(runtime)))
    assert reducer.state is LifecycleState.DEGRADED


async def test_the_service_arms_a_timer_so_degraded_is_prompt(
    service: LifecycleService,
    sink: RecordingSink,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    service.on_runtime_event(docker_event("start", ts=clock.now()))
    service.on_snapshot(await running(runtime))
    assert service.state is LifecycleState.STARTING
    sink.clear()

    await clock.advance(START_DEADLINE_SECONDS + 1)
    assert service.state is LifecycleState.DEGRADED, "the timer fired without any signal arriving"
    assert sink.events == [], "there is no ServerDegraded event to emit"

    await service.aclose()


async def test_readiness_before_the_deadline_disarms_it(
    service: LifecycleService,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    service.on_runtime_event(docker_event("start", ts=clock.now()))
    service.on_snapshot(await running(runtime))
    await clock.advance(32.5)
    service.on_line_event(done_line(ts=clock.now()))
    assert service.state is LifecycleState.READY

    await clock.advance(START_DEADLINE_SECONDS * 2)
    assert service.state is LifecycleState.READY
    await service.aclose()


# ================================================================= readiness, first of three


@pytest.mark.parametrize(
    ("signal_kind", "expected"),
    [
        ("log", ReadySignal.LOG_DONE),
        ("health", ReadySignal.HEALTHCHECK),
        ("probe", ReadySignal.PROBE),
    ],
)
async def test_each_of_the_three_signals_promotes_starting_to_ready(
    signal_kind: str,
    expected: ReadySignal,
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))
    await clock.advance(32.521)

    signal: Signal
    if signal_kind == "log":
        signal = LineSignal(event=done_line(ts=clock.now()))
    elif signal_kind == "health":
        signal = docker_event("health_status: healthy", ts=clock.now())
    else:
        signal = ProbeSignal(result=probe_ok(ts=clock.now()))

    events = reducer.handle(signal)

    assert len(events) == 1
    ready = events[0]
    assert isinstance(ready, ServerReady)
    assert ready.detected_by is expected
    assert ready.server_id == SERVER_ID
    assert reducer.state is LifecycleState.READY
    assert reducer.ready_detected_by is expected


async def test_the_log_line_wins_the_race_and_ready_fires_exactly_once(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """The reason first-of-three exists: ``Done`` at ~32s, healthy no sooner than ~120s."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))

    await clock.advance(32.521)
    first = reducer.handle(LineSignal(event=done_line(ts=clock.now())))
    assert len(first) == 1
    ready = first[0]
    assert isinstance(ready, ServerReady)
    assert ready.startup_seconds == pytest.approx(32.521)

    await clock.advance(120)
    later = reducer.handle(docker_event("health_status: healthy", ts=clock.now()))
    assert later == [], "the healthcheck arriving 120s later must not re-announce readiness"

    await clock.advance(45)
    probe = reducer.handle(ProbeSignal(result=probe_ok(ts=clock.now())))
    assert probe == []
    assert reducer.state is LifecycleState.READY


async def test_ready_carries_the_version_seen_earlier_in_the_run(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))
    reducer.handle(LineSignal(event=version_line(ts=clock.now(), version="26.2")))
    await clock.advance(32.5)
    events = reducer.handle(LineSignal(event=done_line(ts=clock.now())))
    ready = events[0]
    assert isinstance(ready, ServerReady)
    assert ready.version == "26.2"
    assert reducer.version == "26.2"


async def test_strict_health_only_is_a_config_change_not_a_code_change(
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    """The documented way back to the spec-literal behaviour."""
    reducer = LifecycleReducer(
        server_id=SERVER_ID,
        clock=clock,
        ready_signals=(ReadySignal.HEALTHCHECK,),
    )
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))

    await clock.advance(32.5)
    assert reducer.handle(LineSignal(event=done_line(ts=clock.now()))) == []
    assert reducer.handle(ProbeSignal(result=probe_ok(ts=clock.now()))) == []
    assert reducer.state is LifecycleState.STARTING

    await clock.advance(90)
    events = reducer.handle(docker_event("health_status: healthy", ts=clock.now()))
    assert len(events) == 1
    assert reducer.state is LifecycleState.READY


async def test_measured_startup_seconds_when_the_signal_carries_none(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))
    await clock.advance(137.0)
    events = reducer.handle(docker_event("health_status: healthy", ts=clock.now()))
    ready = events[0]
    assert isinstance(ready, ServerReady)
    assert ready.startup_seconds == pytest.approx(137.0)


# ============================================================ health remains authoritative


async def test_unhealthy_inside_the_guard_window_is_suppressed(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """``start_period`` is 120s here; the healthcheck is *expected* to fail inside it."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))

    await clock.advance(GUARD_SECONDS - 1)
    events = reducer.handle(docker_event("health_status: unhealthy", ts=clock.now()))
    assert events == []
    assert reducer.state is LifecycleState.STARTING


async def test_unhealthy_past_the_guard_window_drives_degraded(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))
    await clock.advance(30)
    reducer.handle(LineSignal(event=done_line(ts=clock.now())))
    assert reducer.state is LifecycleState.READY

    await clock.advance(GUARD_SECONDS + 1)
    events = reducer.handle(docker_event("health_status: unhealthy", ts=clock.now()))
    assert events == [], "there is no ServerDegraded event"
    assert reducer.state is LifecycleState.DEGRADED, (
        "a later unhealthy still wins over the log's earlier Done - health is authoritative "
        "for the negative case"
    )


async def test_degraded_recovers_to_ready_without_re_announcing(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """A run that **already announced itself** and then went unhealthy recovers silently.

    There is no ``ServerRecovered`` in the vocabulary, and re-announcing would mean a flapping
    healthcheck posting "server is up!" to Discord every thirty seconds.
    """
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))
    await clock.advance(30)
    announced = reducer.handle(LineSignal(event=done_line(ts=clock.now())))
    assert [type(event).__name__ for event in announced] == ["ServerReady"]

    await clock.advance(GUARD_SECONDS + 5)
    reducer.handle(docker_event("health_status: unhealthy", ts=clock.now()))
    assert reducer.state is LifecycleState.DEGRADED

    events = reducer.handle(docker_event("health_status: healthy", ts=clock.now()))
    assert events == [], "ServerReady fires exactly once per start"
    assert reducer.state is LifecycleState.READY


async def test_a_start_that_overruns_the_deadline_still_reports_ready(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """**First-of-three must not become none-of-three because the clock won a race.**

    A freshly generated world, a chunk pre-generation pass or a post-update migration routinely
    takes longer than the 270-second deadline. The deadline moves ``STARTING -> DEGRADED``, and
    ``handle()`` applies deadlines *before* dispatching the signal - so the ``Done (298.4s)!`` line
    used to arrive in ``DEGRADED`` and be swallowed as a "recovery" from a run that had never
    announced anything. Nothing reached the bus, ``/status`` read ``ready`` with a null
    ``ready_at`` and a null ``ready_detected_by`` forever, and Discord, ``mcmanager events`` and
    the session manager never learned the server came up at all.
    """
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))
    assert reducer.start_deadline_seconds == START_DEADLINE_SECONDS

    await clock.advance(START_DEADLINE_SECONDS + 30.0)
    events = reducer.handle(LineSignal(event=done_line(ts=clock.now(), seconds=298.4)))

    assert [type(event).__name__ for event in events] == ["ServerReady"]
    ready_event = events[0]
    assert isinstance(ready_event, ServerReady)
    assert ready_event.detected_by is ReadySignal.LOG_DONE
    assert ready_event.startup_seconds == pytest.approx(298.4)
    assert reducer.state is LifecycleState.READY
    assert reducer.ready_at is not None
    assert reducer.ready_detected_by is ReadySignal.LOG_DONE


async def test_an_overrun_start_reports_ready_only_once(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """And having announced it, a later unhealthy/healthy flap is still silent."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))
    await clock.advance(START_DEADLINE_SECONDS + 30.0)
    reducer.handle(LineSignal(event=done_line(ts=clock.now(), seconds=298.4)))

    reducer.handle(docker_event("health_status: unhealthy", ts=clock.now()))
    assert reducer.state is LifecycleState.DEGRADED
    assert reducer.handle(docker_event("health_status: healthy", ts=clock.now())) == []


async def test_a_degraded_server_discovered_at_boot_is_never_announced(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """The other side of the same coin, and the reason the gate is "this run was witnessed
    starting" rather than "the state is DEGRADED".

    Redeploying the daemon against a server that has been up for three days *discovers* its state;
    it does not witness a transition into it. A probe succeeding afterwards must not produce a
    "server is up!" for a server that never went anywhere.
    """
    a_day_ago = clock.now() - timedelta(seconds=86400.0)
    snapshot = await running(runtime, health="unhealthy", started_at=a_day_ago)
    reducer.handle(SnapshotSignal(snapshot=snapshot))
    assert reducer.state is LifecycleState.DEGRADED

    events = reducer.handle(ProbeSignal(result=probe_ok(ts=clock.now())))

    assert events == []
    assert reducer.state is LifecycleState.READY


@pytest.mark.parametrize(
    "state_setup",
    ["stopped", "crashed", "absent", "unknown", "stopping"],
)
async def test_health_status_outside_starting_ready_degraded_is_dropped(
    state_setup: str,
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """Docker never clears ``State.Health.Status``, so this arrives for dead containers."""
    if state_setup == "stopped":
        reducer.handle(SnapshotSignal(snapshot=await exited(runtime, exit_code=0)))
        reducer.handle(docker_event("start", ts=clock.now()))
        reducer.handle(docker_event("die", ts=clock.now(), attributes={"exitCode": "0"}))
    elif state_setup == "crashed":
        reducer.handle(SnapshotSignal(snapshot=await exited(runtime, exit_code=0)))
        reducer.handle(docker_event("start", ts=clock.now()))
        reducer.handle(docker_event("die", ts=clock.now(), attributes={"exitCode": "1"}))
    elif state_setup == "absent":
        reducer.handle(SnapshotSignal(snapshot=await absent(runtime)))
    elif state_setup == "stopping":
        reducer.handle(docker_event("start", ts=clock.now()))
        reducer.handle(SnapshotSignal(snapshot=await running(runtime)))
        reducer.handle(IntentSignal(intent=Intent.STOP, requested_by="cli", timeout_seconds=90.0))
    before = reducer.state

    healthy = reducer.handle(docker_event("health_status: healthy", ts=clock.now()))
    unhealthy = reducer.handle(docker_event("health_status: unhealthy", ts=clock.now()))

    assert healthy == []
    assert unhealthy == []
    assert reducer.state is before, "a health event must not move a non-running state"


async def test_the_stale_health_string_on_an_exited_container_is_not_believed(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """The verified gotcha, end to end through the reducer.

    On the real host an *exited* container reported ``State.Health.Status == "unhealthy"`` with
    ``FailingStreak`` 0. ``ContainerSnapshot.health`` covers for it; this asserts the reducer never
    reaches past that property.
    """
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    assert reducer.state is LifecycleState.READY

    reducer.handle(docker_event("die", ts=clock.now(), attributes={"exitCode": "0"}))
    snapshot = await exited(runtime, exit_code=0)
    assert snapshot.health_reported_raw == "healthy", "the fake keeps the stale string, like Docker"

    reducer.handle(SnapshotSignal(snapshot=snapshot))
    assert reducer.state is LifecycleState.STOPPED, "not READY, whatever the health field says"


# ================================================================== boot reconcile is silent


@pytest.mark.parametrize(
    ("setup", "expected_state"),
    [
        ("running_healthy", LifecycleState.READY),
        ("running_starting", LifecycleState.STARTING),
        ("exited_clean", LifecycleState.STOPPED),
        ("exited_crash", LifecycleState.CRASHED),
        ("absent", LifecycleState.ABSENT),
    ],
)
async def test_boot_reconcile_adopts_state_and_emits_nothing(
    setup: str,
    expected_state: LifecycleState,
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
) -> None:
    """Discovering state is not witnessing a transition. No Discord spam on daemon restart."""
    if setup == "running_healthy":
        snapshot = await running(runtime, health="healthy")
    elif setup == "running_starting":
        snapshot = await running(runtime, health="starting")
    elif setup == "exited_clean":
        snapshot = await exited(runtime, exit_code=0)
    elif setup == "exited_crash":
        snapshot = await exited(runtime, exit_code=1)
    else:
        snapshot = await absent(runtime)

    assert reducer.state is LifecycleState.UNKNOWN
    events = reducer.handle(SnapshotSignal(snapshot=snapshot))

    assert events == [], "a boot reconcile emits nothing at all"
    assert reducer.state is expected_state


async def test_boot_reconcile_inside_the_start_period_is_not_degraded(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """The daemon coming up 30s into a container's 120s start_period must not cry wolf."""
    started = clock.now()
    runtime.set_state(
        ContainerState.RUNNING,
        running=True,
        exit_code=None,
        health_raw="unhealthy",
        started_at=started,
    )
    await clock.advance(30)

    reducer.handle(SnapshotSignal(snapshot=await snapshot_of(runtime)))
    assert reducer.state is LifecycleState.STARTING


async def test_boot_reconcile_of_a_long_unhealthy_container_is_degraded(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    started = clock.now()
    runtime.set_state(
        ContainerState.RUNNING,
        running=True,
        exit_code=None,
        health_raw="unhealthy",
        started_at=started,
    )
    await clock.advance(GUARD_SECONDS + 60)

    reducer.handle(SnapshotSignal(snapshot=await snapshot_of(runtime)))
    assert reducer.state is LifecycleState.DEGRADED


async def test_daemon_restart_mid_session_reconciles_from_snapshot_plus_backfill(
    clock: ManualClock,
    runtime: FakeRuntime,
    sink: RecordingSink,
) -> None:
    """The M2 verification step, as a unit test.

    A fresh daemon comes up against a server that has been READY for twenty minutes, then the
    reconnecting log stream replays that run's ``Done (32.521s)!`` out of the json-file ring. The
    correct outcome is READY and **zero** events - not ``STOPPED``, and not a second
    ``ServerReady``.
    """
    started = clock.now()
    runtime.set_state(
        ContainerState.RUNNING,
        running=True,
        exit_code=None,
        health_raw="healthy",
        started_at=started,
    )
    done_at = started + timedelta(seconds=32.521)
    await clock.advance(20 * 60)

    reducer = LifecycleReducer(server_id=SERVER_ID, clock=clock)
    service = LifecycleService(reducer=reducer, sink=sink, clock=clock)

    service.on_snapshot(await snapshot_of(runtime))
    assert service.state is LifecycleState.READY, "not UNKNOWN, and certainly not STOPPED"

    # Backfill: the same run's readiness line, replayed.
    service.on_line_event(done_line(ts=done_at))
    # And the version line from the same run.
    service.on_line_event(version_line(ts=started + timedelta(seconds=1)))

    assert sink.events == [], "backfill re-emits nothing"
    assert service.state is LifecycleState.READY
    await service.aclose()


async def test_a_backfilled_done_line_from_a_previous_run_is_ignored(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """A ``Done`` line older than ``StartedAt`` belongs to a run that has already ended."""
    old_done = clock.now() - timedelta(days=3)
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))

    events = reducer.handle(LineSignal(event=done_line(ts=old_done)))
    assert events == []
    assert reducer.state is LifecycleState.STARTING


# ======================================================================== BLIND and restore


async def test_runtime_unavailable_retains_last_known_state(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    assert reducer.state is LifecycleState.READY

    events = reducer.handle(AvailabilitySignal(available=False, error="socket gone"))

    assert len(events) == 1
    unavailable = events[0]
    assert isinstance(unavailable, RuntimeUnavailable)
    assert unavailable.error == "socket gone"
    assert reducer.state is LifecycleState.BLIND
    assert reducer.last_known_state is LifecycleState.READY
    assert not any(isinstance(e, ServerStopped) for e in events), "no phantom ServerStopped"


async def test_going_blind_is_edge_triggered(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    first = reducer.handle(AvailabilitySignal(available=False, error="boom"))
    second = reducer.handle(AvailabilitySignal(available=False, error="boom again"))
    third = reducer.handle(AvailabilitySignal(available=False, error="still boom"))

    assert len(first) == 1
    assert second == []
    assert third == []


async def test_restore_emits_runtime_restored_with_downtime_and_no_delta(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    reducer.handle(AvailabilitySignal(available=False, error="socket gone"))

    await clock.advance(12.5)
    events = reducer.handle(AvailabilitySignal(available=True))

    assert len(events) == 1
    restored = events[0]
    assert isinstance(restored, RuntimeRestored)
    assert restored.downtime_seconds == pytest.approx(12.5)
    assert reducer.state is LifecycleState.READY, "back to exactly what we last knew"

    # Nothing changed under us, so re-reconciling emits nothing.
    assert reducer.handle(SnapshotSignal(snapshot=await snapshot_of(runtime))) == []


async def test_restore_emits_only_the_genuine_delta(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """The server really did stop during the outage: exactly one ``ServerStopped``, no more."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    reducer.handle(AvailabilitySignal(available=False, error="socket gone"))

    await clock.advance(30)
    await exited(runtime, exit_code=0)

    restored = reducer.handle(AvailabilitySignal(available=True))
    assert len(restored) == 1
    assert isinstance(restored[0], RuntimeRestored)

    delta = reducer.handle(SnapshotSignal(snapshot=await snapshot_of(runtime)))
    assert len(delta) == 1
    stopped = delta[0]
    assert isinstance(stopped, ServerStopped)
    assert stopped.clean is True
    assert reducer.state is LifecycleState.STOPPED


async def test_a_snapshot_arriving_while_blind_restores_implicitly(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """Either ordering of restore-vs-inspect gives exactly one ``RuntimeRestored``."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    reducer.handle(AvailabilitySignal(available=False, error="socket gone"))
    await clock.advance(5)

    events = reducer.handle(SnapshotSignal(snapshot=await snapshot_of(runtime)))
    assert [type(e).__name__ for e in events] == ["RuntimeRestored"]
    assert reducer.state is LifecycleState.READY

    assert reducer.handle(AvailabilitySignal(available=True)) == [], "not a second restore"


@pytest.mark.parametrize("action", ["start", "die", "destroy"])
async def test_docker_events_are_dropped_while_blind(
    action: str,
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    reducer.handle(AvailabilitySignal(available=False, error="socket gone"))

    assert reducer.handle(docker_event(action, ts=clock.now())) == []
    assert reducer.state is LifecycleState.BLIND


async def test_lines_and_probes_are_dropped_while_blind(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))
    reducer.handle(AvailabilitySignal(available=False, error="socket gone"))

    assert reducer.handle(LineSignal(event=done_line(ts=clock.now()))) == []
    assert reducer.handle(ProbeSignal(result=probe_ok(ts=clock.now()))) == []
    assert reducer.state is LifecycleState.BLIND


async def test_available_when_not_blind_is_a_noop(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
) -> None:
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    assert reducer.handle(AvailabilitySignal(available=True)) == []
    assert reducer.state is LifecycleState.READY


async def test_the_service_translates_bus_runtime_status_events(
    service: LifecycleService,
    sink: RecordingSink,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    service.on_snapshot(await running(runtime, health="healthy"))
    sink.clear()

    await service.on_runtime_status(
        RuntimeUnavailable(
            ts=clock.now(),
            server_id=SERVER_ID,
            source=Source.RUNTIME,
            error="socket gone",
            endpoint="unix:///var/run/docker.sock",
        )
    )
    assert service.state is LifecycleState.BLIND
    assert sink.names == ["RuntimeUnavailable"]

    await service.on_runtime_status(
        RuntimeRestored(ts=clock.now(), server_id=SERVER_ID, source=Source.RUNTIME)
    )
    assert service.state is LifecycleState.READY
    assert sink.names == ["RuntimeUnavailable", "RuntimeRestored"]
    await service.aclose()


# ================================================================ stops, crashes, externals


async def test_our_own_stop_emits_stopping_then_stopped(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    await clock.advance(600)

    stopping = reducer.handle(
        IntentSignal(
            intent=Intent.STOP,
            requested_by="kunal",
            reason="idle timeout",
            timeout_seconds=90.0,
        )
    )
    assert len(stopping) == 1
    event = stopping[0]
    assert isinstance(event, ServerStopping)
    assert event.requested_by == "kunal"
    assert event.reason == "idle timeout"
    assert event.timeout_seconds == 90.0
    assert reducer.state is LifecycleState.STOPPING

    stopped = reducer.handle(docker_event("die", ts=clock.now(), attributes={"exitCode": "0"}))
    assert len(stopped) == 1
    final = stopped[0]
    assert isinstance(final, ServerStopped)
    assert final.clean is True
    assert final.forced is False
    assert final.exit_code == 0
    assert final.uptime_seconds == pytest.approx(600.0)
    assert reducer.state is LifecycleState.STOPPED


async def test_an_externally_initiated_docker_stop_reconciles_without_us_asking(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """``docker stop minecraft`` typed by a human: ``kill`` -> ``die`` -> ``stop``."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    await clock.advance(300)
    assert reducer.has_stop_intent is False

    killed = reducer.handle(
        docker_event("kill", ts=clock.now(), attributes={"signal": "15"}),
    )
    assert len(killed) == 1
    stopping = killed[0]
    assert isinstance(stopping, ServerStopping)
    assert stopping.requested_by is None
    assert stopping.reason == "external docker stop"
    assert reducer.state is LifecycleState.STOPPING
    assert reducer.has_stop_intent is True, "so the following die classifies as intentional"

    died = reducer.handle(docker_event("die", ts=clock.now(), attributes={"exitCode": "0"}))
    assert len(died) == 1
    assert isinstance(died[0], ServerStopped)

    trailing = reducer.handle(docker_event("stop", ts=clock.now()))
    assert trailing == [], "docker's trailing `stop` action adds nothing"
    assert reducer.state is LifecycleState.STOPPED


async def test_an_external_stop_seen_only_by_the_reconcile(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """Every Docker event missed; only the 60-second inspect notices. Still exactly one event."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    await clock.advance(60)
    await exited(runtime, exit_code=0)

    events = reducer.handle(SnapshotSignal(snapshot=await snapshot_of(runtime)))
    assert len(events) == 1
    stopped = events[0]
    assert isinstance(stopped, ServerStopped)
    assert stopped.clean is True
    assert reducer.state is LifecycleState.STOPPED


async def test_a_stop_line_in_the_log_enters_stopping(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """``[Rcon: Stopping the server]`` - somebody typed /stop in game."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    await clock.advance(120)

    events = reducer.handle(LineSignal(event=stopping_line(ts=clock.now())))
    assert len(events) == 1
    assert isinstance(events[0], ServerStopping)
    assert reducer.state is LifecycleState.STOPPING
    assert reducer.has_stop_intent is True

    assert reducer.handle(LineSignal(event=stopping_line(ts=clock.now()))) == []


async def test_an_unexpected_exit_is_a_crash_and_carries_the_ring_buffer(
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    tail = ["java.lang.OutOfMemoryError: Java heap space", "\tat net.minecraft.Whatever"]
    reducer = LifecycleReducer(
        server_id=SERVER_ID,
        clock=clock,
        tail_provider=lambda: tail,
    )
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    await clock.advance(45)

    events = reducer.handle(docker_event("die", ts=clock.now(), attributes={"exitCode": "1"}))
    assert len(events) == 1
    crashed = events[0]
    assert isinstance(crashed, ServerCrashed)
    assert crashed.exit_code == 1
    assert crashed.oom_killed is False
    assert crashed.tail == tuple(tail)
    assert reducer.state is LifecycleState.CRASHED


async def test_the_tail_is_capped_and_a_throwing_provider_is_survivable(
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    reducer = LifecycleReducer(
        server_id=SERVER_ID,
        clock=clock,
        tail_provider=lambda: [f"line {i}" for i in range(200)],
        tail_lines=5,
    )
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))
    crashed = reducer.handle(docker_event("die", ts=clock.now(), attributes={"exitCode": "1"}))[0]
    assert isinstance(crashed, ServerCrashed)
    assert crashed.tail == ("line 195", "line 196", "line 197", "line 198", "line 199")

    def boom() -> list[str]:
        raise RuntimeError("the ring buffer is on fire")

    reducer2 = LifecycleReducer(server_id=SERVER_ID, clock=clock, tail_provider=boom)
    reducer2.handle(docker_event("start", ts=clock.now()))
    reducer2.handle(SnapshotSignal(snapshot=await running(runtime)))
    second = reducer2.handle(docker_event("die", ts=clock.now(), attributes={"exitCode": "1"}))[0]
    assert isinstance(second, ServerCrashed)
    assert second.tail == ()


async def test_an_oom_event_makes_the_exit_a_crash_even_at_137_with_intent(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    reducer.handle(IntentSignal(intent=Intent.STOP, requested_by="cli", timeout_seconds=90.0))
    reducer.handle(docker_event("oom", ts=clock.now()))

    events = reducer.handle(docker_event("die", ts=clock.now(), attributes={"exitCode": "137"}))
    crashed = events[0]
    assert isinstance(crashed, ServerCrashed)
    assert crashed.oom_killed is True
    assert reducer.state is LifecycleState.CRASHED


async def test_137_with_our_stop_intent_is_a_forced_but_clean_stop(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    reducer.handle(IntentSignal(intent=Intent.STOP, requested_by="cli", timeout_seconds=90.0))

    events = reducer.handle(docker_event("die", ts=clock.now(), attributes={"exitCode": "137"}))
    stopped = events[0]
    assert isinstance(stopped, ServerStopped)
    assert stopped.clean is True
    assert stopped.forced is True, "the JVM did not finish saving inside 90s - a warning"
    assert reducer.state is LifecycleState.STOPPED


async def test_treat_unexpected_exit_as_crash_can_be_turned_off(
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    reducer = LifecycleReducer(
        server_id=SERVER_ID,
        clock=clock,
        report_unexpected_exit_as_crash=False,
    )
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))
    events = reducer.handle(docker_event("die", ts=clock.now(), attributes={"exitCode": "1"}))
    stopped = events[0]
    assert isinstance(stopped, ServerStopped)
    assert stopped.clean is False
    assert reducer.state is LifecycleState.STOPPED


async def test_a_die_with_a_non_integer_exit_code_does_not_explode(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """Docker sends ``exitCode`` as a string, and a garbage one must not take the daemon down."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))
    events = reducer.handle(docker_event("die", ts=clock.now(), attributes={"exitCode": "wat"}))
    crashed = events[0]
    assert isinstance(crashed, ServerCrashed)
    assert crashed.exit_code is None


# ============================================================================ starts and ids


async def test_a_docker_start_event_emits_exactly_one_server_starting(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(SnapshotSignal(snapshot=await exited(runtime, exit_code=0)))
    events = reducer.handle(docker_event("start", ts=clock.now(), container_id="abc123"))

    assert len(events) == 1
    starting = events[0]
    assert isinstance(starting, ServerStarting)
    assert starting.container_id == "abc123"
    assert reducer.state is LifecycleState.STARTING

    assert reducer.handle(docker_event("start", ts=clock.now())) == [], "duplicate is a no-op"


async def test_the_start_intent_attributes_the_next_server_starting(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(SnapshotSignal(snapshot=await exited(runtime, exit_code=0)))

    intent_events = reducer.handle(IntentSignal(intent=Intent.START, requested_by="@kunal"))
    assert intent_events == [], "a start intent emits nothing on its own"

    events = reducer.handle(docker_event("start", ts=clock.now()))
    starting = events[0]
    assert isinstance(starting, ServerStarting)
    assert starting.requested_by == "@kunal"

    # And the attribution is consumed, not sticky.
    reducer.handle(docker_event("die", ts=clock.now(), attributes={"exitCode": "0"}))
    again = reducer.handle(docker_event("start", ts=clock.now()))
    second = again[0]
    assert isinstance(second, ServerStarting)
    assert second.requested_by is None


async def test_a_reconcile_that_finds_a_newly_running_container_emits_server_starting(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
) -> None:
    reducer.handle(SnapshotSignal(snapshot=await exited(runtime, exit_code=0)))
    events = reducer.handle(SnapshotSignal(snapshot=await running(runtime)))

    assert len(events) == 1
    assert isinstance(events[0], ServerStarting)
    assert reducer.state is LifecycleState.STARTING


async def test_a_recreated_container_closes_the_old_run_and_opens_a_new_one(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """``compose down && compose up``: same name, brand new id.

    The one transition that legitimately emits two events. The session manager needs the close as
    much as the open, and a cached container id is exactly how it gets missed.
    """
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    old_id = reducer.container_id
    await clock.advance(120)

    runtime.recreate("newid0000000000")
    events = reducer.handle(SnapshotSignal(snapshot=await running(runtime)))

    assert [type(e).__name__ for e in events] == ["ServerStopped", "ServerStarting"]
    stopped = events[0]
    starting = events[1]
    assert isinstance(stopped, ServerStopped)
    assert isinstance(starting, ServerStarting)
    assert stopped.clean is False, "nobody asked; the world had no say"
    assert starting.container_id == "newid0000000000"
    assert reducer.container_id != old_id
    assert reducer.state is LifecycleState.STARTING


async def test_a_container_removed_out_from_under_us_stops_rather_than_crashes(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    await clock.advance(60)

    events = reducer.handle(SnapshotSignal(snapshot=await absent(runtime)))
    assert len(events) == 1
    stopped = events[0]
    assert isinstance(stopped, ServerStopped), "nothing crashed; it was removed"
    assert stopped.clean is False
    assert reducer.state is LifecycleState.ABSENT


async def test_destroy_while_running_emits_one_stop_then_absent(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    events = reducer.handle(docker_event("destroy", ts=clock.now()))

    assert [type(e).__name__ for e in events] == ["ServerStopped"]
    assert reducer.state is LifecycleState.ABSENT
    assert reducer.container_id is None


async def test_create_after_absent_is_stopped_not_started(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(SnapshotSignal(snapshot=await absent(runtime)))
    events = reducer.handle(docker_event("create", ts=clock.now()))

    assert events == []
    assert reducer.state is LifecycleState.STOPPED, "a container existing is not a server running"


# ================================================================================== probes


async def test_a_failed_probe_is_never_a_state_change(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """Unreachable means unknown. This is the rule that keeps populated servers alive."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    assert reducer.state is LifecycleState.READY

    for _ in range(10):
        await clock.advance(45)
        assert reducer.handle(ProbeSignal(result=probe_failed(ts=clock.now()))) == []

    assert reducer.state is LifecycleState.READY, "ten failed probes in a row change nothing"


async def test_a_failed_probe_while_stopped_stays_stopped(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(SnapshotSignal(snapshot=await exited(runtime, exit_code=0)))
    assert reducer.handle(ProbeSignal(result=probe_failed(ts=clock.now()))) == []
    assert reducer.state is LifecycleState.STOPPED


async def test_a_successful_probe_while_stopped_does_not_invent_a_start(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """The container is stopped; whatever answered on 25565 is not this server."""
    reducer.handle(SnapshotSignal(snapshot=await exited(runtime, exit_code=0)))
    assert reducer.handle(ProbeSignal(result=probe_ok(ts=clock.now(), online=3))) == []
    assert reducer.state is LifecycleState.STOPPED


# ================================================================================== intents


async def test_a_dry_run_stop_never_enters_stopping(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """The idle soak runs like this for days against a live server."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))

    events = reducer.handle(
        IntentSignal(
            intent=Intent.STOP,
            requested_by="idle-manager",
            reason="idle timeout",
            timeout_seconds=90.0,
            dry_run=True,
        )
    )

    assert events == []
    assert reducer.state is LifecycleState.READY
    assert reducer.has_stop_intent is False, (
        "arming the intent would make a later unrelated exit 137 look clean"
    )


@pytest.mark.parametrize("setup", ["stopped", "absent", "unknown"])
async def test_a_stop_intent_in_a_down_state_is_a_noop(
    setup: str,
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
) -> None:
    if setup == "stopped":
        reducer.handle(SnapshotSignal(snapshot=await exited(runtime, exit_code=0)))
    elif setup == "absent":
        reducer.handle(SnapshotSignal(snapshot=await absent(runtime)))
    before = reducer.state

    events = reducer.handle(IntentSignal(intent=Intent.STOP, requested_by="cli"))
    assert events == []
    assert reducer.state is before


async def test_a_second_stop_intent_while_stopping_is_a_noop(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    assert len(reducer.handle(IntentSignal(intent=Intent.STOP, requested_by="a"))) == 1
    assert reducer.handle(IntentSignal(intent=Intent.STOP, requested_by="b")) == []
    assert reducer.state is LifecycleState.STOPPING


async def test_a_running_snapshot_during_stopping_does_not_restart_the_run(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """The 60s reconcile lands mid-stop; the container is still up. That is not a fresh start."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    reducer.handle(IntentSignal(intent=Intent.STOP, requested_by="cli", timeout_seconds=90.0))

    events = reducer.handle(SnapshotSignal(snapshot=await snapshot_of(runtime)))
    assert events == []
    assert reducer.state is LifecycleState.STOPPING


# ========================================================== exhaustiveness over the matrix


def _every_state() -> list[LifecycleState]:
    return list(LifecycleState)


async def _drive_to(
    state: LifecycleState,
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """Put ``reducer`` into ``state`` using only real signals. No private attribute pokes."""
    if state is LifecycleState.UNKNOWN:
        return
    if state is LifecycleState.ABSENT:
        reducer.handle(SnapshotSignal(snapshot=await absent(runtime)))
        return
    if state is LifecycleState.STOPPED:
        reducer.handle(SnapshotSignal(snapshot=await exited(runtime, exit_code=0)))
        return
    if state is LifecycleState.CRASHED:
        reducer.handle(SnapshotSignal(snapshot=await exited(runtime, exit_code=1)))
        return
    if state is LifecycleState.STARTING:
        reducer.handle(SnapshotSignal(snapshot=await exited(runtime, exit_code=0)))
        reducer.handle(docker_event("start", ts=clock.now()))
        reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="starting")))
        return
    if state is LifecycleState.READY:
        await _drive_to(LifecycleState.STARTING, reducer, runtime, clock)
        await clock.advance(32.5)
        reducer.handle(LineSignal(event=done_line(ts=clock.now())))
        return
    if state is LifecycleState.DEGRADED:
        await _drive_to(LifecycleState.READY, reducer, runtime, clock)
        await clock.advance(GUARD_SECONDS + 1)
        reducer.handle(docker_event("health_status: unhealthy", ts=clock.now()))
        return
    if state is LifecycleState.STOPPING:
        await _drive_to(LifecycleState.READY, reducer, runtime, clock)
        reducer.handle(IntentSignal(intent=Intent.STOP, requested_by="cli", timeout_seconds=90.0))
        return
    if state is LifecycleState.BLIND:
        await _drive_to(LifecycleState.READY, reducer, runtime, clock)
        reducer.handle(AvailabilitySignal(available=False, error="socket gone"))
        return
    raise AssertionError(f"no recipe for {state}")


@pytest.mark.parametrize("state", _every_state(), ids=lambda s: s.value)
async def test_every_state_is_reachable_with_real_signals(
    state: LifecycleState,
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """If a state cannot be reached, the transition table has a hole."""
    await _drive_to(state, reducer, runtime, clock)
    assert reducer.state is state


@pytest.mark.parametrize("state", _every_state(), ids=lambda s: s.value)
async def test_no_state_has_an_unhandled_signal_kind(
    state: LifecycleState,
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """The exhaustiveness test: 9 states x 6 signal kinds, none of which may raise.

    A state machine that throws because Docker sent something unexpected at 3am is not a state
    machine. Every combination must either transition or be a logged no-op, and must leave the
    reducer in a real :class:`LifecycleState`.
    """
    await _drive_to(state, reducer, runtime, clock)
    snapshot = await snapshot_of(runtime)

    signals: list[Signal] = [
        docker_event("die", ts=clock.now(), attributes={"exitCode": "0"}),
        SnapshotSignal(snapshot=snapshot),
        LineSignal(event=done_line(ts=clock.now())),
        ProbeSignal(result=probe_ok(ts=clock.now())),
        IntentSignal(intent=Intent.STOP, requested_by="cli", timeout_seconds=90.0),
        AvailabilitySignal(available=False, error="gone"),
    ]
    assert len(signals) == len(SIGNAL_KINDS), "one representative per signal kind"
    assert [type(s) for s in signals] == list(SIGNAL_KINDS)

    for signal in signals:
        fresh = LifecycleReducer(server_id=SERVER_ID, clock=clock)
        await _drive_to(state, fresh, runtime, clock)
        events = fresh.handle(signal)
        assert isinstance(events, list)
        assert all(isinstance(e, object) for e in events)
        assert fresh.state in set(LifecycleState), f"{state.value} + {type(signal).__name__}"


@pytest.mark.parametrize("action", ["pause", "unpause", "rename", "attach", "exec_start", "top"])
async def test_uninteresting_docker_actions_are_logged_noops(
    action: str,
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    before = reducer.state

    assert reducer.handle(docker_event(action, ts=clock.now())) == []
    assert reducer.state is before


def test_the_signal_union_and_the_kind_tuple_agree() -> None:
    """``SIGNAL_KINDS`` is what the exhaustiveness test iterates; it must not drift.

    ``handle`` itself ends with ``assert_never``, so pyright already proves the union is fully
    handled. This asserts the runtime tuple matches what the module documents.
    """
    assert len(SIGNAL_KINDS) == 6
    assert len(set(SIGNAL_KINDS)) == 6


# ================================================================================== service


async def test_the_service_publishes_exactly_what_the_reducer_returned(
    service: LifecycleService,
    sink: RecordingSink,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    service.on_snapshot(await exited(runtime, exit_code=0))
    assert sink.events == []

    service.on_runtime_event(docker_event("start", ts=clock.now()))
    service.on_snapshot(await running(runtime))
    await clock.advance(32.5)
    service.on_line_event(done_line(ts=clock.now()))
    await clock.advance(600)
    service.on_intent(IntentSignal(intent=Intent.STOP, requested_by="cli", timeout_seconds=90.0))
    service.on_runtime_event(docker_event("die", ts=clock.now(), attributes={"exitCode": "0"}))

    assert sink.names == [
        "ServerStarting",
        "ServerReady",
        "ServerStopping",
        "ServerStopped",
    ]
    await service.aclose()


async def test_feed_returns_the_same_events_it_published(
    service: LifecycleService,
    sink: RecordingSink,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    service.on_snapshot(await exited(runtime, exit_code=0))
    returned = service.feed(docker_event("start", ts=clock.now()))
    assert returned == sink.events
    await service.aclose()


async def test_aclose_is_idempotent(service: LifecycleService) -> None:
    await service.aclose()
    await service.aclose()


async def test_describe_is_json_friendly(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))
    described = reducer.describe()

    assert described["state"] == "ready"
    assert described["guard_seconds"] == GUARD_SECONDS
    assert described["start_deadline_seconds"] == START_DEADLINE_SECONDS
    assert described["container_id"] is not None


async def test_container_name_is_the_one_the_fake_answers_to(runtime: FakeRuntime) -> None:
    """Resolution is by name; a snapshot for another name is absent, not an error."""
    assert (await snapshot_of(runtime, CONTAINER)).exists is True
    assert (await snapshot_of(runtime, "not-minecraft")).absent is True


async def test_every_emitted_event_carries_utc_and_the_server_id(
    service: LifecycleService,
    sink: RecordingSink,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    service.on_snapshot(await exited(runtime, exit_code=0))
    service.on_runtime_event(docker_event("start", ts=clock.now()))
    service.on_snapshot(await running(runtime))
    await clock.advance(32.5)
    service.on_line_event(done_line(ts=clock.now()))
    service.on_availability(available=False, error="gone")
    service.on_availability(available=True)

    assert sink.events, "the setup should have produced events"
    for event in sink.events:
        assert event.server_id == SERVER_ID
        assert event.ts.tzinfo is not None, f"{type(event).__name__} has a naive timestamp"
        assert event.ts.utcoffset() == timedelta(0), "all datetimes are UTC"
    await service.aclose()


def test_events_are_frozen_and_cannot_be_mutated_by_a_subscriber() -> None:
    """A guarantee this module relies on when it hands the same list to several places."""
    from mcmanager.clock import ManualClock as _ManualClock

    clock = _ManualClock()
    event: Event = done_line(ts=clock.now())
    with pytest.raises((AttributeError, TypeError)):
        event.ts = clock.now()  # type: ignore[misc]


async def test_health_status_starting_is_a_noop(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """``health_status: starting`` is Docker narrating the start_period, not a fact about us."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime)))

    assert reducer.handle(docker_event("health_status: starting", ts=clock.now())) == []
    assert reducer.state is LifecycleState.STARTING


async def test_a_sigkill_kill_event_records_no_stop_intent(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """``docker kill`` is not asking politely, so the resulting exit is a crash."""
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))

    assert reducer.handle(docker_event("kill", ts=clock.now(), attributes={"signal": "9"})) == []
    assert reducer.has_stop_intent is False
    assert reducer.state is LifecycleState.READY

    events = reducer.handle(docker_event("die", ts=clock.now(), attributes={"exitCode": "137"}))
    crashed = events[0]
    assert isinstance(crashed, ServerCrashed)
    assert reducer.state is LifecycleState.CRASHED


async def test_a_kill_event_while_stopped_is_ignored(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(SnapshotSignal(snapshot=await exited(runtime, exit_code=0)))
    assert reducer.handle(docker_event("kill", ts=clock.now(), attributes={"signal": "15"})) == []
    assert reducer.state is LifecycleState.STOPPED


async def test_a_backfilled_stop_line_is_ignored(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    old = clock.now() - timedelta(days=1)
    reducer.handle(docker_event("start", ts=clock.now()))
    reducer.handle(SnapshotSignal(snapshot=await running(runtime, health="healthy")))

    assert reducer.handle(LineSignal(event=stopping_line(ts=old))) == []
    assert reducer.state is LifecycleState.READY


async def test_a_destroy_while_already_stopped_just_goes_absent(
    reducer: LifecycleReducer,
    runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    reducer.handle(SnapshotSignal(snapshot=await exited(runtime, exit_code=0)))
    assert reducer.handle(docker_event("destroy", ts=clock.now())) == []
    assert reducer.state is LifecycleState.ABSENT


async def test_a_version_line_is_recorded_even_before_the_container_is_known(
    reducer: LifecycleReducer,
    clock: ManualClock,
) -> None:
    """The version line is the only place the version ever appears; never drop it."""
    assert reducer.handle(LineSignal(event=version_line(ts=clock.now(), version="26.3"))) == []
    assert reducer.version == "26.3"
