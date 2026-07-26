"""Tests for the idle countdown, and above all for its teardown.

The one that matters is
``test_aclose_publishes_idle_cancelled_and_never_stops_the_server``. An idle timer firing during
daemon teardown - and stopping the Minecraft server because the *manager* was restarting - is the
single most dangerous bug this design could have, so it has a named test rather than a comment.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from mcmanager.clock import ManualClock
from mcmanager.containers.dto import ContainerState
from mcmanager.containers.fake import FakeRuntime
from mcmanager.core.bus import EventBus
from mcmanager.core.events import (
    Event,
    IdleCancelled,
    IdleStarted,
    IdleStopTriggered,
    IdleWarning,
    PlayerJoined,
    PlayerLeft,
    ServerReady,
    ServerStopped,
)
from mcmanager.core.types import (
    LifecycleState,
    PlayerRef,
    ReadySignal,
    ServerId,
    Source,
)
from mcmanager.persistence.state_store import StateStore
from mcmanager.services.controller import ServerController
from mcmanager.services.idle import (
    ARMABLE_STATES,
    DAEMON_SHUTDOWN,
    SERVER_NOT_RUNNING,
    IdleManager,
)
from mcmanager.services.lifecycle import LifecycleReducer, LifecycleService
from mcmanager.services.players import PlayerRoster

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

SERVER: ServerId = "minecraft"
TIMEOUT = 900.0
WARN = 120.0


class RecordingSink:
    """Collects published events. The bus satisfies the same one-method protocol."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)

    def of[E: Event](self, kind: type[E]) -> list[E]:
        return [event for event in self.events if isinstance(event, kind)]


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


@pytest.fixture
def roster(clock: ManualClock) -> PlayerRoster:
    return PlayerRoster(clock=clock)


@pytest.fixture
def runtime(clock: ManualClock) -> FakeRuntime:
    return FakeRuntime(clock=clock)


@pytest.fixture
def lifecycle(clock: ManualClock, sink: RecordingSink) -> LifecycleService:
    reducer = LifecycleReducer(server_id=SERVER, clock=clock)
    return LifecycleService(reducer=reducer, sink=sink, clock=clock)


@pytest.fixture
async def controller(
    clock: ManualClock,
    sink: RecordingSink,
    runtime: FakeRuntime,
    lifecycle: LifecycleService,
) -> ServerController:
    """A controller whose lifecycle reports ``READY``.

    Not incidental setup: an empty roster only means "idle" when there is a server for it to be
    empty *of*, so the manager reads the lifecycle state through this controller before arming.
    Leaving the reducer at its ``UNKNOWN`` default here would leave every countdown test asserting
    against a machine that also has nothing to stop.
    """
    await ready_server(runtime, lifecycle)
    sink.events.clear()
    return ServerController(
        runtime=runtime,
        lifecycle=lifecycle,
        sink=sink,
        clock=clock,
        server_id=SERVER,
        container="minecraft",
        stop_timeout_seconds=90,
    )


async def ready_server(runtime: FakeRuntime, lifecycle: LifecycleService) -> None:
    """Put the fake into ``running``/``healthy`` and let the reducer reconcile onto it."""
    runtime.set_state(
        ContainerState.RUNNING,
        running=True,
        exit_code=None,
        health_raw="healthy",
    )
    lifecycle.on_snapshot(await runtime.inspect("minecraft"))


async def stop_server(runtime: FakeRuntime, lifecycle: LifecycleService) -> None:
    """Stop the fake and let the reducer see it, so the lifecycle state is genuinely down."""
    runtime.set_state(ContainerState.EXITED, running=False, exit_code=0)
    lifecycle.on_snapshot(await runtime.inspect("minecraft"))


def build(
    *,
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
    enabled: bool = True,
    dry_run: bool = True,
) -> IdleManager:
    return IdleManager(
        sink=sink,
        clock=clock,
        server_id=SERVER,
        roster=roster,
        controller=controller,
        enabled=enabled,
        dry_run=dry_run,
        timeout_seconds=TIMEOUT,
        warn_seconds=WARN,
        poll_interval_seconds=60.0,
        min_uptime_seconds=1200.0,
    )


def ready(clock: ManualClock) -> ServerReady:
    return ServerReady(
        ts=clock.now(),
        server_id=SERVER,
        source=Source.LOG,
        detected_by=ReadySignal.LOG_DONE,
    )


def left(clock: ManualClock, name: str = "Steve", *, at: datetime | None = None) -> PlayerLeft:
    return PlayerLeft(
        ts=at if at is not None else clock.now(),
        server_id=SERVER,
        source=Source.LOG,
        player=PlayerRef(name=name),
    )


def joined(clock: ManualClock, name: str = "Steve") -> PlayerJoined:
    return PlayerJoined(
        ts=clock.now(),
        server_id=SERVER,
        source=Source.LOG,
        player=PlayerRef(name=name),
    )


# ---------------------------------------------------------------------------- the dangerous one


async def test_aclose_publishes_idle_cancelled_and_never_stops_the_server(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
    runtime: FakeRuntime,
) -> None:
    """Teardown cancels the countdown, says so, and issues no stop. The whole point of the module.

    A daemon restart must never look like an idle timeout to the server.
    """
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    await idle.on_server_event(ready(clock))
    assert idle.armed

    await idle.aclose()

    cancels = sink.of(IdleCancelled)
    assert [event.reason for event in cancels] == [DAEMON_SHUTDOWN]
    assert not idle.armed
    assert runtime.stop_calls == []
    assert sink.of(IdleStopTriggered) == []


async def test_a_timer_that_fires_during_teardown_is_ignored(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
    runtime: FakeRuntime,
) -> None:
    """Even if the deadline handle survives, ``_closing`` shuts the door.

    Belt and braces: cancelling the handle is the first defence, and this is the second.
    """
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    await idle.on_server_event(ready(clock))
    await idle.aclose()

    idle._fire()  # pyright: ignore[reportPrivateUsage]  # the flag alone must be enough

    assert sink.of(IdleStopTriggered) == []
    assert runtime.stop_calls == []


async def test_advancing_past_the_deadline_after_aclose_stops_nothing(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
    runtime: FakeRuntime,
) -> None:
    """The realistic shape: shutdown happens, then virtual time runs past the old deadline."""
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    await idle.on_server_event(ready(clock))
    await idle.aclose()

    await clock.advance(TIMEOUT * 2)

    assert sink.of(IdleStopTriggered) == []
    assert runtime.stop_calls == []


async def test_aclose_is_idempotent(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
) -> None:
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    await idle.on_server_event(ready(clock))
    await idle.aclose()
    await idle.aclose()
    assert len(sink.of(IdleCancelled)) == 1


async def test_aclose_when_never_armed_publishes_nothing(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
) -> None:
    """A disarmed manager has nothing to cancel and must not invent an event."""
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller, enabled=False)
    await idle.aclose()
    assert sink.events == []


async def test_aclose_unsubscribes_from_the_bus(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
) -> None:
    bus = EventBus()
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    subscriptions = idle.subscribe(bus)
    assert all(subscription.active for subscription in subscriptions)

    await idle.aclose()

    assert not any(subscription.active for subscription in subscriptions)


# ------------------------------------------------------------------------------- the countdown


async def test_an_empty_roster_arms_the_countdown(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
) -> None:
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    await idle.on_player_event(left(clock))

    started = sink.of(IdleStarted)
    assert len(started) == 1
    assert started[0].timeout_seconds == TIMEOUT
    assert started[0].deadline == clock.now() + _delta(TIMEOUT)
    assert idle.armed


async def test_a_populated_roster_does_not_arm(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
) -> None:
    roster.join("Alex")
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    await idle.on_player_event(left(clock))
    assert not idle.armed
    assert sink.of(IdleStarted) == []


async def test_a_join_cancels_the_countdown(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
) -> None:
    """``PlayerJoined -> IdleCancelled`` is the causal chain the SEQUENTIAL bus mode exists for."""
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    await idle.on_player_event(left(clock))
    await clock.advance(60.0)
    await idle.on_player_event(joined(clock))

    cancels = sink.of(IdleCancelled)
    assert [event.reason for event in cancels] == ["player_joined"]
    assert cancels[0].idle_seconds == pytest.approx(60.0)
    assert not idle.armed


async def test_a_server_stop_cancels_the_countdown(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
) -> None:
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    await idle.on_player_event(left(clock))
    await idle.on_server_event(
        ServerStopped(ts=clock.now(), server_id=SERVER, source=Source.RUNTIME, clean=True)
    )
    assert [event.reason for event in sink.of(IdleCancelled)] == ["server_stopped"]


async def test_the_warning_fires_before_the_deadline(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
) -> None:
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    await idle.on_player_event(left(clock))

    await clock.advance(TIMEOUT - WARN)

    warnings = sink.of(IdleWarning)
    assert len(warnings) == 1
    assert warnings[0].remaining_seconds == WARN
    assert sink.of(IdleStopTriggered) == []


async def test_the_deadline_publishes_a_trigger_and_still_stops_nothing(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
    runtime: FakeRuntime,
) -> None:
    """M4 fills the body in. Until then the event is published and the server is untouched."""
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    await idle.on_player_event(left(clock))

    await clock.advance(TIMEOUT)

    triggered = sink.of(IdleStopTriggered)
    assert len(triggered) == 1
    assert triggered[0].dry_run is True
    assert triggered[0].idle_seconds == pytest.approx(TIMEOUT)
    assert runtime.stop_calls == []
    assert idle.stats["stops_issued"] == 0


# -------------------------------------------------------------- nothing arms against a dead server


async def test_a_stopped_server_never_arms_the_countdown(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
    runtime: FakeRuntime,
    lifecycle: LifecycleService,
) -> None:
    """An empty roster against a stopped container is not an idle server.

    It is a server that is off, and the roster is empty because nothing is running.
    """
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    await stop_server(runtime, lifecycle)
    sink.events.clear()

    await idle.on_server_event(
        ServerStopped(ts=clock.now(), server_id=SERVER, source=Source.RUNTIME, clean=True)
    )

    assert not idle.armed
    assert sink.of(IdleStarted) == []


async def test_the_poll_loop_does_not_re_arm_while_the_server_is_stopped(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
    runtime: FakeRuntime,
    lifecycle: LifecycleService,
) -> None:
    """**The phantom-idle loop.**

    ``_fire`` clears the timer, so ``armed`` goes false and the next safety-net poll used to arm
    all over again - a fresh ``IdleStarted``/``IdleWarning``/``IdleStopTriggered`` every fifteen
    minutes, forever, aimed at a container that has been down since Tuesday. In M4 that is a real
    ``controller.stop(...)`` against an already-stopped server, and it breaks the plan's phase-4
    cutover gate outright: none of those "would stop" decisions match a running server at all.
    """
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    await stop_server(runtime, lifecycle)
    sink.events.clear()
    await idle.on_server_event(
        ServerStopped(ts=clock.now(), server_id=SERVER, source=Source.RUNTIME, clean=True)
    )

    task = asyncio.create_task(idle.run())
    await clock.tick()
    try:
        await clock.advance(TIMEOUT * 4)
    finally:
        await idle.aclose()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert sink.events == []
    assert idle.stats["armed"] == 0
    assert idle.stats["triggered"] == 0
    assert idle.stats["polls"] > 0, "the poll loop really did run"


async def test_a_server_that_stops_mid_countdown_disarms(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
    runtime: FakeRuntime,
    lifecycle: LifecycleService,
) -> None:
    """Even with no ``ServerStopped`` on the bus - a crash, or an event we missed - the next poll
    notices the lifecycle state and cancels rather than firing at nothing."""
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    await idle.on_player_event(left(clock))
    assert idle.armed

    await stop_server(runtime, lifecycle)
    task = asyncio.create_task(idle.run())
    await clock.tick()
    try:
        await clock.advance(TIMEOUT * 2)
    finally:
        await idle.aclose()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert [event.reason for event in sink.of(IdleCancelled)] == [SERVER_NOT_RUNNING]
    assert sink.of(IdleStopTriggered) == []
    assert not idle.armed


def test_only_ready_and_degraded_can_arm() -> None:
    """The states in which "nobody is online" means "idle", pinned so widening it is deliberate.

    ``STARTING`` is what ``min_uptime_minutes`` exists for; ``STOPPING`` is already going away;
    ``BLIND`` means we cannot see, and unknown is never empty - the same rule that keeps a failed
    probe from being read as zero players.
    """
    assert set(ARMABLE_STATES) == {LifecycleState.READY, LifecycleState.DEGRADED}


async def test_disabled_never_arms(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
) -> None:
    """``idle.enabled = false`` is the default, and it means exactly nothing happens."""
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller, enabled=False)
    await idle.on_player_event(left(clock))
    await clock.advance(TIMEOUT * 3)
    assert sink.events == []
    assert not idle.armed


async def test_arming_twice_is_a_no_op(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
) -> None:
    """Two leaves in a row must not restart the countdown, or an empty server never times out."""
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    await idle.on_player_event(left(clock, "Steve"))
    first_deadline = idle.deadline
    await clock.advance(30.0)
    await idle.on_player_event(left(clock, "Alex"))
    assert idle.deadline == first_deadline
    assert len(sink.of(IdleStarted)) == 1


async def test_empty_since_comes_from_the_event_not_the_clock(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
) -> None:
    """A backfilled leave is dated by Docker's timestamp: the whole point of carrying it."""
    idle = build(clock=clock, sink=sink, roster=roster, controller=controller)
    earlier = clock.now() - _delta(120.0)
    await idle.on_player_event(left(clock, at=earlier))
    assert idle.empty_since == earlier


async def test_the_persisted_deadline_is_ignored_on_start(
    clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
    controller: ServerController,
    tmp_path: Path,
) -> None:
    """The countdown always restarts from now, or a crash-looping manager insta-stops on boot."""
    store = StateStore(path=tmp_path, clock=clock)
    store.load()
    store.set_idle_deadline(clock.now() - _delta(3600.0))

    idle = IdleManager(
        sink=sink,
        clock=clock,
        server_id=SERVER,
        roster=roster,
        controller=controller,
        store=store,
        enabled=True,
        timeout_seconds=TIMEOUT,
        warn_seconds=WARN,
    )
    await idle.start()

    assert store.state.idle_deadline is None
    assert not idle.armed
    assert sink.of(IdleStopTriggered) == []


def _delta(seconds: float) -> timedelta:
    return timedelta(seconds=seconds)
