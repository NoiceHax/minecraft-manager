"""``ServerController``: the one path a start, stop or restart may take.

Everything here runs against :class:`~mcmanager.containers.fake.FakeRuntime` with
``auto_transition=True``, so ``start()`` and ``stop()`` really do move the container and emit the
Docker events the daemon would see in production - which is what makes these end-to-end assertions
about the *emitted event sequence*, not just about which methods were called.

The properties under test, in the order they matter:

- every attempt publishes ``CommandIssued`` first, rejected ones included;
- the stop intent reaches lifecycle **before** the runtime is touched, so an exit during the call
  is already classified as intentional;
- a stop uses the configured 90-second grace period, every time;
- ``restart`` is stop-then-start *here*, so the daemon owns the transition;
- a dry run records and does nothing;
- a runtime that has gone away drives lifecycle into ``BLIND`` instead of raising at the caller.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from builders import (  # sibling test helper module, see builders.py's docstring
    CONTAINER,
    SERVER_ID,
    STOP_TIMEOUT,
    RecordingSink,
    docker_event,
    done_line,
    exited,
    running,
    snapshot_of,
)

from mcmanager.containers.errors import ContainerNotFoundError, RuntimeUnavailableError
from mcmanager.containers.fake import FakeRuntime
from mcmanager.core.events import CommandIssued, ServerStopping
from mcmanager.core.types import ControlAction, LifecycleState, Source
from mcmanager.services.controller import ControlOutcome, ServerController
from mcmanager.services.lifecycle import LifecycleReducer, LifecycleService

if TYPE_CHECKING:
    from mcmanager.clock import ManualClock


# ------------------------------------------------------------------------------------- fixtures
#
# Declared here rather than imported from builders.py, for the reason its docstring gives.


@pytest.fixture
def live_runtime(clock: ManualClock) -> FakeRuntime:
    """The fake with ``auto_transition`` on.

    ``start()``/``stop()`` then move the snapshot and emit the matching Docker events, which is
    what the controller is actually driving in production - so these tests assert on the emitted
    event sequence rather than on which methods were called.
    """
    return FakeRuntime(clock=clock, name=CONTAINER, auto_transition=True)


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


@pytest.fixture
def reducer(clock: ManualClock) -> LifecycleReducer:
    return LifecycleReducer(server_id=SERVER_ID, clock=clock)


@pytest.fixture
def service(
    reducer: LifecycleReducer,
    sink: RecordingSink,
    clock: ManualClock,
) -> LifecycleService:
    return LifecycleService(reducer=reducer, sink=sink, clock=clock)


class EventBridge:
    """Stands in for ``DockerManager``'s ``on_event`` callback.

    ``FakeRuntime(auto_transition=True)`` emits the same Docker events the daemon would see, but
    nothing in a unit test is pumping the event stream. This forwards whatever the fake has emitted
    since the last drain, in order, which is exactly what the manager does in production - and it
    keeps these tests honest about the fact that ``ServerStopped`` comes from Docker, not from the
    controller deciding it happened.
    """

    def __init__(self, runtime: FakeRuntime, service: LifecycleService) -> None:
        self._runtime = runtime
        self._service = service
        self._forwarded = 0

    def drain(self) -> None:
        history = self._runtime.event_history
        for event in history[self._forwarded :]:
            self._service.on_runtime_event(event)
        self._forwarded = len(history)


@pytest.fixture
def bridge(live_runtime: FakeRuntime, service: LifecycleService) -> EventBridge:
    return EventBridge(live_runtime, service)


@pytest.fixture
def controller(
    live_runtime: FakeRuntime,
    service: LifecycleService,
    sink: RecordingSink,
    clock: ManualClock,
) -> ServerController:
    return ServerController(
        runtime=live_runtime,
        lifecycle=service,
        sink=sink,
        clock=clock,
        server_id=SERVER_ID,
        container=CONTAINER,
        stop_timeout_seconds=STOP_TIMEOUT,
    )


def _commands(sink: RecordingSink) -> list[CommandIssued]:
    return [e for e in sink.events if isinstance(e, CommandIssued)]


# ===================================================================================== start


async def test_start_from_stopped_issues_the_command_and_starts_the_container(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
    bridge: EventBridge,
    sink: RecordingSink,
) -> None:
    service.on_snapshot(await exited(live_runtime, exit_code=0))
    sink.clear()

    outcome = await controller.start(actor="@kunal", via=Source.DISCORD)
    bridge.drain()

    assert outcome.ok
    assert outcome.action is ControlAction.START
    assert live_runtime.start_calls == [CONTAINER]

    issued = _commands(sink)
    assert len(issued) == 1
    assert issued[0].action is ControlAction.START
    assert issued[0].actor == "@kunal"
    assert issued[0].via is Source.DISCORD
    assert issued[0].accepted is True
    assert issued[0].rejection is None
    assert sink.names[0] == "CommandIssued", "the audit entry comes first, before anything happens"
    assert "ServerStarting" in sink.names
    assert service.state is LifecycleState.STARTING


async def test_start_attributes_the_resulting_server_starting_to_the_actor(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
    bridge: EventBridge,
    sink: RecordingSink,
) -> None:
    service.on_snapshot(await exited(live_runtime, exit_code=0))
    sink.clear()

    await controller.start(actor="@kunal", via=Source.DISCORD)
    bridge.drain()

    starting = [e for e in sink.events if type(e).__name__ == "ServerStarting"]
    assert len(starting) == 1
    assert getattr(starting[0], "requested_by", None) == "@kunal"


@pytest.mark.parametrize("state", ["starting", "ready", "stopping"])
async def test_start_is_rejected_when_the_server_is_already_up(
    state: str,
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
    sink: RecordingSink,
    clock: ManualClock,
) -> None:
    service.on_runtime_event(docker_event("start", ts=clock.now()))
    service.on_snapshot(await running(live_runtime))
    if state in ("ready", "stopping"):
        await clock.advance(32.5)
        service.on_line_event(done_line(ts=clock.now()))
    if state == "stopping":
        await controller.stop(actor="cli", via=Source.CLI)
    sink.clear()

    outcome = await controller.start(actor="cli", via=Source.CLI)

    assert not outcome.accepted
    assert outcome.rejection is not None
    issued = _commands(sink)
    assert len(issued) == 1, "a rejected command is still audited"
    assert issued[0].accepted is False
    assert issued[0].rejection == outcome.rejection


async def test_start_is_rejected_when_the_container_does_not_exist(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
) -> None:
    live_runtime.set_absent()
    service.on_snapshot(await snapshot_of(live_runtime))

    outcome = await controller.start(actor="cli", via=Source.CLI)

    assert not outcome.accepted
    assert "no container named" in (outcome.rejection or "")
    assert "docker compose up" in (outcome.rejection or ""), "the message says what to do"
    assert live_runtime.start_calls == []


# ====================================================================================== stop


async def test_stop_uses_the_ninety_second_grace_period(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """The number that decides whether a Minecraft world finishes saving."""
    service.on_runtime_event(docker_event("start", ts=clock.now()))
    service.on_snapshot(await running(live_runtime))
    await clock.advance(32.5)
    service.on_line_event(done_line(ts=clock.now()))

    outcome = await controller.stop(actor="@kunal", via=Source.DISCORD, reason="bedtime")

    assert outcome.ok
    live_runtime.assert_stopped_with_timeout(90)
    assert controller.stop_timeout_seconds == 90


async def test_stop_declares_the_intent_before_touching_the_runtime(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
    reducer: LifecycleReducer,
    bridge: EventBridge,
    sink: RecordingSink,
    clock: ManualClock,
) -> None:
    """Ordering, asserted through the emitted sequence rather than by mocking."""
    service.on_runtime_event(docker_event("start", ts=clock.now()))
    service.on_snapshot(await running(live_runtime))
    await clock.advance(32.5)
    service.on_line_event(done_line(ts=clock.now()))
    sink.clear()

    await controller.stop(actor="cli", via=Source.CLI, reason="idle timeout")
    bridge.drain()

    assert sink.names == ["CommandIssued", "ServerStopping", "ServerStopped"]
    stopping = sink.events[1]
    assert isinstance(stopping, ServerStopping)
    assert stopping.requested_by == "cli"
    assert stopping.reason == "idle timeout"
    assert stopping.timeout_seconds == 90.0

    stopped = sink.events[2]
    assert getattr(stopped, "clean", None) is True, "exit 0, classified as ours"
    assert reducer.state is LifecycleState.STOPPED


@pytest.mark.parametrize("state", ["stopped", "crashed", "absent", "unknown"])
async def test_stop_is_rejected_when_there_is_nothing_to_stop(
    state: str,
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
    sink: RecordingSink,
) -> None:
    if state == "stopped":
        service.on_snapshot(await exited(live_runtime, exit_code=0))
    elif state == "crashed":
        service.on_snapshot(await exited(live_runtime, exit_code=1))
    elif state == "absent":
        live_runtime.set_absent()
        service.on_snapshot(await snapshot_of(live_runtime))
    sink.clear()

    outcome = await controller.stop(actor="cli", via=Source.CLI)

    assert not outcome.accepted
    assert live_runtime.stop_calls == []
    live_runtime.assert_never_stopped()
    assert len(_commands(sink)) == 1


async def test_a_second_stop_while_one_is_in_flight_is_rejected(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    service.on_runtime_event(docker_event("start", ts=clock.now()))
    service.on_snapshot(await running(live_runtime))
    await clock.advance(32.5)
    service.on_line_event(done_line(ts=clock.now()))

    first, second = await asyncio.gather(
        controller.stop(actor="a", via=Source.CLI),
        controller.stop(actor="b", via=Source.DISCORD),
    )

    accepted = [o for o in (first, second) if o.accepted]
    assert len(accepted) == 1, "the lock serialises them; the loser sees a stopped server"
    assert len(live_runtime.stop_calls) == 1


# =================================================================================== restart


async def test_restart_is_stop_then_start_in_that_order(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
    bridge: EventBridge,
    sink: RecordingSink,
    clock: ManualClock,
) -> None:
    """The daemon owns the transition, which is why ``ContainerRuntime`` has no ``restart()``."""
    service.on_runtime_event(docker_event("start", ts=clock.now()))
    service.on_snapshot(await running(live_runtime))
    await clock.advance(32.5)
    service.on_line_event(done_line(ts=clock.now()))
    sink.clear()

    outcome = await controller.restart(actor="@kunal", via=Source.DISCORD)
    bridge.drain()

    assert outcome.ok
    assert live_runtime.calls.count("stop") == 1
    assert live_runtime.calls.count("start") == 1
    assert live_runtime.calls.index("stop") < live_runtime.calls.index("start")
    live_runtime.assert_stopped_with_timeout(90)

    assert sink.names == [
        "CommandIssued",
        "ServerStopping",
        "ServerStopped",
        "ServerStarting",
    ]
    issued = _commands(sink)
    assert len(issued) == 1, "one audit entry for what a human actually asked for"
    assert issued[0].action is ControlAction.RESTART


async def test_restart_of_a_stopped_server_degrades_to_a_start(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
    bridge: EventBridge,
    sink: RecordingSink,
) -> None:
    service.on_snapshot(await exited(live_runtime, exit_code=0))
    sink.clear()

    outcome = await controller.restart(actor="cli", via=Source.CLI)
    bridge.drain()

    assert outcome.ok
    assert live_runtime.stop_calls == []
    assert live_runtime.start_calls == [CONTAINER]
    assert sink.names == ["CommandIssued", "ServerStarting"]


async def test_restart_is_rejected_while_blind(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
    sink: RecordingSink,
) -> None:
    service.on_snapshot(await running(live_runtime, health="healthy"))
    service.on_availability(available=False, error="socket gone")
    sink.clear()

    outcome = await controller.restart(actor="cli", via=Source.CLI)

    assert not outcome.accepted
    assert outcome.rejection == "the container runtime is unreachable"
    assert live_runtime.stop_calls == []
    assert live_runtime.start_calls == []


# =================================================================================== dry run


async def test_dry_run_records_everything_and_does_nothing(
    live_runtime: FakeRuntime,
    service: LifecycleService,
    sink: RecordingSink,
    clock: ManualClock,
) -> None:
    """The cutover soak runs like this for days against a live server."""
    controller = ServerController(
        runtime=live_runtime,
        lifecycle=service,
        sink=sink,
        clock=clock,
        server_id=SERVER_ID,
        container=CONTAINER,
        stop_timeout_seconds=STOP_TIMEOUT,
        dry_run=True,
    )
    service.on_runtime_event(docker_event("start", ts=clock.now()))
    service.on_snapshot(await running(live_runtime))
    await clock.advance(32.5)
    service.on_line_event(done_line(ts=clock.now()))
    sink.clear()

    stop = await controller.stop(actor="idle-manager", via=Source.TIMER, reason="idle timeout")
    restart = await controller.restart(actor="cli", via=Source.CLI)

    assert stop.accepted
    assert stop.dry_run
    assert restart.accepted
    assert restart.dry_run
    live_runtime.assert_never_stopped()
    assert live_runtime.start_calls == []
    assert service.state is LifecycleState.READY, "nothing entered STOPPING"

    issued = _commands(sink)
    assert len(issued) == 2
    assert all(command.dry_run for command in issued)
    assert [type(e).__name__ for e in sink.events] == ["CommandIssued", "CommandIssued"], (
        "a dry run publishes the audit trail and nothing else"
    )


# ================================================================= runtime failure handling


async def test_a_runtime_that_went_away_drives_lifecycle_blind_rather_than_raising(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
    sink: RecordingSink,
    clock: ManualClock,
) -> None:
    service.on_runtime_event(docker_event("start", ts=clock.now()))
    service.on_snapshot(await running(live_runtime))
    await clock.advance(32.5)
    service.on_line_event(done_line(ts=clock.now()))
    sink.clear()

    live_runtime.fail_next("stop", RuntimeUnavailableError("socket gone", endpoint="unix://x"))
    outcome = await controller.stop(actor="cli", via=Source.CLI)

    assert outcome.accepted, "we accepted it; the platform then failed"
    assert not outcome.ok
    assert outcome.error is not None
    assert "socket gone" in outcome.error
    assert service.state is LifecycleState.BLIND
    assert "RuntimeUnavailable" in sink.names


async def test_an_ordinary_runtime_error_does_not_go_blind(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
    sink: RecordingSink,
) -> None:
    service.on_snapshot(await exited(live_runtime, exit_code=0))
    sink.clear()

    live_runtime.fail_next("start", ContainerNotFoundError(CONTAINER))
    outcome = await controller.start(actor="cli", via=Source.CLI)

    assert outcome.accepted
    assert not outcome.ok
    assert "no such container" in (outcome.error or "")
    assert service.state is LifecycleState.STOPPED, "not BLIND: the platform answered"
    assert "RuntimeUnavailable" not in sink.names


async def test_a_failed_restart_stop_does_not_go_on_to_start(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
    clock: ManualClock,
) -> None:
    """Half a restart is worse than none: never start a server whose stop failed."""
    service.on_runtime_event(docker_event("start", ts=clock.now()))
    service.on_snapshot(await running(live_runtime))
    await clock.advance(32.5)
    service.on_line_event(done_line(ts=clock.now()))

    live_runtime.fail_next("stop", RuntimeUnavailableError("socket gone"))
    outcome = await controller.restart(actor="cli", via=Source.CLI)

    assert not outcome.ok
    assert live_runtime.start_calls == []


# ==================================================================================== misc


async def test_the_outcome_renders_for_a_human() -> None:
    accepted = ControlOutcome(action=ControlAction.STOP, actor="cli", accepted=True)
    rejected = ControlOutcome(
        action=ControlAction.START,
        actor="cli",
        accepted=False,
        rejection="the server is already ready",
    )
    failed = ControlOutcome(
        action=ControlAction.STOP,
        actor="cli",
        accepted=True,
        error="socket gone",
    )
    dry = ControlOutcome(
        action=ControlAction.STOP,
        actor="cli",
        accepted=True,
        dry_run=True,
    )

    assert str(accepted) == "stop ok"
    assert "already ready" in str(rejected)
    assert "socket gone" in str(failed)
    assert "dry run" in str(dry)
    assert accepted.ok
    assert not rejected.ok
    assert not failed.ok


async def test_command_issued_carries_utc_and_the_server_id(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
    sink: RecordingSink,
) -> None:
    service.on_snapshot(await exited(live_runtime, exit_code=0))
    sink.clear()
    await controller.start(actor="cli", via=Source.CLI)

    issued = _commands(sink)[0]
    assert issued.server_id == SERVER_ID
    assert issued.ts.tzinfo is not None
    assert issued.ts.utcoffset() is not None
    assert issued.ts.utcoffset().total_seconds() == 0  # pyright: ignore[reportOptionalMemberAccess]


async def test_the_controller_never_imports_docker() -> None:
    """The constraint, asserted rather than trusted."""
    import sys

    import mcmanager.services.controller as module

    assert "docker" not in {name.split(".")[0] for name in vars(module) if isinstance(name, str)}
    assert not any(
        getattr(value, "__module__", "").startswith("docker")
        for value in vars(module).values()
        if isinstance(value, type)
    )
    del sys


async def test_a_dry_run_start_records_and_does_nothing(
    live_runtime: FakeRuntime,
    service: LifecycleService,
    sink: RecordingSink,
    clock: ManualClock,
) -> None:
    controller = ServerController(
        runtime=live_runtime,
        lifecycle=service,
        sink=sink,
        clock=clock,
        server_id=SERVER_ID,
        container=CONTAINER,
        stop_timeout_seconds=STOP_TIMEOUT,
        dry_run=True,
    )
    service.on_snapshot(await exited(live_runtime, exit_code=0))
    sink.clear()

    outcome = await controller.start(actor="cli", via=Source.CLI)

    assert outcome.accepted
    assert outcome.dry_run
    assert controller.dry_run
    assert live_runtime.start_calls == []
    assert sink.names == ["CommandIssued"]


async def test_start_and_stop_are_rejected_while_blind(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
) -> None:
    service.on_snapshot(await running(live_runtime, health="healthy"))
    service.on_availability(available=False, error="socket gone")

    start = await controller.start(actor="cli", via=Source.CLI)
    stop = await controller.stop(actor="cli", via=Source.CLI)

    assert start.rejection == "the container runtime is unreachable"
    assert stop.rejection == "the container runtime is unreachable"
    assert live_runtime.start_calls == []
    live_runtime.assert_never_stopped()


async def test_stop_is_rejected_before_the_first_inspect(
    controller: ServerController,
    live_runtime: FakeRuntime,
) -> None:
    """``UNKNOWN`` is "we have not looked yet", which is not licence to stop the server."""
    outcome = await controller.stop(actor="cli", via=Source.CLI)

    assert not outcome.accepted
    assert outcome.state_before is LifecycleState.UNKNOWN
    live_runtime.assert_never_stopped()


async def test_the_controller_reports_the_state_it_will_act_against(
    controller: ServerController,
    service: LifecycleService,
    live_runtime: FakeRuntime,
) -> None:
    assert controller.state is LifecycleState.UNKNOWN
    service.on_snapshot(await exited(live_runtime, exit_code=0))
    assert controller.state is LifecycleState.STOPPED
