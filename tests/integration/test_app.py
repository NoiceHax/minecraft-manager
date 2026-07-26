"""The composition root, booted for real against the fake runtime.

Two things are being proved here and neither is provable in a unit test:

1. **The wiring exists and holds.** Every stub has a real constructor, a real subscription and a
   real teardown, so M4-M6 are body-fills rather than a re-architecture. If a signature drifts,
   this file stops importing.
2. **The shutdown sequence is what the plan says it is**, in order, with the bus closed late - and
   the idle manager cannot stop the server on the way out.
"""

from __future__ import annotations

import signal
from typing import TYPE_CHECKING, final

import pytest

from mcmanager.app import Application, ControlSurface, SignalRelay
from mcmanager.clock import ManualClock
from mcmanager.config import RuntimeUnreachableError, load_settings
from mcmanager.containers.dto import ContainerState
from mcmanager.containers.fake import FakeRuntime
from mcmanager.core.events import (
    Event,
    IdleCancelled,
    IdleStopTriggered,
    PlayerLeft,
    ServerReady,
    ServerStopping,
)
from mcmanager.core.types import PlayerRef, ReadySignal, Source
from mcmanager.services.idle import DAEMON_SHUTDOWN

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from mcmanager.config import Settings

EXPECTED_SHUTDOWN_ORDER: tuple[str, ...] = (
    "discord",
    "idle",
    "poller",
    "docker",
    "session",
    "state",
    "bus",
    "tasks",
    "lifecycle",
    "runtime",
    "signals",
)
"""The sequence from plan section 10, with the three additions this module documents: the control
surface closes alongside Discord; supervised tasks are cancelled after the bus has drained, because
the bus's own dispatch loop is one of them; and the signal handlers are restored **last**, so that
a second SIGINT during a wedged shutdown still reaches ``SignalRelay`` and hard-exits deliberately
instead of raising ``KeyboardInterrupt`` into the middle of teardown."""


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A fully offline configuration: fake runtime, no probe, no Discord, no control socket."""
    for name in ("state", "archives", "daemon-logs", "mc-logs"):
        (tmp_path / name).mkdir()
    return load_settings(
        overrides={
            "runtime": "fake",
            "state": {"dir": str(tmp_path / "state")},
            "probe": {"enabled": False},
            "discord": {"enabled": False, "mode": "disabled"},
            "web": {"enabled": False},
            "idle": {"enabled": True, "dry_run": True},
            "logs": {
                "archive_dir": str(tmp_path / "archives"),
                "daemon_log_dir": str(tmp_path / "daemon-logs"),
                "server_log_dir": str(tmp_path / "mc-logs"),
                "archive_on_stop": False,
            },
        }
    )


@pytest.fixture
def runtime(clock: ManualClock) -> FakeRuntime:
    """A fake with the container **running and healthy**, which is what the daemon manages.

    Not cosmetic. ``follow_logs`` faithfully ends immediately on a stopped container, so a log line
    only reaches the pipeline while the container is up; and the idle manager refuses to arm
    unless the lifecycle says the server is up, because an empty roster against a stopped server
    is not an idle server.
    """
    built = FakeRuntime(clock=clock)
    built.set_state(ContainerState.RUNNING, running=True, exit_code=None, health_raw="healthy")
    return built


@pytest.fixture
async def app(
    settings: Settings,
    clock: ManualClock,
    runtime: FakeRuntime,
) -> AsyncIterator[Application]:
    """A started application, always shut down even when the test fails.

    The teardown matters: :meth:`Application.start` installs process-wide signal handlers, and
    leaking those into the rest of the suite would be a genuinely confusing failure.
    """
    application = Application(settings, clock=clock, runtime=runtime)
    await application.start()
    try:
        yield application
    finally:
        await application.aclose()


async def settle(clock: ManualClock, times: int = 8) -> None:
    """Let the bus dispatch. ``tick`` yields to the loop without moving virtual time."""
    for _ in range(times):
        await clock.tick()


# ------------------------------------------------------------------------------------- wiring


async def test_the_application_builds_without_touching_anything(
    settings: Settings,
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    """Construction is pure: no I/O, no loop, no ping. Everything that can fail is in start()."""
    application = Application(settings, clock=clock, runtime=runtime)
    services = application.services
    assert services.settings is settings
    assert runtime.calls == []
    assert application.shutdown_steps == ()


async def test_every_subsystem_is_subscribed_to_the_bus(app: Application) -> None:
    """The stubs are wired now so M4-M6 cannot quietly need a different shape."""
    names = {subscription.name for subscription in app.services.bus.subscriptions}
    assert {
        "lifecycle.runtime_status",
        "pipeline.stopping_flag",
        "session.server",
        "session.players",
        "session.chat",
        "idle.players",
        "idle.server",
    } <= names


async def test_the_supervisor_owns_every_long_lived_task(app: Application) -> None:
    """``status-poller`` is absent because ``probe.enabled`` is false here; see the next test."""
    assert app.services.supervisor.task_names == (
        "bus",
        "state-checkpoint",
        "docker-logs",
        "docker-events",
        "docker-reconcile",
        "idle",
    )


async def test_a_disabled_probe_does_not_spawn_a_poller_task(app: Application) -> None:
    """A disabled poller returns immediately, and a RESTART policy would then log a restart
    warning every backoff interval forever for a subsystem that is off on purpose."""
    assert "status-poller" not in app.services.supervisor.task_names
    assert app.services.poller.stats["probes"] == 0


async def test_an_unreachable_runtime_is_fatal_rather_than_degraded(
    settings: Settings,
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    """Exit 69, naming the socket. Never a daemon that starts and manages nothing."""
    runtime.set_reachable(False)
    application = Application(settings, clock=clock, runtime=runtime)
    with pytest.raises(RuntimeUnreachableError) as excinfo:
        await application.start()
    assert excinfo.value.exit_code == 69


async def test_a_log_line_becomes_an_event_on_the_bus(
    app: Application,
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    """End to end: docker stream -> pipeline -> parser -> bus, with nothing mocked in between."""
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    app.services.bus.subscribe_all(record, name="test.sink")
    await settle(clock)
    runtime.emit_line("[13:24:37] [Server thread/INFO]: Steve joined the game")
    await settle(clock)

    assert any(event.name == "PlayerJoined" for event in seen)
    assert app.services.roster.online == ("Steve",)


async def test_lifecycle_line_events_do_not_reach_the_bus_twice(
    app: Application,
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    """``ServerReady`` is republished by lifecycle, so the pipeline must not publish it as well.

    Publishing from both places is how a first-of-three readiness policy turns into three
    announcements of the same fact.
    """
    ready: list[Event] = []

    async def record(event: Event) -> None:
        if event.name == "ServerReady":
            ready.append(event)

    app.services.bus.subscribe_all(record, name="test.ready")
    await settle(clock)
    runtime.set_state(ContainerState.RUNNING)
    runtime.emit_line('[13:24:37] [Server thread/INFO]: Done (32.521s)! For help, type "help"')
    runtime.emit_line('[13:25:01] [Server thread/INFO]: Done (12.100s)! For help, type "help"')
    await settle(clock)

    assert len(ready) <= 1


# ----------------------------------------------------------------------------------- shutdown


async def test_shutdown_runs_the_documented_sequence_in_order(
    settings: Settings,
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    application = Application(settings, clock=clock, runtime=runtime)
    await application.start()
    await application.aclose()
    assert application.shutdown_steps == EXPECTED_SHUTDOWN_ORDER


async def test_the_bus_drains_before_the_tasks_are_cancelled(
    settings: Settings,
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    """The bus's dispatch loop is a supervised task; cancelling first would drop final events."""
    application = Application(settings, clock=clock, runtime=runtime)
    steps = list(EXPECTED_SHUTDOWN_ORDER)
    await application.start()
    await application.aclose()
    order = application.shutdown_steps
    assert order.index("bus") < order.index("tasks")
    assert order.index("tasks") < order.index("runtime")
    assert steps.index("session") < steps.index("state") < steps.index("bus")


async def test_shutdown_is_idempotent(app: Application) -> None:
    await app.aclose()
    first = app.shutdown_steps
    await app.aclose()
    assert app.shutdown_steps == first


async def test_the_state_file_is_written_on_shutdown(
    settings: Settings,
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    """A ``docker stop mcmanager`` must leave a state file, or the resume marker is fiction."""
    application = Application(settings, clock=clock, runtime=runtime)
    await application.start()
    await application.aclose()
    assert application.services.store.path.exists()


# ------------------------------------------------------------- the most dangerous bug in the design


async def test_idle_teardown_publishes_cancelled_and_never_stops_the_server(
    app: Application,
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    """**The one that matters.**

    An armed idle countdown, then a daemon shutdown. The countdown must be cancelled, the reason
    must be ``daemon_shutdown``, and the Minecraft server must not be touched. An idle timer firing
    during teardown - stopping the server because the *manager* restarted - would be the single most
    dangerous bug in this design.
    """
    published: list[Event] = []

    async def record(event: Event) -> None:
        published.append(event)

    app.services.bus.subscribe_all(record, name="test.idle")
    app.services.bus.publish(
        PlayerLeft(
            ts=clock.now(),
            server_id=app.services.settings.server.id,
            source=Source.LOG,
            player=PlayerRef(name="Steve"),
        )
    )
    await settle(clock)
    assert app.services.idle.armed

    await app.aclose()

    cancels = [event for event in published if isinstance(event, IdleCancelled)]
    assert [event.reason for event in cancels] == [DAEMON_SHUTDOWN]
    assert not any(isinstance(event, IdleStopTriggered) for event in published)
    assert runtime.stop_calls == []
    assert app.services.idle.stats["stops_issued"] == 0


async def test_the_idle_deadline_cannot_fire_after_shutdown(
    app: Application,
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    """Virtual time runs an hour past the deadline once the daemon has gone. Nothing happens."""
    app.services.bus.publish(
        PlayerLeft(
            ts=clock.now(),
            server_id=app.services.settings.server.id,
            source=Source.LOG,
            player=PlayerRef(name="Steve"),
        )
    )
    await settle(clock)
    assert app.services.idle.armed

    await app.aclose()
    await clock.advance(3600.0)

    assert runtime.stop_calls == []


async def test_a_final_event_still_reaches_subscribers_because_the_bus_closes_late(
    app: Application,
    clock: ManualClock,
) -> None:
    """The reason the bus is not the first thing torn down.

    ``IdleCancelled`` is published *during* teardown, and a subscriber must still see it - which is
    only true because the drain happens after every subsystem has closed.
    """
    delivered: list[str] = []

    async def record(event: Event) -> None:
        delivered.append(event.name)

    app.services.bus.subscribe_all(record, name="test.late")
    app.services.bus.publish(
        PlayerLeft(
            ts=clock.now(),
            server_id=app.services.settings.server.id,
            source=Source.LOG,
            player=PlayerRef(name="Steve"),
        )
    )
    await settle(clock)
    delivered.clear()

    await app.aclose()

    assert "IdleCancelled" in delivered


async def test_a_stopping_event_marks_the_pipeline_as_stopping(
    app: Application,
    clock: ManualClock,
) -> None:
    """The window between "we asked Docker to stop" and the JVM's own log line.

    A disconnect inside it is a shutdown casualty, not a voluntary quit, and this subscription is
    what closes that window.
    """
    app.services.bus.publish(
        ServerStopping(
            ts=clock.now(),
            server_id=app.services.settings.server.id,
            source=Source.INTERNAL,
            reason="test",
        )
    )
    await settle(clock)
    assert app.services.pipeline.stopping is True

    app.services.bus.publish(
        ServerReady(
            ts=clock.now(),
            server_id=app.services.settings.server.id,
            source=Source.LOG,
            detected_by=ReadySignal.LOG_DONE,
        )
    )
    await settle(clock)
    assert app.services.pipeline.stopping is False


# ------------------------------------------------------------------------------ control seam


@final
class _RecordingSurface:
    """A stand-in for whatever ``control/`` builds. Two methods is the whole contract."""

    def __init__(self) -> None:
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def aclose(self) -> None:
        self.closed = True


async def test_an_injected_control_surface_is_started_and_closed_in_order(
    tmp_path: Path,
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    for name in ("state", "archives", "daemon-logs", "mc-logs"):
        (tmp_path / name).mkdir()
    settings = load_settings(
        overrides={
            "runtime": "fake",
            "state": {"dir": str(tmp_path / "state")},
            "probe": {"enabled": False},
            "discord": {"enabled": False, "mode": "disabled"},
            "web": {"enabled": True},
            "logs": {
                "archive_dir": str(tmp_path / "archives"),
                "daemon_log_dir": str(tmp_path / "daemon-logs"),
                "server_log_dir": str(tmp_path / "mc-logs"),
            },
        }
    )
    surface: ControlSurface = _RecordingSurface()
    application = Application(settings, clock=clock, runtime=runtime, control_surface=surface)
    await application.start()
    assert isinstance(surface, _RecordingSurface)
    assert surface.started
    await application.aclose()
    assert surface.closed
    assert application.shutdown_steps.index("control") < application.shutdown_steps.index("idle")


async def test_a_missing_control_surface_is_a_warning_not_a_failure(
    settings: Settings,
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    """``web.enabled = false`` here, but the same is true when ``control/`` has not landed.

    The daemon's job is managing the server; losing ``/healthz`` degrades observability, not
    correctness, and refusing to boot over it would be worse.
    """
    application = Application(settings, clock=clock, runtime=runtime)
    await application.start()
    await application.aclose()
    assert "control" not in application.shutdown_steps


# ---------------------------------------------------------------------------------- signals


def test_the_first_signal_requests_a_graceful_shutdown() -> None:
    requested: list[str] = []
    exits: list[int] = []
    relay = SignalRelay(on_shutdown=requested.append, hard_exit=exits.append)

    relay.handle(signal.SIGTERM)

    assert requested == ["signal:SIGTERM"]
    assert exits == []


def test_a_second_signal_hard_exits_immediately() -> None:
    """The operator has decided the orderly path is not working. Waiting politely is not an answer.

    ``os._exit`` is the real default; the injection point exists so this can be asserted without
    taking pytest with it.
    """
    requested: list[str] = []
    exits: list[int] = []
    relay = SignalRelay(on_shutdown=requested.append, hard_exit=exits.append)

    relay.handle(signal.SIGINT)
    relay.handle(signal.SIGINT)

    assert requested == ["signal:SIGINT"]
    assert exits == [128 + int(signal.SIGINT)]
    assert relay.signals_seen == 2


def test_further_signals_keep_hard_exiting() -> None:
    exits: list[int] = []
    relay = SignalRelay(on_shutdown=lambda _: None, hard_exit=exits.append)
    for _ in range(3):
        relay.handle(signal.SIGTERM)
    assert len(exits) == 2


async def test_a_second_signal_during_teardown_still_escalates(
    settings: Settings,
    clock: ManualClock,
    runtime: FakeRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**The escalation, tested through the wiring rather than in isolation.**

    Calling ``handle()`` twice by hand proves the relay. It does not prove the daemon can ever
    reach it - and it could not: ``aclose()`` used to call ``self._signals.remove()`` as its very
    first action, and ``aclose()`` is the only code that runs after ``wait()`` returns, which only
    returns after signal #1. So the second signal always found the *default* handlers back in
    place. A Ctrl-C during a wedged shutdown raised ``KeyboardInterrupt`` at an arbitrary point
    instead: no ``app.signal_escalated`` line, no deterministic exit code, and possibly before the
    state flush at step 7. Both the ``SignalRelay`` docstring and ``deploy/README.md`` promise
    otherwise.

    Here the second signal is delivered from inside step 7 itself.
    """
    exits: list[int] = []
    handlers_during_teardown: list[object] = []
    application = Application(settings, clock=clock, runtime=runtime)
    relay: SignalRelay = application._signals  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(relay, "_hard_exit", exits.append)
    await application.start()

    relay.handle(signal.SIGTERM)  # #1: the graceful request the daemon is already acting on
    original_flush = application._flush_state  # pyright: ignore[reportPrivateUsage]

    def flush_and_signal(*, force: bool = False) -> None:
        handlers_during_teardown.append(signal.getsignal(signal.SIGINT))
        relay.handle(signal.SIGTERM)  # #2, from the middle of the shutdown sequence
        original_flush(force=force)

    monkeypatch.setattr(application, "_flush_state", flush_and_signal)

    await application.aclose()

    assert relay.signals_seen == 2
    assert exits == [128 + int(signal.SIGTERM)]
    assert application.shutdown_steps[-1] == "signals", "handlers are restored last, not first"
    # And the process-level handler really was still ours while teardown ran, rather than the
    # default one that turns a second Ctrl-C into a KeyboardInterrupt mid-sequence.
    assert handlers_during_teardown != [signal.getsignal(signal.SIGINT)]


async def test_signal_handlers_are_installed_and_removed(
    settings: Settings,
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    """Windows takes the ``signal.signal`` fallback; Unix takes ``loop.add_signal_handler``.

    Either way the previous handler must be back afterwards, or the rest of the test suite inherits
    ours.
    """
    before = signal.getsignal(signal.SIGINT)
    application = Application(settings, clock=clock, runtime=runtime)
    await application.start()
    await application.aclose()
    assert signal.getsignal(signal.SIGINT) is before


async def test_a_failed_start_still_closes_the_runtime(
    settings: Settings,
    clock: ManualClock,
    runtime: FakeRuntime,
) -> None:
    """Exit 69 must not leak a docker connection pool: run() tears down a failed start too."""
    runtime.set_reachable(False)
    application = Application(settings, clock=clock, runtime=runtime)
    with pytest.raises(RuntimeUnreachableError):
        await application.run()
    assert "runtime" in application.shutdown_steps
    assert await runtime.ping() is False
