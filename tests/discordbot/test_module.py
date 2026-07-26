"""Tests for the Discord subsystem's wiring, its mode gate and its admin gate.

Two properties matter more than the rest and each has a named test:

- **``dryrun`` sends nothing.** It is the setting the whole cutover soak runs under, so a
  ``dryrun`` that quietly posted to a real channel would be discovered by the channel filling up.
- **A non-admin cannot reach the controller.** Not "is refused after being run" - never reaches it.

Nothing here imports discord.py, which is the point of the ``Gateway`` protocol.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from mcmanager.containers.dto import ContainerState
from mcmanager.containers.fake import FakeRuntime
from mcmanager.core.bus import EventBus
from mcmanager.core.events import Event, PlayerJoined
from mcmanager.core.types import DispatchMode, PlayerRef, ServerId, Source
from mcmanager.discordbot.client import CommandSpec, InvocationContext
from mcmanager.discordbot.module import MODE_DISABLED, MODE_DRYRUN, MODE_LIVE, DiscordModule
from mcmanager.services.controller import ServerController
from mcmanager.services.lifecycle import LifecycleReducer, LifecycleService
from mcmanager.services.players import PlayerRoster

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mcmanager.clock import Clock, ManualClock
    from mcmanager.discordbot.client import Gateway

SERVER: ServerId = "minecraft"

# Deliberately invented snowflakes. Real ids from one deployment do not belong in a test suite:
# they are not secret, but they encode whose server this is, and a test that reads like it only
# applies to one guild is a test nobody trusts to run anywhere else.
GUILD = 100000000000000001
CHANNEL = 100000000000000002
ADMIN_ROLE = 100000000000000003
ADMIN_USER = 100000000000000004
RANDO = 42


class FakeGateway:
    """Satisfies ``Gateway`` structurally. No discord.py, no network, no subclassing."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []
        self.commands: dict[str, CommandSpec] = {}
        self.registered: list[str] = []
        self.connected_calls = 0
        self.closed = 0
        self.deliver = True

    @property
    def connected(self) -> bool:
        return self.connected_calls > 0 and self.closed == 0

    def add_command(self, spec: CommandSpec) -> None:
        self.commands[spec.name] = spec

    async def connect(self) -> None:
        self.connected_calls += 1

    async def register_commands(self, names: Sequence[str]) -> None:
        self.registered = list(names)

    async def send(self, channel_id: int, content: str) -> bool:
        if not self.deliver:
            return False
        self.sent.append((channel_id, content))
        return True

    async def aclose(self) -> None:
        self.closed += 1


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


@pytest.fixture
def roster(clock: ManualClock) -> PlayerRoster:
    return PlayerRoster(clock=clock)


@pytest.fixture
async def controller(clock: ManualClock, sink: RecordingSink) -> ServerController:
    runtime = FakeRuntime(clock=clock)
    runtime.set_state(ContainerState.RUNNING, running=True, exit_code=None, health_raw="healthy")
    reducer = LifecycleReducer(server_id=SERVER, clock=clock)
    lifecycle = LifecycleService(reducer=reducer, sink=sink, clock=clock)
    lifecycle.on_snapshot(await runtime.inspect("minecraft"))
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


def build(
    *,
    clock: ManualClock,
    controller: ServerController,
    roster: PlayerRoster,
    gateway: FakeGateway | None = None,
    mode: str = MODE_DRYRUN,
    enabled: bool = True,
    admin_role_id: int = ADMIN_ROLE,
    console_tail: Callable[[int], Sequence[str]] | None = None,
) -> DiscordModule:
    class Token:
        def get_secret_value(self) -> str:
            return "not-a-real-token"

    fixed = gateway

    def factory(*, token: str, clock: Clock, guild_id: int) -> Gateway:
        assert fixed is not None
        return fixed

    return DiscordModule(
        clock=clock,
        server_id=SERVER,
        controller=controller,
        roster=roster,
        console_tail=console_tail,
        mode=mode,
        enabled=enabled,
        token=Token(),
        guild_id=GUILD,
        channel_id=CHANNEL,
        admin_role_id=admin_role_id,
        gateway_factory=factory if gateway is not None else None,
    )


def joined(clock: ManualClock) -> PlayerJoined:
    return PlayerJoined(
        ts=clock.now(),
        server_id=SERVER,
        source=Source.LOG,
        raw="Steve joined the game",
        player=PlayerRef(name="Steve"),
        online_count=1,
    )


def invocation(command: str, *, roles: tuple[int, ...], user_id: int = RANDO) -> InvocationContext:
    return InvocationContext(
        command=command,
        user_id=user_id,
        user_name="tester",
        role_ids=roles,
        channel_id=CHANNEL,
    )


# ------------------------------------------------------------------------------- mode gate


async def test_dry_run_renders_everything_and_sends_nothing(
    clock: ManualClock, controller: ServerController, roster: PlayerRoster
) -> None:
    """The setting the whole cutover soak runs under."""
    gateway = FakeGateway()
    module = build(clock=clock, controller=controller, roster=roster, gateway=gateway)
    await module.start()
    await module.on_event(joined(clock))

    assert gateway.sent == [], "dryrun must not send"
    assert gateway.connected_calls == 0, "dryrun must not even open the gateway"
    assert module.stats["messages_suppressed"] == 1
    assert module.stats["messages_sent"] == 0


async def test_disabled_does_not_even_subscribe(
    clock: ManualClock, controller: ServerController, roster: PlayerRoster
) -> None:
    module = build(clock=clock, controller=controller, roster=roster, mode=MODE_DISABLED)
    bus = EventBus()
    assert module.subscribe(bus) == ()
    await module.on_event(joined(clock))
    assert module.stats["events_seen"] == 0


async def test_live_actually_sends_and_registers_commands(
    clock: ManualClock, controller: ServerController, roster: PlayerRoster
) -> None:
    gateway = FakeGateway()
    module = build(
        clock=clock, controller=controller, roster=roster, gateway=gateway, mode=MODE_LIVE
    )
    await module.start()

    assert gateway.connected_calls == 1
    assert set(gateway.registered) == set(gateway.commands)
    assert "stop" in gateway.commands

    await module.on_event(joined(clock))
    assert len(gateway.sent) == 1
    channel, content = gateway.sent[0]
    assert channel == CHANNEL
    assert "Steve" in content
    assert module.stats["messages_sent"] == 1


async def test_the_relay_subscription_is_concurrent(
    clock: ManualClock, controller: ServerController, roster: PlayerRoster
) -> None:
    """A Discord round trip on the bus's sequential path would stall every other subscriber."""
    module = build(clock=clock, controller=controller, roster=roster)
    bus = EventBus()
    subs = module.subscribe(bus)
    assert len(subs) == 1
    assert subs[0].mode is DispatchMode.CONCURRENT


# ------------------------------------------------------------------------------ admin gate


async def test_a_non_admin_never_reaches_the_controller(
    clock: ManualClock, controller: ServerController, roster: PlayerRoster
) -> None:
    """Refused *before* the command body, not after."""
    gateway = FakeGateway()
    module = build(
        clock=clock, controller=controller, roster=roster, gateway=gateway, mode=MODE_LIVE
    )
    await module.start()

    reply = await gateway.commands["stop"].handler(invocation("stop", roles=(999,)))

    assert "do not hold the admin role" in reply
    assert controller.state.value == "ready", "the server must be untouched"
    assert module.stats["commands_refused"] == 1


async def test_an_admin_reaches_the_controller(
    clock: ManualClock, controller: ServerController, roster: PlayerRoster
) -> None:
    gateway = FakeGateway()
    module = build(
        clock=clock, controller=controller, roster=roster, gateway=gateway, mode=MODE_LIVE
    )
    await module.start()

    reply = await gateway.commands["stop"].handler(
        invocation("stop", roles=(ADMIN_ROLE,), user_id=ADMIN_USER)
    )

    assert "accepted" in reply.lower() or "refused" in reply.lower()
    assert module.stats["commands_refused"] == 0
    assert module.stats["commands_handled"] == 1


async def test_an_unconfigured_admin_role_refuses_even_the_owner(
    clock: ManualClock, controller: ServerController, roster: PlayerRoster
) -> None:
    """A gate that fails open is worse than no gate."""
    gateway = FakeGateway()
    module = build(
        clock=clock,
        controller=controller,
        roster=roster,
        gateway=gateway,
        mode=MODE_LIVE,
        admin_role_id=0,
    )
    await module.start()

    reply = await gateway.commands["start"].handler(
        invocation("start", roles=(ADMIN_ROLE,), user_id=ADMIN_USER)
    )
    assert "not configured" in reply


async def test_read_only_commands_need_no_role(
    clock: ManualClock, controller: ServerController, roster: PlayerRoster
) -> None:
    gateway = FakeGateway()
    module = build(
        clock=clock, controller=controller, roster=roster, gateway=gateway, mode=MODE_LIVE
    )
    await module.start()

    reply = await gateway.commands["players"].handler(invocation("players", roles=()))
    assert "Nobody is online" in reply


async def test_a_handler_that_raises_is_reported_not_propagated(
    clock: ManualClock, controller: ServerController, roster: PlayerRoster
) -> None:
    """An exception must become a message, not an unanswered interaction."""

    def boom(_: int) -> Sequence[str]:
        raise RuntimeError("ring buffer exploded")

    gateway = FakeGateway()
    module = build(
        clock=clock,
        controller=controller,
        roster=roster,
        gateway=gateway,
        mode=MODE_LIVE,
        console_tail=boom,
    )
    await module.start()

    reply = await gateway.commands["logs"].handler(invocation("logs", roles=()))
    assert "failed" in reply
    assert module.stats["errors"] == 1


# --------------------------------------------------------------------------------- teardown


async def test_aclose_is_idempotent_and_closes_the_gateway(
    clock: ManualClock, controller: ServerController, roster: PlayerRoster
) -> None:
    gateway = FakeGateway()
    module = build(
        clock=clock, controller=controller, roster=roster, gateway=gateway, mode=MODE_LIVE
    )
    await module.start()
    await module.aclose()
    await module.aclose()
    assert gateway.closed == 1


async def test_events_after_close_are_ignored(
    clock: ManualClock, controller: ServerController, roster: PlayerRoster
) -> None:
    gateway = FakeGateway()
    module = build(
        clock=clock, controller=controller, roster=roster, gateway=gateway, mode=MODE_LIVE
    )
    await module.start()
    await module.aclose()
    await module.on_event(joined(clock))
    assert gateway.sent == []


def test_importing_the_daemon_does_not_import_discord_py() -> None:
    """``discord.enabled = false`` must cost nothing, and the parser must stay library-free.

    The gateway import lives inside ``DiscordGateway.connect``, reached only from the ``live``
    branch. If somebody moves it to module scope this test fails, and so does the claim that
    ``mcmanager replay`` and every service test are free of discord.py.

    Run in a subprocess: this test session has almost certainly imported discord.py already, via
    some other test, so checking ``sys.modules`` in-process would prove nothing.
    """
    probe = (
        "import sys;"
        "import mcmanager.app, mcmanager.cli.main, mcmanager.discordbot.module;"
        "import mcmanager.games.minecraft.parser;"
        "leaked=[m for m in sys.modules if m=='discord' or m.startswith('discord.')];"
        "print(','.join(sorted(leaked)))"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "", f"discord.py was imported at module scope: {result.stdout}"
