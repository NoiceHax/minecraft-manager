"""Fixtures and builders shared by the lifecycle and controller tests.

Deliberately **not** a ``conftest.py``. Other agents own other files in ``tests/services/`` and a
shared conftest is the one file two people cannot both create; a plain module that each test file
imports from has the same effect with none of the collision.

It holds constants, the recording sink and pure builder *functions* only. The pytest fixtures live
in the test modules that use them, because a fixture imported into a test module reads to ruff as
an unused import and gets deleted by ``--fix`` - a failure mode that produces a hundred
"fixture not found" errors and no obvious cause.

Everything here runs on :class:`~mcmanager.clock.ManualClock` and
:class:`~mcmanager.containers.fake.FakeRuntime`. Nothing sleeps, nothing opens a socket.

The fake's defaults mirror the verified homelab container, so ``guard_window`` here is a real 210
seconds (``start_period=120s`` + ``interval=30s`` x ``(retries=2 + 1)``) rather than a number this
test suite invented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcmanager.containers.dto import ContainerState, RuntimeEvent
from mcmanager.core.events import Event, ServerReady, ServerStarting, ServerStopping
from mcmanager.core.types import ReadySignal, ServerId, Source
from mcmanager.games.base import ProbeResult

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from mcmanager.containers.dto import ContainerSnapshot
    from mcmanager.containers.fake import FakeRuntime

SERVER_ID: ServerId = "minecraft"
CONTAINER = "minecraft"

CONTAINER_ID = "fake00000000000000000000000000000000000000000000000000000000"
"""``FakeRuntime``'s default container id.

Docker events and inspects have to agree about the id, or the reducer correctly concludes the
container was replaced and emits the ``compose down/up`` pair. That is a real transition with a
real test of its own; it must not turn up by accident in every other one.
"""

GUARD_SECONDS = 210.0
"""``start_period(120) + interval(30) * (retries(2) + 1)``, off the fake's real timings."""

START_DEADLINE_SECONDS = 270.0
"""``GUARD_SECONDS + 60``. Both are asserted against the reducer rather than assumed."""

STOP_TIMEOUT = 90
"""How long a Minecraft world save can take. Formerly a bare ``-t 90`` in a bash script."""


class RecordingSink:
    """Collects published events. Structurally an ``EventSink``; no dispatch loop involved."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)

    @property
    def names(self) -> list[str]:
        return [type(event).__name__ for event in self.events]

    def clear(self) -> None:
        self.events.clear()


# ------------------------------------------------------------------------------------ builders


async def snapshot_of(runtime: FakeRuntime, name: str = CONTAINER) -> ContainerSnapshot:
    """Inspect the fake. Goes through the real ``ContainerRuntime`` interface on purpose."""
    return await runtime.inspect(name)


async def running(
    runtime: FakeRuntime,
    *,
    health: str | None = "starting",
    started_at: datetime | None = None,
) -> ContainerSnapshot:
    """Put the fake into ``running`` with a given raw health string, and inspect it."""
    runtime.set_state(
        ContainerState.RUNNING,
        running=True,
        exit_code=None,
        health_raw=health,
        started_at=started_at,
    )
    return await snapshot_of(runtime)


async def exited(
    runtime: FakeRuntime,
    *,
    exit_code: int = 0,
    oom_killed: bool = False,
    health: str | None = None,
) -> ContainerSnapshot:
    """Put the fake into ``exited``.

    ``health`` is left alone by default, which reproduces Docker's real behaviour: it never clears
    ``State.Health.Status`` on stop, so an exited container can still report ``"unhealthy"``.
    """
    runtime.set_state(
        ContainerState.EXITED,
        running=False,
        exit_code=exit_code,
        oom_killed=oom_killed,
        health_raw=health,
    )
    return await snapshot_of(runtime)


async def absent(runtime: FakeRuntime) -> ContainerSnapshot:
    runtime.set_absent()
    return await snapshot_of(runtime)


def docker_event(
    action: str,
    *,
    ts: datetime,
    container_id: str = CONTAINER_ID,
    attributes: Mapping[str, str] | None = None,
) -> RuntimeEvent:
    """A Docker daemon event.

    Every attribute value is a **string**, exactly as Docker sends them - ``exitCode`` included,
    which is the landmine ``RuntimeEvent.exit_code`` exists to defuse.
    """
    return RuntimeEvent(
        action=action,
        container_id=container_id,
        container_name=CONTAINER,
        ts=ts,
        attributes={"name": CONTAINER, **(attributes or {})},
    )


def done_line(*, ts: datetime, seconds: float = 32.521) -> Event:
    """What the parser makes of ``Done (32.521s)! For help, type "help"``."""
    return ServerReady(
        ts=ts,
        server_id=SERVER_ID,
        source=Source.LOG,
        raw=f'Done ({seconds}s)! For help, type "help"',
        startup_seconds=seconds,
        detected_by=ReadySignal.LOG_DONE,
    )


def stopping_line(*, ts: datetime) -> Event:
    """What the parser makes of ``[Rcon: Stopping the server]``."""
    return ServerStopping(
        ts=ts,
        server_id=SERVER_ID,
        source=Source.LOG,
        raw="[Rcon: Stopping the server]",
        reason="stopping the server",
    )


def version_line(*, ts: datetime, version: str = "26.2") -> Event:
    """What the parser makes of ``Starting minecraft server version 26.2``."""
    return ServerStarting(
        ts=ts,
        server_id=SERVER_ID,
        source=Source.LOG,
        raw=f"Starting minecraft server version {version}",
        version=version,
    )


def probe_ok(*, ts: datetime, online: int = 0, sample: tuple[str, ...] = ()) -> ProbeResult:
    return ProbeResult(
        reachable=True,
        players_online=online,
        players_max=5,
        sample=sample,
        version="26.2",
        latency_ms=4.2,
        probed_at=ts,
    )


def probe_failed(*, ts: datetime, error: str = "timed out") -> ProbeResult:
    """A probe that did not answer. Unknown: never "stopped", and never "zero players"."""
    return ProbeResult(reachable=False, error=error, probed_at=ts)
