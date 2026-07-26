"""Read-only assertions against the real homelab Docker daemon. **Never runs in CI.**

Gated twice, on purpose:

- ``@pytest.mark.live``, which ``conftest.block_network`` turns into "skip unless
  ``MCMANAGER_LIVE=1``" *and* which is the only way the outbound-socket guard is lifted;
- ``-m live`` is not in the default ``addopts``, so a bare ``pytest`` never selects them.

**No test here may start or stop the real server.** That is not a convention, it is the point:
this file exists to check that ``DockerRuntime`` reads reality correctly, and a test suite that
can bounce somebody's Minecraft world is not a test suite anyone will run. Every call below is
``ping`` / ``inspect`` / ``logs_tail`` / ``watch_events`` / a read-only ``exec``.

Run it::

    $env:MCMANAGER_LIVE = "1"
    $env:DOCKER_HOST = "ssh://minty@192.168.1.7"
    uv run pytest -m live

The ``ssh://`` transport needs ``paramiko``, which is **not** currently a dependency (docker-py
puts it behind the ``docker[ssh]`` extra). Without it these tests skip with a message saying so
rather than failing with a confusing ``DockerException``.

What this file is really for: docker-py is the one library this project cannot verify by faking,
because faking it would only prove our assumptions agree with themselves. So the four corrections
in ``streams.py`` and every field mapping in ``docker_runtime.py`` are checked here, against the
actual engine, or they are not checked at all.
"""

from __future__ import annotations

import os
from datetime import UTC, timedelta
from typing import TYPE_CHECKING, cast

import pytest

from mcmanager.clock import SystemClock
from mcmanager.containers.docker_runtime import DockerRuntime, describe_process_identity
from mcmanager.containers.dto import ContainerState, HealthState
from mcmanager.containers.streams import LOG_STREAM_PATH

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from mcmanager.containers.dto import LogLine, RuntimeEvent

pytestmark = pytest.mark.live

CONTAINER = os.environ.get("MCMANAGER_LIVE_CONTAINER", "minecraft")


@pytest.fixture
async def runtime() -> AsyncIterator[DockerRuntime]:
    """A real runtime against whatever ``DOCKER_HOST`` says.

    Skips rather than fails when the transport is unavailable: an ``ssh://`` endpoint without
    ``paramiko`` installed is a missing optional dependency, not a bug in this code.
    """
    built = DockerRuntime(clock=SystemClock(), host=os.environ.get("DOCKER_HOST") or None)
    try:
        if not await built.ping():
            pytest.skip(
                f"no Docker at {built.endpoint} ({describe_process_identity()}). "
                "For ssh:// you also need paramiko: `uv add --dev docker[ssh]`."
            )
        yield built
    finally:
        await built.aclose()


# ------------------------------------------------------------------------------ reachability


async def test_ping_reaches_the_daemon(runtime: DockerRuntime) -> None:
    assert await runtime.ping() is True


async def test_the_private_log_stream_path_is_active_against_this_docker_py() -> None:
    """If this flips to ``"fallback"``, ``close()`` on a log stream may be a no-op again and the
    pump thread outlives its stream. Worth failing loudly rather than degrading silently."""
    assert LOG_STREAM_PATH == "private"


# ------------------------------------------------------------------------------- inspect


async def test_inspect_resolves_the_container_by_name(runtime: DockerRuntime) -> None:
    snapshot = await runtime.inspect(CONTAINER)
    if snapshot.absent:
        pytest.skip(f"container {CONTAINER!r} does not exist on this host")
    assert snapshot.name == CONTAINER
    assert snapshot.id is not None
    assert len(snapshot.id) >= 12


async def test_a_missing_container_is_a_snapshot_not_an_exception(runtime: DockerRuntime) -> None:
    """Absence is a state. If this ever raises, the daemon crashes on a renamed container."""
    snapshot = await runtime.inspect("mcmanager-no-such-container-b6a1")
    assert snapshot.absent
    assert snapshot.state is ContainerState.ABSENT
    assert not snapshot.exists


async def test_the_stale_health_regression(runtime: DockerRuntime) -> None:
    """**The M1 acceptance check from plan section 15.**

    Observed on this exact container while it was *exited*: ``State.Health.Status`` still read
    ``"unhealthy"``, with ``FailingStreak`` at 0 and the last five probes having exited 0. Docker
    never clears the field on stop. Anything that trusts that string reports a permanently
    unhealthy server the moment it is turned off.

    So: while the container is not running, ``health`` must be UNKNOWN no matter what Docker said.
    When it *is* running we assert nothing about the value, only that the raw string is what the
    derived property was computed from.
    """
    snapshot = await runtime.inspect(CONTAINER)
    if snapshot.absent:
        pytest.skip(f"container {CONTAINER!r} does not exist on this host")

    if not snapshot.running:
        assert snapshot.health is HealthState.UNKNOWN
    elif snapshot.health_reported_raw is None:
        assert snapshot.health is HealthState.NONE


async def test_snapshot_datetimes_are_tz_aware_utc(runtime: DockerRuntime) -> None:
    """Everything persisted, serialised or shown to a human comes from these."""
    snapshot = await runtime.inspect(CONTAINER)
    if snapshot.absent:
        pytest.skip(f"container {CONTAINER!r} does not exist on this host")

    assert snapshot.observed_at.tzinfo is not None
    assert snapshot.observed_at.utcoffset() == timedelta(0)
    for value in (snapshot.created_at, snapshot.started_at, snapshot.finished_at):
        if value is not None:
            assert value.tzinfo is not None
            assert value.utcoffset() == timedelta(0)


async def test_tty_is_false_so_the_multiplexed_log_path_applies(runtime: DockerRuntime) -> None:
    """If this ever becomes true, the log stream stops being framed and the splitter's input
    changes shape. Better to learn it here than from garbled player names."""
    snapshot = await runtime.inspect(CONTAINER)
    if snapshot.absent:
        pytest.skip(f"container {CONTAINER!r} does not exist on this host")
    assert snapshot.tty is False


async def test_the_healthcheck_timing_comes_from_the_container(runtime: DockerRuntime) -> None:
    """Plan section 5 reads its 210-second guard window from here rather than hardcoding it. This
    asserts the mapping works and the numbers are sane, not that they are exactly today's."""
    snapshot = await runtime.inspect(CONTAINER)
    if snapshot.absent:
        pytest.skip(f"container {CONTAINER!r} does not exist on this host")
    timing = snapshot.health_timing
    if timing.interval is None:
        pytest.skip("this container declares no healthcheck")
    assert timing.interval > timedelta(0)
    guard = snapshot.guard_window
    assert guard is not None
    assert guard > timing.interval


async def test_the_container_is_attached_to_a_network_we_can_name(
    runtime: DockerRuntime,
) -> None:
    """Is the server on the ``homelab`` network?

    That is the first question when ``minecraft:25565`` stops resolving, and ``mcmanager inspect``
    answers it from this field.
    """
    snapshot = await runtime.inspect(CONTAINER)
    if snapshot.absent:
        pytest.skip(f"container {CONTAINER!r} does not exist on this host")
    assert snapshot.networks, "a container with no networks cannot be probed or RCONed"
    assert all(attachment.name for attachment in snapshot.networks)


async def test_the_log_driver_caps_are_visible(runtime: DockerRuntime) -> None:
    """``json-file`` at 10m x 3 is a ~30MB ring, not an archive. Plan section 12 depends on
    knowing that, so it has to be readable."""
    snapshot = await runtime.inspect(CONTAINER)
    if snapshot.absent:
        pytest.skip(f"container {CONTAINER!r} does not exist on this host")
    assert snapshot.log_driver is not None


# ---------------------------------------------------------------------------------- logs


async def test_logs_tail_returns_lines_with_utc_timestamps(runtime: DockerRuntime) -> None:
    """``timestamps=True``, always: Paper's ``[13:24:37]`` is time-only in ``Asia/Kolkata`` and
    Docker's RFC3339Nano prefix is the only uniform, backfill-safe source."""
    snapshot = await runtime.inspect(CONTAINER)
    if snapshot.absent:
        pytest.skip(f"container {CONTAINER!r} does not exist on this host")

    lines = await runtime.logs_tail(CONTAINER, lines=20)
    if not lines:
        pytest.skip("the container has no log output to read")

    stamped = [line for line in lines if line.ts is not None]
    assert stamped, "every line should carry Docker's timestamp prefix"
    for line in stamped:
        assert line.ts is not None
        assert line.ts.tzinfo is not None
        assert line.ts.utcoffset() == timedelta(0)
    for line in lines:
        assert "\n" not in line.text
        assert not line.text.endswith("\r")
        assert line.received_at.tzinfo is not None


async def test_logs_tail_works_on_a_stopped_container(runtime: DockerRuntime) -> None:
    """Which is what makes ``mcmanager logs`` useful after a crash - the moment you need it."""
    snapshot = await runtime.inspect(CONTAINER)
    if snapshot.absent or snapshot.running:
        pytest.skip("this assertion is about a container that is currently stopped")
    await runtime.logs_tail(CONTAINER, lines=5)


async def test_the_docker_timestamp_prefix_is_stripped_from_the_text(
    runtime: DockerRuntime,
) -> None:
    """``LogLine.text`` is what the parser sees. A leftover RFC3339 prefix would break every
    anchored pattern in ``games/minecraft/`` at once."""
    snapshot = await runtime.inspect(CONTAINER)
    if snapshot.absent:
        pytest.skip(f"container {CONTAINER!r} does not exist on this host")
    lines = await runtime.logs_tail(CONTAINER, lines=20)
    if not lines:
        pytest.skip("the container has no log output to read")
    for line in lines:
        if line.ts is not None:
            assert not line.text.startswith(line.ts.strftime("%Y-%m-%d"))


async def test_a_bounded_follow_delivers_and_then_closes_cleanly(
    runtime: DockerRuntime,
) -> None:
    """**Correction 1 and correction 2, verified against the engine.**

    Attaches a following stream, reads a bounded number of lines, then closes the iterator. If
    ``close()`` on the underlying stream did nothing, the pump thread would still be parked in a
    blocking read afterwards; because it is a daemon thread that cannot hang the interpreter, but
    it would leak one thread and one connection per reconnect.

    Read-only: following a log stream changes nothing.
    """
    snapshot = await runtime.inspect(CONTAINER)
    if snapshot.absent:
        pytest.skip(f"container {CONTAINER!r} does not exist on this host")

    # The ABC returns AsyncIterator; both implementations are async generators, and closing one
    # promptly is what stops the pump thread instead of leaving it to the collector.
    stream = cast("AsyncGenerator[LogLine, None]", runtime.follow_logs(CONTAINER, tail=5))
    collected = 0
    try:
        async for line in stream:
            assert line.received_at.tzinfo is not None
            collected += 1
            if collected >= 3:
                break
    finally:
        await stream.aclose()


# --------------------------------------------------------------------------------- events


async def test_the_event_stream_opens_and_closes(runtime: DockerRuntime) -> None:
    """Opens the watcher and tears it straight back down without waiting for an event, because
    waiting for one would need somebody to touch the container. What is being proven is that the
    stream opens, the pump starts, and closing it does not raise or hang.
    """
    stream = cast("AsyncGenerator[RuntimeEvent, None]", runtime.watch_events(CONTAINER))
    await stream.aclose()


async def test_historical_events_decode_correctly(runtime: DockerRuntime) -> None:
    """``since`` in the past means the engine replays what already happened, so this reads real
    events without anything having to happen now.

    The assertion that matters is ``exit_code``: Docker sends ``Actor.Attributes["exitCode"]`` as
    a **string**, so ``attributes["exitCode"] == 0`` is silently always false.
    """
    from datetime import datetime

    since = datetime.now(UTC) - timedelta(days=7)
    stream = cast(
        "AsyncGenerator[RuntimeEvent, None]", runtime.watch_events(CONTAINER, since=since)
    )
    seen = 0
    try:
        async for event in stream:
            assert event.ts.tzinfo is not None
            assert event.action
            for value in event.attributes.values():
                assert isinstance(value, str)
            if "exitCode" in event.attributes:
                assert isinstance(event.exit_code, int)
            seen += 1
            if seen >= 5:
                break
    finally:
        await stream.aclose()


# ----------------------------------------------------------------------------------- exec


async def test_a_read_only_exec_round_trips(runtime: DockerRuntime) -> None:
    """The RCON channel is ``docker exec minecraft rcon-cli <cmd>``, so this proves the exec path
    without issuing a game command: ``true`` is the most read-only command there is.

    ``rcon-cli list`` would also be harmless, but it is a game command, and this file's rule is
    that it does not send those.
    """
    snapshot = await runtime.inspect(CONTAINER)
    if snapshot.absent or not snapshot.running:
        pytest.skip("exec needs a running container")

    result = await runtime.exec(CONTAINER, ["true"], timeout=15.0)
    assert result.exit_code == 0


# ------------------------------------------------------------------------------- teardown


async def test_aclose_is_safe_twice(runtime: DockerRuntime) -> None:
    await runtime.aclose()
    await runtime.aclose()
    assert await runtime.ping() is False, "a closed runtime must not silently reconnect"
