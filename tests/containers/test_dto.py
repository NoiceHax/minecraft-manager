"""Container DTOs, and the stale-health regression in particular."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mcmanager.containers.dto import (
    ContainerSnapshot,
    ContainerState,
    ExecResult,
    HealthState,
    HealthTiming,
    LogLine,
    MountInfo,
    NetworkAttachment,
    RuntimeEvent,
)
from mcmanager.core.types import Stream

OBSERVED_AT = datetime(2026, 7, 25, 18, 30, tzinfo=UTC)


def test_health_is_unknown_on_a_stopped_container_even_when_docker_says_unhealthy() -> None:
    """The observed gotcha, encoded once so no caller can reproduce it.

    On the real homelab container, while ``minecraft`` was *exited*, ``State.Health.Status`` still
    read ``"unhealthy"`` - with ``FailingStreak`` at 0 and the last five probes having exited 0.
    Docker never clears the field on stop. Any readiness or alerting path that trusts that string
    reports a permanently unhealthy server the moment it is turned off.
    """
    snapshot = ContainerSnapshot(
        name="minecraft",
        id="abc123",
        exists=True,
        state=ContainerState.EXITED,
        running=False,
        _health_raw="unhealthy",
        health_failing_streak=0,
        exit_code=0,
        observed_at=OBSERVED_AT,
    )

    assert snapshot.health is HealthState.UNKNOWN
    # ...and the raw value is still available for `mcmanager inspect`, just never for a decision.
    assert snapshot.health_reported_raw == "unhealthy"


def test_health_is_unknown_on_a_stopped_container_whatever_docker_said() -> None:
    for raw in ("healthy", "unhealthy", "starting", "none", None):
        snapshot = ContainerSnapshot(
            name="minecraft",
            state=ContainerState.EXITED,
            running=False,
            _health_raw=raw,
            observed_at=OBSERVED_AT,
        )
        assert snapshot.health is HealthState.UNKNOWN


def test_health_is_reported_when_the_container_is_running() -> None:
    snapshot = ContainerSnapshot(
        name="minecraft",
        state=ContainerState.RUNNING,
        running=True,
        _health_raw="unhealthy",
        observed_at=OBSERVED_AT,
    )

    assert snapshot.health is HealthState.UNHEALTHY


def test_a_running_container_with_no_healthcheck_reports_none_not_unknown() -> None:
    snapshot = ContainerSnapshot(
        name="minecraft",
        state=ContainerState.RUNNING,
        running=True,
        _health_raw=None,
        observed_at=OBSERVED_AT,
    )

    assert snapshot.health is HealthState.NONE


def test_an_unrecognised_health_string_is_unknown_not_a_crash() -> None:
    snapshot = ContainerSnapshot(
        name="minecraft",
        state=ContainerState.RUNNING,
        running=True,
        _health_raw="probably fine",
        observed_at=OBSERVED_AT,
    )

    assert snapshot.health is HealthState.UNKNOWN


def test_guard_window_is_read_from_the_container_not_hardcoded() -> None:
    """120s start_period + 30s interval * (2 retries + 1) = 210s, on this deployment."""
    timing = HealthTiming(
        interval=timedelta(seconds=30),
        timeout=timedelta(seconds=10),
        start_period=timedelta(seconds=120),
        retries=2,
    )

    assert timing.guard_window == timedelta(seconds=210)


def test_guard_window_is_none_when_the_container_declares_no_healthcheck() -> None:
    assert HealthTiming().guard_window is None
    assert HealthTiming(interval=timedelta(seconds=30)).guard_window is None


def test_missing_snapshot_is_absent_and_never_running() -> None:
    snapshot = ContainerSnapshot.missing("minecraft", observed_at=OBSERVED_AT)

    assert snapshot.absent is True
    assert snapshot.running is False
    assert snapshot.state is ContainerState.ABSENT
    assert snapshot.health is HealthState.UNKNOWN
    assert snapshot.uptime(OBSERVED_AT) is None


def test_uptime_needs_a_running_container_and_a_start_time() -> None:
    started = datetime(2026, 7, 25, 18, 0, tzinfo=UTC)
    running = ContainerSnapshot(
        name="minecraft",
        state=ContainerState.RUNNING,
        running=True,
        started_at=started,
        observed_at=OBSERVED_AT,
    )
    stopped = ContainerSnapshot(
        name="minecraft",
        state=ContainerState.EXITED,
        running=False,
        started_at=started,
        observed_at=OBSERVED_AT,
    )

    assert running.uptime(OBSERVED_AT) == timedelta(minutes=30)
    assert stopped.uptime(OBSERVED_AT) is None


def test_network_aliases_are_looked_up_by_network_name() -> None:
    snapshot = ContainerSnapshot(
        name="minecraft",
        running=True,
        state=ContainerState.RUNNING,
        networks=(
            NetworkAttachment(name="minecraft_default", aliases=("minecraft",)),
            NetworkAttachment(name="homelab", ip_address="172.20.0.5", aliases=("minecraft",)),
        ),
        mounts=(MountInfo(source="/home/minty/homelab/data/minecraft", destination="/data"),),
        observed_at=OBSERVED_AT,
    )

    assert snapshot.alias_for("homelab") == ("minecraft",)
    assert snapshot.alias_for("nonexistent") == ()
    assert snapshot.mounts[0].destination == "/data"


def test_log_line_falls_back_to_received_at_when_docker_gave_no_timestamp() -> None:
    stamped = LogLine(
        text="[13:24:37] [Server thread/INFO]: Done (32.521s)!",
        ts=datetime(2026, 7, 25, 13, 24, 37, tzinfo=UTC),
        received_at=OBSERVED_AT,
    )
    unstamped = LogLine(
        text="partial chunk",
        ts=None,
        received_at=OBSERVED_AT,
        stream=Stream.STDERR,
    )

    assert stamped.event_ts == datetime(2026, 7, 25, 13, 24, 37, tzinfo=UTC)
    assert unstamped.event_ts == OBSERVED_AT
    assert unstamped.stream is Stream.STDERR


def test_runtime_event_parses_the_string_exit_code() -> None:
    """``Actor.Attributes["exitCode"]`` is a string. ``attrs["exitCode"] == 0`` is always false."""
    died = RuntimeEvent(
        action="die",
        container_name="minecraft",
        ts=OBSERVED_AT,
        attributes={"exitCode": "137", "name": "minecraft"},
    )

    assert died.exit_code == 137
    assert died.is_lifecycle is True
    assert died.health_status is None


def test_runtime_event_tolerates_a_missing_or_junk_exit_code() -> None:
    assert RuntimeEvent(action="die", ts=OBSERVED_AT).exit_code is None
    assert RuntimeEvent(action="die", ts=OBSERVED_AT, attributes={"exitCode": ""}).exit_code is None


def test_runtime_event_decodes_health_status_actions() -> None:
    healthy = RuntimeEvent(action="health_status: healthy", ts=OBSERVED_AT)
    unhealthy = RuntimeEvent(action="health_status: unhealthy", ts=OBSERVED_AT)
    weird = RuntimeEvent(action="health_status: sideways", ts=OBSERVED_AT)

    assert healthy.health_status is HealthState.HEALTHY
    assert unhealthy.health_status is HealthState.UNHEALTHY
    assert weird.health_status is HealthState.UNKNOWN
    assert healthy.is_lifecycle is False


def test_exec_result_ok_and_output_fallback() -> None:
    ok = ExecResult(exit_code=0, stdout="There are 0 of a max of 5 players online:")
    failed = ExecResult(exit_code=1, stderr="rcon-cli: connection refused")

    assert ok.ok is True
    assert ok.output.startswith("There are 0")
    assert failed.ok is False
    assert failed.output == "rcon-cli: connection refused"
