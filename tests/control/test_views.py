"""The read-model: round trips, and the readiness policy.

Two properties matter here and nothing else really does:

1. **Every view survives ``to_dict`` -> ``from_dict`` unchanged.** These structures cross the HTTP
   boundary in both directions, so a field that encodes but does not decode is a field the CLI
   silently prints as ``unknown`` forever.
2. **The readiness policy says which subsystem is red.** ``/readyz`` returning a bare 503 is the
   failure mode this endpoint exists to avoid.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mcmanager.containers.dto import HealthState
from mcmanager.control.views import (
    ControlResultView,
    IdleView,
    LivenessView,
    PlayerView,
    ProbeView,
    ReadinessView,
    SessionView,
    StatusView,
    SubsystemView,
    evaluate_readiness,
)
from mcmanager.core.events import PlayerJoined
from mcmanager.core.serde import SerdeError
from mcmanager.core.types import LifecycleState, PlayerRef, Source

NOW = datetime(2026, 7, 25, 22, 58, 20, tzinfo=UTC)


def _status() -> StatusView:
    return StatusView(
        server_id="minecraft",
        container="minecraft",
        state=LifecycleState.READY,
        observed_at=NOW,
        running=True,
        health=HealthState.HEALTHY,
        health_reported_raw="healthy",
        container_id="5ff1a3c2deadbeef",
        image="itzg/minecraft-server:latest",
        started_at=NOW - timedelta(hours=3),
        uptime_seconds=10800.0,
        version="26.2",
        ready_at=NOW - timedelta(hours=3),
        ready_detected_by="log",
        startup_seconds=32.521,
        players_online=2,
        players_max=5,
        roster=(
            PlayerView(name="Steve", online_since=NOW, session_seconds=3600.0),
            PlayerView(name="bharath_720", first_seen=True, source="probe"),
        ),
        probe=ProbeView(
            reachable=True,
            players_online=2,
            players_max=5,
            sample=("Steve", "bharath_720"),
            sample_is_complete=True,
            latency_ms=18.4,
            motd="A Minecraft Server",
            probed_at=NOW,
        ),
        idle=IdleView(
            enabled=True,
            dry_run=True,
            armed=True,
            deadline=NOW + timedelta(minutes=12),
            seconds_remaining=750.0,
            timeout_seconds=900.0,
            empty_since=NOW - timedelta(minutes=3),
        ),
        session=SessionView(
            id="s-1",
            server_id="minecraft",
            started_at=NOW - timedelta(hours=3),
            open=True,
            players=("Steve",),
            peak_online=2,
            joins=5,
            deaths=2,
            chat_messages=41,
        ),
        last_event=PlayerJoined(
            ts=NOW,
            server_id="minecraft",
            source=Source.LOG,
            raw="Steve joined the game",
            player=PlayerRef(name="Steve"),
            online_count=2,
        ),
        stop_timeout_seconds=90,
        notes=("a note",),
    )


class TestRoundTrips:
    def test_a_full_status_view_survives_a_round_trip(self) -> None:
        view = _status()
        assert StatusView.from_dict(view.to_dict()) == view

    def test_a_minimal_status_view_survives_a_round_trip(self) -> None:
        view = StatusView(server_id="mc", container="minecraft", observed_at=NOW)
        assert StatusView.from_dict(view.to_dict()) == view

    def test_the_embedded_event_round_trips_through_serde(self) -> None:
        view = _status()
        restored = StatusView.from_dict(view.to_dict())
        assert restored.last_event == view.last_event

    def test_a_player_view_round_trips(self) -> None:
        view = PlayerView(name="Steve", uuid="abc", online_since=NOW, session_seconds=1.5)
        assert PlayerView.from_dict(view.to_dict()) == view
        bare = PlayerView(name="Steve")
        assert PlayerView.from_dict(bare.to_dict()) == bare

    def test_a_probe_view_round_trips(self) -> None:
        view = ProbeView(reachable=False, error="timed out")
        assert ProbeView.from_dict(view.to_dict()) == view

    def test_an_idle_view_round_trips(self) -> None:
        assert IdleView.from_dict(IdleView().to_dict()) == IdleView()

    def test_a_session_view_round_trips(self) -> None:
        view = SessionView(id="s-2", started_at=NOW, ended_at=NOW, partial=True, clean=False)
        assert SessionView.from_dict(view.to_dict()) == view
        bare = SessionView(id="s-1")
        assert SessionView.from_dict(bare.to_dict()) == bare

    def test_a_subsystem_view_round_trips(self) -> None:
        view = SubsystemView(name="runtime", ok=False, detail="socket gone")
        assert SubsystemView.from_dict(view.to_dict()) == view

    def test_a_liveness_view_round_trips(self) -> None:
        view = LivenessView(loop_lag_seconds=0.004, tasks=7)
        assert LivenessView.from_dict(view.to_dict()) == view

    def test_a_control_result_round_trips(self) -> None:
        view = ControlResultView(action="stop", accepted=True, ok=True, message="stop ok")
        assert ControlResultView.from_dict(view.to_dict()) == view

    def test_a_readiness_report_round_trips(self) -> None:
        view = evaluate_readiness(runtime_available=True, log_stream_attached=True)
        assert ReadinessView.from_dict(view.to_dict()) == view


class TestDecodingIsStrictAboutShape:
    def test_a_missing_key_falls_back_to_the_default(self) -> None:
        # Lenient on absence: the two ends ship together, so a missing key is an older daemon.
        assert PlayerView.from_dict({"name": "Steve"}).source == "log"

    def test_a_wrong_type_raises_rather_than_coercing(self) -> None:
        with pytest.raises(SerdeError, match="must be a string"):
            PlayerView.from_dict({"name": 17})

    def test_a_naive_timestamp_is_refused(self) -> None:
        with pytest.raises(SerdeError, match="naive"):
            PlayerView.from_dict({"name": "Steve", "online_since": "2026-07-25T22:58:20"})

    def test_a_required_timestamp_must_be_present(self) -> None:
        with pytest.raises(SerdeError, match="observed_at"):
            StatusView.from_dict({"server_id": "mc", "container": "minecraft"})

    def test_an_unknown_enum_value_is_refused(self) -> None:
        with pytest.raises(SerdeError, match="LifecycleState"):
            StatusView.from_dict(
                {
                    "server_id": "mc",
                    "container": "mc",
                    "observed_at": "2026-01-01T00:00:00Z",
                    "state": "vibing",
                }
            )


class TestSessionElapsed:
    def test_a_closed_session_uses_its_own_endpoints(self) -> None:
        session = SessionView(id="s", started_at=NOW - timedelta(hours=1), ended_at=NOW)
        assert session.elapsed_seconds(NOW) == pytest.approx(3600.0)

    def test_an_open_session_measures_against_the_supplied_now(self) -> None:
        session = SessionView(id="s", started_at=NOW - timedelta(minutes=5), open=True)
        assert session.elapsed_seconds(NOW) == pytest.approx(300.0)

    def test_a_recorded_duration_wins_over_recomputation(self) -> None:
        session = SessionView(id="s", started_at=NOW, duration_seconds=42.0)
        assert session.elapsed_seconds(NOW + timedelta(hours=9)) == 42.0

    def test_a_session_with_no_start_has_no_duration(self) -> None:
        assert SessionView(id="s").elapsed_seconds(NOW) is None


class TestReadinessPolicy:
    def test_everything_healthy_is_ready(self) -> None:
        view = evaluate_readiness(runtime_available=True, log_stream_attached=True)
        assert view.ready
        assert view.failing() == ()

    def test_a_dead_runtime_names_the_runtime(self) -> None:
        view = evaluate_readiness(
            runtime_available=False,
            runtime_detail="permission denied on /var/run/docker.sock",
            log_stream_attached=True,
        )
        assert not view.ready
        assert [check.name for check in view.failing()] == ["runtime"]
        assert "permission denied" in (view.failing()[0].detail or "")

    def test_a_stale_runtime_reading_is_not_ready_even_though_it_answered_once(self) -> None:
        view = evaluate_readiness(
            runtime_available=True,
            runtime_age_seconds=600.0,
            runtime_max_age_seconds=120.0,
            log_stream_attached=True,
        )
        assert not view.ready
        assert "600s ago" in (view.failing()[0].detail or "")

    def test_a_fresh_runtime_reading_passes(self) -> None:
        view = evaluate_readiness(
            runtime_available=True,
            runtime_age_seconds=30.0,
            log_stream_attached=True,
        )
        assert view.ready

    def test_discord_disabled_is_ready_not_degraded(self) -> None:
        # "connected OR deliberately off" - a daemon with Discord off is not half broken.
        view = evaluate_readiness(
            runtime_available=True,
            discord_enabled=False,
            discord_connected=False,
            log_stream_attached=True,
        )
        assert view.ready
        discord = next(check for check in view.checks if check.name == "discord")
        assert "disabled" in (discord.detail or "")

    def test_discord_enabled_but_disconnected_is_not_ready(self) -> None:
        view = evaluate_readiness(
            runtime_available=True,
            discord_enabled=True,
            discord_connected=False,
            log_stream_attached=True,
        )
        assert not view.ready
        assert [check.name for check in view.failing()] == ["discord"]

    def test_a_detached_log_stream_is_fine_when_the_container_is_not_running(self) -> None:
        # The load-bearing clause: a stopped server has no log stream to attach to, and a daemon
        # that reads red for the whole week the server is off trains everyone to ignore it.
        view = evaluate_readiness(
            runtime_available=True,
            log_stream_attached=False,
            log_stream_expected=False,
        )
        assert view.ready
        stream = next(check for check in view.checks if check.name == "log_stream")
        assert "idle" in (stream.detail or "")

    def test_a_detached_log_stream_is_not_ready_while_the_container_is_up(self) -> None:
        view = evaluate_readiness(
            runtime_available=True,
            log_stream_attached=False,
            log_stream_expected=True,
        )
        assert not view.ready
        assert [check.name for check in view.failing()] == ["log_stream"]

    def test_a_dropped_critical_event_makes_the_daemon_unready(self) -> None:
        # bus.dropped_critical being non-zero means the queue sizing is wrong. Surfacing it here
        # rather than only in a log line is the entire point of counting it.
        view = evaluate_readiness(
            runtime_available=True,
            log_stream_attached=True,
            dropped_critical=3,
        )
        assert not view.ready
        assert "3 non-console event(s) dropped" in (view.failing()[0].detail or "")

    def test_every_subsystem_is_reported_even_when_ready(self) -> None:
        view = evaluate_readiness(runtime_available=True, log_stream_attached=True)
        assert {check.name for check in view.checks} == {"runtime", "discord", "log_stream", "bus"}
