"""The read model, against a real (unstarted) :class:`~mcmanager.app.Application`.

``build_status_view`` and friends are the "status aggregator": the one place that knows how a
lifecycle reducer, a roster, a poller and an idle manager add up to one view. Testing them against
the actual composition root rather than a hand-made stub is the point - a field that quietly stops
existing on ``AppServices`` should fail here, not in a Discord embed three weeks later.

The application is **built but never started**, so no socket is bound, no task is spawned and no
signal handler is installed. Construction is pure by design; everything that can fail lives in
``start()``.
"""

from __future__ import annotations

import asyncio
import socket
from typing import TYPE_CHECKING

import pytest

from mcmanager.app import Application
from mcmanager.config import load_settings
from mcmanager.containers.dto import HealthState
from mcmanager.containers.fake import FakeRuntime
from mcmanager.control.server import (
    ControlServer,
    build_control_surface,
    build_liveness_view,
    build_player_views,
    build_readiness_view,
    build_session_views,
    build_status_view,
    disabled_surface,
)
from mcmanager.control.sse import SseChannel
from mcmanager.core.events import PlayerJoined
from mcmanager.core.types import LifecycleState, PlayerRef, Source

if TYPE_CHECKING:
    from pathlib import Path

    from mcmanager.app import AppServices
    from mcmanager.clock import ManualClock
    from mcmanager.config import Settings


def _free_port() -> int:
    """A loopback port nothing is using. ``web.port`` must be >= 1, so 0 is not an option."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        chosen: int = probe.getsockname()[1]
    return chosen


def _settings(tmp_path: Path, *, web: bool) -> Settings:
    for name in ("state", "archives", "daemon-logs", "mc-logs"):
        (tmp_path / name).mkdir(exist_ok=True)
    return load_settings(
        overrides={
            "runtime": "fake",
            "state": {"dir": str(tmp_path / "state")},
            "probe": {"enabled": False},
            "discord": {"enabled": False, "mode": "disabled"},
            "web": {"enabled": web, "port": _free_port()},
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
def services(tmp_path: Path, clock: ManualClock) -> AppServices:
    """A fully built, never started daemon."""
    settings = _settings(tmp_path, web=True)
    return Application(settings, clock=clock, runtime=FakeRuntime(clock=clock)).services


class TestFactory:
    def test_web_disabled_returns_a_no_op_surface(self, tmp_path: Path, clock: ManualClock) -> None:
        settings = _settings(tmp_path, web=False)
        services = Application(settings, clock=clock, runtime=FakeRuntime(clock=clock)).services
        surface = build_control_surface(services)
        assert not isinstance(surface, ControlServer)
        assert type(surface) is type(disabled_surface())

    def test_web_enabled_returns_a_control_server_bound_to_the_configured_address(
        self, services: AppServices
    ) -> None:
        surface = build_control_surface(services)
        assert isinstance(surface, ControlServer)
        assert surface.url.startswith("http://127.0.0.1:")

    def test_the_factory_attaches_the_sse_channel_to_the_bus(self, services: AppServices) -> None:
        """Otherwise ``/events`` is a stream that never produces a frame."""
        before = {subscription.name for subscription in services.bus.subscriptions}
        build_control_surface(services)
        after = {subscription.name for subscription in services.bus.subscriptions}
        assert after - before == {"sse"}

    def test_a_control_server_gets_the_configured_token(self, services: AppServices) -> None:
        surface = build_control_surface(services)
        assert isinstance(surface, ControlServer)
        # No token in this config, so the mutating endpoints refuse everything - which is what
        # `check-config` warns about at startup.
        assert surface.app is not None


class TestStatusView:
    def test_a_fresh_daemon_reports_unknown_rather_than_inventing_a_state(
        self, services: AppServices
    ) -> None:
        view = build_status_view(services)
        assert view.state is LifecycleState.UNKNOWN
        assert view.daemon_online is True
        assert view.health is HealthState.UNKNOWN
        assert view.players_online == 0

    def test_the_container_and_server_ids_come_from_config(self, services: AppServices) -> None:
        view = build_status_view(services)
        assert view.server_id == services.settings.server.id
        assert view.container == services.settings.server.container

    def test_the_roster_reaches_the_view(self, services: AppServices) -> None:
        services.roster.join("Steve")
        view = build_status_view(services)
        assert [player.name for player in view.roster] == ["Steve"]
        assert view.players_online == 1

    def test_a_probe_that_never_ran_is_none_rather_than_zero_players(
        self, services: AppServices
    ) -> None:
        # The poller is disabled in this config, so there is no reading - and "no reading" must
        # not render as "nobody is online".
        assert build_status_view(services).probe is None

    def test_the_idle_view_reflects_the_manager(self, services: AppServices) -> None:
        view = build_status_view(services)
        assert view.idle is not None
        assert view.idle.enabled is True
        assert view.idle.dry_run is True
        assert view.idle.armed is False
        assert view.idle.timeout_seconds == services.settings.idle.timeout_seconds

    def test_no_session_in_flight_means_no_session_view(self, services: AppServices) -> None:
        assert build_status_view(services).session is None
        assert build_session_views(services, limit=10) == []

    def test_the_stop_timeout_is_surfaced_from_the_controller(self, services: AppServices) -> None:
        view = build_status_view(services)
        assert view.stop_timeout_seconds == services.settings.server.lifecycle.stop_timeout_seconds

    def test_without_a_channel_there_is_no_last_event(self, services: AppServices) -> None:
        assert build_status_view(services).last_event is None

    async def test_the_last_event_comes_from_the_channels_replay_ring(
        self, services: AppServices, clock: ManualClock
    ) -> None:
        """The SSE ring is the single source of "most recent event".

        The alternative - a variable somewhere that every publisher has to remember to update - is
        one more thing to keep in step, and it would be wrong in exactly the cases nobody tests.
        """
        channel = SseChannel(clock=clock)
        channel.attach(services.bus)
        pump = asyncio.create_task(services.bus.run())
        await asyncio.sleep(0)

        services.bus.publish(
            PlayerJoined(
                ts=clock.now(),
                server_id="minecraft",
                source=Source.LOG,
                raw="Steve joined the game",
                player=PlayerRef(name="Steve"),
            )
        )
        for _ in range(6):
            await asyncio.sleep(0)

        last = build_status_view(services, channel=channel).last_event
        assert last is not None
        assert last.name == "PlayerJoined"

        await channel.aclose()
        await services.bus.aclose()
        await pump

    def test_the_view_round_trips_through_its_own_codec(self, services: AppServices) -> None:
        from mcmanager.control.views import StatusView

        services.roster.join("Steve")
        view = build_status_view(services)
        assert StatusView.from_dict(view.to_dict()) == view


class TestPlayerViews:
    async def test_durations_are_measured_monotonically(
        self, services: AppServices, clock: ManualClock
    ) -> None:
        """Wall clocks jump. A negative session length in a Discord message would be a bad look."""
        services.roster.join("Steve")
        await clock.advance(90.0)
        players = build_player_views(services)
        assert players[0].session_seconds == pytest.approx(90.0)
        assert players[0].session_seconds is not None
        assert players[0].session_seconds >= 0.0

    def test_a_probe_sourced_player_says_so(self, services: AppServices) -> None:
        services.roster.join("Alex", source="probe")
        views = {player.name: player.source for player in build_player_views(services)}
        assert views["Alex"] == "probe"


class TestReadinessAndLiveness:
    def test_a_fresh_daemon_is_ready_because_nothing_is_expected_yet(
        self, services: AppServices
    ) -> None:
        """The container is not running, so a detached log stream is legitimate idleness."""
        view = build_readiness_view(services)
        stream = next(check for check in view.checks if check.name == "log_stream")
        assert stream.ok
        assert view.ready

    def test_discord_disabled_reads_as_satisfied(self, services: AppServices) -> None:
        discord = next(
            check for check in build_readiness_view(services).checks if check.name == "discord"
        )
        assert discord.ok
        assert "disabled" in (discord.detail or "")

    def test_liveness_counts_supervised_tasks_and_ignores_docker(
        self, services: AppServices
    ) -> None:
        view = build_liveness_view(services)
        assert view.alive
        assert view.supervisor_ok
        assert view.tasks == len(services.supervisor.task_names)

    def test_liveness_goes_red_when_the_supervisor_gives_up(self, services: AppServices) -> None:
        # The one condition where restarting the process is the correct response.
        services.supervisor.request_shutdown("bus dispatch died")
        view = build_liveness_view(services)
        assert not view.supervisor_ok
        assert view.detail == "shutdown requested"
