"""The command modules, against the fake runtime and a real control server.

The split the plan draws is what these tests assert:

- **standalone** (``inspect``, ``replay``, ``sessions``, ``logs`` without ``--follow``) build a
  ``ContainerRuntime`` from the factory and nothing else;
- **daemon-preferred** (``status``, ``players``) fall back with a visible banner;
- **daemon-required** (``events``, ``logs --follow``, the control commands) exit 69 rather than
  degrading.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp.test_utils import TestServer

from mcmanager.cli import (
    cmd_control,
    cmd_events,
    cmd_inspect,
    cmd_logs,
    cmd_sessions,
    cmd_status,
)
from mcmanager.containers.dto import ContainerState, HealthState, LogLine
from mcmanager.containers.fake import FakeRuntime
from mcmanager.control.routes import ControlContext
from mcmanager.control.server import build_app
from mcmanager.control.sse import SseChannel
from mcmanager.control.views import (
    LivenessView,
    PlayerView,
    SessionView,
    StatusView,
    evaluate_readiness,
)
from mcmanager.core.bus import EventBus
from mcmanager.core.events import ConsoleLog, PlayerJoined
from mcmanager.core.types import ControlAction, LifecycleState, PlayerRef, Source
from mcmanager.errors import EXIT_OK, EXIT_UNAVAILABLE
from mcmanager.games.base import ProbeResult
from mcmanager.services.controller import ControlOutcome

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from mcmanager.cli.main import CliContext
    from mcmanager.clock import ManualClock

NOW = datetime(2026, 7, 25, 22, 58, 20, tzinfo=UTC)
TOKEN = "s3cret"  # noqa: S105 - a literal for a test server, not a credential


@pytest.fixture
def fake_runtime(clock: ManualClock) -> FakeRuntime:
    return FakeRuntime(clock=clock, name="minecraft")


@pytest.fixture
def use_fake(
    monkeypatch: pytest.MonkeyPatch,
    fake_runtime: FakeRuntime,
) -> FakeRuntime:
    """Hand every standalone command the *same* fake, so a test can script it first.

    The commands build their runtime through ``containers.factory.build_runtime``; patching the
    name each module imported is the smallest intervention that leaves the production path intact.
    """

    def factory(*args: object, **kwargs: object) -> FakeRuntime:
        del args, kwargs
        return fake_runtime

    for module in (cmd_inspect, cmd_logs, cmd_status):
        monkeypatch.setattr(module, "build_runtime", factory)
    return fake_runtime


# ------------------------------------------------------------------------------------ inspect


class TestInspect:
    async def test_it_prints_the_derived_health_of_a_stopped_container(
        self, ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """§15's M3.5 pass condition, as a unit test.

        The fake keeps a stale ``health_raw`` across a stop exactly as Docker does, so this is the
        real regression and not a staged one.
        """
        use_fake.set_health("unhealthy")
        use_fake.set_state(ContainerState.EXITED, running=False, exit_code=0)
        assert await cmd_inspect.run(ctx) == EXIT_OK
        out = capsys.readouterr().out
        assert "health          unknown" in out
        assert "docker still reports 'unhealthy'" in out

    async def test_json_output_carries_both_health_values(
        self, json_ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        use_fake.set_health("unhealthy")
        use_fake.set_state(ContainerState.EXITED, running=False, exit_code=0)
        assert await cmd_inspect.run(json_ctx) == EXIT_OK
        payload: dict[str, Any] = json.loads(capsys.readouterr().out)
        assert payload["health"] == HealthState.UNKNOWN.value
        assert payload["health_reported_raw"] == "unhealthy"
        assert payload["healthcheck"]["guard_window_seconds"] == 210.0

    async def test_an_absent_container_is_not_an_error(
        self, ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A container that does not exist is a legitimate state: mcmanager owns a container's
        # lifecycle, not its existence.
        use_fake.set_absent()
        assert await cmd_inspect.run(ctx) == EXIT_OK
        assert "no container with this name exists" in capsys.readouterr().out

    async def test_an_unreachable_runtime_exits_69_naming_the_process_identity(
        self, ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        use_fake.set_reachable(reachable=False)
        assert await cmd_inspect.run(ctx) == EXIT_UNAVAILABLE
        err = capsys.readouterr().err
        assert "Docker is unreachable" in err
        assert "group_add" in err

    async def test_an_explicit_container_name_overrides_the_config(
        self, ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert await cmd_inspect.run(ctx, container="something-else") == EXIT_OK
        assert "something-else" in capsys.readouterr().out


# ------------------------------------------------------------------------------------- status


class TestStandaloneStatus:
    def test_a_running_container_answering_slp_is_ready(self) -> None:
        snapshot = _snapshot(running=True, state=ContainerState.RUNNING, health_raw="healthy")
        probe = ProbeResult(reachable=True, players_online=0, probed_at=NOW)
        view = cmd_status.build_standalone_view(snapshot, probe, server_id="mc", now=NOW)
        assert view.state is LifecycleState.READY
        assert view.daemon_online is False

    def test_a_running_container_not_yet_answering_is_starting_not_ready(self) -> None:
        # Claiming a server is up while it is still loading chunks is the worse error.
        snapshot = _snapshot(running=True, state=ContainerState.RUNNING, health_raw="starting")
        probe = ProbeResult(reachable=False, error="timed out", probed_at=NOW)
        view = cmd_status.build_standalone_view(snapshot, probe, server_id="mc", now=NOW)
        assert view.state is LifecycleState.STARTING

    def test_a_failed_probe_leaves_the_player_count_unknown_not_zero(self) -> None:
        """The failure mode that loses somebody's build session, asserted at the view layer."""
        snapshot = _snapshot(running=True, state=ContainerState.RUNNING)
        probe = ProbeResult(reachable=False, error="timed out", probed_at=NOW)
        view = cmd_status.build_standalone_view(snapshot, probe, server_id="mc", now=NOW)
        assert view.players_online is None
        assert any("UNKNOWN, not zero" in note for note in view.notes)

    def test_a_clean_exit_reads_as_stopped(self) -> None:
        view = cmd_status.build_standalone_view(
            _snapshot(exit_code=0), None, server_id="mc", now=NOW
        )
        assert view.state is LifecycleState.STOPPED

    @pytest.mark.parametrize("exit_code", [1, 137, None])
    def test_an_unexpected_exit_reads_as_crashed(self, exit_code: int | None) -> None:
        view = cmd_status.build_standalone_view(
            _snapshot(exit_code=exit_code), None, server_id="mc", now=NOW
        )
        assert view.state is LifecycleState.CRASHED

    def test_an_oom_kill_is_a_crash_whatever_the_exit_code(self) -> None:
        view = cmd_status.build_standalone_view(
            _snapshot(exit_code=0, oom_killed=True), None, server_id="mc", now=NOW
        )
        assert view.state is LifecycleState.CRASHED

    def test_an_absent_container_maps_to_absent(self) -> None:
        from mcmanager.containers.dto import ContainerSnapshot

        view = cmd_status.build_standalone_view(
            ContainerSnapshot.missing("minecraft", observed_at=NOW),
            None,
            server_id="mc",
            now=NOW,
        )
        assert view.state is LifecycleState.ABSENT

    def test_a_truncated_sample_is_called_out_rather_than_silently_partial(self) -> None:
        snapshot = _snapshot(running=True, state=ContainerState.RUNNING)
        probe = ProbeResult(reachable=True, players_online=20, sample=("a", "b"), probed_at=NOW)
        view = cmd_status.build_standalone_view(snapshot, probe, server_id="mc", now=NOW)
        assert any("truncated" in note for note in view.notes)

    def test_the_roster_from_a_probe_says_where_it_came_from(self) -> None:
        snapshot = _snapshot(running=True, state=ContainerState.RUNNING)
        probe = ProbeResult(reachable=True, players_online=1, sample=("Steve",), probed_at=NOW)
        view = cmd_status.build_standalone_view(snapshot, probe, server_id="mc", now=NOW)
        assert view.roster == (PlayerView(name="Steve", source="probe"),)

    def test_the_standalone_view_never_invents_a_session_or_an_idle_countdown(self) -> None:
        view = cmd_status.build_standalone_view(_snapshot(), None, server_id="mc", now=NOW)
        assert view.session is None
        assert view.idle is None


class TestStatusCommand:
    async def test_it_falls_back_with_a_visible_banner(
        self, ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert await cmd_status.run(ctx) == EXIT_OK
        captured = capsys.readouterr()
        assert "daemon: offline" in captured.out
        assert "falling back to a standalone view" in captured.err

    async def test_standalone_skips_the_daemon_entirely(
        self, ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert await cmd_status.run(ctx, standalone=True) == EXIT_OK
        captured = capsys.readouterr()
        assert "daemon: offline" in captured.out
        assert "falling back" not in captured.err

    async def test_players_only_prints_the_roster(
        self, ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert await cmd_status.run(ctx, players_only=True, standalone=True) == EXIT_OK
        assert "nobody is online" in capsys.readouterr().out

    async def test_an_unreachable_runtime_in_standalone_mode_exits_69(
        self, ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        use_fake.set_reachable(reachable=False)
        assert await cmd_status.run(ctx, standalone=True) == EXIT_UNAVAILABLE
        assert "Docker is unreachable" in capsys.readouterr().err

    async def test_json_output_of_the_standalone_view_round_trips(
        self, json_ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        assert await cmd_status.run(json_ctx, standalone=True) == EXIT_OK
        payload: dict[str, Any] = json.loads(capsys.readouterr().out)
        assert StatusView.from_dict(payload).daemon_online is False

    async def test_the_daemon_path_prints_the_daemons_own_json(
        self, ctx: CliContext, clock: ManualClock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--json`` returns what the endpoint said, not a locally rebuilt copy."""
        import json

        async for url in _control_server(clock):
            code = await cmd_status.run(replace(ctx, url=url, json_output=True))
            assert code == EXIT_OK
            payload: dict[str, Any] = json.loads(capsys.readouterr().out)
            assert payload["daemon_online"] is True
            assert payload["server_id"] == "minecraft"

    async def test_the_daemon_path_renders_the_online_banner(
        self, ctx: CliContext, clock: ManualClock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        async for url in _control_server(clock):
            assert await cmd_status.run(replace(ctx, url=url)) == EXIT_OK
            assert "daemon: online" in capsys.readouterr().out


# --------------------------------------------------------------------------------------- logs


class TestLogs:
    async def test_raw_mode_prints_the_original_line_untouched(
        self, ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--raw`` answers "why did the parser not match this", so it must not sanitise."""
        use_fake.emit_line("[13:24:37] [Server thread/INFO]: \x1b[93mSteve joined the game\x1b[0m")
        assert await cmd_logs.run(ctx, raw=True) == EXIT_OK
        assert "\x1b[93m" in capsys.readouterr().out

    async def test_parsed_mode_strips_ansi_and_renders_events(
        self, ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        use_fake.emit_line("[13:24:37] [Server thread/INFO]: \x1b[93mSteve joined the game\x1b[0m")
        assert await cmd_logs.run(ctx) == EXIT_OK
        out = capsys.readouterr().out
        assert "\x1b[93m" not in out
        assert "Steve joined" in out

    async def test_a_console_line_renders_with_its_thread_and_level(
        self, ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        use_fake.emit_line("[13:24:37] [Server thread/INFO]: Preparing spawn area: 2%")
        assert await cmd_logs.run(ctx) == EXIT_OK
        assert "[Server thread/INFO] Preparing spawn area: 2%" in capsys.readouterr().out

    async def test_json_output_is_serde_encoded_events(
        self, json_ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        from mcmanager.core.serde import event_from_dict

        use_fake.emit_line("[13:24:37] [Server thread/INFO]: Steve joined the game")
        assert await cmd_logs.run(json_ctx) == EXIT_OK
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert [event_from_dict(json.loads(line)).name for line in lines] == ["PlayerJoined"]

    async def test_a_bad_since_is_reported_before_any_docker_call(
        self, ctx: CliContext, use_fake: FakeRuntime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = await cmd_logs.run(ctx, since="whenever")
        assert code != EXIT_OK
        assert "relative age" in capsys.readouterr().err

    async def test_a_since_switches_the_tail_to_everything_after_it(
        self, ctx: CliContext, use_fake: FakeRuntime
    ) -> None:
        """Docker applies ``tail`` *after* ``since``, so the pair must be ``tail=-1``.

        Getting this wrong returns nothing and silently kills the backfill.
        """
        captured: list[tuple[int, object]] = []
        original = use_fake.logs_tail

        async def spy(name: str, *, lines: int = 100, since: object = None) -> list[LogLine]:
            captured.append((lines, since))
            return await original(name, lines=lines, since=since)  # pyright: ignore[reportArgumentType]

        use_fake.logs_tail = spy
        assert await cmd_logs.run(ctx, since="5m") == EXIT_OK
        assert captured[0][0] == -1
        assert captured[0][1] is not None

    async def test_follow_requires_the_daemon(self, ctx: CliContext) -> None:
        from mcmanager.errors import DaemonUnreachableError

        with pytest.raises(DaemonUnreachableError):
            await cmd_logs.run(ctx, follow=True)

    async def test_raw_with_follow_says_what_it_can_and_cannot_show(
        self, ctx: CliContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from mcmanager.errors import DaemonUnreachableError

        with pytest.raises(DaemonUnreachableError):
            await cmd_logs.run(ctx, follow=True, raw=True)
        assert "already-sanitised" in capsys.readouterr().err


# ----------------------------------------------------------------------------------- sessions


class TestSessions:
    def test_records_are_listed_newest_first(
        self, ctx: CliContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_sessions(
            ctx,
            [
                {"id": "old", "started_at": "2026-07-24T10:00:00Z"},
                {"id": "new", "started_at": "2026-07-25T10:00:00Z"},
            ],
        )
        assert cmd_sessions.run(ctx) == EXIT_OK
        lines = capsys.readouterr().out.splitlines()
        assert lines[1].endswith("new")
        assert lines[2].endswith("old")

    def test_a_jsonl_stream_is_read_too(
        self, ctx: CliContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        path = ctx.settings.state.dir / "sessions.jsonl"
        path.write_text(
            "\n".join(
                json.dumps({"id": f"s-{index}", "started_at": "2026-07-25T10:00:00Z"})
                for index in range(3)
            ),
            encoding="utf-8",
        )
        assert cmd_sessions.run(ctx) == EXIT_OK
        assert capsys.readouterr().out.count("s-") == 3

    def test_a_partial_record_is_labelled(
        self, ctx: CliContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Under-reporting somebody's playtime without saying so is the dishonest option.
        _write_sessions(ctx, [{"id": "s-1", "started_at": "2026-07-25T10:00:00Z", "partial": True}])
        assert cmd_sessions.run(ctx) == EXIT_OK
        assert "partial" in capsys.readouterr().out

    def test_one_corrupt_record_does_not_hide_the_others(
        self, ctx: CliContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_sessions(ctx, [{"id": "good", "started_at": "2026-07-25T10:00:00Z"}])
        (ctx.settings.state.dir / "sessions" / "torn.json").write_text("{oh no", encoding="utf-8")
        assert cmd_sessions.run(ctx) == EXIT_OK
        captured = capsys.readouterr()
        assert "good" in captured.out
        assert "not valid JSON" in captured.err

    def test_a_record_with_a_wrong_type_is_skipped_with_a_reason(
        self, ctx: CliContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_sessions(ctx, [{"id": "bad", "joins": "many"}])
        assert cmd_sessions.run(ctx) == EXIT_OK
        assert "must be an integer" in capsys.readouterr().err

    def test_showing_one_session_by_id(
        self, ctx: CliContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_sessions(ctx, [{"id": "s-1", "started_at": "2026-07-25T10:00:00Z", "joins": 4}])
        assert cmd_sessions.run(ctx, session_id="s-1") == EXIT_OK
        assert "joins 4" in capsys.readouterr().out

    def test_an_unknown_id_exits_69_with_a_hint(
        self, ctx: CliContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cmd_sessions.run(ctx, session_id="nope") == EXIT_UNAVAILABLE
        assert "mcmanager sessions" in capsys.readouterr().err

    def test_no_records_is_not_an_error(
        self, ctx: CliContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cmd_sessions.run(ctx) == EXIT_OK
        assert "no session records yet" in capsys.readouterr().out

    def test_a_missing_state_dir_explains_itself(
        self, ctx: CliContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from mcmanager.config import StateConfig

        elsewhere = StateConfig(dir=ctx.settings.state.dir.parent / "definitely-not-here")
        settings = ctx.settings.model_copy(update={"state": elsewhere})
        assert cmd_sessions.run(replace(ctx, settings=settings)) == EXIT_OK
        assert "does not exist yet" in capsys.readouterr().out

    def test_the_limit_is_honoured(
        self, ctx: CliContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_sessions(
            ctx,
            [{"id": f"s-{index}", "started_at": "2026-07-25T10:00:00Z"} for index in range(5)],
        )
        assert cmd_sessions.run(ctx, limit=2) == EXIT_OK
        assert len(capsys.readouterr().out.splitlines()) == 3  # header + 2

    def test_json_output_is_a_document(
        self, json_ctx: CliContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        _write_sessions(json_ctx, [{"id": "s-1", "started_at": "2026-07-25T10:00:00Z"}])
        assert cmd_sessions.run(json_ctx) == EXIT_OK
        payload: dict[str, Any] = json.loads(capsys.readouterr().out)
        assert payload["count"] == 1


# ------------------------------------------------------------------------------------- events


class TestEvents:
    """``events`` against a real daemon. It has no standalone mode, by design."""

    async def test_it_prints_events_from_the_replay_ring(
        self, ctx: CliContext, clock: ManualClock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        async with _bus_and_channel(clock) as (bus, channel):
            async for url in _control_server(clock, channel=channel):
                bus.publish(_joined("Steve"))
                await _settle()
                code = await cmd_events.run(replace(ctx, url=url), since="15m", idle_timeout=0.2)
                assert code == EXIT_OK
                assert "Steve joined" in capsys.readouterr().out

    async def test_json_output_round_trips_through_serde(
        self, ctx: CliContext, clock: ManualClock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        from mcmanager.core.serde import event_from_dict

        async with _bus_and_channel(clock) as (bus, channel):
            async for url in _control_server(clock, channel=channel):
                bus.publish(_joined("Steve"))
                await _settle()
                code = await cmd_events.run(
                    replace(ctx, url=url, json_output=True), since="15m", idle_timeout=0.2
                )
                assert code == EXIT_OK
                lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
                assert [event_from_dict(json.loads(line)).name for line in lines] == [
                    "PlayerJoined"
                ]

    async def test_a_type_filter_is_applied_by_the_daemon(
        self, ctx: CliContext, clock: ManualClock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        async with _bus_and_channel(clock) as (bus, channel):
            async for url in _control_server(clock, channel=channel):
                bus.publish(_joined("Steve"))
                bus.publish(
                    ConsoleLog(
                        ts=NOW,
                        server_id="minecraft",
                        source=Source.LOG,
                        raw="Preparing spawn area",
                        message="Preparing spawn area",
                    )
                )
                await _settle()
                code = await cmd_events.run(
                    replace(ctx, url=url), types=["PlayerEvent"], since="15m", idle_timeout=0.2
                )
                assert code == EXIT_OK
                out = capsys.readouterr().out
                assert "Steve joined" in out
                assert "Preparing spawn" not in out

    async def test_an_unknown_type_fails_before_connecting(
        self, ctx: CliContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A filter that silently matches nothing looks exactly like a broken daemon.
        code = await cmd_events.run(ctx, types=["NoSuchEvent"])
        assert code != EXIT_OK
        assert "unknown event type" in capsys.readouterr().err

    async def test_the_limit_stops_the_stream(
        self, ctx: CliContext, clock: ManualClock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        async with _bus_and_channel(clock) as (bus, channel):
            async for url in _control_server(clock, channel=channel):
                for name in ("a", "b", "c"):
                    bus.publish(_joined(name))
                await _settle()
                code = await cmd_events.run(
                    replace(ctx, url=url), since="15m", limit=2, idle_timeout=0.2
                )
                assert code == EXIT_OK
                assert len(capsys.readouterr().out.splitlines()) == 2

    async def test_an_empty_ring_says_so_rather_than_looking_broken(
        self, ctx: CliContext, clock: ManualClock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        async with _bus_and_channel(clock) as (_bus, channel):
            async for url in _control_server(clock, channel=channel):
                code = await cmd_events.run(replace(ctx, url=url), idle_timeout=0.2)
                assert code == EXIT_OK
                assert "replay ring" in capsys.readouterr().err

    async def test_it_requires_the_daemon(self, ctx: CliContext) -> None:
        from mcmanager.errors import DaemonUnreachableError

        with pytest.raises(DaemonUnreachableError):
            await cmd_events.run(ctx, idle_timeout=0.2)


# ------------------------------------------------------------------------------------ control


class TestControl:
    async def test_no_token_refuses_before_connecting(
        self, ctx: CliContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert await cmd_control.run(ctx, action="stop") == EXIT_UNAVAILABLE
        assert "web.token" in capsys.readouterr().err

    async def test_a_successful_stop_exits_zero(
        self, ctx: CliContext, clock: ManualClock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        outcome = ControlOutcome(action=ControlAction.STOP, actor="tester", accepted=True)
        async for url in _control_server(clock, outcome=outcome):
            code = await cmd_control.run(
                replace(ctx, url=url, token=TOKEN), action="stop", reason="dinner"
            )
            assert code == EXIT_OK
            assert "stop ok" in capsys.readouterr().out

    async def test_a_refusal_exits_one_and_prints_the_reason(
        self, ctx: CliContext, clock: ManualClock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        outcome = ControlOutcome(
            action=ControlAction.START,
            actor="tester",
            accepted=False,
            rejection="the server is already ready",
        )
        async for url in _control_server(clock, outcome=outcome):
            code = await cmd_control.run(replace(ctx, url=url, token=TOKEN), action="start")
            assert code == 1
            assert "already ready" in capsys.readouterr().out

    async def test_a_runtime_failure_exits_69(
        self, ctx: CliContext, clock: ManualClock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        outcome = ControlOutcome(
            action=ControlAction.STOP, actor="tester", accepted=True, error="docker said no"
        )
        async for url in _control_server(clock, outcome=outcome):
            code = await cmd_control.run(replace(ctx, url=url, token=TOKEN), action="stop")
            assert code == EXIT_UNAVAILABLE
            assert "docker said no" in capsys.readouterr().out

    async def test_the_actor_from_the_context_reaches_the_daemon(
        self, ctx: CliContext, clock: ManualClock
    ) -> None:
        seen: list[str] = []
        outcome = ControlOutcome(action=ControlAction.STOP, actor="tester", accepted=True)
        async for url in _control_server(clock, outcome=outcome, record=seen):
            await cmd_control.run(replace(ctx, url=url, token=TOKEN), action="stop")
        assert seen == ["tester"]


# ------------------------------------------------------------------------------------ helpers


def _no_sessions(limit: int) -> list[SessionView]:
    """A sessions provider with nothing to give."""
    del limit
    return []


def _write_sessions(ctx: CliContext, records: list[dict[str, Any]]) -> None:
    import json

    directory = ctx.settings.state.dir / "sessions"
    directory.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        (directory / f"{index:03d}.json").write_text(json.dumps(record), encoding="utf-8")


def _snapshot(
    *,
    running: bool = False,
    state: ContainerState = ContainerState.EXITED,
    health_raw: str | None = None,
    exit_code: int | None = 0,
    oom_killed: bool = False,
) -> Any:
    from mcmanager.containers.dto import ContainerSnapshot

    return ContainerSnapshot(
        name="minecraft",
        id="abc123",
        state=state,
        running=running,
        _health_raw=health_raw,
        exit_code=exit_code,
        oom_killed=oom_killed,
        started_at=NOW - timedelta(hours=1) if running else None,
        observed_at=NOW,
    )


class _StubController:
    def __init__(self, outcome: ControlOutcome, record: list[str] | None = None) -> None:
        self._outcome = outcome
        self._record = record

    async def start(self, *, actor: str, via: Source, reason: str | None) -> ControlOutcome:
        del via, reason
        if self._record is not None:
            self._record.append(actor)
        return self._outcome

    async def stop(self, *, actor: str, via: Source, reason: str | None) -> ControlOutcome:
        return await self.start(actor=actor, via=via, reason=reason)

    async def restart(self, *, actor: str, via: Source, reason: str | None) -> ControlOutcome:
        return await self.start(actor=actor, via=via, reason=reason)


def _joined(name: str) -> PlayerJoined:
    return PlayerJoined(
        ts=NOW,
        server_id="minecraft",
        source=Source.LOG,
        raw=f"{name} joined the game",
        player=PlayerRef(name=name),
        online_count=1,
    )


async def _settle(times: int = 8) -> None:
    """Let the bus dispatch into the SSE channel's replay ring."""
    for _ in range(times):
        await asyncio.sleep(0)


@asynccontextmanager
async def _bus_and_channel(clock: ManualClock) -> AsyncGenerator[tuple[EventBus, SseChannel]]:
    """A running bus with an attached SSE channel, torn down in the right order."""
    bus = EventBus()
    pump = asyncio.create_task(bus.run())
    await asyncio.sleep(0)
    channel = SseChannel(clock=clock)
    channel.attach(bus)
    try:
        yield bus, channel
    finally:
        await channel.aclose()
        await bus.aclose()
        await pump


async def _control_server(
    clock: ManualClock,
    *,
    outcome: ControlOutcome | None = None,
    record: list[str] | None = None,
    channel: SseChannel | None = None,
) -> AsyncIterator[str]:
    """A real control server on an ephemeral loopback port, for the daemon-path tests."""
    context = ControlContext(
        server_id="minecraft",
        clock=clock,
        channel=channel if channel is not None else SseChannel(clock=clock),
        status=lambda: StatusView(
            server_id="minecraft",
            container="minecraft",
            state=LifecycleState.READY,
            observed_at=NOW,
            running=True,
            players_online=0,
        ),
        players=lambda: (),
        sessions=_no_sessions,
        readiness=lambda: evaluate_readiness(runtime_available=True, log_stream_attached=True),
        liveness=lambda: LivenessView(),
        controller=None if outcome is None else _StubController(outcome, record),
        token=TOKEN,
    )
    server = TestServer(build_app(context))
    await server.start_server()
    try:
        yield str(server.make_url("")).rstrip("/")
    finally:
        await server.close()
