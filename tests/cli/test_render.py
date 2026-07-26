r"""Golden-output tests for the only module allowed to print.

Two things are being protected:

1. **The exact rendered blocks.** ``render.py`` is pure - DTOs in, strings out - so a golden
   comparison is cheap and catches an accidental layout change in review rather than in a
   screenshot three weeks later.
2. **No ANSI in non-tty output.** This project exists partly because escape codes corrupted player
   names; a CLI that pipes them into ``grep`` would be a poor joke. There is a test asserting the
   byte ``\x1b`` appears nowhere in any rendered block when colour is off, across every renderer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from io import StringIO
from typing import TYPE_CHECKING

import pytest

from mcmanager.cli import render
from mcmanager.containers.dto import (
    ContainerSnapshot,
    ContainerState,
    HealthState,
    HealthTiming,
    MountInfo,
    NetworkAttachment,
)
from mcmanager.control.views import (
    ControlResultView,
    IdleView,
    LivenessView,
    PlayerView,
    ProbeView,
    SessionView,
    StatusView,
    evaluate_readiness,
)
from mcmanager.core.events import (
    ChatMessage,
    CommandIssued,
    ConsoleLog,
    PlayerAdvancement,
    PlayerDeath,
    PlayerJoined,
    PlayerLeft,
    ServerCrashed,
    ServerReady,
    ServerStopped,
)
from mcmanager.core.types import (
    AdvancementKind,
    ChatKind,
    ControlAction,
    LeaveReason,
    LifecycleState,
    PlayerRef,
    ReadySignal,
    Source,
)

if TYPE_CHECKING:
    from mcmanager.core.events import Event

NOW = datetime(2026, 7, 25, 22, 58, 20, tzinfo=UTC)
ESC = "\x1b"


def _running_status() -> StatusView:
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
        ready_detected_by="log",
        startup_seconds=32.521,
        players_online=2,
        players_max=5,
        roster=(
            PlayerView(name="Steve", online_since=NOW, session_seconds=3660.0),
            PlayerView(name="bharath_720", session_seconds=720.0, first_seen=True),
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
        stop_timeout_seconds=90,
    )


def _stopped_status() -> StatusView:
    return StatusView(
        server_id="minecraft",
        container="minecraft",
        state=LifecycleState.STOPPED,
        observed_at=NOW,
        daemon_online=False,
        running=False,
        health=HealthState.UNKNOWN,
        health_reported_raw="unhealthy",
        container_id="5ff1a3c2deadbeef",
        exit_code=0,
        probe=ProbeView(reachable=False, error="connection refused", probed_at=NOW),
        notes=("state is inferred",),
    )


def _snapshot(**overrides: object) -> ContainerSnapshot:
    defaults: dict[str, object] = {
        "name": "minecraft",
        "id": "5ff1a3c2deadbeef",
        "state": ContainerState.EXITED,
        "status_text": "exited (0) 4 hours ago",
        "running": False,
        "_health_raw": "unhealthy",
        "health_failing_streak": 0,
        "health_timing": HealthTiming(
            interval=timedelta(seconds=30),
            timeout=timedelta(seconds=5),
            start_period=timedelta(seconds=120),
            retries=2,
        ),
        "tty": False,
        "exit_code": 0,
        "image": "itzg/minecraft-server:latest",
        "created_at": NOW - timedelta(days=30),
        "log_driver": "json-file",
        "log_options": {"max-size": "10m", "max-file": "3"},
        "restart_policy": "no",
        "stop_signal": "SIGTERM",
        "networks": (
            NetworkAttachment(name="homelab", ip_address="172.20.0.9", aliases=("minecraft",)),
        ),
        "mounts": (
            MountInfo(source="/home/minty/homelab/data/minecraft", destination="/data", rw=True),
        ),
        "labels": {"com.docker.compose.project": "minecraft"},
        "observed_at": NOW,
    }
    defaults.update(overrides)
    return ContainerSnapshot(**defaults)  # pyright: ignore[reportArgumentType]


# ------------------------------------------------------------------------------------- colour


class TestColourPolicy:
    def test_a_non_tty_stream_gets_no_colour(self) -> None:
        assert not render.supports_color(StringIO(), env={})

    def test_no_color_wins_even_on_a_tty(self) -> None:
        class Tty(StringIO):
            def isatty(self) -> bool:
                return True

        assert render.supports_color(Tty(), env={})
        assert not render.supports_color(Tty(), env={"NO_COLOR": ""})
        assert not render.supports_color(Tty(), env={"NO_COLOR": "1"})

    def test_the_flag_wins_over_everything(self) -> None:
        class Tty(StringIO):
            def isatty(self) -> bool:
                return True

        assert not render.supports_color(Tty(), no_color=True, env={})

    def test_a_closed_stream_is_treated_as_not_a_terminal(self) -> None:
        stream = StringIO()
        stream.close()
        assert not render.supports_color(stream, env={})

    def test_a_plain_palette_cannot_emit_an_escape_byte(self) -> None:
        pal = render.Palette.plain()
        for method in (pal.bold, pal.dim, pal.red, pal.green, pal.yellow, pal.cyan, pal.grey):
            assert method("text") == "text"
        assert pal.state(LifecycleState.CRASHED) == "CRASHED"
        assert pal.verdict(False) == "FAIL"

    def test_an_enabled_palette_does_emit_colour(self) -> None:
        pal = render.Palette(enabled=True)
        assert pal.red("boom") == "\x1b[31mboom\x1b[0m"

    def test_an_unknown_colour_name_is_a_no_op_rather_than_a_crash(self) -> None:
        assert render.Palette(enabled=True).paint("text", "chartreuse") == "text"

    @pytest.mark.parametrize(
        "block",
        [
            render.render_status(_running_status()),
            render.render_status(_stopped_status()),
            render.render_players(_running_status().roster),
            render.render_sessions([SessionView(id="s", started_at=NOW)], now=NOW),
            render.render_session(SessionView(id="s", started_at=NOW, partial=True), now=NOW),
            render.render_inspect(_snapshot()),
            render.render_readiness(evaluate_readiness(runtime_available=False)),
            render.render_liveness(LivenessView(loop_lag_seconds=0.004)),
            render.render_control_result(ControlResultView(action="stop", accepted=False)),
            render.render_event(
                PlayerJoined(
                    ts=NOW, server_id="mc", source=Source.LOG, player=PlayerRef(name="Steve")
                )
            ),
            render.render_log_line(ts=NOW, message="Done", level="ERROR", thread="Server thread"),
        ],
    )
    def test_no_renderer_emits_an_escape_byte_without_a_palette(self, block: str) -> None:
        """The regression this project is named after, one layer up.

        Every renderer defaults to ``Palette.plain()``, so piped output is escape-free by
        construction rather than by a caller remembering to ask.
        """
        assert ESC not in block


# --------------------------------------------------------------------------------- formatting


class TestFormatting:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (None, "unknown"),
            (0, "0s"),
            (-5, "0s"),
            (45, "45s"),
            (59.9, "59s"),
            (60, "1m 00s"),
            (750, "12m 30s"),
            (3600, "1h 00m"),
            (10800, "3h 00m"),
            (86400, "1d 00h"),
            (183600, "2d 03h"),
        ],
    )
    def test_durations_are_human(self, seconds: float | None, expected: str) -> None:
        assert render.format_duration(seconds) == expected

    def test_timestamps_are_utc_with_a_z(self) -> None:
        assert render.format_timestamp(NOW) == "2026-07-25 22:58:20Z"

    def test_a_non_utc_timestamp_is_converted_rather_than_relabelled(self) -> None:
        """The homelab runs Asia/Kolkata. Rendering local time would give three clocks per bug."""
        kolkata = NOW.astimezone(timezone(timedelta(hours=5, minutes=30)))
        assert kolkata.hour == 4  # the same instant, spelled differently
        assert render.format_timestamp(kolkata) == "2026-07-25 22:58:20Z"

    def test_a_missing_timestamp_reads_unknown(self) -> None:
        assert render.format_timestamp(None) == "unknown"
        assert render.format_clock(None) == "--:--:--"


# ------------------------------------------------------------------------------------ goldens


EXPECTED_STATUS = """\
minecraft  READY    daemon: online
  container       minecraft (5ff1a3c2dead)
  image           itzg/minecraft-server:latest
  health          healthy
  uptime          3h 00m  since 2026-07-25 19:58:20Z
  version         26.2   ready in 32.5s (via log)
  players         2/5
                  Steve             1h 01m
                  bharath_720       12m 00s  (first visit)
  probe           reachable  18ms  "A Minecraft Server"
  idle            stops in 12m 30s at 2026-07-25 23:10:20Z  [dry run: nothing will be stopped]
  session         2026-07-25 19:58Z  open
                  joins 5  deaths 2  chat 41  peak 2"""

EXPECTED_STOPPED = """\
minecraft  STOPPED    daemon: offline (standalone: container inspect + status probe only)
  container       minecraft (5ff1a3c2dead)
  health          unknown  (docker still reports 'unhealthy'; that value is stale)
  uptime          not running
  exit            0
  players         unknown
  probe           unreachable  - player count unknown, not zero: connection refused
  note            state is inferred"""

EXPECTED_PLAYERS = """\
player            online for  since                   source
Steve             1h 01m      2026-07-25 22:58:20Z    log
bharath_720       12m 00s     unknown                 log  first visit"""

EXPECTED_SESSIONS = """\
started             duration    players  deaths  chat   id
2026-07-25 20:58Z   2h 00m      2        3       17     s-1
2026-07-25 22:58Z   0s          0        0       0      s-2  open partial"""

EXPECTED_READINESS = """\
readiness: NOT ready
  runtime     FAIL  socket gone
  discord     ok  disabled by config
  log_stream  ok  attached
  bus         ok  no critical drops"""


class TestGoldens:
    def test_the_running_status_block(self) -> None:
        assert render.render_status(_running_status()) == EXPECTED_STATUS

    def test_the_stopped_status_block(self) -> None:
        assert render.render_status(_stopped_status()) == EXPECTED_STOPPED

    def test_the_players_table(self) -> None:
        assert render.render_players(_running_status().roster) == EXPECTED_PLAYERS

    def test_the_sessions_table(self) -> None:
        sessions = [
            SessionView(
                id="s-1",
                started_at=NOW - timedelta(hours=2),
                ended_at=NOW,
                players=("Steve", "Alex"),
                deaths=3,
                chat_messages=17,
            ),
            SessionView(id="s-2", started_at=NOW, open=True, partial=True),
        ]
        assert render.render_sessions(sessions, now=NOW) == EXPECTED_SESSIONS

    def test_the_readiness_block(self) -> None:
        view = evaluate_readiness(
            runtime_available=False,
            runtime_detail="socket gone",
            log_stream_attached=True,
        )
        assert render.render_readiness(view) == EXPECTED_READINESS


# ------------------------------------------------------------------------- the stale-health fix


class TestStaleHealthIsVisible:
    def test_inspect_shows_derived_unknown_and_names_dockers_stale_string(self) -> None:
        """The user-visible half of the fix, as run against the real exited homelab container."""
        block = render.render_inspect(_snapshot())
        assert "health          unknown" in block
        assert "docker still reports 'unhealthy'; that value is stale" in block

    def test_a_running_healthy_container_prints_no_disclaimer(self) -> None:
        block = render.render_inspect(
            _snapshot(running=True, state=ContainerState.RUNNING, _health_raw="healthy")
        )
        assert "health          healthy" in block
        assert "stale" not in block

    def test_status_shows_the_same_disclaimer(self) -> None:
        assert "that value is stale" in render.render_status(_stopped_status())


class TestInspect:
    def test_the_guard_window_is_shown_as_read_from_the_container(self) -> None:
        block = render.render_inspect(_snapshot())
        assert "guard         210s" in block
        assert "read from the container" in block

    def test_tty_false_explains_what_it_means_for_the_log_stream(self) -> None:
        assert "multiplexed frames" in render.render_inspect(_snapshot())

    def test_the_json_file_driver_is_labelled_a_ring(self) -> None:
        # 10m x 3 is not an archive, and a reader who assumes otherwise loses a session.
        assert "(a ring, not an archive)" in render.render_inspect(_snapshot())

    def test_network_aliases_are_listed(self) -> None:
        assert "aliases: minecraft" in render.render_inspect(_snapshot())

    def test_no_networks_is_called_out_as_a_problem(self) -> None:
        block = render.render_inspect(_snapshot(networks=()))
        assert "SLP and RCON by DNS name cannot work" in block

    def test_a_read_only_mount_is_marked(self) -> None:
        block = render.render_inspect(
            _snapshot(mounts=(MountInfo(source="/mnt/logs", destination="/logs", rw=False),))
        )
        assert "/mnt/logs -> /logs  bind ro" in block

    def test_an_absent_container_says_so_without_pretending_it_is_an_error(self) -> None:
        block = render.render_inspect(ContainerSnapshot.missing("minecraft", observed_at=NOW))
        assert "no container with this name exists" in block
        assert "lifecycle, not its existence" in block

    def test_extra_diagnostics_are_passed_through(self) -> None:
        block = render.render_inspect(_snapshot(), extra={"log stream": "private"})
        assert "log stream      private" in block


# ------------------------------------------------------------------------------------- events


class TestEventRendering:
    @pytest.mark.parametrize(
        ("event", "expected"),
        [
            (
                PlayerJoined(
                    ts=NOW,
                    server_id="mc",
                    source=Source.LOG,
                    player=PlayerRef(name="Steve"),
                    online_count=2,
                ),
                "PlayerJoined       Steve joined  online 2",
            ),
            (
                PlayerLeft(
                    ts=NOW,
                    server_id="mc",
                    source=Source.LOG,
                    player=PlayerRef(name="Steve"),
                    reason=LeaveReason.SERVER_CLOSED,
                    session_seconds=3660.0,
                    online_count=0,
                ),
                "PlayerLeft         Steve left (server_closed) after 1h 01m  online 0",
            ),
            (
                ChatMessage(
                    ts=NOW,
                    server_id="mc",
                    source=Source.LOG,
                    player=PlayerRef(name="Steve"),
                    message="hello @everyone",
                    kind=ChatKind.CHAT,
                ),
                "ChatMessage        <Steve> hello @everyone",
            ),
            (
                PlayerDeath(
                    ts=NOW,
                    server_id="mc",
                    source=Source.LOG,
                    player=PlayerRef(name="Steve"),
                    message="Steve was blown up by Creeper",
                    template="%1$s was blown up by %2$s",
                ),
                "PlayerDeath        Steve was blown up by Creeper",
            ),
            (
                PlayerAdvancement(
                    ts=NOW,
                    server_id="mc",
                    source=Source.LOG,
                    player=PlayerRef(name="Steve"),
                    title="Stone Age",
                    kind=AdvancementKind.ADVANCEMENT,
                ),
                "PlayerAdvancement  Steve earned advancement [Stone Age]",
            ),
            (
                ServerReady(
                    ts=NOW,
                    server_id="mc",
                    source=Source.LOG,
                    startup_seconds=32.521,
                    detected_by=ReadySignal.LOG_DONE,
                ),
                "ServerReady        server ready in 32.5s (via log)",
            ),
            (
                ServerStopped(
                    ts=NOW, server_id="mc", source=Source.RUNTIME, exit_code=0, clean=True
                ),
                "ServerStopped      server stopped (clean, exit 0)",
            ),
            (
                ServerCrashed(
                    ts=NOW, server_id="mc", source=Source.RUNTIME, exit_code=137, oom_killed=True
                ),
                "ServerCrashed      server crashed (exit 137, OOM-killed)",
            ),
            (
                CommandIssued(
                    ts=NOW,
                    server_id="mc",
                    source=Source.CLI,
                    action=ControlAction.STOP,
                    actor="kunal",
                    via=Source.CLI,
                    accepted=False,
                    rejection="already stopped",
                ),
                "CommandIssued      stop by kunal via cli - rejected: already stopped",
            ),
            (
                ConsoleLog(
                    ts=NOW,
                    server_id="mc",
                    source=Source.LOG,
                    message="Preparing spawn area: 2%",
                    level="INFO",
                ),
                "ConsoleLog         Preparing spawn area: 2%",
            ),
        ],
    )
    def test_one_event_renders_as_one_line(self, event: Event, expected: str) -> None:
        assert render.render_event(event) == expected

    def test_a_timestamp_can_be_prefixed(self) -> None:
        event = PlayerJoined(
            ts=NOW, server_id="mc", source=Source.LOG, player=PlayerRef(name="Steve")
        )
        assert render.render_event(event, timestamp=True).startswith("22:58:20  ")

    def test_a_players_ip_address_is_never_rendered(self) -> None:
        """``PlayerJoined.address`` exists for idle diagnostics and abuse investigation only."""
        event = PlayerJoined(
            ts=NOW,
            server_id="mc",
            source=Source.LOG,
            player=PlayerRef(name="Steve"),
            address="115.99.245.156:49237",
        )
        assert "115.99" not in render.render_event(event)

    def test_a_tier_two_death_is_marked_as_having_matched_no_template(self) -> None:
        event = PlayerDeath(
            ts=NOW,
            server_id="mc",
            source=Source.LOG,
            player=PlayerRef(name="Steve"),
            message="Steve was impaled by a plugin",
            template=None,
        )
        assert "(unmatched template)" in render.render_event(event)

    def test_a_chat_message_is_never_interpreted_as_a_join(self) -> None:
        # The old bridge script's substring dispatch, restated as a rendering assertion.
        event = ChatMessage(
            ts=NOW,
            server_id="mc",
            source=Source.LOG,
            player=PlayerRef(name="Bob"),
            message="Alice joined the game",
        )
        assert render.render_event(event) == "ChatMessage        <Bob> Alice joined the game"


class TestLogLines:
    def test_a_console_line_carries_its_thread_and_level(self) -> None:
        line = render.render_log_line(ts=NOW, message="Done", level="INFO", thread="Server thread")
        assert line == "22:58:20  [Server thread/INFO] Done"

    def test_the_time_can_be_suppressed(self) -> None:
        line = render.render_log_line(ts=NOW, message="Done", show_time=False)
        assert line == "Done"

    def test_a_line_with_no_metadata_is_just_the_message(self) -> None:
        assert render.render_log_line(ts=None, message="raw", show_time=False) == "raw"


class TestControlResult:
    def test_a_refusal_names_the_reason(self) -> None:
        result = ControlResultView(action="start", accepted=False, rejection="already ready")
        assert render.render_control_result(result) == "start refused: already ready"

    def test_a_runtime_failure_is_distinct_from_a_refusal(self) -> None:
        result = ControlResultView(action="stop", accepted=True, error="docker said no")
        assert render.render_control_result(result) == "stop failed: docker said no"

    def test_a_dry_run_says_nothing_was_done(self) -> None:
        result = ControlResultView(action="stop", accepted=True, dry_run=True)
        assert "nothing was done" in render.render_control_result(result)

    def test_success_uses_the_daemons_own_message(self) -> None:
        result = ControlResultView(action="stop", accepted=True, ok=True, message="stop ok")
        assert render.render_control_result(result) == "stop ok"


class TestEmit:
    def test_emit_writes_a_line_to_the_given_stream(self) -> None:
        stream = StringIO()
        render.emit("hello", stream=stream)
        assert stream.getvalue() == "hello\n"

    def test_emit_error_goes_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        render.emit_error("careful")
        captured = capsys.readouterr()
        assert captured.err == "careful\n"
        assert captured.out == ""


class TestSessionDetail:
    def test_a_partial_session_says_its_counts_are_incomplete(self) -> None:
        block = render.render_session(
            SessionView(id="s-1", started_at=NOW, partial=True, joins=3), now=NOW
        )
        assert "the daemon missed part of this session" in block

    def test_an_open_session_reports_its_end_as_open(self) -> None:
        block = render.render_session(SessionView(id="s-1", started_at=NOW, open=True), now=NOW)
        assert "ended           open" in block

    def test_an_empty_list_says_so_rather_than_printing_a_header(self) -> None:
        assert render.render_sessions([], now=NOW) == "no session records yet"


class TestPlayersTable:
    def test_an_empty_roster_says_nobody_is_online(self) -> None:
        assert render.render_players([]) == "nobody is online"

    def test_the_standalone_banner_warns_the_names_come_from_the_probe(self) -> None:
        block = render.render_players([], daemon_online=False)
        assert "status probe's sample only" in block


class TestReplaySummary:
    def test_the_ratio_verdict_flips_at_the_threshold(self) -> None:
        from mcmanager.games.minecraft.parser import scan

        stats = scan(["[13:24:37] [Server thread/INFO]: Steve joined the game"])
        block = render.render_replay_summary(stats, source="x.log", threshold=0.01)
        assert "ratio 0.0000" in block
        assert "ok" in block
