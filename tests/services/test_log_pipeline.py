"""Tests for :mod:`mcmanager.services.log_pipeline`.

Every log line here is either sampled verbatim from the real archives at
``tests/fixtures/logs/archives/`` or is one of the three adversarial shapes the old
``minecraft-discord-bridge.py`` got wrong. Those three have named regression tests:
:func:`test_chat_can_never_forge_a_join`, :func:`test_a_prefixed_join_captures_only_the_name` and
:func:`test_an_ansi_coloured_join_captures_a_clean_name`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, final

import pytest

from mcmanager.containers.dto import LogLine
from mcmanager.core.events import (
    ChatMessage,
    ConsoleLog,
    Event,
    PlayerDeath,
    PlayerJoined,
    PlayerLeft,
    ServerStarting,
)
from mcmanager.core.types import LeaveReason, Source, Stream
from mcmanager.games.minecraft.adapter import MinecraftAdapter
from mcmanager.services.log_pipeline import REDACTED_ADDRESS, LogPipeline, TokenBucket
from mcmanager.services.players import PlayerRoster

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mcmanager.clock import Clock, ManualClock
    from mcmanager.core.types import ReadySignal, ServerId
    from mcmanager.games.base import ProbeResult

SERVER_ID = "minecraft"

# ------------------------------------------------------------------------------- real log lines

VERSION = "[13:24:00] [Server thread/INFO]: Starting minecraft server version 26.2"
READY = '[13:24:37] [Server thread/INFO]: Done (32.521s)! For help, type "help"'
UUID_LINE = (
    "[13:29:00] [User Authenticator #1/INFO]: "
    "UUID of player Hypixelite is 403d4fb2-f466-3716-b9cb-a3e769bb40c9"
)
LOGIN = (
    "[13:24:37] [Server thread/INFO]: Hypixelite[/115.99.245.156:49237] "
    "logged in with entity id 74 at (12.5, 64.0, -3.5)"
)
JOIN = "[13:24:37] [Server thread/INFO]: Hypixelite joined the game"
JOIN_ANSI = "[13:24:37] [Server thread/INFO]: \x1b[93mHypixelite joined the game\x1b[0m"
LEAVE = "[13:30:00] [Server thread/INFO]: Hypixelite left the game"
LOST_QUIT = "[13:30:00] [Server thread/INFO]: Hypixelite lost connection: Disconnected"
LOST_SERVER_CLOSED = "[13:31:01] [Server thread/INFO]: Hypixelite lost connection: Server closed"
KICKED = "[13:30:00] [Server thread/INFO]: Hypixelite was kicked due to keepalive timeout!"
STOPPING = "[13:31:00] [Server thread/INFO]: [Rcon: Stopping the server]"
CHAT = "[14:48:06] [Async Chat Thread - #0/INFO]: [Not Secure] <Hypixelite> yo"
CHAT_FORGERY = (
    "[14:48:06] [Async Chat Thread - #0/INFO]: [Not Secure] <Hypixelite> Alex joined the game"
)
COMMAND = "[14:48:06] [Server thread/INFO]: Hypixelite issued server command: /gamemode creative"
VANILLA_DEATH = "[03:01:01] [Server thread/INFO]: Hypixelite was blown up by Creeper"
PLUGIN_DEATH = "[03:01:02] [Server thread/INFO]: Hypixelite was disintegrated by a Void Reaver"
"""A death message no vanilla template covers, phrased the way the game phrases one.

That second half is load-bearing: the tier-2 fallback requires the remainder to open with a
word the vanilla death table uses, because "begins with an online player's name" alone makes
every autosave a death for anybody who logs in as `Saving`.
"""
CONSOLE_NOISE = "[13:24:10] [Server thread/INFO]: Preparing spawn area: 2%"


# ------------------------------------------------------------------------------------- harness


@final
class RecordingSink:
    """Collects published events. Structurally an ``EventSink``; no bus required."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)

    def of[E: Event](self, kind: type[E]) -> list[E]:
        return [event for event in self.events if isinstance(event, kind)]

    @property
    def names(self) -> list[str]:
        return [event.name for event in self.events]


@final
class BrokenAdapter:
    """A ``GameAdapter`` whose parser raises. Nothing else about it is interesting."""

    def __init__(self) -> None:
        self._inner = MinecraftAdapter()

    @property
    def game_id(self) -> str:
        return self._inner.game_id

    @property
    def default_port(self) -> int:
        return self._inner.default_port

    @property
    def default_stop_timeout(self) -> int:
        return self._inner.default_stop_timeout

    def parse_line(
        self,
        raw: str,
        *,
        ts: datetime,
        server_id: ServerId,
        stream: Stream,
    ) -> Event:
        del raw, ts, server_id, stream
        msg = "adapter exploded"
        raise RuntimeError(msg)

    def ready_signal(self, event: Event) -> ReadySignal | None:
        return self._inner.ready_signal(event)

    def stop_signal(self, event: Event) -> bool:
        return self._inner.stop_signal(event)

    async def probe(
        self,
        host: str,
        port: int,
        *,
        timeout: float,  # noqa: ASYNC109 - the GameAdapter contract; handed to the query library
    ) -> ProbeResult:
        del host, port, timeout
        raise NotImplementedError


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


@pytest.fixture
def roster(manual_clock: ManualClock) -> PlayerRoster:
    return PlayerRoster(clock=manual_clock)


def build(
    clock: Clock,
    sink: RecordingSink,
    roster: PlayerRoster,
    **kwargs: object,
) -> LogPipeline:
    """A pipeline wired to the real Minecraft adapter."""
    return LogPipeline(
        adapter=MinecraftAdapter(clock=clock),
        sink=sink,
        clock=clock,
        server_id=SERVER_ID,
        roster=roster,
        **kwargs,  # pyright: ignore[reportArgumentType]
    )


@pytest.fixture
def pipeline(
    manual_clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> LogPipeline:
    return build(manual_clock, sink, roster)


def feed(pipeline: LogPipeline, lines: Iterable[str]) -> None:
    for text in lines:
        pipeline.handle_raw(text)


# ------------------------------------------------------------------------------ the enrichers


def test_the_login_line_stitches_an_address_onto_the_following_join(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    feed(pipeline, [LOGIN, JOIN])

    joined = sink.of(PlayerJoined)
    assert len(joined) == 1
    assert joined[0].address == "115.99.245.156:49237"


def test_the_address_is_never_put_in_the_login_console_event(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    """**The mechanism behind "never relay raw".**

    ``ConsoleLog`` carries no sensitivity marker, so whatever is in it reaches ``/logs`` SSE and
    its 512-entry replay ring, ``mcmanager logs --follow``, and the M6 Discord console relay -
    which subscribes to ``Event`` and enqueues ``event.raw`` verbatim. A comment saying the
    address is never relayed is not a mechanism; removing it from the event is.
    """
    feed(pipeline, [LOGIN])

    console = sink.of(ConsoleLog)
    assert len(console) == 1
    assert not isinstance(console[0], PlayerJoined)
    assert "115.99.245.156" not in console[0].message
    assert "115.99.245.156" not in (console[0].raw or "")
    assert REDACTED_ADDRESS in console[0].message
    assert "logged in with entity id 74" in console[0].message, "the rest of the line survives"
    assert pipeline.stats["addresses_redacted"] == 1


def test_no_published_event_carries_an_ip_from_a_full_connect_disconnect(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    """The other spelling: ``KittyScan (/176.65.148.158:58184) lost connection: Disconnected`` is
    a real line from this deployment's archives, and that parenthesised address is an IP too.

    Asserted across every published event's ``message`` *and* ``raw``, because the relay renders
    ``raw`` and the SSE frame carries both.
    """
    feed(
        pipeline,
        [
            LOGIN,
            JOIN,
            "[13:30:00] [Server thread/INFO]: KittyScan (/176.65.148.158:58184) "
            "lost connection: Disconnected",
        ],
    )

    for event in sink.events:
        text = f"{event.raw or ''} {getattr(event, 'message', '')}"
        assert "115.99.245.156" not in text, type(event).__name__
        assert "176.65.148.158" not in text, type(event).__name__

    # ...and the address is still available on the one field documented as carrying it.
    assert sink.of(PlayerJoined)[0].address == "115.99.245.156:49237"


def test_the_uuid_line_stitches_an_offline_uuid_onto_the_following_join(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    feed(pipeline, [UUID_LINE, LOGIN, JOIN])

    joined = sink.of(PlayerJoined)[0]
    assert joined.player.uuid == "403d4fb2-f466-3716-b9cb-a3e769bb40c9"
    assert joined.player.name == "Hypixelite"


def test_the_lost_connection_line_stitches_a_reason_onto_the_following_leave(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    feed(pipeline, [JOIN, LOST_QUIT, LEAVE])

    left = sink.of(PlayerLeft)[0]
    assert left.reason is LeaveReason.QUIT


def test_a_kick_line_stitches_a_kicked_reason_onto_the_following_leave(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    feed(pipeline, [JOIN, KICKED, LEAVE])

    assert sink.of(PlayerLeft)[0].reason is LeaveReason.KICKED


def test_an_enrichment_fact_is_consumed_once(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    # A stale reason applying to the *next* session would attribute an old kick to a fresh quit.
    feed(pipeline, [JOIN, KICKED, LEAVE, JOIN, LEAVE])

    reasons = [event.reason for event in sink.of(PlayerLeft)]
    assert reasons == [LeaveReason.KICKED, LeaveReason.UNKNOWN]


def test_a_command_line_is_recognised_and_stays_a_console_log(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    feed(pipeline, [COMMAND])

    assert sink.names == ["ConsoleLog"]
    assert pipeline.stats["commands_seen"] == 1
    assert pipeline.stats["recognised_console_lines"] == 1


# ------------------------------------------------------- shutdown versus voluntary disconnect


def test_server_closed_is_never_a_voluntary_quit(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    """``bharath_720 lost connection: Server closed`` is a real line from this server.

    Counting it as a quit corrupts every session summary that follows.
    """
    feed(pipeline, [JOIN, STOPPING, LOST_SERVER_CLOSED, LEAVE])

    assert sink.of(PlayerLeft)[0].reason is LeaveReason.SERVER_CLOSED


def test_server_closed_is_classified_even_without_a_stop_being_known(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    # Two independent defences. This one is the reason string itself, which does not need the
    # pipeline to have noticed the shutdown.
    feed(pipeline, [JOIN, LOST_SERVER_CLOSED, LEAVE])

    assert pipeline.stopping is False
    assert sink.of(PlayerLeft)[0].reason is LeaveReason.SERVER_CLOSED


def test_a_bare_leave_during_a_shutdown_is_a_shutdown_casualty(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    # No "lost connection" line at all. Outside a shutdown that is UNKNOWN; during one it is what
    # a shutdown casualty looks like, and UNKNOWN would let a summary count it as a quit.
    feed(pipeline, [JOIN, STOPPING, LEAVE])

    assert sink.of(PlayerLeft)[0].reason is LeaveReason.SERVER_CLOSED


def test_a_bare_leave_outside_a_shutdown_is_unknown_not_a_guess(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    feed(pipeline, [JOIN, LEAVE])

    assert sink.of(PlayerLeft)[0].reason is LeaveReason.UNKNOWN


def test_an_explicit_quit_during_a_shutdown_is_still_a_quit(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    # Deliberately not over-reaching: somebody who disconnects two seconds before a shutdown really
    # did quit, and the log said so in as many words.
    feed(pipeline, [JOIN, STOPPING, LOST_QUIT, LEAVE])

    assert sink.of(PlayerLeft)[0].reason is LeaveReason.QUIT


def test_the_controller_can_declare_a_stop_before_the_server_logs_one(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    # We ask Docker to stop the container, then the JVM takes a moment to say so. Leaves in that
    # window must not be counted as quits.
    feed(pipeline, [JOIN])
    pipeline.set_stopping(True)
    feed(pipeline, [LEAVE])

    assert sink.of(PlayerLeft)[0].reason is LeaveReason.SERVER_CLOSED


def test_a_server_start_clears_the_stopping_flag_and_the_roster(
    pipeline: LogPipeline,
    roster: PlayerRoster,
) -> None:
    feed(pipeline, [JOIN, STOPPING])
    assert pipeline.stopping is True
    assert roster.count == 1

    feed(pipeline, [VERSION])

    assert pipeline.stopping is False
    assert roster.count == 0


def test_a_ready_line_clears_the_stopping_flag(pipeline: LogPipeline) -> None:
    feed(pipeline, [STOPPING, READY])

    assert pipeline.stopping is False


# ------------------------------------------------------------------------- roster enrichment


def test_a_join_carries_the_roster_count_and_first_seen(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    feed(pipeline, [JOIN])
    joined = sink.of(PlayerJoined)[0]

    assert joined.online_count == 1
    assert joined.first_seen is True


async def test_a_leave_carries_its_session_duration_and_the_new_count(
    manual_clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    pipeline = build(manual_clock, sink, roster)
    feed(pipeline, [JOIN])
    await manual_clock.advance(600.0)
    feed(pipeline, [LEAVE])

    left = sink.of(PlayerLeft)[0]
    assert left.session_seconds == pytest.approx(600.0)
    assert left.online_count == 0


def test_a_leave_for_an_unseen_player_has_no_fabricated_duration(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    # The daemon started mid-session. None is the honest answer; a zero would read as a real
    # zero-second session in a summary.
    feed(pipeline, [LEAVE])

    assert sink.of(PlayerLeft)[0].session_seconds is None


def test_the_roster_is_the_pipelines_own_source_of_truth(
    pipeline: LogPipeline,
    roster: PlayerRoster,
) -> None:
    feed(pipeline, [JOIN])
    assert roster.online == ("Hypixelite",)

    feed(pipeline, [LEAVE])
    assert roster.count == 0


# ------------------------------------------------------------------ the old script's three bugs


def test_chat_can_never_forge_a_join(pipeline: LogPipeline, sink: RecordingSink) -> None:
    """The old script dispatched on ``if "joined the game" in line``.

    Any player could type ``Alex joined the game`` and produce a join notification for somebody who
    was not there. Chat is matched before join and every pattern is anchored, so this line is
    exactly one chat message and zero joins.
    """
    feed(pipeline, [CHAT_FORGERY])

    assert sink.names == ["ChatMessage"]
    assert sink.of(ChatMessage)[0].message == "Alex joined the game"
    assert sink.of(PlayerJoined) == []


def test_a_prefixed_join_captures_only_the_name(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    """The old script's ``re.compile(r"(.+) joined the game").search(line)``.

    Greedy, unanchored, and happy to start at column zero, so it captured
    ``[22:41:23] [Server thread/INFO]: Steve`` as the player's name.
    """
    feed(pipeline, [JOIN])

    name = sink.of(PlayerJoined)[0].player.name
    assert name == "Hypixelite"
    assert "[" not in name
    assert ":" not in name


def test_an_ansi_coloured_join_captures_a_clean_name(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    """ANSI is confirmed in both sinks on this server; the old script stripped none of it."""
    feed(pipeline, [JOIN_ANSI])

    joined = sink.of(PlayerJoined)[0]
    assert joined.player.name == "Hypixelite"
    assert joined.raw is not None
    assert "\x1b" not in joined.raw


# --------------------------------------------------------------------- tier-2 death fallback


def test_an_unmatched_line_from_an_online_player_becomes_a_tier_2_death(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    feed(pipeline, [JOIN, PLUGIN_DEATH])

    deaths = sink.of(PlayerDeath)
    assert len(deaths) == 1
    assert deaths[0].template is None
    assert deaths[0].message == "Hypixelite was disintegrated by a Void Reaver"
    assert deaths[0].player.name == "Hypixelite"
    assert pipeline.stats["deaths_tier2"] == 1


def test_a_vanilla_death_is_matched_by_the_parser_not_the_fallback(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    feed(pipeline, [JOIN, VANILLA_DEATH])

    death = sink.of(PlayerDeath)[0]
    assert death.template == "%1$s was blown up by %2$s"
    assert pipeline.stats["deaths_tier2"] == 0


def test_an_unmatched_line_from_an_offline_player_stays_a_console_log(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    # This is the whole reason the fallback lives in the pipeline: the parser cannot know who is
    # online, and without that fact "Hypixelite was disintegrated by a Void Reaver" is a line.
    feed(pipeline, [PLUGIN_DEATH])

    assert sink.names == ["ConsoleLog"]


def test_console_chatter_is_never_upgraded_to_a_death(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    feed(pipeline, [JOIN, CONSOLE_NOISE])

    assert sink.of(PlayerDeath) == []


def test_a_player_named_after_console_chatter_fabricates_no_deaths(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    """**The offline-mode name collision.**

    ``online-mode=false`` accepts any name without authentication, and ``Saving`` is six legal
    characters. The two save lines below are verbatim from
    ``tests/fixtures/logs/archives/2026-07-25-9.log`` and are emitted on every autosave - roughly
    every five minutes while anybody is playing - plus on every ``/save-all`` and every shutdown.
    With only an online-name test in front of the fallback, joining as ``Saving`` posts three
    fabricated deaths to Discord per autosave and corrupts the session summary's death count.

    ``Preparing``, ``Time``, ``Flushing`` and ``Closing`` are the other viable collisions in the
    same corpus.
    """
    feed(
        pipeline,
        [
            "[00:18:20] [Server thread/INFO]: Saving joined the game",
            "[00:18:28] [Server thread/INFO]: Saving players",
            "[00:18:29] [Server thread/INFO]: Saving chunks for level "
            "'ServerLevel[world]'/minecraft:overworld",
            "[00:18:30] [Server thread/INFO]: Saving worlds",
        ],
    )

    assert sink.of(PlayerDeath) == []
    assert pipeline.stats["deaths_tier2"] == 0
    assert pipeline.stats["deaths_tier2_rejected"] == 3
    assert len(sink.of(PlayerJoined)) == 1, "the join itself is real and must survive"


def test_a_rejected_death_candidate_is_still_counted_so_the_table_can_grow(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    """A genuine plugin death phrased in a way the vanilla table never uses is missed - by design,
    because a false death is a lie about a player - but it is counted and logged, which is the
    evidence for widening ``DEATH_TEMPLATES``."""
    feed(pipeline, [JOIN, "[03:01:02] [Server thread/INFO]: Hypixelite exploded into confetti"])

    assert sink.of(PlayerDeath) == []
    assert pipeline.stats["deaths_tier2_rejected"] == 1


def test_the_tier_2_death_carries_the_known_uuid(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    feed(pipeline, [UUID_LINE, JOIN, PLUGIN_DEATH])

    assert sink.of(PlayerDeath)[0].player.uuid == "403d4fb2-f466-3716-b9cb-a3e769bb40c9"


def test_a_recognised_enrichment_line_never_reaches_the_death_fallback(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    # "Hypixelite lost connection: ..." starts with an online player's name, so without the
    # earlier matchers it would be reported as a death.
    feed(pipeline, [JOIN, LOST_QUIT, COMMAND])

    assert sink.of(PlayerDeath) == []


# ------------------------------------------------------------------------------- ring buffer


def test_the_ring_buffer_keeps_the_last_forty_lines(pipeline: LogPipeline) -> None:
    feed(
        pipeline,
        [f"[00:00:{index:02d}] [Server thread/INFO]: line {index}" for index in range(60)],
    )

    tail = pipeline.tail()
    assert len(tail) == 40
    assert tail[-1].endswith("line 59")
    assert tail[0].endswith("line 20")


def test_the_ring_buffer_survives_a_crash_and_is_available_after_eof(
    manual_clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    """``ServerCrashed.tail`` is captured from here, not re-fetched.

    The stream EOF and the ``die`` event race each other; a post-hoc ``logs_tail()`` can lose to
    the container going away entirely, and the buffer is already populated for free.
    """
    pipeline = build(manual_clock, sink, roster, tail_lines=5)
    crash = [
        "[03:14:15] [Server thread/ERROR]: Exception ticking world",
        "\tat net.minecraft.server.MinecraftServer.tickServer(MinecraftServer.java:1)",
        "\tat net.minecraft.server.MinecraftServer.run(MinecraftServer.java:2)",
    ]
    feed(pipeline, crash)
    pipeline.on_eof()

    assert pipeline.tail() == tuple(crash)
    assert pipeline.tail(2) == tuple(crash[-2:])


def test_the_ring_buffer_holds_sanitised_text(pipeline: LogPipeline) -> None:
    # It is rendered into Discord as a crash tail. Escape bytes there would re-create, one layer
    # down, the exact defect this project exists to fix.
    feed(pipeline, [JOIN_ANSI])

    assert "\x1b" not in pipeline.tail()[0]


def test_reset_does_not_discard_the_crash_tail(pipeline: LogPipeline) -> None:
    feed(pipeline, [JOIN])
    pipeline.reset()

    assert len(pipeline.tail()) == 1


def test_tail_of_zero_is_empty(pipeline: LogPipeline) -> None:
    feed(pipeline, [JOIN])

    assert pipeline.tail(0) == ()


# ------------------------------------------------------------------------------ rate limiting


async def test_the_rate_limiter_coalesces_overflow_into_one_notice(
    manual_clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    pipeline = build(manual_clock, sink, roster, rate_per_second=1.0, rate_burst=3.0)
    noise = [f"[00:00:00] [Server thread/INFO]: chatter {index}" for index in range(8)]

    feed(pipeline, noise)

    assert len(sink.events) == 3
    assert pipeline.pending_suppressed == 5
    assert pipeline.stats["suppressed"] == 5

    # One token refills, and the very next surviving line is preceded by a single notice for the
    # whole burst - not five notices, and not silence.
    await manual_clock.advance(1.0)
    feed(pipeline, ["[00:00:01] [Server thread/INFO]: chatter 8"])

    messages = [event.message for event in sink.of(ConsoleLog)]
    assert messages[3] == "... 5 lines suppressed ..."
    assert messages[4] == "chatter 8"
    assert pipeline.pending_suppressed == 0
    assert pipeline.stats["suppression_notices"] == 1


def test_the_suppression_notice_is_internally_sourced(
    manual_clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    pipeline = build(manual_clock, sink, roster, rate_per_second=1.0, rate_burst=1.0)
    feed(pipeline, [CONSOLE_NOISE, CONSOLE_NOISE, CONSOLE_NOISE])
    pipeline.flush()

    notice = sink.of(ConsoleLog)[-1]
    assert notice.source is Source.INTERNAL
    assert notice.message == "... 2 lines suppressed ..."


def test_flush_emits_a_pending_notice_even_with_an_empty_bucket(
    manual_clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    pipeline = build(manual_clock, sink, roster, rate_per_second=1.0, rate_burst=1.0)
    feed(pipeline, [CONSOLE_NOISE, CONSOLE_NOISE])

    pipeline.on_eof()

    assert sink.of(ConsoleLog)[-1].message == "... 1 lines suppressed ..."


def test_flush_is_a_no_op_when_nothing_was_suppressed(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    feed(pipeline, [CONSOLE_NOISE])
    pipeline.flush()

    assert len(sink.events) == 1


def test_a_player_event_is_never_dropped_by_the_rate_limiter(
    manual_clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    """Only ConsoleLog is suppressible.

    Dropping a ``PlayerJoined`` corrupts the roster, which arms the idle timer against a populated
    server. The events that can flood are exactly the ones that can be dropped.
    """
    pipeline = build(manual_clock, sink, roster, rate_per_second=1.0, rate_burst=1.0)
    feed(pipeline, [CONSOLE_NOISE, CONSOLE_NOISE, CONSOLE_NOISE])
    assert pipeline.pending_suppressed == 2

    feed(pipeline, [JOIN, CHAT, LEAVE])

    assert sink.of(PlayerJoined) != []
    assert sink.of(ChatMessage) != []
    assert sink.of(PlayerLeft) != []


def test_the_token_bucket_refills_at_its_rate(manual_clock: ManualClock) -> None:
    bucket = TokenBucket(rate=2.0, burst=4.0, clock=manual_clock)
    for _ in range(4):
        assert bucket.take() is True
    assert bucket.take() is False
    assert bucket.tokens == pytest.approx(0.0)


async def test_the_token_bucket_never_exceeds_its_burst(manual_clock: ManualClock) -> None:
    bucket = TokenBucket(rate=100.0, burst=2.0, clock=manual_clock)
    await manual_clock.advance(60.0)

    assert bucket.tokens == pytest.approx(2.0)


@pytest.mark.parametrize(("rate", "burst"), [(0.0, 10.0), (-1.0, 10.0), (10.0, 0.0)])
def test_a_degenerate_token_bucket_is_rejected_at_construction(
    manual_clock: ManualClock,
    rate: float,
    burst: float,
) -> None:
    # A zero-rate bucket suppresses every line forever after the burst. Better to fail loudly at
    # startup than to discover it from an empty console channel three days later.
    with pytest.raises(ValueError, match="must be positive"):
        TokenBucket(rate=rate, burst=burst, clock=manual_clock)


# --------------------------------------------------------------------------------- robustness


def test_handle_line_uses_dockers_timestamp_not_the_clock(
    pipeline: LogPipeline,
    sink: RecordingSink,
    manual_clock: ManualClock,
) -> None:
    # Paper's "[13:24:37]" is time-only in Asia/Kolkata and is simply wrong for a backfilled line.
    docker_ts = manual_clock.now() - timedelta(hours=3)
    pipeline.handle_line(
        LogLine(text=JOIN, ts=docker_ts, received_at=manual_clock.now(), stream=Stream.STDOUT)
    )

    assert sink.of(PlayerJoined)[0].ts == docker_ts


def test_handle_line_falls_back_to_received_at_when_docker_gave_no_prefix(
    pipeline: LogPipeline,
    sink: RecordingSink,
    manual_clock: ManualClock,
) -> None:
    received = manual_clock.now()
    pipeline.handle_line(LogLine(text=JOIN, ts=None, received_at=received, stream=Stream.STDOUT))

    assert sink.of(PlayerJoined)[0].ts == received


def test_handle_lines_ingests_a_batch_in_order(
    pipeline: LogPipeline,
    sink: RecordingSink,
    manual_clock: ManualClock,
) -> None:
    now = manual_clock.now()
    pipeline.handle_lines(
        LogLine(text=text, ts=now, received_at=now, stream=Stream.STDOUT) for text in (JOIN, LEAVE)
    )

    assert sink.names == ["PlayerJoined", "PlayerLeft"]


def test_a_bracketed_line_from_an_online_player_is_not_a_death(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    # Redundant by construction - chat, /say, /me and the [Rcon: ...] echo are all matched earlier
    # - and kept because the failure it guards against is somebody's words being republished as a
    # death message.
    feed(pipeline, [JOIN, "[13:24:10] [Server thread/INFO]: (Hypixelite) says hello"])

    assert sink.of(PlayerDeath) == []


def test_a_single_word_line_is_not_a_death(pipeline: LogPipeline, sink: RecordingSink) -> None:
    feed(pipeline, [JOIN, "[13:24:10] [Server thread/INFO]: Flushing"])

    assert sink.of(PlayerDeath) == []


def test_pending_enrichment_facts_are_bounded(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    """A port scanner produces login lines nothing ever consumes.

    ``KittyScan (/176.65.148.158:58184) lost connection: Disconnected`` is a real archive line from
    somebody who authenticated and hung up without ever joining, so an uncapped map would grow for
    as long as the daemon runs. Eviction is not a correctness concern: a pending address that has
    survived 200 other logins was never going to be claimed.
    """
    feed(
        pipeline,
        [
            f"[13:24:37] [Server thread/INFO]: scanner{index}[/10.0.0.{index % 250}:5] "
            "logged in with entity id 1 at (0, 0, 0)"
            for index in range(200)
        ],
    )

    feed(
        pipeline,
        [
            "[13:24:37] [Server thread/INFO]: scanner199 joined the game",
            "[13:24:37] [Server thread/INFO]: scanner0 joined the game",
        ],
    )

    addresses = {event.player.name: event.address for event in sink.of(PlayerJoined)}
    assert addresses["scanner199"] is not None, "the most recent login must still be claimable"
    assert addresses["scanner0"] is None, "the oldest pending fact must have been evicted"


def test_a_stderr_line_keeps_its_stream(pipeline: LogPipeline, sink: RecordingSink) -> None:
    pipeline.handle_line(
        LogLine(
            text=CONSOLE_NOISE,
            ts=datetime(2026, 1, 1, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, tzinfo=UTC),
            stream=Stream.STDERR,
        )
    )

    assert sink.of(ConsoleLog)[0].stream is Stream.STDERR


def test_a_raising_adapter_never_takes_the_log_pump_down(
    manual_clock: ManualClock,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    """A raising line handler would kill the pump thread and stop the daemon seeing anything.

    The input is attacker-influenced - chat is a player-controlled string that reaches these
    regexes - so the callback counts and logs instead of propagating.
    """
    pipeline = LogPipeline(
        adapter=BrokenAdapter(),
        sink=sink,
        clock=manual_clock,
        server_id=SERVER_ID,
        roster=roster,
    )

    pipeline.handle_raw(JOIN)
    pipeline.handle_line(
        LogLine(text=JOIN, ts=None, received_at=manual_clock.now(), stream=Stream.STDOUT)
    )

    assert pipeline.stats["handler_errors"] == 2
    assert sink.events == []


def test_a_4kb_chat_message_is_published_intact(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    body = "@everyone `x` " * 300
    feed(pipeline, [f"[14:48:06] [Async Chat Thread - #0/INFO]: [Not Secure] <Hypixelite> {body}"])

    assert sink.of(ChatMessage)[0].message == body


def test_stats_count_what_flowed_through(pipeline: LogPipeline) -> None:
    feed(pipeline, [VERSION, LOGIN, JOIN, CHAT, LEAVE])

    stats = pipeline.stats
    assert stats["lines"] == 5
    assert stats["published"] == 5
    assert stats["addresses_stashed"] == 1
    assert stats["handler_errors"] == 0


def test_a_server_starting_event_still_reaches_the_bus(
    pipeline: LogPipeline,
    sink: RecordingSink,
) -> None:
    feed(pipeline, [VERSION])

    starting = sink.of(ServerStarting)
    assert len(starting) == 1
    assert starting[0].version == "26.2"
