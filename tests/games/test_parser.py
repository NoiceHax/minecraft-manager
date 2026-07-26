"""The parser: purity, the three named regressions, adversarial input, and the real corpus.

The three regression tests at the top of this file are why the project exists. Each names the
defect in ``~/homelab/scripts/minecraft-discord-bridge.py`` that it pins down.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from mcmanager.core.events import (
    ChatMessage,
    ConsoleLog,
    Event,
    PlayerAdvancement,
    PlayerDeath,
    PlayerJoined,
    PlayerLeft,
    ServerReady,
    ServerStarting,
    ServerStopping,
)
from mcmanager.core.types import (
    AdvancementKind,
    ChatKind,
    LeaveReason,
    LineOrigin,
    ReadySignal,
    Source,
    Stream,
)
from mcmanager.games.minecraft.parser import parse, scan
from mcmanager.games.minecraft.patterns import (
    MATCH_ORDER,
    match_command,
    match_kick,
    match_login,
    match_lost_connection,
    match_uuid,
)

if TYPE_CHECKING:
    from collections.abc import Callable

ESC = "\x1b"
TS = datetime(2026, 7, 25, 17, 11, 33, tzinfo=UTC)


def p(raw: str) -> Event:
    """Parse one line with the fixed test timestamp."""
    return parse(raw, ts=TS, server_id="minecraft", stream=Stream.STDOUT)


# =====================================================================================
# The three regressions. Each one is a defect in the script this project replaces.
# =====================================================================================


def test_regression_1_join_captures_only_the_name_not_the_whole_prefix() -> None:
    """OLD BUG: unanchored greedy capture.

    ``~/homelab/scripts/minecraft-discord-bridge.py`` did::

        re.compile(r"(.+) joined the game").search(line)

    Against the line below, ``.+`` is greedy, the pattern is unanchored and ``search`` happily
    starts at column zero, so the captured "player name" was
    ``[22:41:23] [Server thread/INFO]: Steve`` and that is what went to Discord.

    Two independent defences make it structurally impossible here: the prefix is removed before
    any pattern runs, and the name group is ``[A-Za-z0-9_]{1,16}``, which cannot contain ``[``,
    ``]``, ``:`` or a space.
    """
    event = p("[22:41:23] [Server thread/INFO]: Steve joined the game")

    assert isinstance(event, PlayerJoined)
    assert event.player.name == "Steve"
    assert "[" not in event.player.name
    assert ":" not in event.player.name
    assert "Server thread" not in event.player.name
    assert "22:41:23" not in event.player.name


def test_regression_2_ansi_coloured_name_is_clean() -> None:
    """OLD BUG: no ANSI stripping.

    The bridge script matched against the raw line, so on a coloured join the captured name was
    ``\\x1b[93mHypixelite`` and Discord rendered the escape bytes.

    The fixture below is built from explicit ``\\x1b`` bytes, not from a printable ``^[``: a test
    written with the pretty form asserts nothing about the real file. This is the exact byte
    sequence in ``tests/fixtures/logs/archives/2026-07-23-22.log``.
    """
    raw = f"[13:24:37] [Server thread/INFO]: {ESC}[93mHypixelite joined the game{ESC}[0m"
    event = p(raw)

    assert isinstance(event, PlayerJoined)
    assert event.player.name == "Hypixelite"
    assert ESC not in event.player.name
    assert "\x1b" not in (event.raw or "")
    assert "93m" not in event.player.name


def test_regression_3_chat_cannot_forge_a_join() -> None:
    """OLD BUG: substring dispatch.

    The bridge script did ``if "joined the game" in line``, so the chat message below produced a
    join notification for ``Alex``. Any player could announce anyone's arrival, or their
    departure, by typing it.

    Fixed by two things together: CHAT is matched before JOIN in the fixed order, and JOIN is
    anchored to the payload at both ends so it can never see inside a chat message.
    """
    event = p("[14:48:06] [Async Chat Thread - #0/INFO]: [Not Secure] <Steve> Alex joined the game")

    assert isinstance(event, ChatMessage)
    assert not isinstance(event, PlayerJoined)
    assert event.player.name == "Steve"
    assert event.message == "Alex joined the game"
    assert event.kind is ChatKind.CHAT


def test_regression_3_chat_cannot_forge_a_leave_or_a_death_either() -> None:
    """The same defect, exercised across the other event families it also broke."""
    for payload, expected_message in [
        ("<Steve> Alex left the game", "Alex left the game"),
        ("<Steve> Alex was slain by Zombie", "Alex was slain by Zombie"),
        (
            "<Steve> Alex has made the advancement [Stone Age]",
            "Alex has made the advancement [Stone Age]",
        ),
        ('<Steve> Done (32.521s)! For help, type "help"', 'Done (32.521s)! For help, type "help"'),
        ("<Steve> [Rcon: Stopping the server]", "[Rcon: Stopping the server]"),
    ]:
        event = p(f"[14:48:06] [Async Chat Thread - #0/INFO]: [Not Secure] {payload}")
        assert isinstance(event, ChatMessage), payload
        assert event.player.name == "Steve"
        assert event.message == expected_message


# =====================================================================================
# The purity contract
# =====================================================================================


@pytest.mark.parametrize(
    "hostile",
    [
        pytest.param("", id="empty"),
        pytest.param(" ", id="space"),
        pytest.param("\x00", id="nul"),
        pytest.param("\x00" * 100, id="many-nul"),
        pytest.param("\ud800", id="lone-high-surrogate"),
        pytest.param("\udfff\udfff", id="lone-low-surrogates"),
        pytest.param(
            b"\xff\xfe\x80".decode("utf-8", "surrogateescape"), id="invalid-utf8-surrogateescape"
        ),
        pytest.param(
            "[13:24:37] [Server thread/INFO]: \x00Steve\ud800 joined the game",
            id="join-line-with-nul-and-surrogate",
        ),
        pytest.param(f"{ESC}[", id="truncated-csi"),
        pytest.param(f"{ESC}[38;2;", id="truncated-truecolor"),
        pytest.param("[" * 10_000, id="brackets"),
        pytest.param("]" * 10_000, id="close-brackets"),
        pytest.param("<" * 10_000, id="chat-markers"),
        # A 10MB line with no newline. Explicit id: pytest exports the test id through
        # PYTEST_CURRENT_TEST, and Windows caps an environment variable at 32,767 characters.
        pytest.param(
            "[13:24:37] [Server thread/INFO]: " + "a" * 10_000_000, id="ten-megabyte-payload"
        ),
        pytest.param("\t" * 5000, id="tabs"),
        pytest.param("§" * 5000, id="section-signs"),
    ],
)
def test_parse_is_total(hostile: str) -> None:
    """It always returns an Event, never None, never raises.

    A parser that can throw takes the log pipeline down with it, and the input is
    attacker-influenced: chat is a player-controlled string that reaches these regexes.
    """
    event = parse(hostile, ts=TS, server_id="minecraft", stream=Stream.STDOUT)
    assert isinstance(event, Event)


def test_parse_never_reads_a_clock() -> None:
    """``ts`` is injected, and nothing else is allowed to decide it.

    The line below carries ``[13:24:37]``. The event carries the timestamp the caller passed, and
    the clock hint never becomes authoritative: it has no date, it is in ``Asia/Kolkata``, and it
    is meaningless for a backfilled line.
    """
    event = p("[13:24:37] [Server thread/INFO]: Steve joined the game")
    assert event.ts == TS
    assert event.ts.tzinfo is not None


def test_parse_is_deterministic() -> None:
    raw = f"[13:24:37] [Server thread/INFO]: {ESC}[93mHypixelite joined the game{ESC}[0m"
    assert p(raw) == p(raw)


def test_the_implemented_match_order_matches_the_documented_one() -> None:
    """Guards against the docstring and the code drifting apart."""
    assert MATCH_ORDER == (
        "READY",
        "VERSION",
        "STOPPING",
        "CHAT",
        "EMOTE",
        "SAY",
        "RCON_ECHO",
        "JOIN",
        "LEAVE",
        "LOGIN",
        "LOST_CONN",
        "KICKED",
        "COMMAND",
        "UUID",
        "ADVANCEMENT",
        "DEATHS",
    )
    assert MATCH_ORDER.index("CHAT") < MATCH_ORDER.index("JOIN")
    assert MATCH_ORDER.index("SAY") < MATCH_ORDER.index("RCON_ECHO")


# =====================================================================================
# The thread guard
# =====================================================================================


@pytest.mark.parametrize(
    "raw",
    [
        # Right payload, wrong thread.
        "[13:24:37] [ServerMain/INFO]: Steve joined the game",
        "[13:24:37] [Worker-Main-1/INFO]: Steve joined the game",
        "[13:24:37] [RCON Listener #1/INFO]: Steve joined the game",
        # Right thread, wrong level. "moved too quickly" is a real WARN line starting with a name.
        "[13:24:37] [Server thread/WARN]: Steve joined the game",
        "[13:24:37] [Server thread/ERROR]: Steve joined the game",
        # Not a server line at all.
        "\tat Steve joined the game",
        "Steve joined the game",
    ],
)
def test_a_join_payload_on_the_wrong_thread_or_level_is_console_output(raw: str) -> None:
    assert isinstance(p(raw), ConsoleLog)


def test_chat_is_accepted_from_the_async_chat_pool() -> None:
    """Verbatim archive line. The blanket ``Server thread`` guard would have dropped all chat."""
    event = p("[14:48:06] [Async Chat Thread - #0/INFO]: [Not Secure] <bharath_720> yo")
    assert isinstance(event, ChatMessage)
    assert event.player.name == "bharath_720"
    assert event.message == "yo"


def test_a_join_payload_on_the_chat_thread_is_not_a_join() -> None:
    """Chat threads carry chat. Nothing else about a player comes from them."""
    event = p("[14:48:06] [Async Chat Thread - #0/INFO]: Steve joined the game")
    assert isinstance(event, ConsoleLog)


# =====================================================================================
# Server lifecycle
# =====================================================================================


def test_ready() -> None:
    event = p('[22:41:47] [Server thread/INFO]: Done (14.117s)! For help, type "help"')
    assert isinstance(event, ServerReady)
    assert event.startup_seconds == pytest.approx(14.117)
    assert event.detected_by is ReadySignal.LOG_DONE
    assert event.source is Source.LOG


def test_ready_tolerates_a_locale_thousands_separator() -> None:
    """log4j2 renders durations with the JVM's locale.

    ``Finished initialising converters for DataConverter in 1,044.8ms`` is in the real archives,
    so a slow start really can print ``Done (1,032.5s)!``.
    """
    event = p('[22:41:47] [Server thread/INFO]: Done (1,032.5s)! For help, type "help"')
    assert isinstance(event, ServerReady)
    assert event.startup_seconds == pytest.approx(1032.5)


def test_version() -> None:
    event = p("[22:41:33] [Server thread/INFO]: Starting minecraft server version 26.2")
    assert isinstance(event, ServerStarting)
    assert event.version == "26.2"


@pytest.mark.parametrize(
    "payload",
    ["[Rcon: Stopping the server]", "Stopping the server", "Stopping server"],
)
def test_stopping(payload: str) -> None:
    event = p(f"[00:18:28] [Server thread/INFO]: {payload}")
    assert isinstance(event, ServerStopping)
    assert event.reason == payload


def test_stopping_with_ansi_around_it() -> None:
    """The real archive spelling: italic + grey."""
    raw = f"[00:18:28] [Server thread/INFO]: {ESC}[3m{ESC}[37m[Rcon: Stopping the server]{ESC}[0m"
    assert isinstance(p(raw), ServerStopping)


def test_a_non_stopping_rcon_echo_is_console_output() -> None:
    event = p(f"[12:48:26] [Server thread/INFO]: {ESC}[3m{ESC}[37m[Rcon: Saved the game]{ESC}[0m")
    assert isinstance(event, ConsoleLog)
    assert event.message == "[Rcon: Saved the game]"


# =====================================================================================
# Player lifecycle
# =====================================================================================


def test_join_and_leave() -> None:
    joined = p(f"[13:24:37] [Server thread/INFO]: {ESC}[93mHypixelite joined the game{ESC}[0m")
    left = p(f"[13:25:53] [Server thread/INFO]: {ESC}[93mHypixelite left the game{ESC}[0m")
    assert isinstance(joined, PlayerJoined)
    assert isinstance(left, PlayerLeft)
    assert joined.player.name == left.player.name == "Hypixelite"


def test_leave_reason_is_unknown_in_the_parser() -> None:
    """The reason arrives on a different line, and correlating lines is state.

    The parser is stateless by contract; ``services/log_pipeline.py`` stitches the two together
    using :func:`~mcmanager.games.minecraft.patterns.match_lost_connection`.
    """
    left = p("[13:25:53] [Server thread/INFO]: Hypixelite left the game")
    assert isinstance(left, PlayerLeft)
    assert left.reason is LeaveReason.UNKNOWN


def test_advancement_kinds() -> None:
    for payload, kind, title in [
        (
            "Hypixelite has made the advancement [Stone Age]",
            AdvancementKind.ADVANCEMENT,
            "Stone Age",
        ),
        (
            "Hypixelite has reached the goal [Adventuring Time]",
            AdvancementKind.GOAL,
            "Adventuring Time",
        ),
        (
            "Hypixelite has completed the challenge [Beaconator]",
            AdvancementKind.CHALLENGE,
            "Beaconator",
        ),
    ]:
        event = p(f"[13:19:07] [Server thread/INFO]: {payload}")
        assert isinstance(event, PlayerAdvancement)
        assert event.kind is kind
        assert event.title == title


def test_advancement_with_ansi_around_the_title() -> None:
    """Verbatim archive spelling: only the bracketed title is coloured."""
    raw = (
        f"[13:19:07] [Server thread/INFO]: "
        f"Hypixelite has made the advancement {ESC}[92m[Stone Age]{ESC}[0m"
    )
    event = p(raw)
    assert isinstance(event, PlayerAdvancement)
    assert event.title == "Stone Age"


def test_advancement_title_containing_a_bracket() -> None:
    event = p("[13:19:07] [Server thread/INFO]: Steve has made the advancement [A [Weird] Name]")
    assert isinstance(event, PlayerAdvancement)
    assert event.title == "A [Weird] Name"


def test_death() -> None:
    event = p("[03:01:01] [Server thread/INFO]: Hypixelite was blown up by Creeper")
    assert isinstance(event, PlayerDeath)
    assert event.player.name == "Hypixelite"
    assert event.killer == "Creeper"
    assert event.template == "%1$s was blown up by %2$s"
    assert event.message == "Hypixelite was blown up by Creeper"


# =====================================================================================
# Chat family
# =====================================================================================


def test_chat_emote_say_and_rcon() -> None:
    cases = [
        (
            "[14:48:06] [Async Chat Thread - #0/INFO]: [Not Secure] <Steve> hello",
            ChatKind.CHAT,
            "Steve",
            "hello",
        ),
        (
            "[14:48:06] [Server thread/INFO]: * Steve waves",
            ChatKind.EMOTE,
            "Steve",
            "waves",
        ),
        (
            "[14:48:06] [Server thread/INFO]: [Steve] an announcement",
            ChatKind.SAY,
            "Steve",
            "an announcement",
        ),
        (
            "[14:48:06] [Server thread/INFO]: [Not Secure] [Rcon] Hello from HCP",
            ChatKind.RCON,
            "Rcon",
            "Hello from HCP",
        ),
    ]
    for raw, kind, name, message in cases:
        event = p(raw)
        assert isinstance(event, ChatMessage), raw
        assert event.kind is kind
        assert event.player.name == name
        assert event.message == message


def test_say_does_not_swallow_the_rcon_broadcast() -> None:
    """``SAY`` sits before ``RCON_ECHO`` in the fixed order; a lookahead is what keeps it honest.

    Without it, ``[Rcon] Hello`` is attributed to a player named ``Rcon`` with kind ``SAY``.
    """
    event = p("[14:48:06] [Server thread/INFO]: [Rcon] Hello from HCP")
    assert isinstance(event, ChatMessage)
    assert event.kind is ChatKind.RCON


def test_chat_containing_more_chat_markers_is_one_message() -> None:
    """``<Steve> hello <Alex> hi`` is one message from Steve, not two."""
    event = p("[14:48:06] [Async Chat Thread - #0/INFO]: [Not Secure] <Steve> hello <Alex> hi")
    assert isinstance(event, ChatMessage)
    assert event.player.name == "Steve"
    assert event.message == "hello <Alex> hi"


def test_chat_carries_hostile_payloads_verbatim_for_the_presenter_to_escape() -> None:
    """Escaping is the presenter's job; mangling the message here would be a silent data loss.

    ``@everyone``, backticks, markdown and 4KB of text are all expected inputs. What must hold at
    *this* layer is that the parser attributes them to the right player and does not choke.
    """
    payloads = [
        "@everyone get in here",
        "@here",
        "`rm -rf /` ```py\nprint(1)\n```",
        "**bold** _italic_ ||spoiler|| [link](http://x)",
        "<@1234567890> <#98765>",
        "A" * 4096,
        "\\" * 200,
        "https://example.com/?a=1&b=2#frag",
    ]
    for payload in payloads:
        event = p(f"[14:48:06] [Async Chat Thread - #0/INFO]: [Not Secure] <Steve> {payload}")
        assert isinstance(event, ChatMessage), payload[:40]
        assert event.player.name == "Steve"
        assert event.message == payload


def test_chat_from_a_player_named_after_a_command() -> None:
    """A player can call themselves ``joined`` and the anchors still hold."""
    event = p("[14:48:06] [Async Chat Thread - #0/INFO]: [Not Secure] <joined> joined the game")
    assert isinstance(event, ChatMessage)
    assert event.player.name == "joined"

    real = p("[14:48:06] [Server thread/INFO]: joined joined the game")
    assert isinstance(real, PlayerJoined)
    assert real.player.name == "joined"


# =====================================================================================
# Recognised lines that enrich a neighbour rather than becoming an event
# =====================================================================================


def test_login_line_becomes_console_output_and_is_matchable_by_the_pipeline() -> None:
    """It carries a player IP, which is why the pipeline wants it and Discord must never see it."""
    raw = (
        "[13:24:37] [Server thread/INFO]: Hypixelite[/115.99.245.156:49237] logged in "
        "with entity id 53 at ([minecraft:overworld]-16.5, 76.0, 383.5)"
    )
    event = p(raw)
    assert isinstance(event, ConsoleLog)

    found = match_login(event.message)
    assert found is not None
    assert found.name == "Hypixelite"
    assert found.address == "115.99.245.156:49237"
    assert found.ip == "115.99.245.156"
    assert found.entity_id == 53


@pytest.mark.parametrize(
    ("payload", "name", "reason"),
    [
        # All three spellings are real, and two of them were found by the canary, not by hand.
        ("Hypixelite lost connection: Disconnected", "Hypixelite", LeaveReason.QUIT),
        ("bharath_720 lost connection: Server closed", "bharath_720", LeaveReason.SERVER_CLOSED),
        (
            "Hypixelite (403d4fb2-f466-3716-b9cb-a3e769bb40c9) lost connection: Disconnected",
            "Hypixelite",
            LeaveReason.QUIT,
        ),
        (
            "KittyScan (/176.65.148.158:58184) lost connection: Disconnected",
            "KittyScan",
            LeaveReason.QUIT,
        ),
        ("Steve lost connection: Timed out", "Steve", LeaveReason.TIMED_OUT),
        ("Steve lost connection: Something novel", "Steve", LeaveReason.UNKNOWN),
    ],
)
def test_lost_connection_is_matchable_and_classified(
    payload: str, name: str, reason: LeaveReason
) -> None:
    event = p(f"[13:25:53] [Server thread/INFO]: {payload}")
    assert isinstance(event, ConsoleLog)
    found = match_lost_connection(event.message)
    assert found is not None
    assert found.name == name
    assert found.reason is reason


def test_server_closed_is_never_a_voluntary_quit() -> None:
    """The one classification that has to be right.

    ``bharath_720 lost connection: Server closed`` means the player was dropped by a shutdown.
    Counting it as a quit corrupts every session summary.
    """
    found = match_lost_connection("bharath_720 lost connection: Server closed")
    assert found is not None
    assert found.reason is LeaveReason.SERVER_CLOSED
    assert found.reason is not LeaveReason.QUIT


def test_kick_line() -> None:
    event = p("[17:39:25] [Server thread/INFO]: bharath_720 was kicked due to keepalive timeout!")
    assert isinstance(event, ConsoleLog)
    found = match_kick(event.message)
    assert found is not None
    assert found.name == "bharath_720"
    assert found.reason is LeaveReason.KICKED


def test_kick_line_is_not_read_as_a_death() -> None:
    assert not isinstance(
        p("[17:39:25] [Server thread/INFO]: bharath_720 was kicked due to keepalive timeout!"),
        PlayerDeath,
    )


def test_command_line() -> None:
    event = p(
        "[13:24:37] [Server thread/INFO]: Hypixelite issued server command: /gamemode creative"
    )
    assert isinstance(event, ConsoleLog)
    found = match_command(event.message)
    assert found is not None
    assert found.name == "Hypixelite"
    assert found.command == "/gamemode creative"


def test_uuid_line() -> None:
    """Offline v3 UUID. Stable per name, not a Mojang UUID; never send it to a Mojang API."""
    event = p(
        "[00:49:05] [User Authenticator #0/INFO]: "
        "UUID of player Hypixelite is 403d4fb2-f466-3716-b9cb-a3e769bb40c9"
    )
    assert isinstance(event, ConsoleLog)
    found = match_uuid(event.message)
    assert found is not None
    assert found.name == "Hypixelite"
    assert found.uuid == "403d4fb2-f466-3716-b9cb-a3e769bb40c9"


def test_a_command_echo_containing_a_death_message_is_not_a_death() -> None:
    """``/say Notch_fell fell from a high place`` echoes on Server thread before it broadcasts."""
    event = p(
        "[13:24:37] [Server thread/INFO]: Steve issued server command: "
        "/say Notch_fell fell from a high place"
    )
    assert isinstance(event, ConsoleLog)


# =====================================================================================
# ConsoleLog shape
# =====================================================================================


def test_console_log_carries_the_grammar_it_came_from() -> None:
    event = p("[13:24:37] [Server thread/INFO]: Preparing spawn area: 2%")
    assert isinstance(event, ConsoleLog)
    assert event.origin is LineOrigin.SERVER
    assert event.thread == "Server thread"
    assert event.level == "INFO"
    assert event.message == "Preparing spawn area: 2%"


def test_wrapper_and_continuation_lines_keep_their_origin() -> None:
    wrapper = p(
        "2026-07-25T22:58:18.720+0530\tINFO\tmc-server-runner\tgracefully stopping server..."
    )
    assert isinstance(wrapper, ConsoleLog)
    assert wrapper.origin is LineOrigin.WRAPPER
    assert wrapper.message == "gracefully stopping server..."

    frame = p("\tat net.minecraft.server.MinecraftServer.runServer(MinecraftServer.java:1385)")
    assert isinstance(frame, ConsoleLog)
    assert frame.origin is LineOrigin.CONTINUATION


def test_every_event_carries_raw_and_it_is_ansi_free() -> None:
    """``raw`` is what the Discord console relay renders, so it must not carry escape bytes.

    Deliberate deviation: the plan calls ``raw`` "the original line". Keeping the escape bytes
    there would re-create, one layer down, the exact defect this project exists to fix.
    ``mcmanager logs --raw`` bypasses the parser and still shows the true original.
    """
    for raw in [
        f"[13:24:37] [Server thread/INFO]: {ESC}[93mHypixelite joined the game{ESC}[0m",
        f"[14:48:06] [Async Chat Thread - #0/INFO]: [Not Secure] <Steve> {ESC}[31mred{ESC}[0m",
        f"[13:24:37] [Server thread/INFO]: {ESC}[93mnonsense",
    ]:
        event = p(raw)
        assert event.raw is not None
        assert ESC not in event.raw


def test_the_stream_is_carried_onto_console_logs() -> None:
    event = parse("boom", ts=TS, server_id="minecraft", stream=Stream.STDERR)
    assert isinstance(event, ConsoleLog)
    assert event.stream is Stream.STDERR


# =====================================================================================
# The real corpus
# =====================================================================================

MAX_UNRECOGNISED_RATIO = 0.01
"""Measured **0.0000** over all 3,419 committed archive lines on 2026-07-26.

The threshold is 1%, not the measured zero, so that a single genuinely novel line does not turn
CI red. Anything above it means Paper changed a message and events are silently disappearing.
"""


def test_unrecognised_line_ratio_over_every_real_archive(all_archive_lines: list[str]) -> None:
    """The canary. It fails **with a sample** so the failure names what stopped matching.

    Note what the denominator is, because the obvious choice was wrong. Measuring unrecognised
    lines against *all* ``Server thread/INFO`` lines gives 0.895 on this corpus: nine in ten are
    legitimately console chatter (``Preparing spawn area: 2%``, ``Saving chunks for level ...``)
    and always will be. A canary that idles at 0.895 cannot detect nineteen join lines going
    quiet. So the population is lines that begin with a player the same corpus proved real.

    This test has already paid for itself: it found two log spellings that were not in the plan's
    sampled lines - ``Hypixelite (uuid) lost connection: ...`` and
    ``KittyScan (/ip:port) lost connection: ...`` - plus the ``issued server command`` family.
    """
    stats = scan(all_archive_lines)

    assert stats.total > 3000, "the archive fixtures are missing"
    assert stats.player_lines > 100, "no player lines found: the fixtures or the guard are broken"
    assert stats.unrecognised_ratio <= MAX_UNRECOGNISED_RATIO, (
        f"unrecognised player-line ratio {stats.unrecognised_ratio:.4f} exceeds "
        f"{MAX_UNRECOGNISED_RATIO}. Paper's log format has probably drifted. "
        f"{stats.unrecognised_player_lines} of {stats.player_lines} lines about "
        f"{sorted(stats.known_players)} were not attributed. Samples:\n  "
        + "\n  ".join(repr(sample) for sample in stats.samples)
    )


def test_event_distribution_over_the_real_archives(all_archive_lines: list[str]) -> None:
    """The second canary, and the more direct one.

    If Paper renames ``joined the game``, the ratio above moves by less than one percent but
    ``PlayerJoined`` drops to zero. Pinned to floors rather than exact counts so that adding a
    fixture is not a test change.
    """
    stats = scan(all_archive_lines)
    floors = {
        "PlayerJoined": 19,
        "PlayerLeft": 19,
        "ChatMessage": 300,
        "PlayerDeath": 13,
        "PlayerAdvancement": 25,
        "ServerReady": 31,
        "ServerStarting": 34,
        "ServerStopping": 54,
    }
    for name, floor in floors.items():
        assert stats.by_event.get(name, 0) >= floor, (
            f"{name}: expected at least {floor} over the real archives, got "
            f"{stats.by_event.get(name, 0)}. Distribution: {stats.by_event}"
        )


def test_no_player_name_in_the_real_corpus_is_ever_polluted(all_archive_lines: list[str]) -> None:
    """The regression, asserted over 3,419 real lines rather than one hand-written one."""
    for raw in all_archive_lines:
        event = p(raw)
        name = getattr(getattr(event, "player", None), "name", None)
        if name is None:
            continue
        assert ESC not in name
        assert "[" not in name
        assert "]" not in name
        assert ":" not in name
        assert " " not in name
        assert name == name.strip()
        assert len(name) <= 16


def test_parsing_the_whole_corpus_raises_nothing(all_archive_lines: list[str]) -> None:
    for raw in all_archive_lines:
        assert isinstance(p(raw), Event)


# =====================================================================================
# Golden files
# =====================================================================================


@pytest.mark.parametrize(
    "name",
    ["run_normal_session", "run_deaths_session", "docker_stream"],
)
def test_golden(
    name: str,
    load_fixture: Callable[[str], list[str]],
    check_golden: Callable[[str, list[str]], None],
) -> None:
    """Byte-for-byte comparison against committed JSON.

    Plain sorted JSON rather than ``syrupy``: a snapshot library that writes its own format is one
    more thing to be wrong about, and ``git diff`` on a JSON file is already the review tool.
    Regenerate with ``MCMANAGER_UPDATE_GOLDENS=1 pytest tests/games`` and read the diff.
    """
    check_golden(name, load_fixture(f"{name}.log"))
