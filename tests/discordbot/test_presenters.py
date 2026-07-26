"""Tests for the one module where attacker-controlled text becomes a Discord message.

The server runs ``online-mode=false``. A player picks their own name with no authentication, and
that name is embedded verbatim in join, leave, death and advancement messages. Chat is worse still.
So the adversarial cases here are not hypothetical: they are the reachable ones.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mcmanager.core.events import (
    EVENT_TYPES,
    ChatMessage,
    CommandIssued,
    ConsoleLog,
    Event,
    PlayerDeath,
    PlayerJoined,
    ServerReady,
)
from mcmanager.core.types import ChatKind, ControlAction, PlayerRef, ReadySignal, Source
from mcmanager.discordbot.presenters import (
    MAX_MESSAGE_LENGTH,
    escape_markdown,
    neutralise_mentions,
    render_event,
    render_raw,
    sanitize,
)

TS = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)


def chat(message: str, *, name: str = "Steve") -> ChatMessage:
    return ChatMessage(
        ts=TS,
        server_id="minecraft",
        source=Source.LOG,
        raw=f"<{name}> {message}",
        player=PlayerRef(name=name),
        message=message,
        kind=ChatKind.CHAT,
    )


# ------------------------------------------------------------------------------- mentions


@pytest.mark.parametrize("payload", ["@everyone", "@here", "hey @everyone look"])
def test_mass_mentions_are_neutralised(payload: str) -> None:
    """A player types @everyone in chat. It must reach Discord inert."""
    out = render_event(chat(payload))
    assert out is not None
    # The literal sequence Discord parses must not survive: there is always a zero-width space
    # between the @ and the keyword.
    assert "@everyone" not in out
    assert "@here" not in out


def test_user_and_role_mentions_are_neutralised() -> None:
    out = render_event(chat("ping <@123456789> and <@&987654321> and <#555>"))
    assert out is not None
    assert "<@123456789>" not in out
    assert "<@&987654321>" not in out
    assert "<#555>" not in out


def test_neutralise_keeps_the_text_readable() -> None:
    """Inert, but a human still reads it as what the player typed."""
    assert neutralise_mentions("@everyone").replace("​", "") == "@everyone"


# ------------------------------------------------------------------------------- markdown


def test_backticks_and_fences_are_escaped() -> None:
    out = render_event(chat("```py\nimport os\n```"))
    assert out is not None
    assert "```" not in out


def test_markdown_cannot_restyle_the_message() -> None:
    assert escape_markdown("**bold**") == "\\*\\*bold\\*\\*"
    assert escape_markdown("[link](http://x)") == "\\[link\\]\\(http://x\\)"


# ------------------------------------------------------------------------------- robustness


def test_a_very_long_chat_message_is_truncated() -> None:
    out = render_event(chat("A" * 4096))
    assert out is not None
    assert len(out) <= MAX_MESSAGE_LENGTH


def test_null_bytes_and_control_characters_are_stripped() -> None:
    out = render_event(chat("hel\x00lo\x07 there\x1b[31m"))
    assert out is not None
    assert "\x00" not in out
    assert "\x07" not in out


def test_a_lone_surrogate_does_not_raise_and_can_be_encoded() -> None:
    """A lone surrogate would raise at JSON-encode time and take out the whole batch."""
    out = render_event(chat("bad \udcff end"))
    assert out is not None
    out.encode("utf-8")  # must not raise


def test_sanitize_keeps_newlines_and_tabs() -> None:
    assert sanitize("a\nb\tc") == "a\nb\tc"


# ------------------------------------------------------------------------------- privacy


def test_a_players_ip_address_is_never_rendered() -> None:
    """``PlayerJoined.address`` exists for idle logic and abuse investigation, not for chat."""
    event = PlayerJoined(
        ts=TS,
        server_id="minecraft",
        source=Source.LOG,
        raw="Hypixelite[/115.99.245.156:49237] logged in",
        player=PlayerRef(name="Hypixelite"),
        online_count=1,
        address="115.99.245.156:49237",
    )
    out = render_event(event)
    assert out is not None
    assert "115.99.245.156" not in out
    assert "49237" not in out


# ------------------------------------------------------------------------------- coverage


def test_every_event_type_renders_without_raising() -> None:
    """``render_event`` is total. A new event must not fall through to a base-class rendering.

    ``assert_never`` makes that a type error at build time; this makes it a test failure too, for
    the case where somebody adds an event and a matching case that returns the wrong shape.
    """
    unhandled: list[str] = []
    samples = _build_samples()
    for event_type in EVENT_TYPES:
        if event_type.__name__ not in samples:
            unhandled.append(event_type.__name__)
    assert not unhandled, f"no sample event for: {unhandled}"

    for name, event in samples.items():
        result = render_event(event)  # pyright: ignore[reportArgumentType]
        assert result is None or isinstance(result, str), name
        if isinstance(result, str):
            assert len(result) <= MAX_MESSAGE_LENGTH, name


def test_console_log_is_not_announced_in_the_events_channel() -> None:
    event = ConsoleLog(
        ts=TS,
        server_id="minecraft",
        source=Source.LOG,
        raw="[12:00:00] [Server thread/INFO]: Saving chunks",
        message="Saving chunks",
    )
    assert render_event(event) is None
    # ...but the console relay still gets it, which is the whole reason Event.raw exists.
    assert render_raw(event) is not None


def test_an_accepted_command_is_silent_but_a_refused_one_is_not() -> None:
    accepted = CommandIssued(
        ts=TS,
        server_id="minecraft",
        source=Source.DISCORD,
        action=ControlAction.STOP,
        actor="kunal",
        accepted=True,
    )
    refused = CommandIssued(
        ts=TS,
        server_id="minecraft",
        source=Source.DISCORD,
        action=ControlAction.STOP,
        actor="kunal",
        accepted=False,
        rejection="the server is already stopped",
    )
    assert render_event(accepted) is None
    out = render_event(refused)
    assert out is not None
    assert "refused" in out
    assert "already stopped" in out


def test_render_raw_breaks_a_fence_so_console_output_cannot_escape_the_block() -> None:
    event = ConsoleLog(
        ts=TS,
        server_id="minecraft",
        source=Source.LOG,
        raw="``` now I am outside the block",
        message="x",
    )
    out = render_raw(event)
    assert out is not None
    assert "```" not in out


def _build_samples() -> dict[str, Event]:
    """One sample of every event type, so the coverage test above is honest.

    Written out rather than built from a shared ``**kwargs`` dict: pyright cannot narrow a
    heterogeneous mapping through an unpack, and losing type checking on the very table that
    guards ``render_event``'s exhaustiveness would defeat the point of it.
    """
    from mcmanager.core import events as e
    from mcmanager.core.types import AdvancementKind, LeaveReason

    sid = "minecraft"
    src = Source.LOG
    steve = PlayerRef(name="Steve")
    return {
        "ServerStarting": e.ServerStarting(ts=TS, server_id=sid, source=src, raw="x"),
        "ServerReady": ServerReady(
            ts=TS, server_id=sid, source=src, raw="x", detected_by=ReadySignal.LOG_DONE
        ),
        "ServerStopping": e.ServerStopping(ts=TS, server_id=sid, source=src, raw="x"),
        "ServerStopped": e.ServerStopped(ts=TS, server_id=sid, source=src, raw="x", clean=True),
        "ServerCrashed": e.ServerCrashed(ts=TS, server_id=sid, source=src, raw="x", tail=("boom",)),
        "PlayerJoined": e.PlayerJoined(ts=TS, server_id=sid, source=src, raw="x", player=steve),
        "PlayerLeft": e.PlayerLeft(
            ts=TS, server_id=sid, source=src, raw="x", player=steve, reason=LeaveReason.QUIT
        ),
        "PlayerDeath": PlayerDeath(
            ts=TS,
            server_id=sid,
            source=src,
            raw="x",
            player=steve,
            message="Steve was blown up by Creeper",
        ),
        "PlayerAdvancement": e.PlayerAdvancement(
            ts=TS,
            server_id=sid,
            source=src,
            raw="x",
            player=steve,
            title="Stone Age",
            kind=AdvancementKind.ADVANCEMENT,
        ),
        "ChatMessage": chat("hi"),
        "ConsoleLog": e.ConsoleLog(ts=TS, server_id=sid, source=src, raw="x", message="x"),
        "IdleStarted": e.IdleStarted(
            ts=TS,
            server_id=sid,
            source=src,
            raw="x",
            deadline=TS,
            empty_since=TS,
            timeout_seconds=900.0,
        ),
        "IdleWarning": e.IdleWarning(
            ts=TS, server_id=sid, source=src, raw="x", remaining_seconds=120.0, deadline=TS
        ),
        "IdleCancelled": e.IdleCancelled(
            ts=TS, server_id=sid, source=src, raw="x", reason="player_joined"
        ),
        "IdleStopTriggered": e.IdleStopTriggered(
            ts=TS, server_id=sid, source=src, raw="x", idle_seconds=900.0
        ),
        "RuntimeUnavailable": e.RuntimeUnavailable(
            ts=TS, server_id=sid, source=src, raw="x", error="socket gone"
        ),
        "RuntimeRestored": e.RuntimeRestored(
            ts=TS, server_id=sid, source=src, raw="x", downtime_seconds=12.0
        ),
        "CommandIssued": CommandIssued(
            ts=TS,
            server_id=sid,
            source=src,
            raw="x",
            action=ControlAction.STOP,
            actor="kunal",
            accepted=True,
        ),
    }
