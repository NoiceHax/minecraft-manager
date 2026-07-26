"""The event hierarchy's structural guarantees.

These are cheap tests of properties everything else assumes: immutability, no ``__dict__``,
subclass grouping for bus matching, and the registry staying in step with the module.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from mcmanager.core import events as ev
from mcmanager.core.events import (
    EVENT_BY_NAME,
    EVENT_TYPES,
    ChatMessage,
    CommandIssued,
    ConsoleLog,
    Event,
    IdleEvent,
    PlayerEvent,
    PlayerJoined,
    RuntimeStatusEvent,
    ServerEvent,
    ServerReady,
)
from mcmanager.core.types import ChatKind, ControlAction, PlayerRef, ReadySignal, Source

TS = datetime(2026, 7, 25, 18, 30, tzinfo=UTC)


def _joined() -> PlayerJoined:
    return PlayerJoined(
        ts=TS,
        server_id="minecraft",
        source=Source.LOG,
        raw="[13:24:37] [Server thread/INFO]: Hypixelite joined the game",
        player=PlayerRef(name="Hypixelite"),
        online_count=1,
    )


def test_events_are_immutable() -> None:
    event = _joined()
    field_name = "online_count"

    with pytest.raises(FrozenInstanceError):
        setattr(event, field_name, 99)


def test_events_are_slotted() -> None:
    """No ``__dict__``: construction stays cheap on a path that runs once per log line."""
    assert not hasattr(_joined(), "__dict__")


def test_with_seq_returns_a_stamped_copy_and_leaves_the_original_alone() -> None:
    event = _joined()

    stamped = event.with_seq(42)

    assert stamped.seq == 42
    assert event.seq == 0
    assert stamped.player == event.player
    assert type(stamped) is PlayerJoined


def test_name_is_the_type_name() -> None:
    assert _joined().name == "PlayerJoined"


def test_category_bases_group_events_for_subscription() -> None:
    """The bus matches by type with subclass semantics, so these relationships are the API."""
    assert issubclass(PlayerJoined, PlayerEvent)
    assert issubclass(ev.PlayerLeft, PlayerEvent)
    assert issubclass(ev.PlayerDeath, PlayerEvent)
    assert issubclass(ev.PlayerAdvancement, PlayerEvent)

    assert issubclass(ServerReady, ServerEvent)
    assert issubclass(ev.ServerCrashed, ServerEvent)
    assert issubclass(ev.IdleStopTriggered, IdleEvent)
    assert issubclass(ev.RuntimeUnavailable, RuntimeStatusEvent)

    # Everything is an Event, which is what makes a wildcard console relay possible.
    for event_type in EVENT_TYPES:
        assert issubclass(event_type, Event)


def test_chat_and_console_are_not_player_events() -> None:
    """A subscriber asking "what did players do" must not be handed the chat firehose."""
    assert not issubclass(ChatMessage, PlayerEvent)
    assert not issubclass(ConsoleLog, PlayerEvent)
    assert not issubclass(ConsoleLog, ServerEvent)


def test_player_event_group_has_exactly_four_members() -> None:
    members = [t for t in EVENT_TYPES if issubclass(t, PlayerEvent)]
    assert len(members) == 4


def test_registry_covers_every_concrete_event_exactly_once() -> None:
    assert len(EVENT_TYPES) == len(set(EVENT_TYPES))
    assert set(EVENT_BY_NAME) == {t.__name__ for t in EVENT_TYPES}
    # The 15 specified events plus RuntimeUnavailable, RuntimeRestored and CommandIssued.
    assert len(EVENT_TYPES) == 18


def test_category_bases_are_not_in_the_registry() -> None:
    """``--type PlayerEvent`` is resolved by the bus's subclass matching, not by this table."""
    for base in (Event, ServerEvent, PlayerEvent, IdleEvent, RuntimeStatusEvent):
        assert base.__name__ not in EVENT_BY_NAME


def test_fields_that_exist_so_no_consumer_re_derives_anything() -> None:
    ready = ServerReady(
        ts=TS,
        server_id="minecraft",
        source=Source.LOG,
        raw='[13:24:37] [Server thread/INFO]: Done (32.521s)! For help, type "help"',
        startup_seconds=32.521,
        version="26.2",
        detected_by=ReadySignal.LOG_DONE,
    )
    stopped = ev.ServerStopped(
        ts=TS,
        server_id="minecraft",
        source=Source.RUNTIME,
        exit_code=0,
        clean=True,
        forced=False,
        uptime_seconds=3600.0,
    )
    crashed = ev.ServerCrashed(
        ts=TS,
        server_id="minecraft",
        source=Source.RUNTIME,
        exit_code=137,
        oom_killed=True,
        tail=("java.lang.OutOfMemoryError: Java heap space",),
    )

    assert ready.detected_by is ReadySignal.LOG_DONE
    assert ready.startup_seconds == 32.521
    assert stopped.clean is True
    assert crashed.tail[0].startswith("java.lang.OutOfMemoryError")


def test_a_chat_message_carries_the_kind_that_produced_it() -> None:
    message = ChatMessage(
        ts=TS,
        server_id="minecraft",
        source=Source.LOG,
        raw="[13:24:37] [Server thread/INFO]: <Bob> Alice joined the game",
        player=PlayerRef(name="Bob"),
        message="Alice joined the game",
        kind=ChatKind.CHAT,
    )

    assert message.kind is ChatKind.CHAT
    assert message.player.name == "Bob"


def test_command_issued_records_rejections_too() -> None:
    rejected = CommandIssued(
        ts=TS,
        server_id="minecraft",
        source=Source.DISCORD,
        action=ControlAction.STOP,
        actor="someone",
        via=Source.DISCORD,
        accepted=False,
        rejection="not an admin",
    )

    assert rejected.accepted is False
    assert rejected.action is ControlAction.STOP


def test_player_ref_stringifies_to_the_name() -> None:
    assert str(PlayerRef(name="Hypixelite", uuid="0d0c4f0e-0000-3000-8000-000000000000")) == (
        "Hypixelite"
    )
