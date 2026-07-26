"""The event wire format.

Two guarantees matter here and nothing else does:

1. **Every concrete event round-trips exactly.** ``event_from_dict(event_to_dict(e)) == e`` for a
   *fully populated* instance of all 18 types, so no field is silently dropped on the way to a
   session record or an SSE client.
2. **Coverage is checkable.** :data:`SAMPLES` is asserted to name every type in ``EVENT_TYPES``,
   and each payload is asserted to carry every dataclass field of its event. Adding a field to an
   existing event without teaching ``serde`` about it fails here; adding a whole new event without
   teaching ``serde`` about it fails pyright, at the ``assert_never``.

The instances below deliberately set *every* field to a non-default value. A sample that leans on
defaults would pass even if the encoder dropped that field and the decoder re-defaulted it.
"""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import UTC, datetime
from typing import Any

import pytest

from mcmanager.core.events import (
    EVENT_TYPES,
    ChatMessage,
    CommandIssued,
    ConsoleLog,
    Event,
    IdleCancelled,
    IdleStarted,
    IdleStopTriggered,
    IdleWarning,
    PlayerAdvancement,
    PlayerDeath,
    PlayerEvent,
    PlayerJoined,
    PlayerLeft,
    RuntimeRestored,
    RuntimeUnavailable,
    ServerCrashed,
    ServerReady,
    ServerStarting,
    ServerStopped,
    ServerStopping,
)
from mcmanager.core.serde import SerdeError, event_from_dict, event_to_dict, event_to_json
from mcmanager.core.types import (
    AdvancementKind,
    ChatKind,
    ControlAction,
    LeaveReason,
    LineOrigin,
    PlayerRef,
    ReadySignal,
    Source,
    Stream,
)

TS = datetime(2026, 7, 25, 22, 58, 20, 809000, tzinfo=UTC)
PLAYER = PlayerRef(name="Hypixelite", uuid="8f4a2b1c-0000-3000-8000-0123456789ab")

_BASE: dict[str, Any] = {
    "ts": TS,
    "server_id": "minecraft",
    "source": Source.LOG,
    "raw": "[13:24:37] [Server thread/INFO]: something happened",
    "seq": 4171,
}


SAMPLES: tuple[Event, ...] = (
    ServerStarting(**_BASE, version="26.2", container_id="c0ffee1234", requested_by="cli"),
    ServerReady(**_BASE, startup_seconds=32.521, version="26.2", detected_by=ReadySignal.LOG_DONE),
    ServerStopping(**_BASE, reason="idle timeout", requested_by="idle-manager", timeout_seconds=90),
    ServerStopped(**_BASE, exit_code=0, clean=True, forced=True, uptime_seconds=7231.5),
    ServerCrashed(**_BASE, exit_code=137, oom_killed=True, tail=("boom", "\tat net.minecraft")),
    PlayerJoined(
        **_BASE,
        player=PLAYER,
        online_count=3,
        address="115.99.245.156:49237",
        first_seen=True,
    ),
    PlayerLeft(
        **_BASE,
        player=PLAYER,
        reason=LeaveReason.SERVER_CLOSED,
        session_seconds=612.25,
        online_count=2,
    ),
    PlayerDeath(
        **_BASE,
        player=PLAYER,
        message="Hypixelite was blown up by Creeper",
        killer="Creeper",
        item="Sharpness Sword",
        template="%1$s was blown up by %2$s",
    ),
    PlayerAdvancement(**_BASE, player=PLAYER, title="Stone Age", kind=AdvancementKind.CHALLENGE),
    ChatMessage(**_BASE, player=PLAYER, message="<@everyone> `hi`", kind=ChatKind.EMOTE),
    ConsoleLog(
        **_BASE,
        message="Done (32.521s)!",
        level="WARN",
        thread="Server thread",
        origin=LineOrigin.WRAPPER,
        stream=Stream.STDERR,
    ),
    IdleStarted(
        **_BASE,
        deadline=datetime(2026, 7, 25, 23, 13, 20, tzinfo=UTC),
        empty_since=datetime(2026, 7, 25, 22, 58, 20, tzinfo=UTC),
        timeout_seconds=900.0,
        dry_run=True,
    ),
    IdleWarning(
        **_BASE,
        remaining_seconds=120.0,
        deadline=datetime(2026, 7, 25, 23, 13, 20, tzinfo=UTC),
        dry_run=True,
    ),
    IdleCancelled(**_BASE, reason="daemon_shutdown", idle_seconds=61.5),
    IdleStopTriggered(**_BASE, idle_seconds=900.0, dry_run=True, uptime_seconds=4200.0),
    RuntimeUnavailable(**_BASE, error="permission denied", endpoint="unix:///var/run/docker.sock"),
    RuntimeRestored(**_BASE, downtime_seconds=12.75),
    CommandIssued(
        **_BASE,
        action=ControlAction.RESTART,
        actor="kunal",
        via=Source.DISCORD,
        dry_run=True,
        accepted=False,
        rejection="server is already stopping",
    ),
)

IDS: tuple[str, ...] = tuple(type(event).__name__ for event in SAMPLES)


def test_samples_cover_every_event_type() -> None:
    """If this fails, an event was added without a serde sample, and its wire format is untested."""
    assert {type(event) for event in SAMPLES} == set(EVENT_TYPES)


@pytest.mark.parametrize("event", SAMPLES, ids=IDS)
def test_round_trip_is_exact(event: Event) -> None:
    assert event_from_dict(event_to_dict(event)) == event


@pytest.mark.parametrize("event", SAMPLES, ids=IDS)
def test_every_field_reaches_the_payload(event: Event) -> None:
    """Catches the drift the ``assert_never`` cannot: a new *field* on an existing event."""
    payload = event_to_dict(event)
    for field in fields(event):
        assert field.name in payload, f"{type(event).__name__}.{field.name} is not serialised"


@pytest.mark.parametrize("event", SAMPLES, ids=IDS)
def test_payload_is_json_safe_and_stable(event: Event) -> None:
    text = event_to_json(event)
    assert json.loads(text) == event_to_dict(event)
    # sort_keys defaults true, so two encodings of the same event are byte-identical.
    assert text == event_to_json(event)


@pytest.mark.parametrize("event", SAMPLES, ids=IDS)
def test_type_discriminator_is_the_class_name(event: Event) -> None:
    assert event_to_dict(event)["type"] == type(event).__name__


def test_timestamps_render_as_rfc3339_zulu() -> None:
    payload = event_to_dict(SAMPLES[0])
    assert payload["ts"] == "2026-07-25T22:58:20.809000Z"


def test_non_utc_timestamps_are_normalised_to_utc() -> None:
    """The homelab is Asia/Kolkata; a +0530 timestamp must not reach the wire as +05:30."""
    kolkata = TS.astimezone(UTC).replace(tzinfo=UTC)
    event = RuntimeRestored(
        ts=kolkata,
        server_id="minecraft",
        source=Source.RUNTIME,
        downtime_seconds=1.0,
    )
    assert event_to_dict(event)["ts"].endswith("Z")


def test_enums_render_as_their_value_not_their_repr() -> None:
    payload = event_to_dict(SAMPLES[6])
    assert payload["source"] == "log"
    assert payload["reason"] == "server_closed"


def test_tuples_render_as_arrays_and_decode_back_to_tuples() -> None:
    payload = event_to_dict(
        ServerCrashed(**_BASE, exit_code=1, oom_killed=False, tail=("a", "b")),
    )
    assert payload["tail"] == ["a", "b"]
    decoded = event_from_dict(payload)
    assert isinstance(decoded, ServerCrashed)
    assert decoded.tail == ("a", "b")


def test_nulls_are_present_rather_than_omitted() -> None:
    """A missing key and a null mean different things; only one of them is true."""
    payload = event_to_dict(
        ServerStarting(ts=TS, server_id="minecraft", source=Source.RUNTIME),
    )
    assert payload["version"] is None
    assert payload["raw"] is None


# ------------------------------------------------------------------------------- refusals


def test_a_base_event_is_refused() -> None:
    """Nothing should publish a bare ``Event``; serialising one would lose the discriminator."""
    with pytest.raises(SerdeError, match="not a concrete event type"):
        event_to_dict(Event(ts=TS, server_id="minecraft", source=Source.INTERNAL))


def test_a_category_base_is_refused() -> None:
    with pytest.raises(SerdeError, match="not a concrete event type"):
        event_to_dict(
            PlayerEvent(ts=TS, server_id="minecraft", source=Source.LOG, player=PLAYER),
        )


def test_a_naive_datetime_is_refused() -> None:
    naive = datetime(2026, 7, 25, 22, 58, 20)  # noqa: DTZ001 - the point of the test
    event = RuntimeRestored(ts=naive, server_id="minecraft", source=Source.RUNTIME)
    with pytest.raises(SerdeError, match="naive datetime"):
        event_to_dict(event)


def test_unknown_type_is_refused() -> None:
    payload = event_to_dict(SAMPLES[0])
    payload["type"] = "PlayerAscended"
    with pytest.raises(SerdeError, match="unknown event type"):
        event_from_dict(payload)


def test_missing_key_is_refused_rather_than_defaulted() -> None:
    payload = event_to_dict(SAMPLES[0])
    del payload["server_id"]
    with pytest.raises(SerdeError, match="missing required key"):
        event_from_dict(payload)


def test_wrong_scalar_type_is_refused() -> None:
    payload = event_to_dict(SAMPLES[3])
    payload["exit_code"] = "0"
    with pytest.raises(SerdeError, match="must be an integer"):
        event_from_dict(payload)


def test_a_bool_does_not_decode_as_an_integer() -> None:
    """``bool`` is an ``int`` subclass, so ``true`` would otherwise silently become ``1``."""
    payload = event_to_dict(SAMPLES[3])
    payload["exit_code"] = True
    with pytest.raises(SerdeError, match="must be an integer"):
        event_from_dict(payload)


def test_unknown_enum_member_lists_the_allowed_values() -> None:
    payload = event_to_dict(SAMPLES[6])
    payload["reason"] = "vanished"
    with pytest.raises(SerdeError, match="server_closed"):
        event_from_dict(payload)


def test_naive_timestamp_on_the_wire_is_refused() -> None:
    payload = event_to_dict(SAMPLES[0])
    payload["ts"] = "2026-07-25T22:58:20.809000"
    with pytest.raises(SerdeError, match="naive"):
        event_from_dict(payload)


def test_unparsable_timestamp_is_refused() -> None:
    payload = event_to_dict(SAMPLES[0])
    payload["ts"] = "yesterday"
    with pytest.raises(SerdeError, match="not an RFC3339 timestamp"):
        event_from_dict(payload)


def test_player_must_be_an_object() -> None:
    payload = event_to_dict(SAMPLES[5])
    payload["player"] = "Hypixelite"
    with pytest.raises(SerdeError, match="must be an object"):
        event_from_dict(payload)


def test_tail_must_be_an_array_of_strings() -> None:
    payload = event_to_dict(SAMPLES[4])
    payload["tail"] = ["ok", 3]
    with pytest.raises(SerdeError, match="array of strings"):
        event_from_dict(payload)
