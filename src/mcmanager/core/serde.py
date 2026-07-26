"""JSON serialisation of events.

Why this module exists rather than ``json.dumps(asdict(event))``: the CLI's ``--json`` output, the
``/events`` SSE stream and the persisted session records must be the *same* bytes, or the CLI and
the web API drift and the golden files stop meaning anything. ``render.py`` is skipped entirely on
the ``--json`` path for exactly this reason.

**Explicit per-type encoding, not reflection.** ``dataclasses.asdict`` would be shorter and would
also mean that renaming a field silently rewrites every golden file and every persisted session
record. The wire format is a compatibility surface, so it is written down. The cost of writing it
down is bounded by a :func:`typing.assert_never` at the end of the encoder's ``match``: adding an
event to :data:`~mcmanager.core.events.AnyEvent` without adding it here is a **type error**, not a
silent gap where the new event serialises as its base class. The matching runtime guarantee for
*fields* is ``tests/core/test_serde.py``, which round-trips a fully populated instance of every
type in ``EVENT_TYPES`` and asserts that every dataclass field appears in the payload.

Pydantic is deliberately absent. It is the right tool at the config boundary, where the input is
untrusted text; here both directions are over a closed set of frozen dataclasses whose fields are
already the validated ones, and a ``TypeAdapter`` over an 18-way union would need a discriminator
field added to every event purely to serve serialisation.

Shape of the wire format:

.. code-block:: json

    {"type": "PlayerJoined", "ts": "2026-07-25T22:58:20.809000Z", "server_id": "mc",
     "source": "log", "raw": "...", "seq": 41, "player": {"name": "Steve", "uuid": null},
     "online_count": 1, "address": null, "first_seen": true}

Datetimes are RFC3339 with an explicit ``Z``; enums are their ``value``; tuples are arrays;
``None`` is ``null``. Nothing is dropped for being falsy - a missing key and a null mean different
things to a consumer, and only one of them is true.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, assert_never, cast

from mcmanager.core.events import (
    EVENT_BY_NAME,
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

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mcmanager.core.events import AnyEvent

__all__ = ["SerdeError", "event_from_dict", "event_to_dict", "event_to_json"]


class SerdeError(ValueError):
    """A payload could not be encoded or decoded.

    A ``ValueError`` rather than an ``McManagerError``: this is a data-shape problem at a boundary,
    and every caller (the SSE client, ``mcmanager replay``, the session-record loader) wants to
    report the offending payload and carry on, not exit the process.
    """


# ------------------------------------------------------------------------------------ encoding


def event_to_dict(event: Event) -> dict[str, Any]:
    """Convert to a plain JSON-safe dict, including a ``"type"`` discriminator.

    Datetimes render as RFC3339 with an explicit ``Z``; enums render as their value.

    Raises:
        SerdeError: ``event`` is not one of the concrete types in
            :data:`~mcmanager.core.events.EVENT_TYPES` - a bare ``Event`` or a category base, which
            nothing should ever publish.
    """
    known = _narrow(event)
    if known is None:
        msg = (
            f"{type(event).__name__} is not a concrete event type; "
            "only members of EVENT_TYPES can be serialised"
        )
        raise SerdeError(msg)
    payload: dict[str, Any] = {
        "type": known.name,
        "ts": _iso(known.ts),
        "server_id": known.server_id,
        "source": known.source.value,
        "raw": known.raw,
        "seq": known.seq,
    }
    payload.update(_encode_specific(known))
    return payload


def event_to_json(event: Event, *, sort_keys: bool = True) -> str:
    """Serialise. ``sort_keys`` defaults true so golden-file comparison is byte-stable."""
    return json.dumps(
        event_to_dict(event),
        sort_keys=sort_keys,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _narrow(event: Event) -> AnyEvent | None:
    """Narrow a base-typed event to the closed union, so the encoder can be exhaustive.

    ``isinstance`` against the union literal is the one place this list is repeated; the test suite
    asserts it agrees with ``EVENT_TYPES``.
    """
    if isinstance(
        event,
        ServerStarting
        | ServerReady
        | ServerStopping
        | ServerStopped
        | ServerCrashed
        | PlayerJoined
        | PlayerLeft
        | PlayerDeath
        | PlayerAdvancement
        | ChatMessage
        | ConsoleLog
        | IdleStarted
        | IdleWarning
        | IdleCancelled
        | IdleStopTriggered
        | RuntimeUnavailable
        | RuntimeRestored
        | CommandIssued,
    ):
        return event
    return None


def _encode_specific(event: AnyEvent) -> dict[str, Any]:
    """Everything beyond the five base fields, per concrete type.

    The ``assert_never`` at the end is the point of the whole function: a nineteenth event type
    makes this fail to type-check rather than silently losing its fields on the wire.
    """
    match event:
        case ServerStarting():
            return {
                "version": event.version,
                "container_id": event.container_id,
                "requested_by": event.requested_by,
            }
        case ServerReady():
            return {
                "startup_seconds": event.startup_seconds,
                "version": event.version,
                "detected_by": event.detected_by.value,
            }
        case ServerStopping():
            return {
                "reason": event.reason,
                "requested_by": event.requested_by,
                "timeout_seconds": event.timeout_seconds,
            }
        case ServerStopped():
            return {
                "exit_code": event.exit_code,
                "clean": event.clean,
                "forced": event.forced,
                "uptime_seconds": event.uptime_seconds,
            }
        case ServerCrashed():
            return {
                "exit_code": event.exit_code,
                "oom_killed": event.oom_killed,
                "tail": list(event.tail),
            }
        case PlayerJoined():
            return {
                "player": _encode_player(event.player),
                "online_count": event.online_count,
                "address": event.address,
                "first_seen": event.first_seen,
            }
        case PlayerLeft():
            return {
                "player": _encode_player(event.player),
                "reason": event.reason.value,
                "session_seconds": event.session_seconds,
                "online_count": event.online_count,
            }
        case PlayerDeath():
            return {
                "player": _encode_player(event.player),
                "message": event.message,
                "killer": event.killer,
                "item": event.item,
                "template": event.template,
            }
        case PlayerAdvancement():
            return {
                "player": _encode_player(event.player),
                "title": event.title,
                "kind": event.kind.value,
            }
        case ChatMessage():
            return {
                "player": _encode_player(event.player),
                "message": event.message,
                "kind": event.kind.value,
            }
        case ConsoleLog():
            return {
                "message": event.message,
                "level": event.level,
                "thread": event.thread,
                "origin": event.origin.value,
                "stream": event.stream.value,
            }
        case IdleStarted():
            return {
                "deadline": _iso(event.deadline),
                "empty_since": _iso(event.empty_since),
                "timeout_seconds": event.timeout_seconds,
                "dry_run": event.dry_run,
            }
        case IdleWarning():
            return {
                "remaining_seconds": event.remaining_seconds,
                "deadline": _iso(event.deadline),
                "dry_run": event.dry_run,
            }
        case IdleCancelled():
            return {"reason": event.reason, "idle_seconds": event.idle_seconds}
        case IdleStopTriggered():
            return {
                "idle_seconds": event.idle_seconds,
                "dry_run": event.dry_run,
                "uptime_seconds": event.uptime_seconds,
            }
        case RuntimeUnavailable():
            return {"error": event.error, "endpoint": event.endpoint}
        case RuntimeRestored():
            return {"downtime_seconds": event.downtime_seconds}
        case CommandIssued():
            return {
                "action": event.action.value,
                "actor": event.actor,
                "via": event.via.value,
                "dry_run": event.dry_run,
                "accepted": event.accepted,
                "rejection": event.rejection,
            }
        case _:
            assert_never(event)


def _encode_player(player: PlayerRef) -> dict[str, Any]:
    return {"name": player.name, "uuid": player.uuid}


def _iso(value: datetime) -> str:
    """RFC3339 with an explicit ``Z``.

    ``isoformat()`` renders UTC as ``+00:00``; every consumer of this format so far - jq filters,
    JavaScript ``Date``, and a human reading ``docker logs`` - reads ``Z`` more easily, and it is
    two characters shorter in a stream that carries one of these per log line.
    """
    if value.tzinfo is None:
        msg = "refusing to serialise a naive datetime; every timestamp in this project is UTC"
        raise SerdeError(msg)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


# ------------------------------------------------------------------------------------ decoding


def event_from_dict(payload: dict[str, Any]) -> Event:
    """Rebuild an event from :func:`event_to_dict` output, resolved via ``EVENT_BY_NAME``.

    Round-trips exactly: ``event_from_dict(event_to_dict(e)) == e`` for every concrete event.

    Raises:
        SerdeError: unknown ``type``, a missing required key, or a value of the wrong shape. The
            decoder never guesses - a payload it does not understand is a bug somewhere, and
            filling in a default would hide it inside a session record forever.
    """
    name = _req_str(payload, "type")
    if name not in EVENT_BY_NAME:
        msg = f"unknown event type {name!r}"
        raise SerdeError(msg)
    base = _base_kwargs(payload)

    match name:
        case "ServerStarting":
            return ServerStarting(
                **base,
                version=_opt_str(payload, "version"),
                container_id=_opt_str(payload, "container_id"),
                requested_by=_opt_str(payload, "requested_by"),
            )
        case "ServerReady":
            return ServerReady(
                **base,
                startup_seconds=_opt_float(payload, "startup_seconds"),
                version=_opt_str(payload, "version"),
                detected_by=_enum(ReadySignal, payload, "detected_by"),
            )
        case "ServerStopping":
            return ServerStopping(
                **base,
                reason=_opt_str(payload, "reason"),
                requested_by=_opt_str(payload, "requested_by"),
                timeout_seconds=_opt_float(payload, "timeout_seconds"),
            )
        case "ServerStopped":
            return ServerStopped(
                **base,
                exit_code=_opt_int(payload, "exit_code"),
                clean=_req_bool(payload, "clean"),
                forced=_req_bool(payload, "forced"),
                uptime_seconds=_opt_float(payload, "uptime_seconds"),
            )
        case "ServerCrashed":
            return ServerCrashed(
                **base,
                exit_code=_opt_int(payload, "exit_code"),
                oom_killed=_req_bool(payload, "oom_killed"),
                tail=_req_str_tuple(payload, "tail"),
            )
        case "PlayerJoined":
            return PlayerJoined(
                **base,
                player=_player(payload),
                online_count=_req_int(payload, "online_count"),
                address=_opt_str(payload, "address"),
                first_seen=_req_bool(payload, "first_seen"),
            )
        case "PlayerLeft":
            return PlayerLeft(
                **base,
                player=_player(payload),
                reason=_enum(LeaveReason, payload, "reason"),
                session_seconds=_opt_float(payload, "session_seconds"),
                online_count=_req_int(payload, "online_count"),
            )
        case "PlayerDeath":
            return PlayerDeath(
                **base,
                player=_player(payload),
                message=_req_str(payload, "message"),
                killer=_opt_str(payload, "killer"),
                item=_opt_str(payload, "item"),
                template=_opt_str(payload, "template"),
            )
        case "PlayerAdvancement":
            return PlayerAdvancement(
                **base,
                player=_player(payload),
                title=_req_str(payload, "title"),
                kind=_enum(AdvancementKind, payload, "kind"),
            )
        case "ChatMessage":
            return ChatMessage(
                **base,
                player=_player(payload),
                message=_req_str(payload, "message"),
                kind=_enum(ChatKind, payload, "kind"),
            )
        case "ConsoleLog":
            return ConsoleLog(
                **base,
                message=_req_str(payload, "message"),
                level=_opt_str(payload, "level"),
                thread=_opt_str(payload, "thread"),
                origin=_enum(LineOrigin, payload, "origin"),
                stream=_enum(Stream, payload, "stream"),
            )
        case "IdleStarted":
            return IdleStarted(
                **base,
                deadline=_req_dt(payload, "deadline"),
                empty_since=_req_dt(payload, "empty_since"),
                timeout_seconds=_req_float(payload, "timeout_seconds"),
                dry_run=_req_bool(payload, "dry_run"),
            )
        case "IdleWarning":
            return IdleWarning(
                **base,
                remaining_seconds=_req_float(payload, "remaining_seconds"),
                deadline=_req_dt(payload, "deadline"),
                dry_run=_req_bool(payload, "dry_run"),
            )
        case "IdleCancelled":
            return IdleCancelled(
                **base,
                reason=_req_str(payload, "reason"),
                idle_seconds=_opt_float(payload, "idle_seconds"),
            )
        case "IdleStopTriggered":
            return IdleStopTriggered(
                **base,
                idle_seconds=_req_float(payload, "idle_seconds"),
                dry_run=_req_bool(payload, "dry_run"),
                uptime_seconds=_opt_float(payload, "uptime_seconds"),
            )
        case "RuntimeUnavailable":
            return RuntimeUnavailable(
                **base,
                error=_req_str(payload, "error"),
                endpoint=_opt_str(payload, "endpoint"),
            )
        case "RuntimeRestored":
            return RuntimeRestored(
                **base,
                downtime_seconds=_opt_float(payload, "downtime_seconds"),
            )
        case "CommandIssued":
            return CommandIssued(
                **base,
                action=_enum(ControlAction, payload, "action"),
                actor=_req_str(payload, "actor"),
                via=_enum(Source, payload, "via"),
                dry_run=_req_bool(payload, "dry_run"),
                accepted=_req_bool(payload, "accepted"),
                rejection=_opt_str(payload, "rejection"),
            )
        case _:  # pragma: no cover - EVENT_BY_NAME membership was checked above
            msg = f"event type {name!r} is in EVENT_BY_NAME but has no decoder"
            raise SerdeError(msg)


def _base_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ts": _req_dt(payload, "ts"),
        "server_id": _req_str(payload, "server_id"),
        "source": _enum(Source, payload, "source"),
        "raw": _opt_str(payload, "raw"),
        "seq": _req_int(payload, "seq"),
    }


def _player(payload: Mapping[str, Any]) -> PlayerRef:
    raw: object = _require(payload, "player")
    if not isinstance(raw, dict):
        msg = f"'player' must be an object, got {type(raw).__name__}"
        raise SerdeError(msg)
    nested = cast("Mapping[str, Any]", raw)
    return PlayerRef(name=_req_str(nested, "name"), uuid=_opt_str(nested, "uuid"))


# -- primitive readers -----------------------------------------------------------------------
#
# Every one of these raises rather than defaulting. A decoder that quietly substitutes a default
# turns a wire-format mismatch into a session record that is subtly wrong forever.


def _require(payload: Mapping[str, Any], key: str) -> object:
    if key not in payload:
        msg = f"missing required key {key!r}"
        raise SerdeError(msg)
    return payload[key]


def _type_error(key: str, expected: str, value: object) -> SerdeError:
    return SerdeError(f"{key!r} must be {expected}, got {type(value).__name__}")


def _req_str(payload: Mapping[str, Any], key: str) -> str:
    value = _require(payload, key)
    if not isinstance(value, str):
        raise _type_error(key, "a string", value)
    return value


def _opt_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = _require(payload, key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _type_error(key, "a string or null", value)
    return value


def _req_int(payload: Mapping[str, Any], key: str) -> int:
    value = _require(payload, key)
    # bool is an int subclass; accepting it here would let `true` decode as `1`.
    if isinstance(value, bool) or not isinstance(value, int):
        raise _type_error(key, "an integer", value)
    return value


def _opt_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = _require(payload, key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _type_error(key, "an integer or null", value)
    return value


def _req_float(payload: Mapping[str, Any], key: str) -> float:
    value = _require(payload, key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _type_error(key, "a number", value)
    return float(value)


def _opt_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = _require(payload, key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _type_error(key, "a number or null", value)
    return float(value)


def _req_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = _require(payload, key)
    if not isinstance(value, bool):
        raise _type_error(key, "a boolean", value)
    return value


def _req_str_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = _require(payload, key)
    if not isinstance(value, list):
        raise _type_error(key, "an array of strings", value)
    items = cast("list[Any]", value)
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise _type_error(key, "an array of strings", item)
        out.append(item)
    return tuple(out)


def _req_dt(payload: Mapping[str, Any], key: str) -> datetime:
    text = _req_str(payload, key)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        msg = f"{key!r} is not an RFC3339 timestamp: {text!r}"
        raise SerdeError(msg) from exc
    if parsed.tzinfo is None:
        msg = f"{key!r} is naive; every timestamp in this project is tz-aware UTC: {text!r}"
        raise SerdeError(msg)
    return parsed.astimezone(UTC)


def _enum[E: StrEnum](enum_type: type[E], payload: Mapping[str, Any], key: str) -> E:
    text = _req_str(payload, key)
    try:
        return enum_type(text)
    except ValueError as exc:
        allowed = ", ".join(sorted(str(member.value) for member in enum_type))
        msg = f"{key!r} must be one of ({allowed}), got {text!r}"
        raise SerdeError(msg) from exc
