"""The read-model: what ``/status``, ``/players``, ``/sessions`` and ``/readyz`` return.

These are the DTOs that cross the HTTP boundary in both directions. The daemon builds them from
live state and serialises them; the CLI decodes them and hands them to ``cli/render.py``, which is
pure. The CLI's ``--json`` path prints the same JSON the endpoint returned, so a rendered view and
a scripted one can never disagree about what the daemon said.

**Why a module the plan's file list does not name.** ``routes.py`` needs these types to encode,
``cli/client.py`` needs them to decode, and ``cli/render.py`` needs them to format. Putting them in
``routes.py`` would make every CLI invocation import ``aiohttp.web``; putting them in ``render.py``
would make the server import the CLI. A third module in the directory that already owns the wire
format is the only arrangement where the dependency arrows all point one way:

    views  <-  sse  <-  routes  <-  server
      ^
      +-------------  cli/client, cli/render

Nothing here imports aiohttp, docker, discord or the services layer, which is what lets
``cli/render.py``'s golden tests construct a full status view in four lines.

**Events are not re-encoded here.** :attr:`StatusView.last_event` is a real
:class:`~mcmanager.core.events.Event` and round-trips through :mod:`mcmanager.core.serde`. There is
exactly one wire format for an event in this project.

Decoding is deliberately lenient about *missing* keys and strict about *wrong types*. The two ends
ship together, so a missing key is almost always an older daemon rather than corruption, and the
CLI degrading to "unknown" beats it refusing to print a status. A key of the wrong type is a bug
and says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

from mcmanager.containers.dto import HealthState
from mcmanager.core.events import Event
from mcmanager.core.serde import SerdeError, event_from_dict, event_to_dict
from mcmanager.core.types import LifecycleState

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "ControlResultView",
    "IdleView",
    "LivenessView",
    "PlayerView",
    "ProbeView",
    "ReadinessView",
    "SessionView",
    "StatusView",
    "SubsystemView",
    "evaluate_readiness",
]


# ------------------------------------------------------------------------------- subsystems


@dataclass(frozen=True, slots=True, kw_only=True)
class SubsystemView:
    """One line of ``/readyz``.

    Attributes:
        name: ``runtime``, ``discord``, ``log_stream``, ``bus``.
        ok: Whether this subsystem is satisfied.
        required: False for advisory checks, which are reported but never make the daemon
            unready. A red advisory check is information; a red required one is a rollout gate.
        detail: One sentence saying *why*, which is the entire reason this endpoint returns a
            body instead of a bare status code. "not ready" without a subsystem name is a support
            ticket.
    """

    name: str
    ok: bool
    required: bool = True
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "required": self.required, "detail": self.detail}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            name=_s(payload, "name", "?"),
            ok=_b(payload, "ok"),
            required=_b(payload, "required", default=True),
            detail=_os(payload, "detail"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadinessView:
    """``GET /readyz``: can this daemon do its job right now?

    Distinct from :class:`LivenessView` on purpose, and the distinction is the whole design of the
    two endpoints. Liveness answers "should I restart this process"; readiness answers "is it
    working". A dead Docker socket must never restart the daemon - restarting fixes nothing and
    crash-looping hides the real problem behind a container that never stays up long enough to
    read its logs.
    """

    ready: bool
    checks: tuple[SubsystemView, ...] = ()

    def failing(self) -> tuple[SubsystemView, ...]:
        """The required checks that are red. Empty when ready."""
        return tuple(check for check in self.checks if check.required and not check.ok)

    def to_dict(self) -> dict[str, Any]:
        return {"ready": self.ready, "checks": [check.to_dict() for check in self.checks]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            ready=_b(payload, "ready"),
            checks=tuple(SubsystemView.from_dict(item) for item in _objects(payload, "checks")),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LivenessView:
    """``GET /healthz``: is the event loop responsive and the supervisor alive?

    **Deliberately does not consult Docker or Discord.** This endpoint is what compose's
    healthcheck polls, and a healthcheck that goes red because the Docker socket lost its group
    membership would restart the daemon in a loop while the actual fault sat untouched on the
    host.

    Attributes:
        alive: Always true if this was produced at all; it is a field so the JSON shape is
            self-describing rather than "you got a 200, infer the rest".
        loop_lag_seconds: How long a ``call_later(0)`` actually took to fire. The one honest
            measure of a blocked loop, which is the failure a liveness probe *should* catch.
        supervisor_ok: False when a task marked ``critical`` has died. That does warrant a
            restart, which is why it is here and Docker is not.
        tasks: Supervised task count, for the log line.
        detail: Free text when something is off.
    """

    alive: bool = True
    loop_lag_seconds: float | None = None
    supervisor_ok: bool = True
    tasks: int = 0
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alive": self.alive,
            "loop_lag_seconds": self.loop_lag_seconds,
            "supervisor_ok": self.supervisor_ok,
            "tasks": self.tasks,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            alive=_b(payload, "alive", default=True),
            loop_lag_seconds=_of(payload, "loop_lag_seconds"),
            supervisor_ok=_b(payload, "supervisor_ok", default=True),
            tasks=_i(payload, "tasks"),
            detail=_os(payload, "detail"),
        )


def evaluate_readiness(
    *,
    runtime_available: bool,
    runtime_age_seconds: float | None = None,
    runtime_max_age_seconds: float = 120.0,
    runtime_detail: str | None = None,
    discord_enabled: bool = False,
    discord_connected: bool = False,
    discord_detail: str | None = None,
    log_stream_attached: bool = False,
    log_stream_expected: bool = True,
    log_stream_detail: str | None = None,
    dropped_critical: int = 0,
) -> ReadinessView:
    """The readiness policy, as one pure function.

    Ready means: the runtime answered recently, **and** Discord is connected or deliberately off,
    **and** the log streamer is either attached or legitimately idle because the container is not
    running.

    ``log_stream_expected`` is what makes the third clause honest. A stopped Minecraft server has
    no log stream to attach to, and a daemon that reports itself unready for the entire time the
    server is off would be red most of the week - which trains everybody to ignore it, which is
    worse than not having the check.

    ``dropped_critical`` comes from ``bus.stats``. It being non-zero means the queue evicted an
    event that was not a ``ConsoleLog``, i.e. the sizing is wrong somewhere. It is surfaced here
    rather than merely logged, because a counter nobody looks at is a counter that does not exist.
    """
    checks: list[SubsystemView] = []

    runtime_ok = runtime_available
    detail = runtime_detail
    if (
        runtime_ok
        and runtime_age_seconds is not None
        and runtime_age_seconds > runtime_max_age_seconds
    ):
        runtime_ok = False
        detail = (
            f"last successful runtime call was {runtime_age_seconds:.0f}s ago "
            f"(stale after {runtime_max_age_seconds:.0f}s)"
        )
    if runtime_ok and detail is None:
        detail = "reachable"
    checks.append(SubsystemView(name="runtime", ok=runtime_ok, detail=detail))

    if not discord_enabled:
        checks.append(
            SubsystemView(name="discord", ok=True, detail=discord_detail or "disabled by config")
        )
    else:
        checks.append(
            SubsystemView(
                name="discord",
                ok=discord_connected,
                detail=discord_detail
                or ("gateway connected" if discord_connected else "gateway not connected"),
            )
        )

    if not log_stream_expected:
        checks.append(
            SubsystemView(
                name="log_stream",
                ok=True,
                detail=log_stream_detail or "idle: the container is not running",
            )
        )
    else:
        checks.append(
            SubsystemView(
                name="log_stream",
                ok=log_stream_attached,
                detail=log_stream_detail
                or ("attached" if log_stream_attached else "detached while the container is up"),
            )
        )

    checks.append(
        SubsystemView(
            name="bus",
            ok=dropped_critical == 0,
            detail=(
                "no critical drops"
                if dropped_critical == 0
                else f"{dropped_critical} non-console event(s) dropped: the queue is undersized"
            ),
        )
    )

    ready = all(check.ok for check in checks if check.required)
    return ReadinessView(ready=ready, checks=tuple(checks))


# ----------------------------------------------------------------------------------- players


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayerView:
    """One player, as ``/players`` and the roster inside ``/status`` report them.

    Note what is **absent**: the client address. ``PlayerJoined.address`` is carried on the event
    because idle diagnostics and abuse investigation want it, and it is dropped here because this
    is the structure that reaches a terminal, a Discord embed and eventually a browser. It is not
    a presenter's job to remember that.

    Attributes:
        source: ``log`` for a player we watched join, ``probe`` for one the status query found.
            A probe-derived session has an approximate join time and saying so is better than
            printing a confident wrong number.
    """

    name: str
    uuid: str | None = None
    online_since: datetime | None = None
    session_seconds: float | None = None
    first_seen: bool = False
    source: str = "log"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "uuid": self.uuid,
            "online_since": _iso(self.online_since),
            "session_seconds": self.session_seconds,
            "first_seen": self.first_seen,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            name=_s(payload, "name", "?"),
            uuid=_os(payload, "uuid"),
            online_since=_odt(payload, "online_since"),
            session_seconds=_of(payload, "session_seconds"),
            first_seen=_b(payload, "first_seen"),
            source=_s(payload, "source", "log"),
        )


# ------------------------------------------------------------------------------ control result


@dataclass(frozen=True, slots=True, kw_only=True)
class ControlResultView:
    """The result of a ``POST /control/*``, which is a rendering of a ``ControlOutcome``.

    A DTO rather than a loose dict because ``cli/render.py`` takes DTOs, and because the three
    outcomes a caller must distinguish - refused, attempted-and-failed, done - are three different
    exit paths in the CLI and three different messages in Discord.

    Attributes:
        accepted: False means the daemon refused before touching Docker; :attr:`rejection` says
            why, in a sentence somebody can act on.
        error: Set when the command was accepted and the runtime then failed. Distinct from a
            rejection, and the distinction is worth an exit code.
        message: The daemon's own one-line rendering, so the CLI and Discord say the same thing.
    """

    action: str
    actor: str = ""
    accepted: bool = False
    ok: bool = False
    rejection: str | None = None
    dry_run: bool = False
    error: str | None = None
    state_before: str = LifecycleState.UNKNOWN.value
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "actor": self.actor,
            "accepted": self.accepted,
            "ok": self.ok,
            "rejection": self.rejection,
            "dry_run": self.dry_run,
            "error": self.error,
            "state_before": self.state_before,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            action=_s(payload, "action", "?"),
            actor=_s(payload, "actor", ""),
            accepted=_b(payload, "accepted"),
            ok=_b(payload, "ok"),
            rejection=_os(payload, "rejection"),
            dry_run=_b(payload, "dry_run"),
            error=_os(payload, "error"),
            state_before=_s(payload, "state_before", LifecycleState.UNKNOWN.value),
            message=_s(payload, "message", ""),
        )


# ------------------------------------------------------------------------------------ probe


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbeView:
    """The last Server List Ping, flattened.

    ``reachable=False`` means **unknown**, never "zero players". The renderer says so in those
    words, because that distinction is the difference between an idle timer that works and one
    that stops a server full of people.
    """

    reachable: bool
    players_online: int | None = None
    players_max: int | None = None
    sample: tuple[str, ...] = ()
    sample_is_complete: bool = False
    version: str | None = None
    motd: str | None = None
    latency_ms: float | None = None
    error: str | None = None
    probed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "players_online": self.players_online,
            "players_max": self.players_max,
            "sample": list(self.sample),
            "sample_is_complete": self.sample_is_complete,
            "version": self.version,
            "motd": self.motd,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "probed_at": _iso(self.probed_at),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            reachable=_b(payload, "reachable"),
            players_online=_oi(payload, "players_online"),
            players_max=_oi(payload, "players_max"),
            sample=_strs(payload, "sample"),
            sample_is_complete=_b(payload, "sample_is_complete"),
            version=_os(payload, "version"),
            motd=_os(payload, "motd"),
            latency_ms=_of(payload, "latency_ms"),
            error=_os(payload, "error"),
            probed_at=_odt(payload, "probed_at"),
        )


# ------------------------------------------------------------------------------------- idle


@dataclass(frozen=True, slots=True, kw_only=True)
class IdleView:
    """The idle-shutdown countdown, as far as anyone outside it needs to know.

    Attributes:
        armed: A countdown is running right now.
        deadline: When the stop would happen. Note that on daemon boot this is restarted from
            "now" rather than resumed, so a crash-looping manager cannot repeatedly insta-stop the
            server; the value here is always the *live* deadline.
        dry_run: The countdown is computed and logged and nothing is ever stopped.
    """

    enabled: bool = False
    dry_run: bool = True
    armed: bool = False
    deadline: datetime | None = None
    seconds_remaining: float | None = None
    timeout_seconds: float | None = None
    empty_since: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "armed": self.armed,
            "deadline": _iso(self.deadline),
            "seconds_remaining": self.seconds_remaining,
            "timeout_seconds": self.timeout_seconds,
            "empty_since": _iso(self.empty_since),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            enabled=_b(payload, "enabled"),
            dry_run=_b(payload, "dry_run", default=True),
            armed=_b(payload, "armed"),
            deadline=_odt(payload, "deadline"),
            seconds_remaining=_of(payload, "seconds_remaining"),
            timeout_seconds=_of(payload, "timeout_seconds"),
            empty_since=_odt(payload, "empty_since"),
        )


# ---------------------------------------------------------------------------------- sessions


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionView:
    """One archived or in-flight play session.

    A session spans the *Minecraft server's* lifetime, not the daemon's, and its identity is
    ``(container_id, started_at)``: stable across a daemon restart, changed by a
    ``compose down/up``.

    Attributes:
        open: The session had not ended when this record was last written. On a clean shutdown the
            daemon writes ``open=true`` deliberately, so a resumed daemon can tell "still running"
            from "we missed the ending".
        partial: The counts are known to be incomplete - the daemon missed part of the session.
            **Rendered prominently.** The docker log ring is ~30MB, so a long session genuinely
            cannot be reconstructed after the fact; under-reporting somebody's playtime without
            saying so is the dishonest option.
    """

    id: str
    server_id: str = ""
    container_id: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    open: bool = False
    partial: bool = False
    duration_seconds: float | None = None
    players: tuple[str, ...] = ()
    peak_online: int = 0
    joins: int = 0
    leaves: int = 0
    deaths: int = 0
    advancements: int = 0
    chat_messages: int = 0
    stop_reason: str | None = None
    stopped_by: str | None = None
    exit_code: int | None = None
    clean: bool | None = None
    archive: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "server_id": self.server_id,
            "container_id": self.container_id,
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at),
            "open": self.open,
            "partial": self.partial,
            "duration_seconds": self.duration_seconds,
            "players": list(self.players),
            "peak_online": self.peak_online,
            "joins": self.joins,
            "leaves": self.leaves,
            "deaths": self.deaths,
            "advancements": self.advancements,
            "chat_messages": self.chat_messages,
            "stop_reason": self.stop_reason,
            "stopped_by": self.stopped_by,
            "exit_code": self.exit_code,
            "clean": self.clean,
            "archive": self.archive,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Decode a session record.

        Every field but ``id`` has a default, because this also reads records written by the
        session manager (M5) directly off disk, and a reader that refuses a record it half
        understands loses history that cannot be regenerated.
        """
        return cls(
            id=_s(payload, "id", ""),
            server_id=_s(payload, "server_id", ""),
            container_id=_os(payload, "container_id"),
            started_at=_odt(payload, "started_at"),
            ended_at=_odt(payload, "ended_at"),
            open=_b(payload, "open"),
            partial=_b(payload, "partial"),
            duration_seconds=_of(payload, "duration_seconds"),
            players=_strs(payload, "players"),
            peak_online=_i(payload, "peak_online"),
            joins=_i(payload, "joins"),
            leaves=_i(payload, "leaves"),
            deaths=_i(payload, "deaths"),
            advancements=_i(payload, "advancements"),
            chat_messages=_i(payload, "chat_messages"),
            stop_reason=_os(payload, "stop_reason"),
            stopped_by=_os(payload, "stopped_by"),
            exit_code=_oi(payload, "exit_code"),
            clean=_ob(payload, "clean"),
            archive=_os(payload, "archive"),
        )

    def elapsed_seconds(self, now: datetime) -> float | None:
        """Duration, computed against ``now`` for a still-open session.

        Takes ``now`` rather than reading a clock: nothing in this module is allowed one.
        """
        if self.duration_seconds is not None:
            return self.duration_seconds
        if self.started_at is None:
            return None
        end = self.ended_at if self.ended_at is not None else now
        return max((end - self.started_at).total_seconds(), 0.0)


# ------------------------------------------------------------------------------------ status


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusView:
    """Everything ``mcmanager status`` and ``/status`` show.

    Built two ways, and it matters which:

    - by the daemon, from the lifecycle reducer, the roster and the poller;
    - by the **standalone** CLI, from one container inspect plus one status probe, with
      :attr:`daemon_online` false.

    The second form is missing everything only a running daemon can know - the session, the idle
    countdown, session durations - and every one of those is ``None`` rather than zero, so the
    renderer prints "unknown" instead of a confident lie. The ``daemon: offline`` banner is not
    decoration; it is the caveat on all of it.
    """

    server_id: str
    container: str
    state: LifecycleState = LifecycleState.UNKNOWN
    daemon_online: bool = True
    observed_at: datetime

    exists: bool = True
    running: bool = False
    health: HealthState = HealthState.UNKNOWN
    health_reported_raw: str | None = None
    container_id: str | None = None
    image: str | None = None
    started_at: datetime | None = None
    uptime_seconds: float | None = None
    exit_code: int | None = None
    oom_killed: bool = False
    restart_count: int = 0

    version: str | None = None
    ready_at: datetime | None = None
    ready_detected_by: str | None = None
    startup_seconds: float | None = None

    players_online: int | None = None
    players_max: int | None = None
    roster: tuple[PlayerView, ...] = ()

    probe: ProbeView | None = None
    idle: IdleView | None = None
    session: SessionView | None = None
    last_event: Event | None = None

    stop_timeout_seconds: int | None = None
    dry_run: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "container": self.container,
            "state": self.state.value,
            "daemon_online": self.daemon_online,
            "observed_at": _iso(self.observed_at),
            "exists": self.exists,
            "running": self.running,
            "health": self.health.value,
            "health_reported_raw": self.health_reported_raw,
            "container_id": self.container_id,
            "image": self.image,
            "started_at": _iso(self.started_at),
            "uptime_seconds": self.uptime_seconds,
            "exit_code": self.exit_code,
            "oom_killed": self.oom_killed,
            "restart_count": self.restart_count,
            "version": self.version,
            "ready_at": _iso(self.ready_at),
            "ready_detected_by": self.ready_detected_by,
            "startup_seconds": self.startup_seconds,
            "players_online": self.players_online,
            "players_max": self.players_max,
            "roster": [player.to_dict() for player in self.roster],
            "probe": None if self.probe is None else self.probe.to_dict(),
            "idle": None if self.idle is None else self.idle.to_dict(),
            "session": None if self.session is None else self.session.to_dict(),
            "last_event": None if self.last_event is None else event_to_dict(self.last_event),
            "stop_timeout_seconds": self.stop_timeout_seconds,
            "dry_run": self.dry_run,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            server_id=_s(payload, "server_id", ""),
            container=_s(payload, "container", ""),
            state=_enum(LifecycleState, payload, "state", LifecycleState.UNKNOWN),
            daemon_online=_b(payload, "daemon_online", default=True),
            observed_at=_dt(payload, "observed_at"),
            exists=_b(payload, "exists", default=True),
            running=_b(payload, "running"),
            health=_enum(HealthState, payload, "health", HealthState.UNKNOWN),
            health_reported_raw=_os(payload, "health_reported_raw"),
            container_id=_os(payload, "container_id"),
            image=_os(payload, "image"),
            started_at=_odt(payload, "started_at"),
            uptime_seconds=_of(payload, "uptime_seconds"),
            exit_code=_oi(payload, "exit_code"),
            oom_killed=_b(payload, "oom_killed"),
            restart_count=_i(payload, "restart_count"),
            version=_os(payload, "version"),
            ready_at=_odt(payload, "ready_at"),
            ready_detected_by=_os(payload, "ready_detected_by"),
            startup_seconds=_of(payload, "startup_seconds"),
            players_online=_oi(payload, "players_online"),
            players_max=_oi(payload, "players_max"),
            roster=tuple(PlayerView.from_dict(item) for item in _objects(payload, "roster")),
            probe=_nested(payload, "probe", ProbeView),
            idle=_nested(payload, "idle", IdleView),
            session=_nested(payload, "session", SessionView),
            last_event=_event(payload, "last_event"),
            stop_timeout_seconds=_oi(payload, "stop_timeout_seconds"),
            dry_run=_b(payload, "dry_run"),
            notes=_strs(payload, "notes"),
        )


# ------------------------------------------------------------------------ decoding primitives
#
# Lenient on absence, strict on shape. Anything missing falls back to the field's default;
# anything present with the wrong type raises, because that is a bug on the other end and
# coercing it would move the failure somewhere unhelpful.


def _iso(value: datetime | None) -> str | None:
    """RFC3339 with an explicit ``Z``, matching ``core/serde``'s rendering exactly."""
    if value is None:
        return None
    if value.tzinfo is None:
        msg = "refusing to serialise a naive datetime; every timestamp in this project is UTC"
        raise SerdeError(msg)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _wrong(key: str, expected: str, value: object) -> SerdeError:
    return SerdeError(f"{key!r} must be {expected}, got {type(value).__name__}")


def _s(payload: Mapping[str, Any], key: str, default: str) -> str:
    value: object = payload.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise _wrong(key, "a string", value)
    return value


def _os(payload: Mapping[str, Any], key: str) -> str | None:
    value: object = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _wrong(key, "a string or null", value)
    return value


def _b(payload: Mapping[str, Any], key: str, *, default: bool = False) -> bool:
    value: object = payload.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise _wrong(key, "a boolean", value)
    return value


def _ob(payload: Mapping[str, Any], key: str) -> bool | None:
    value: object = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise _wrong(key, "a boolean or null", value)
    return value


def _i(payload: Mapping[str, Any], key: str, default: int = 0) -> int:
    value: object = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise _wrong(key, "an integer", value)
    return value


def _oi(payload: Mapping[str, Any], key: str) -> int | None:
    value: object = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _wrong(key, "an integer or null", value)
    return value


def _of(payload: Mapping[str, Any], key: str) -> float | None:
    value: object = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _wrong(key, "a number or null", value)
    return float(value)


def _odt(payload: Mapping[str, Any], key: str) -> datetime | None:
    text = _os(payload, key)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        msg = f"{key!r} is not an RFC3339 timestamp: {text!r}"
        raise SerdeError(msg) from exc
    if parsed.tzinfo is None:
        msg = f"{key!r} is naive; every timestamp in this project is tz-aware UTC"
        raise SerdeError(msg)
    return parsed.astimezone(UTC)


def _dt(payload: Mapping[str, Any], key: str) -> datetime:
    value = _odt(payload, key)
    if value is None:
        msg = f"missing required key {key!r}"
        raise SerdeError(msg)
    return value


def _strs(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value: object = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _wrong(key, "an array of strings", value)
    out: list[str] = []
    for entry in cast("list[object]", value):
        if not isinstance(entry, str):
            raise _wrong(key, "an array of strings", entry)
        out.append(entry)
    return tuple(out)


def _objects(payload: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    value: object = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _wrong(key, "an array of objects", value)
    out: list[Mapping[str, Any]] = []
    for entry in cast("list[object]", value):
        if not isinstance(entry, dict):
            raise _wrong(key, "an array of objects", entry)
        out.append(cast("Mapping[str, Any]", entry))
    return tuple(out)


class _FromDict[T](Protocol):
    """The shape every view in this module satisfies, so ``_nested`` can stay generic."""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> T: ...


def _object(payload: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value: object = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _wrong(key, "an object or null", value)
    return cast("Mapping[str, Any]", value)


def _nested[T](payload: Mapping[str, Any], key: str, factory: type[_FromDict[T]]) -> T | None:
    """Decode an optional nested object through ``factory.from_dict``."""
    nested = _object(payload, key)
    if nested is None:
        return None
    return factory.from_dict(nested)


def _event(payload: Mapping[str, Any], key: str) -> Event | None:
    nested = _object(payload, key)
    if nested is None:
        return None
    return event_from_dict(dict(nested))


def _enum[E: StrEnum](enum_type: type[E], payload: Mapping[str, Any], key: str, default: E) -> E:
    text = _os(payload, key)
    if text is None:
        return default
    try:
        return enum_type(text)
    except ValueError as exc:
        msg = f"{key!r} is not a valid {enum_type.__name__}: {text!r}"
        raise SerdeError(msg) from exc
