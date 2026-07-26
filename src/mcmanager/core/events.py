"""The event vocabulary. Everything the daemon knows, expressed as immutable facts.

**Frozen slotted dataclasses, never pydantic.** These are built from already-validated inputs on
the hot path - one per log line, and a Paper crash emits thousands per second - where pydantic
validation buys nothing and costs roughly an order of magnitude. ``frozen=True`` means no
subscriber can mutate what another subscriber is about to see; ``slots=True`` keeps construction
cheap; ``kw_only=True`` means adding a field to a base class never silently reorders a subclass's
positional arguments. Pydantic stays at the two real boundaries: config load, and JSON
serialisation in :mod:`mcmanager.core.serde`.

Every event carries :attr:`Event.raw` - the original line or a rendering of the originating fact.
That is what lets a Discord console relay subscribe to ``Event`` and show recognised lines too:
recognised lines deliberately do *not* also emit a :class:`ConsoleLog`, so a relay subscribed only
to ``ConsoleLog`` would show gaps exactly where the joins and the chat were.

Field choices follow one rule: **no consumer should have to re-derive anything.** If a presenter
needs a duration, the event carries the duration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Self

from mcmanager.core.types import (
    AdvancementKind,
    ChatKind,
    ControlAction,
    LeaveReason,
    LineOrigin,
    PlayerRef,
    ReadySignal,
    ServerId,
    Source,
    Stream,
)

__all__ = [
    "EVENT_BY_NAME",
    "EVENT_TYPES",
    "AnyEvent",
    "ChatMessage",
    "CommandIssued",
    "ConsoleLog",
    "Event",
    "IdleCancelled",
    "IdleEvent",
    "IdleEventU",
    "IdleStarted",
    "IdleStopTriggered",
    "IdleWarning",
    "PlayerAdvancement",
    "PlayerDeath",
    "PlayerEvent",
    "PlayerEventU",
    "PlayerJoined",
    "PlayerLeft",
    "RuntimeEventU",
    "RuntimeRestored",
    "RuntimeStatusEvent",
    "RuntimeUnavailable",
    "ServerCrashed",
    "ServerEvent",
    "ServerEventU",
    "ServerReady",
    "ServerStarting",
    "ServerStopped",
    "ServerStopping",
]


# --------------------------------------------------------------------------------------- bases


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Base of every event.

    Attributes:
        ts: When the fact happened, always tz-aware UTC. For log-derived events this is the
            timestamp Docker prefixed onto the line (``logs(timestamps=True)``), falling back to
            the moment we received it. It is never parsed out of the line text: Paper's
            ``[13:24:37]`` is time-only in ``Asia/Kolkata``, which needs midnight-rollover handling
            and is simply wrong for backfilled lines.
        server_id: Which managed server this is about. Present from day one so a second server is
            configuration, not a refactor.
        source: Provenance - how we know this.
        raw: The original line, or a short rendering of the originating fact. Never ``None`` for
            log-derived events.
        seq: Monotonic sequence number stamped by the bus at publish time. Total order across all
            events, which is what makes ``PlayerJoined -> IdleCancelled -> PlayerLeft ->
            IdleStarted`` a checkable causal chain rather than four independent observations.
    """

    ts: datetime
    server_id: ServerId
    source: Source
    raw: str | None = None
    seq: int = 0

    @property
    def name(self) -> str:
        """The event's type name, e.g. ``"PlayerJoined"``. Used in logs, filters and serde."""
        return type(self).__name__

    def with_seq(self, seq: int) -> Self:
        """Return a copy stamped with ``seq``. The bus's only mutation of a published event."""
        return replace(self, seq=seq)


@dataclass(frozen=True, slots=True, kw_only=True)
class ServerEvent(Event):
    """Lifecycle of the game server process itself.

    A category base so a subscriber can say "everything about the server coming up or going down"
    in one subscription; the bus matches by type with subclass semantics.

    Deliberately does **not** include :class:`ConsoleLog` or :class:`ChatMessage`: those are
    unbounded-volume, and a Discord status subscriber that accidentally got the console firehose
    would be a bad surprise.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayerEvent(Event):
    """Something a specific player did. Subscribing to this gets all four player events."""

    player: PlayerRef


@dataclass(frozen=True, slots=True, kw_only=True)
class IdleEvent(Event):
    """The idle-shutdown state machine narrating itself."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeStatusEvent(Event):
    """The container runtime's availability, as distinct from the server's."""


# ------------------------------------------------------------------------------ server events


@dataclass(frozen=True, slots=True, kw_only=True)
class ServerStarting(ServerEvent):
    """The container is running; the server has not answered yet.

    Attributes:
        version: Parsed from ``Starting minecraft server version 26.2`` when that line is what
            produced this event; ``None`` when the trigger was a docker ``start`` event.
        container_id: The freshly resolved container id. Worth carrying because a
            ``compose down/up`` changes it, and it is the first half of the session identity.
        requested_by: Who asked, when we asked. ``None`` for an out-of-band start.
    """

    version: str | None = None
    container_id: str | None = None
    requested_by: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ServerReady(ServerEvent):
    """The server is answering. Fired exactly once per start.

    Attributes:
        startup_seconds: How long the server took to come up. Parsed from ``Done (32.521s)!`` when
            :attr:`detected_by` is ``LOG_DONE``, otherwise measured from the container's
            ``StartedAt``.
        version: Server version, if we have seen it this run.
        detected_by: Which of the three independent signals fired first. All three assert the same
            fact (``mc-health`` is itself an SLP query), but the log line arrives around 90 seconds
            before the healthcheck can, and telling Discord the server is up 90 seconds late is bad
            UX for zero correctness gain. Recording the signal keeps that policy auditable and
            makes reverting to health-only a one-line change.
    """

    startup_seconds: float | None = None
    version: str | None = None
    detected_by: ReadySignal


@dataclass(frozen=True, slots=True, kw_only=True)
class ServerStopping(ServerEvent):
    """A stop is in flight. Emitted on our own stop intent and on ``[Rcon: Stopping the server]``.

    While this is the current state, a ``lost connection: Server closed`` is a shutdown casualty,
    not a voluntary leave - see :class:`~mcmanager.core.types.LeaveReason.SERVER_CLOSED`.

    Attributes:
        reason: Free text, e.g. ``"idle timeout"``, ``"discord /stop"``.
        requested_by: Actor, for the audit trail and the session summary.
        timeout_seconds: The grace period the JVM is being given to save the world.
    """

    reason: str | None = None
    requested_by: str | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ServerStopped(ServerEvent):
    """The container exited, and we are content that it was meant to.

    Attributes:
        exit_code: Verified on this container: a graceful stop exits **0**, because
            ``mc-server-runner`` traps SIGTERM, writes ``stop`` and exits cleanly. 143 (SIGTERM) is
            also clean. 137 is SIGKILL after the grace period expired.
        clean: The world had time to save.
        forced: Docker had to SIGKILL. With a stop intent that means the JVM did not finish saving
            inside the timeout - a warning worth surfacing, not a crash.
        uptime_seconds: How long the server ran, for the session summary.
    """

    exit_code: int | None = None
    clean: bool
    forced: bool = False
    uptime_seconds: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ServerCrashed(ServerEvent):
    """The container died without a stop intent, or the OOM killer took it.

    Attributes:
        exit_code: Docker reports ``Actor.Attributes["exitCode"]`` as a *string*; it is parsed to
            ``int`` at the runtime boundary and is ``None`` if it was missing or unparsable.
        oom_killed: ``State.OOMKilled``. Decisive on its own: OOM is never a clean stop.
        tail: The last lines of console output, taken from the log pipeline's ring buffer
            **before** the stream closed. A post-hoc ``logs_tail()`` races the container going
            away; this does not, and is free.
    """

    exit_code: int | None = None
    oom_killed: bool = False
    tail: tuple[str, ...] = ()


# ------------------------------------------------------------------------------ player events


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayerJoined(PlayerEvent):
    """A player finished connecting.

    Attributes:
        online_count: Roster size *after* this join, so a presenter never has to ask.
        address: The client IP, stitched on by the log pipeline from the preceding
            ``Steve[/115.99.245.156:49237] logged in`` line. **Never relay this to Discord.** It is
            carried because idle logic and abuse investigation want it and re-deriving it later is
            impossible; presenters must drop it.
        first_seen: True if this name has no prior session record.
    """

    online_count: int = 0
    address: str | None = None
    first_seen: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayerLeft(PlayerEvent):
    """A player stopped being online.

    Attributes:
        reason: Classified, not raw. ``SERVER_CLOSED`` means they were dropped by a shutdown and
            must not be counted as having quit.
        session_seconds: How long they were on this session, or ``None`` if we never saw them join
            (daemon started mid-session).
        online_count: Roster size *after* this leave. Zero is what arms the idle timer.
    """

    reason: LeaveReason = LeaveReason.UNKNOWN
    session_seconds: float | None = None
    online_count: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayerDeath(PlayerEvent):
    """A player died.

    Attributes:
        message: The full rendered death message, ready to display.
        killer: The entity or player responsible, when the template has one.
        item: The named weapon, when the template has one (``... using [Sharpness Sword]``).
        template: The vanilla ``%1$s/%2$s/%3$s`` template that matched, or ``None`` for a tier-2
            match. ``None`` is the signal that this came from the pipeline's
            "message starts with an online player's name" fallback rather than the death table, and
            those are logged under ``mcmanager.deaths.unmatched`` so the table can grow.
    """

    message: str
    killer: str | None = None
    item: str | None = None
    template: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayerAdvancement(PlayerEvent):
    """A player earned an advancement, goal or challenge.

    Attributes:
        title: The bracketed title with the brackets removed, e.g. ``"Stone Age"``.
        kind: Which of the three tiers, because each has its own wording and colour.
    """

    title: str
    kind: AdvancementKind = AdvancementKind.ADVANCEMENT


# ----------------------------------------------------------------------- chat and console
#
# Neither is a PlayerEvent: PlayerEvent is exactly the four lifecycle-of-a-player facts, and a
# subscriber asking for "what players did" should not be handed the chat firehose.


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatMessage(Event):
    """A player said something.

    ``<Bob> Alice joined the game`` produces exactly one of these and **zero**
    :class:`PlayerJoined` events. That is not an incidental property: the old bridge script
    dispatched on ``"joined the game" in line``, which let any player forge a join notification.
    Chat is matched before join, and every pattern is anchored to the message payload.

    Attributes:
        player: The speaker. For ``RCON``-kind messages this may be a console pseudo-name.
        message: The message body, ANSI already stripped. Presenters are responsible for escaping
            it - ``@everyone``, backticks and 4KB of text are all expected inputs.
        kind: Chat, emote, /say or rcon.
    """

    player: PlayerRef
    message: str
    kind: ChatKind = ChatKind.CHAT


@dataclass(frozen=True, slots=True, kw_only=True)
class ConsoleLog(Event):
    """A log line that matched no specific pattern.

    The only unbounded-volume event, and therefore the only one the bus is allowed to drop: on a
    full queue an incoming ``ConsoleLog`` is dropped, and for any other event the oldest
    ``ConsoleLog`` is evicted to make room. The log pipeline additionally token-buckets before
    publishing, so the queue is never the thing that saves you.

    Attributes:
        message: The line with the prefix and ANSI removed.
        level: ``INFO`` / ``WARN`` / ``ERROR`` where the grammar provides one.
        thread: e.g. ``"Server thread"``. Only ``Server thread`` + ``INFO`` lines are ever
            candidates for player events, and that single guard kills a whole class of false
            positives.
        origin: Which of the three interleaved grammars this line came from. A rising ratio of
            :attr:`~mcmanager.core.types.LineOrigin.RAW` is the canary for a Paper upgrade breaking
            the patterns.
        stream: stdout or stderr.
    """

    message: str
    level: str | None = None
    thread: str | None = None
    origin: LineOrigin = LineOrigin.RAW
    stream: Stream = Stream.STDOUT


# -------------------------------------------------------------------------------- idle events


@dataclass(frozen=True, slots=True, kw_only=True)
class IdleStarted(IdleEvent):
    """The roster emptied and the shutdown countdown is armed.

    Attributes:
        deadline: When the stop would happen, tz-aware UTC. Persisted, but deliberately
            **restarted from now** on daemon boot - otherwise a crash-looping manager would
            repeatedly insta-stop the server.
        empty_since: When the last player left.
        timeout_seconds: The configured window, so a presenter can say "15 minutes" without
            reading config.
        dry_run: True while ``idle.dry_run`` is set. Everything is computed and logged; nothing is
            stopped.
    """

    deadline: datetime
    empty_since: datetime
    timeout_seconds: float
    dry_run: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class IdleWarning(IdleEvent):
    """The countdown is nearly up. Announced so a player about to reconnect has a chance.

    Attributes:
        remaining_seconds: Time left before the stop.
        deadline: The same deadline :class:`IdleStarted` carried.
    """

    remaining_seconds: float
    deadline: datetime
    dry_run: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class IdleCancelled(IdleEvent):
    """The countdown was disarmed.

    Attributes:
        reason: ``"player_joined"``, ``"server_stopped"``, ``"disabled"``, or
            ``"daemon_shutdown"``. That last one matters: on shutdown the idle manager cancels its
            timer, publishes this, and explicitly does **not** trigger a stop. An idle timer firing
            during teardown and killing the server because the manager restarted would be the
            single most dangerous bug in this design.
        idle_seconds: How long the server had been empty when the countdown was cancelled.
    """

    reason: str
    idle_seconds: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class IdleStopTriggered(IdleEvent):
    """The idle deadline fired.

    Attributes:
        idle_seconds: How long the server was empty.
        dry_run: When true, this event was published and **no stop was issued**. The cutover soak
            runs in this mode for days: every one of these must correspond to a genuinely empty
            server.
        uptime_seconds: Server uptime at trigger time, for auditing ``min_uptime_minutes``.
    """

    idle_seconds: float
    dry_run: bool = False
    uptime_seconds: float | None = None


# ----------------------------------------------------------------------------- runtime events


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeUnavailable(RuntimeStatusEvent):
    """The Docker daemon went away.

    The most important of the three additions to the specified event set. Without it, lifecycle
    cannot tell "inspect failed" from "the server stopped", and would emit a spurious
    :class:`ServerStopped` every time the socket hiccups. The lifecycle reducer enters ``BLIND``
    here and retains last-known state.

    Published **edge-triggered, once per outage**, never once per failed poll.

    Attributes:
        error: The exception's message. Not the traceback.
        endpoint: Which socket or URL we failed against - the single most useful thing to print,
            because this error is a permissions problem on ``/var/run/docker.sock`` most of the
            time.
    """

    error: str
    endpoint: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeRestored(RuntimeStatusEvent):
    """The Docker daemon answered again. Lifecycle re-reconciles and emits only the real delta.

    Attributes:
        downtime_seconds: How long we were blind, measured on the monotonic clock.
    """

    downtime_seconds: float | None = None


# -------------------------------------------------------------------------------------- audit


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandIssued(Event):
    """Somebody asked for a mutating action. The audit trail.

    Emitted before the action is attempted, so a session summary can say "stopped by @kunal" and a
    stop that never completed is still visible.

    Attributes:
        action: start / stop / restart.
        actor: Human-readable: a Discord user, ``"cli"``, ``"idle-manager"``.
        via: Which interface it arrived through.
        dry_run: The action was recorded but deliberately not performed.
        accepted: False when the controller rejected it (wrong state, missing permission), with
            :attr:`rejection` saying why.
        rejection: Why it was rejected, or ``None``.
    """

    action: ControlAction
    actor: str
    via: Source = Source.INTERNAL
    dry_run: bool = False
    accepted: bool = True
    rejection: str | None = None


# ------------------------------------------------------------------------------ union aliases
#
# These exist so `match` statements in presenters and serde can end with `assert_never(event)` and
# have pyright prove exhaustiveness. Adding an event without adding it here is a type error at
# every consumer, which is exactly the failure we want.

type ServerEventU = ServerStarting | ServerReady | ServerStopping | ServerStopped | ServerCrashed
type PlayerEventU = PlayerJoined | PlayerLeft | PlayerDeath | PlayerAdvancement
type IdleEventU = IdleStarted | IdleWarning | IdleCancelled | IdleStopTriggered
type RuntimeEventU = RuntimeUnavailable | RuntimeRestored
type AnyEvent = (
    ServerEventU
    | PlayerEventU
    | IdleEventU
    | RuntimeEventU
    | ChatMessage
    | ConsoleLog
    | CommandIssued
)


EVENT_TYPES: tuple[type[Event], ...] = (
    # server
    ServerStarting,
    ServerReady,
    ServerStopping,
    ServerStopped,
    ServerCrashed,
    # player
    PlayerJoined,
    PlayerLeft,
    PlayerDeath,
    PlayerAdvancement,
    # chat / console
    ChatMessage,
    ConsoleLog,
    # idle
    IdleStarted,
    IdleWarning,
    IdleCancelled,
    IdleStopTriggered,
    # runtime + audit
    RuntimeUnavailable,
    RuntimeRestored,
    CommandIssued,
)
"""Every concrete event type, in a stable order. Serde builds its discriminated union from this
and the CLI's ``--type`` filter resolves against it, so a new event needs exactly one edit."""

EVENT_BY_NAME: dict[str, type[Event]] = {t.__name__: t for t in EVENT_TYPES}
"""Name -> type, for deserialisation and for ``mcmanager events --type PlayerJoined``.

The category bases are intentionally absent: ``--type PlayerEvent`` is resolved by
:mod:`mcmanager.core.bus`'s subclass matching, not by this table.
"""
