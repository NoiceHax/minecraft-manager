"""Log line to published event.

``line -> parse -> enrich -> rate limit -> publish``, plus the ring buffer that feeds
``ServerCrashed.tail``.

**All the statefulness the parser refuses to have lives here.** That is what keeps ``parse()`` a
pure total function and ``mcmanager replay`` possible with no daemon, no Docker and no clock:

- ``LOGIN`` stashes ``name -> address`` so the following
  :class:`~mcmanager.core.events.PlayerJoined` can carry it. **The address is a player's IP**, so
  it is carried on exactly one field of exactly one event - the one documented as holding it, which
  every presenter drops - and is **replaced with** :data:`REDACTED_ADDRESS` **in the published
  ``ConsoleLog``'s ``message`` and ``raw``**. That substitution is the mechanism behind the plan's
  "never relay raw": ``ConsoleLog`` has no sensitivity marker, so anything that reaches it reaches
  ``/logs`` SSE, its replay ring, ``mcmanager logs --follow`` and the M6 Discord console relay,
  which enqueues ``Event.raw`` verbatim. The same applies to the parenthesised address in the
  ``lost connection`` spelling. ``mcmanager logs --raw`` reads Docker directly and still shows the
  true line, which is where an unredacted copy belongs.
- ``UUID of player X is ...`` stashes the offline v3 UUID for the same join.
- ``LOST_CONN`` and the kick line stash the disconnect reason for the following
  :class:`~mcmanager.core.events.PlayerLeft`.
- **A leave during a shutdown is not a voluntary quit.** ``bharath_720 lost connection: Server
  closed`` is a real line from this server, and counting it as a quit corrupts every session
  summary. Two independent defences: the reason string itself classifies as ``SERVER_CLOSED``, and
  a ``left the game`` with *no* reason line at all while the pipeline knows a stop is in flight is
  also ``SERVER_CLOSED`` rather than ``UNKNOWN``. What is deliberately *not* done is overriding an
  explicit ``Disconnected`` during a stop: somebody who quits two seconds before a shutdown really
  did quit, and the log said so.
- A ``ConsoleLog`` whose message begins with a name in the **current online set** *and* whose
  remainder reads like a death (:func:`~mcmanager.games.minecraft.deaths.looks_like_death`) is
  upgraded to ``PlayerDeath(template=None)`` - the tier-2 fallback that covers plugin and modded
  death messages - and logged under ``mcmanager.deaths.unmatched`` so the vanilla table can be
  grown from real data. ``template is None`` is the unambiguous marker for a tier-2 match. This is
  the clean answer to "the parser cannot know who is online": it does not need to, the enricher
  does. The second half of the test is not optional: ``online-mode=false`` means a player may pick
  a name that prefixes console chatter, and the online-name test alone turns every autosave into
  fabricated deaths.

**Ring buffer, not a post-hoc fetch.** ``ServerCrashed.tail`` comes from the 40 lines already in
this buffer. Calling ``logs_tail()`` after the fact races the container going away and the stream
closing; the buffer is already populated by the time the ``die`` event lands, so it is both free
and race-free.

**Token bucket before publishing** (200/s, burst 1000), coalescing overflow into a single
``... N lines suppressed ...`` line. A Paper crash dumps thousands of stack frames per second. The
bus has its own eviction policy for ``ConsoleLog``, but that is the second line of defence: by the
time the bus is evicting, every subscriber has already been woken thousands of times. Only
``ConsoleLog`` is ever suppressed - a ``PlayerJoined`` is never dropped, because losing one
silently corrupts the roster and therefore the idle timer.

**On the game seam.** This module imports two things from ``games/minecraft/``: the stateless
enrichment matchers in :mod:`~mcmanager.games.minecraft.patterns` and the thread-name guards in
:mod:`~mcmanager.games.minecraft.lines`. That is a real dent in "services are game-agnostic", and
it is inherited from the design: the plan puts stateful enrichment here precisely so the parser can
stay pure, and the alternative - a second copy of ``LOGIN_RE`` in this file - is strictly worse,
because the two copies would drift. The clean fix is a small addition to
:class:`~mcmanager.games.base.GameAdapter` (an ``enrichment`` accessor returning the matchers), at
which point this module imports nothing game-specific again.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING, Final, Protocol, final

from mcmanager.core.events import (
    ChatMessage,
    ConsoleLog,
    Event,
    PlayerAdvancement,
    PlayerDeath,
    PlayerEvent,
    PlayerJoined,
    PlayerLeft,
    ServerReady,
    ServerStarting,
    ServerStopping,
)
from mcmanager.core.types import LeaveReason, LineOrigin, PlayerRef, Source, Stream
from mcmanager.games.minecraft.deaths import looks_like_death
from mcmanager.games.minecraft.lines import SERVER_THREAD, USER_AUTHENTICATOR_THREAD_RE
from mcmanager.games.minecraft.patterns import (
    PLAYER_NAME,
    match_command,
    match_kick,
    match_login,
    match_lost_connection,
    match_uuid,
)
from mcmanager.logging_setup import get_logger, log_chat, sanitise_for_log

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mcmanager.clock import Clock
    from mcmanager.containers.dto import LogLine
    from mcmanager.core.types import ServerId
    from mcmanager.games.base import GameAdapter
    from mcmanager.services.players import PlayerRoster

__all__ = [
    "DEFAULT_RATE_BURST",
    "DEFAULT_RATE_PER_SECOND",
    "DEFAULT_TAIL_LINES",
    "REDACTED_ADDRESS",
    "EventSink",
    "LogPipeline",
    "TokenBucket",
]

_log: Final = get_logger("mcmanager.log_pipeline")
_deaths_log: Final = get_logger("mcmanager.deaths.unmatched")
"""Where tier-2 death matches are recorded, named in the plan so the vanilla template table can be
grown from lines this deployment actually produced."""

DEFAULT_TAIL_LINES: Final = 40
"""How many recent lines are kept for ``ServerCrashed.tail``."""

DEFAULT_RATE_PER_SECOND: Final = 200.0
DEFAULT_RATE_BURST: Final = 1000.0

REDACTED_ADDRESS: Final = "<redacted>"
"""What a player's IP is replaced with in every event this pipeline publishes.

Deliberately visible rather than blank: an operator reading ``/logs`` should be able to tell that
an address was there and was removed, not wonder whether Paper changed its log format again.
"""

_PENDING_MAX: Final = 64
"""Cap on each ``name -> pending fact`` map.

Not paranoia: the real archives contain ``KittyScan (/176.65.148.158:58184) lost connection:
Disconnected`` - a port scanner that authenticated and hung up without ever joining. Those produce
pending facts that nothing ever consumes, and an uncapped dict would grow for as long as the daemon
runs.
"""

_LEADING_NAME_RE: Final = re.compile(rf"^(?P<name>{PLAYER_NAME}) (?P<rest>\S.*)$", re.DOTALL)
"""``Steve was blown up by Creeper`` split into name and remainder. Reuses the parser's own
character class, which cannot contain ``[``, ``]``, ``:`` or a space - the load-bearing half of the
fix for the old bridge script's greedy-capture bug."""

_NON_DEATH_PREFIXES: Final = "<[*("
"""First characters that disqualify a line from the tier-2 death fallback.

Chat, ``/say``, ``/me`` and the ``[Rcon: ...]`` echo are all recognised earlier and can never reach
here, so this is redundant by construction. It is kept because the failure it guards against -
somebody's words being republished as a death message - is exactly the class of bug this project
exists to stop repeating, and redundancy is cheap.
"""


class EventSink(Protocol):
    """The half of :class:`~mcmanager.core.bus.EventBus` this module needs.

    Narrowed to one method deliberately, matching
    :class:`mcmanager.containers.manager.EventSink`: the pipeline publishes and subscribes to
    nothing, so depending on the whole bus would overstate the coupling and would make every test
    here need a running dispatch loop. ``EventBus`` satisfies it structurally, and a test passes a
    list's ``append`` wrapped in one line.
    """

    def publish(self, event: Event) -> None: ...


@final
class TokenBucket:
    """Classic token bucket on an injected clock.

    Measured on :meth:`~mcmanager.clock.Clock.monotonic`, never the wall clock: an NTP step during
    a log storm must not hand out a million tokens at once.
    """

    __slots__ = ("_burst", "_clock", "_rate", "_tokens", "_updated")

    def __init__(self, *, rate: float, burst: float, clock: Clock) -> None:
        """Create a bucket.

        Args:
            rate: Tokens refilled per second.
            burst: Bucket capacity, and the number of tokens it starts full with.
            clock: Injected time.

        Raises:
            ValueError: If ``rate`` or ``burst`` is not positive. A zero-rate bucket would suppress
                every line forever after the burst, which is a configuration mistake worth failing
                on rather than discovering from an empty console channel.
        """
        if rate <= 0:
            msg = "TokenBucket rate must be positive"
            raise ValueError(msg)
        if burst <= 0:
            msg = "TokenBucket burst must be positive"
            raise ValueError(msg)
        self._rate = rate
        self._burst = burst
        self._clock = clock
        self._tokens = burst
        self._updated = clock.monotonic()

    @property
    def tokens(self) -> float:
        """Tokens available right now, after refilling. For assertions and diagnostics."""
        self._refill()
        return self._tokens

    def take(self, tokens: float = 1.0) -> bool:
        """Consume ``tokens`` if they are available. Returns False without consuming otherwise."""
        self._refill()
        if self._tokens < tokens:
            return False
        self._tokens -= tokens
        return True

    def _refill(self) -> None:
        now = self._clock.monotonic()
        elapsed = now - self._updated
        if elapsed <= 0:
            # A monotonic clock never goes backwards, but it can report the same value twice, and
            # a negative elapsed would silently drain the bucket.
            self._updated = now
            return
        self._updated = now
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)


@final
class LogPipeline:
    """Turns container log lines into published events, and holds the state the parser will not.

    Wire it as ``DockerManager(on_line=pipeline.handle_line, on_log_eof=pipeline.on_eof)``. Both
    are synchronous, because the bus's ``publish`` is synchronous and the manager's pump calls back
    from the loop thread without awaiting.
    """

    __slots__ = (
        "_adapter",
        "_bucket",
        "_clock",
        "_pending_address",
        "_pending_leave",
        "_pending_uuid",
        "_roster",
        "_server_id",
        "_sink",
        "_stats",
        "_stopping",
        "_suppressed",
        "_tail",
    )

    def __init__(
        self,
        *,
        adapter: GameAdapter,
        sink: EventSink,
        clock: Clock,
        server_id: ServerId,
        roster: PlayerRoster,
        tail_lines: int = DEFAULT_TAIL_LINES,
        rate_per_second: float = DEFAULT_RATE_PER_SECOND,
        rate_burst: float = DEFAULT_RATE_BURST,
    ) -> None:
        """Wire the pipeline.

        Args:
            adapter: The game adapter whose ``parse_line`` turns a raw line into exactly one event.
            sink: Where events go. The bus, in production.
            clock: Injected time. Used for the token bucket and for the timestamps on the events
                this module synthesises; parsed events carry Docker's timestamp instead.
            server_id: Stamped onto synthesised events.
            roster: The authoritative online set. The pipeline both updates it (from join and leave
                lines) and reads it (for ``online_count`` and the tier-2 death fallback).
            tail_lines: Ring buffer size for ``ServerCrashed.tail``.
            rate_per_second: Token bucket refill rate.
            rate_burst: Token bucket capacity.
        """
        self._adapter = adapter
        self._sink = sink
        self._clock = clock
        self._server_id = server_id
        self._roster = roster
        self._tail: deque[str] = deque(maxlen=max(tail_lines, 0))
        self._bucket = TokenBucket(rate=rate_per_second, burst=rate_burst, clock=clock)

        self._pending_address: dict[str, str] = {}
        self._pending_uuid: dict[str, str] = {}
        self._pending_leave: dict[str, LeaveReason] = {}
        self._stopping = False
        self._suppressed = 0

        self._stats: dict[str, int] = {
            "lines": 0,
            "published": 0,
            "suppressed": 0,
            "suppression_notices": 0,
            "server_lines": 0,
            "console_lines": 0,
            "recognised_console_lines": 0,
            "deaths_tier2": 0,
            "deaths_tier2_rejected": 0,
            "addresses_redacted": 0,
            "addresses_stashed": 0,
            "leave_reasons_stashed": 0,
            "uuids_stashed": 0,
            "commands_seen": 0,
            "handler_errors": 0,
        }

    # -------------------------------------------------------------------------- introspection

    @property
    def stopping(self) -> bool:
        """True while a shutdown is believed to be in flight.

        Set by the ``Stopping the server`` line and by :meth:`set_stopping` when the controller
        issues a stop of its own; cleared when the server starts or reports ready.
        """
        return self._stopping

    @property
    def stats(self) -> Mapping[str, int]:
        """Counters, polled rather than pushed. ``suppressed`` climbing means the console is
        flooding and the bus is being protected from it."""
        return dict(self._stats)

    @property
    def pending_suppressed(self) -> int:
        """Lines dropped since the last ``... N lines suppressed ...`` notice."""
        return self._suppressed

    def tail(self, limit: int | None = None) -> tuple[str, ...]:
        """The most recent lines, oldest first. This is ``ServerCrashed.tail``.

        Captured from the ring buffer rather than re-fetched: the stream EOF and the ``die`` event
        race each other, and a ``logs_tail()`` issued after the fact can lose to the container
        going away entirely.
        """
        if limit is None or limit >= len(self._tail):
            return tuple(self._tail)
        if limit <= 0:
            return ()
        return tuple(self._tail)[-limit:]

    # -------------------------------------------------------------------------------- control

    def set_stopping(self, stopping: bool) -> None:
        """Tell the pipeline a stop is (or is no longer) in flight.

        Called by the controller when *we* issue the stop, which happens before the server has
        logged anything about it. Without this, the window between "we asked Docker to stop it" and
        "the JVM logged ``Stopping the server``" would classify leaves as voluntary quits.
        """
        self._stopping = stopping

    def reset(self) -> None:
        """Forget every pending enrichment fact. Called when the server run changes.

        Not the ring buffer: a crash tail is most useful precisely when everything else has been
        reset.
        """
        self._pending_address.clear()
        self._pending_uuid.clear()
        self._pending_leave.clear()

    def on_eof(self) -> None:
        """Called on a clean end of the log stream. Flushes any pending suppression notice.

        A clean EOF means the container stopped; it corroborates the ``die`` event and is not an
        error. Flushing here is what stops a suppression notice sitting in memory until the next
        server start, where it would be attributed to the wrong run.
        """
        self.flush()

    def flush(self) -> None:
        """Publish a pending ``... N lines suppressed ...`` notice, if there is one."""
        self._flush_suppressed()

    # ------------------------------------------------------------------------------- ingestion

    def handle_line(self, line: LogLine) -> None:
        """The ``DockerManager`` line callback. Never raises.

        A raising line handler takes the log pump down with it, and the input is
        attacker-influenced: chat is a player-controlled string that reaches these regexes. The
        parser is already total, so anything caught here is a bug in *this* module, counted and
        logged rather than propagated into the reconnect loop.
        """
        try:
            self._handle(line.text, ts_source=line, stream=line.stream)
        except Exception:
            self._stats["handler_errors"] += 1
            _log.exception("log_pipeline.handle_line_failed", stream=line.stream.value)

    def handle_raw(self, text: str, *, stream: Stream = Stream.STDOUT) -> None:
        """Ingest a bare line, timestamping it from the clock.

        For callers that have text but no :class:`~mcmanager.containers.dto.LogLine`: replaying a
        fixture, or a test. Production always goes through :meth:`handle_line`, because Docker's
        timestamp is the only one that survives backfill.
        """
        try:
            self._handle(text, ts_source=None, stream=stream)
        except Exception:
            self._stats["handler_errors"] += 1
            _log.exception("log_pipeline.handle_raw_failed")

    def handle_lines(self, lines: Iterable[LogLine]) -> None:
        """Ingest a batch, in order."""
        for line in lines:
            self.handle_line(line)

    def _handle(self, text: str, *, ts_source: LogLine | None, stream: Stream) -> None:
        self._stats["lines"] += 1
        ts = ts_source.event_ts if ts_source is not None else self._clock.now()
        event = self._adapter.parse_line(
            text,
            ts=ts,
            server_id=self._server_id,
            stream=stream,
        )
        # The ring buffer holds the *sanitised* line: it is rendered into Discord as a crash tail,
        # and putting escape bytes there would re-create, one layer down, the exact defect this
        # project exists to fix. `mcmanager logs --raw` still shows the true original.
        self._tail.append(event.raw if event.raw is not None else text)
        self._emit(self._enrich(event))

    # ------------------------------------------------------------------------------ enrichment

    def _enrich(self, event: Event) -> Event:
        """Apply every stateful transformation. Returns the event to publish.

        Ordered by frequency: ``ConsoleLog`` is the overwhelming majority of lines (2,905 of 3,419
        in the real archives), chat is next, and the lifecycle events are rare.
        """
        if isinstance(event, ConsoleLog):
            return self._enrich_console(event)
        if isinstance(event, ChatMessage):
            log_chat(
                _log,
                kind=event.kind.value,
                player=event.player.name,
                message=event.message,
            )
            return event
        if isinstance(event, PlayerJoined):
            return self._enrich_join(event)
        if isinstance(event, PlayerLeft):
            return self._enrich_leave(event)
        if isinstance(event, PlayerDeath | PlayerAdvancement):
            return self._with_known_uuid(event)
        if isinstance(event, ServerStopping):
            self._stopping = True
            return event
        if isinstance(event, ServerStarting):
            # A new run of the server. Nobody carried over from the last one, and a stale roster
            # entry would keep the idle timer disarmed forever.
            self._stopping = False
            self._roster.clear()
            self.reset()
            return event
        if isinstance(event, ServerReady):
            self._stopping = False
        return event

    def _enrich_join(self, event: PlayerJoined) -> PlayerJoined:
        """Attach the address, the uuid, the roster count and whether this name is new."""
        name = event.player.name
        address = self._pending_address.pop(name, None)
        uuid = event.player.uuid or self._pending_uuid.pop(name, None)
        session = self._roster.join(name, at=event.ts, uuid=uuid, address=address)
        _log.info(
            "player.joined",
            player=name,
            online=self._roster.count,
            first_seen=session.first_seen,
            # The address itself is never logged: it is a player's IP, and an ops log is not a
            # place to republish one. Whether we have it is the operationally useful half.
            has_address=address is not None,
        )
        return replace(
            event,
            player=PlayerRef(name=name, uuid=session.uuid),
            online_count=self._roster.count,
            address=session.address,
            first_seen=session.first_seen,
        )

    def _enrich_leave(self, event: PlayerLeft) -> PlayerLeft:
        """Attach the classified reason, the session duration and the roster count."""
        name = event.player.name
        reason = self._pending_leave.pop(name, None)
        if reason is None:
            # No "lost connection" line preceded this one. During a shutdown that is exactly what a
            # shutdown casualty looks like, and calling it UNKNOWN would let a session summary
            # count it as a voluntary quit.
            reason = LeaveReason.SERVER_CLOSED if self._stopping else LeaveReason.UNKNOWN
        session = self._roster.leave(name, reason=reason)
        seconds = session.seconds_at(self._clock.monotonic()) if session is not None else None
        _log.info(
            "player.left",
            player=name,
            reason=reason.value,
            online=self._roster.count,
            session_seconds=seconds,
        )
        return replace(
            event,
            player=PlayerRef(name=name, uuid=session.uuid if session is not None else None),
            reason=reason,
            session_seconds=seconds,
            online_count=self._roster.count,
        )

    def _with_known_uuid(self, event: PlayerEvent) -> PlayerEvent:
        """Fill in the offline v3 UUID from the live session, when the line did not carry one."""
        if event.player.uuid is not None:
            return event
        session = self._roster.session(event.player.name)
        if session is None or session.uuid is None:
            return event
        return replace(event, player=PlayerRef(name=event.player.name, uuid=session.uuid))

    def _enrich_console(self, event: ConsoleLog) -> Event:
        """Stash enrichment facts, and apply the tier-2 death fallback.

        Every branch here returns the ``ConsoleLog`` unchanged except the death upgrade: these
        lines are recognised, they simply have no event of their own, and the facts they carry
        belong to a *neighbouring* event. Inventing four more event types for them would push the
        stitching into every consumer instead of keeping it in the one component allowed to hold
        state.
        """
        self._stats["console_lines"] += 1
        if event.origin is not LineOrigin.SERVER:
            return event

        message = event.message
        if event.thread is not None and USER_AUTHENTICATOR_THREAD_RE.match(event.thread):
            uuid_match = match_uuid(message) if event.level == "INFO" else None
            if uuid_match is not None:
                _stash(self._pending_uuid, uuid_match.name, uuid_match.uuid)
                self._stats["uuids_stashed"] += 1
                self._stats["recognised_console_lines"] += 1
            return event

        if event.thread != SERVER_THREAD or event.level != "INFO":
            return event
        self._stats["server_lines"] += 1

        login = match_login(message)
        if login is not None:
            # A player IP. Stashed so the next PlayerJoined can carry it, and *removed from the
            # event itself* - see _redact_address.
            _stash(self._pending_address, login.name, login.address)
            self._stats["addresses_stashed"] += 1
            self._stats["recognised_console_lines"] += 1
            return self._redact_address(event, login.address)

        lost = match_lost_connection(message)
        if lost is not None:
            _stash(self._pending_leave, lost.name, lost.reason)
            self._stats["leave_reasons_stashed"] += 1
            self._stats["recognised_console_lines"] += 1
            _log.debug(
                "player.lost_connection",
                player=lost.name,
                reason=lost.reason.value,
                raw_reason=sanitise_for_log(lost.raw_reason, limit=120),
            )
            # `KittyScan (/176.65.148.158:58184) lost connection: Disconnected` is a real line
            # from this deployment, and that parenthesised address is a player IP too.
            return event if lost.address is None else self._redact_address(event, lost.address)

        kick = match_kick(message)
        if kick is not None:
            _stash(self._pending_leave, kick.name, kick.reason)
            self._stats["leave_reasons_stashed"] += 1
            self._stats["recognised_console_lines"] += 1
            return event

        command = match_command(message)
        if command is not None:
            self._stats["commands_seen"] += 1
            self._stats["recognised_console_lines"] += 1
            _log.info(
                "player.command",
                player=command.name,
                command=sanitise_for_log(command.command, limit=200),
            )
            return event

        death = self._tier2_death(event)
        return death if death is not None else event

    def _redact_address(self, event: ConsoleLog, address: str) -> ConsoleLog:
        """Replace a player's IP with :data:`REDACTED_ADDRESS` in both ``message`` and ``raw``.

        **This is the mechanism behind "never relay raw".** Without it the guarantee was a comment:
        ``ConsoleLog`` carries no sensitivity marker, ``/logs`` serves it over SSE together with a
        512-entry replay ring, ``mcmanager logs --follow`` prints it, and the M6 Discord console
        relay - which subscribes to ``Event`` and enqueues ``event.raw``, precisely so recognised
        lines are not missing from the console channel - would post player IPs into a Discord
        channel. Dropping it in the presenter cannot work either, because every one of those
        consumers is a separate presenter and the address only has to survive one of them.

        Redacting here, once, at the only point that has already extracted the address, means no
        consumer can leak what it never receives. The true address is not lost: it is stashed for
        the following ``PlayerJoined.address``, which is the one field documented as carrying it
        and which presenters are required to drop. ``mcmanager logs --raw`` reads Docker directly
        and is unaffected, which is the correct place for the unredacted line to remain available.
        """
        self._stats["addresses_redacted"] += 1
        return replace(
            event,
            message=event.message.replace(address, REDACTED_ADDRESS),
            raw=None if event.raw is None else event.raw.replace(address, REDACTED_ADDRESS),
        )

    def _tier2_death(self, event: ConsoleLog) -> PlayerDeath | None:
        """Upgrade an unrecognised line that begins with an online player's name to a death.

        This is how plugin and modded death messages are covered while the parser stays pure and
        stateless. The guards, and why each one is here:

        - The line already failed every pattern in the fixed match order, including the ~100
          vanilla death templates.
        - The name must be **currently online**. This is the fact the parser cannot know.
        - There must be a remainder. ``Steve`` alone is not a death message.
        - **The remainder must read like a death**, per
          :func:`~mcmanager.games.minecraft.deaths.looks_like_death`. The online-name test is not
          sufficient on its own: this server runs ``online-mode=false``, so a player may simply
          choose a name that prefixes routine console chatter, and ``Saving``, ``Preparing``,
          ``Time``, ``Flushing`` and ``Closing`` are all legal names that begin real lines in this
          deployment's own archives.
        - The line must not begin with a chat or bracket marker. Redundant, and kept anyway.

        ``PlayerDeath.template is None`` is the unambiguous marker that this path produced the
        event, and every one is logged under ``mcmanager.deaths.unmatched`` so the vanilla table can
        be grown from lines this deployment really produced.
        """
        message = event.message
        if not message or message[0] in _NON_DEATH_PREFIXES:
            return None
        found = _LEADING_NAME_RE.match(message)
        if found is None:
            return None
        name = found["name"]
        session = self._roster.session(name)
        if session is None:
            return None
        if not looks_like_death(found["rest"]):
            # The name test alone is not a test. `online-mode=false` means anybody may log in as
            # `Saving`, and `Saving players` / `Saving chunks for level ...` are emitted on every
            # autosave - roughly every five minutes while somebody is playing, plus on every
            # /save-all and every shutdown. Recorded rather than dropped silently, because a
            # genuine plugin death phrased in a way the vanilla table has never used shows up
            # here, and that is the evidence for widening DEATH_TEMPLATES.
            self._stats["deaths_tier2_rejected"] += 1
            _deaths_log.debug(
                "death.candidate_rejected",
                player=name,
                message=sanitise_for_log(message),
                hint="starts with an online player's name but does not read like a death",
            )
            return None

        self._stats["deaths_tier2"] += 1
        _deaths_log.info(
            "death.unmatched",
            player=name,
            message=sanitise_for_log(message),
            hint="no vanilla template matched; add it to games/minecraft/deaths.py if it recurs",
        )
        return PlayerDeath(
            ts=event.ts,
            server_id=event.server_id,
            source=Source.LOG,
            raw=event.raw,
            player=PlayerRef(name=name, uuid=session.uuid),
            message=message,
            template=None,
        )

    # -------------------------------------------------------------------------------- emission

    def _emit(self, event: Event) -> None:
        """Rate limit, then publish.

        Only :class:`~mcmanager.core.events.ConsoleLog` is suppressible. Everything else consumes a
        token when one is available but is **always** published: dropping a ``PlayerJoined``
        corrupts the roster, and therefore the idle timer, and therefore eventually stops a
        populated server. The events that can flood are exactly the ones that can be dropped.
        """
        if isinstance(event, ConsoleLog):
            if not self._bucket.take():
                self._suppressed += 1
                self._stats["suppressed"] += 1
                return
            # The take succeeded, which is itself the proof that the bucket has recovered, so the
            # backlog notice goes out now - immediately before the line that survived, where it
            # reads as "and here is what you missed".
            self._flush_suppressed()
        else:
            self._bucket.take()
        self._publish(event)

    def _flush_suppressed(self) -> None:
        """Emit the coalesced ``... N lines suppressed ...`` notice.

        One notice per burst, not one per dropped line, which is the entire point. It is published
        without consuming a token: it is bounded by the number of times the bucket recovers, and a
        console that goes quiet with no explanation is worse than one extra line.
        """
        count = self._suppressed
        if count == 0:
            return
        self._suppressed = 0
        self._stats["suppression_notices"] += 1
        self._publish(
            ConsoleLog(
                ts=self._clock.now(),
                server_id=self._server_id,
                source=Source.INTERNAL,
                raw=None,
                message=f"... {count} lines suppressed ...",
                level="WARNING",
                origin=LineOrigin.RAW,
                stream=Stream.STDOUT,
            )
        )
        _log.warning("log_pipeline.rate_limited", suppressed=count)

    def _publish(self, event: Event) -> None:
        self._stats["published"] += 1
        self._sink.publish(event)


def _stash[T](mapping: dict[str, T], key: str, value: T) -> None:
    """Record a pending fact, evicting the oldest entry once the map is full.

    Insertion-ordered dicts make this a two-line LRU-ish cache with no dependency. The eviction is
    not a correctness concern: a pending fact that survives 64 other players' logins was never
    going to be consumed.
    """
    mapping.pop(key, None)
    mapping[key] = value
    while len(mapping) > _PENDING_MAX:
        mapping.pop(next(iter(mapping)))
