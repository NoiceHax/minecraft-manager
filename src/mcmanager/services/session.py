"""Session tracking and summaries. **Stub: real interface and wiring now, body in M5.**

A session spans the *game server's* lifetime, not the daemon's. Identity is
``(container_id, started_at)`` taken from the snapshot: stable across a daemon restart, and changed
by a ``compose down/up``, which is exactly the semantics wanted.

- On shutdown, the in-flight record is written with ``open=true``.
- On boot, it resumes if the identity still matches; otherwise the stale record is closed as
  ``daemon_missed_shutdown`` and a fresh one is opened with ``partial=true``, so the summary says
  honestly that its counts are incomplete.

This persistence is load-bearing rather than a nicety: the docker log driver is a ~30MB ring, so a
long session genuinely **cannot** be reconstructed after the fact.

**What is real here and what is not.** Identity, resume-versus-missed-shutdown, the counters, the
checkpoint and the shutdown ordering are implemented, because those are what ``app.py``'s wiring
has to exercise now. What M5 adds is the *record*: a durable per-session document with the roster,
the death list, the chat volume and the log-archive correlation, written through
:mod:`mcmanager.persistence.session_log`. :meth:`SessionManager._write_summary` is the seam, and it
currently logs ``session.summary_not_implemented`` and writes nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, final

import structlog

from mcmanager.core.events import (
    ChatMessage,
    Event,
    PlayerAdvancement,
    PlayerDeath,
    PlayerEvent,
    PlayerJoined,
    PlayerLeft,
    ServerCrashed,
    ServerEvent,
    ServerReady,
    ServerStopped,
)
from mcmanager.persistence.state_store import SessionIdentity

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from mcmanager.clock import Clock
    from mcmanager.containers.dto import ContainerSnapshot
    from mcmanager.core.bus import EventBus, Subscription
    from mcmanager.core.types import ServerId
    from mcmanager.persistence.session_log import SessionLog
    from mcmanager.persistence.state_store import StateStore
    from mcmanager.services.players import PlayerRoster

__all__ = ["DAEMON_MISSED_SHUTDOWN", "EventSink", "SessionManager"]

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.session")

DAEMON_MISSED_SHUTDOWN = "daemon_missed_shutdown"
"""How a session that was open when the daemon vanished is closed on the next boot. The successor
session is then marked ``partial``, so its summary says honestly that its counts are incomplete."""


class EventSink(Protocol):
    """The narrow slice of :class:`~mcmanager.core.bus.EventBus` this module needs."""

    def publish(self, event: Event) -> None: ...


@final
class SessionManager:
    """Opens, counts and closes one session per run of the game server."""

    def __init__(
        self,
        *,
        sink: EventSink,
        clock: Clock,
        server_id: ServerId,
        roster: PlayerRoster,
        store: StateStore,
        session_log: SessionLog | None = None,
    ) -> None:
        """Wire the manager.

        Args:
            sink: Where session events would go. Held for M5; this stub publishes nothing, because
                inventing a ``SessionClosed`` event before the record exists would put a fact on
                the bus that nothing can substantiate.
            clock: Injected time. Session durations are measured on ``monotonic``; the timestamps
                written into the record come from ``now``.
            server_id: Stamped onto anything this manager records.
            roster: Read at checkpoint time so ``known_players`` survives a restart, which is what
                makes ``PlayerJoined.first_seen`` mean "first time on this server".
            store: The daemon state file: the identity, the open flag, and the known-player set.
            session_log: Where durable session records go in M5. Optional, and unused today.
        """
        self._sink = sink
        self._clock = clock
        self._server_id = server_id
        self._roster = roster
        self._store = store
        self._session_log = session_log

        self._identity: SessionIdentity | None = None
        self._open = False
        self._partial = False
        self._opened_at: datetime | None = None
        self._opened_monotonic: float | None = None
        self._subscriptions: tuple[Subscription, ...] = ()
        self._closing = False

        self._counters: dict[str, int] = _fresh_counters()
        self._stats: dict[str, int] = {
            "opened": 0,
            "closed": 0,
            "resumed": 0,
            "missed_shutdowns": 0,
            "checkpoints": 0,
        }

    # -------------------------------------------------------------------------- introspection

    @property
    def identity(self) -> SessionIdentity | None:
        """``(container_id, started_at)`` of the session in flight, or ``None``."""
        return self._identity

    @property
    def is_open(self) -> bool:
        """True while a session is in flight."""
        return self._open

    @property
    def partial(self) -> bool:
        """True when this session started mid-run, so its counts are known to be incomplete."""
        return self._partial

    @property
    def counters(self) -> Mapping[str, int]:
        """Joins, leaves, deaths, advancements and chat lines seen this session."""
        return dict(self._counters)

    @property
    def stats(self) -> Mapping[str, int]:
        """Lifetime counters for this process. ``missed_shutdowns`` is the interesting one."""
        return dict(self._stats)

    def describe(self) -> dict[str, object]:
        """A JSON-safe summary, for ``mcmanager status`` and ``/status``."""
        return {
            "open": self._open,
            "partial": self._partial,
            "container_id": None if self._identity is None else self._identity.container_id,
            "started_at": (
                None if self._identity is None else self._identity.started_at.isoformat()
            ),
            "opened_at": None if self._opened_at is None else self._opened_at.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "online": self._roster.count,
            "counters": dict(self._counters),
        }

    @property
    def uptime_seconds(self) -> float | None:
        """How long the current session has been observed by *this* daemon process.

        Deliberately not "how long the server has run": after a resume those are different numbers,
        and conflating them is how a summary claims a four-hour session lasted ninety seconds.
        """
        if self._opened_monotonic is None:
            return None
        return max(self._clock.monotonic() - self._opened_monotonic, 0.0)

    # --------------------------------------------------------------------------------- wiring

    def subscribe(self, bus: EventBus) -> tuple[Subscription, ...]:
        """Register the subscriptions this manager needs, and return them for the tests."""
        self._subscriptions = (
            bus.subscribe(ServerEvent, self.on_server_event, name="session.server"),
            bus.subscribe(PlayerEvent, self.on_player_event, name="session.players"),
            bus.subscribe(ChatMessage, self.on_chat, name="session.chat"),
        )
        return self._subscriptions

    async def start(self) -> None:
        """Report what the state file says, before any snapshot has arrived."""
        state = self._store.state
        _log.info(
            "session.started",
            stored_open=state.session_open,
            stored_container_id=None if state.session is None else state.session.container_id,
            known_players=len(state.known_players),
        )

    # ------------------------------------------------------------------------------- handlers

    def on_snapshot(self, snapshot: ContainerSnapshot) -> None:
        """``DockerManager``'s ``on_snapshot`` fan-out. Where identity actually comes from.

        Synchronous, because ``DockerManager``'s callbacks are. A running container with an id and
        a ``StartedAt`` is a session; anything else is left alone, and the session is closed by the
        ``ServerStopped`` / ``ServerCrashed`` events rather than by an inspect, so a momentarily
        unreachable daemon cannot close a session that is still running.
        """
        if self._closing:
            return
        if not snapshot.running or snapshot.id is None or snapshot.started_at is None:
            return
        identity = SessionIdentity(container_id=snapshot.id, started_at=snapshot.started_at)
        if self._identity is not None and self._identity == identity:
            return
        if self._open:
            # A new identity while one is open means we missed the stop entirely.
            self._close("container_replaced")
        self._open_session(identity)

    async def on_server_event(self, event: ServerEvent) -> None:
        """Bus handler for the server's lifecycle."""
        if self._closing:
            return
        if isinstance(event, ServerStopped):
            self._close("server_stopped")
        elif isinstance(event, ServerCrashed):
            self._close("server_crashed")
        elif isinstance(event, ServerReady):
            self._counters["ready_signals"] += 1

    async def on_player_event(self, event: PlayerEvent) -> None:
        """Bus handler. Counting only; the roster is authoritative for who is online."""
        if self._closing:
            return
        if isinstance(event, PlayerJoined):
            self._counters["joins"] += 1
        elif isinstance(event, PlayerLeft):
            self._counters["leaves"] += 1
        elif isinstance(event, PlayerDeath):
            self._counters["deaths"] += 1
        elif isinstance(event, PlayerAdvancement):
            self._counters["advancements"] += 1

    async def on_chat(self, event: ChatMessage) -> None:
        """Bus handler. The **count** is recorded; the message body never is.

        Chat bodies are attacker-controlled text and are not the session record's business. M5's
        record keeps this same rule.
        """
        if self._closing:
            return
        _ = event
        self._counters["chat_lines"] += 1

    # ------------------------------------------------------------------------- checkpointing

    async def checkpoint(self) -> None:
        """Fold the in-memory session state into the state store. Does not write to disk.

        Writing is the daemon's separate "state flush" step, so a checkpoint during shutdown and
        the periodic one behave identically and only one place owns the ``fsync``.
        """
        self._stats["checkpoints"] += 1
        self._store.remember_players(self._roster.known_names)
        for name, value in self._counters.items():
            self._store.set_counter(f"session.{name}", value)
        if self._identity is not None and self._open:
            # Deliberately re-asserts open=true: a session spans the server's life, not ours.
            self._store.begin_session(self._identity)
        _log.debug(
            "session.checkpoint",
            open=self._open,
            partial=self._partial,
            counters=dict(self._counters),
        )

    async def aclose(self) -> None:
        """Checkpoint with the session still marked open, then stop listening.

        The session is **not** closed here. The daemon going away does not end the server's run,
        and writing ``open=false`` would make the next boot believe it had seen the whole session
        and report counts it never observed.

        Safe to call twice, and never raises.
        """
        if self._closing:
            return
        await self.checkpoint()
        self._closing = True
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions = ()
        _log.info(
            "session.closing_daemon",
            open=self._open,
            container_id=None if self._identity is None else self._identity.container_id,
            hint="the session stays open: the server outlives the daemon",
        )

    # ------------------------------------------------------------------------------ internals

    def _open_session(self, identity: SessionIdentity) -> None:
        """Open, resume, or close a stale record and open a partial successor."""
        state = self._store.state
        stored = state.session
        resumed = (
            state.session_open
            and stored is not None
            and stored.matches(
                container_id=identity.container_id,
                started_at=identity.started_at,
            )
        )
        if resumed:
            self._partial = True
            self._stats["resumed"] += 1
            _log.info(
                "session.resumed",
                container_id=identity.container_id,
                started_at=identity.started_at.isoformat(),
                hint="counts from before this daemon started are not ours to claim",
            )
        else:
            if state.session_open and stored is not None:
                self._stats["missed_shutdowns"] += 1
                _log.warning(
                    "session.stale_record_closed",
                    reason=DAEMON_MISSED_SHUTDOWN,
                    stale_container_id=stored.container_id,
                    new_container_id=identity.container_id,
                )
                self._write_summary(stored, reason=DAEMON_MISSED_SHUTDOWN, partial=True)
            self._partial = False
            self._counters = _fresh_counters()

        self._identity = identity
        self._open = True
        self._opened_at = self._clock.now()
        self._opened_monotonic = self._clock.monotonic()
        self._stats["opened"] += 1
        self._store.begin_session(identity)
        _log.info(
            "session.opened",
            container_id=identity.container_id,
            started_at=identity.started_at.isoformat(),
            partial=self._partial,
        )

    def _close(self, reason: str) -> None:
        if not self._open or self._identity is None:
            return
        identity = self._identity
        self._open = False
        self._stats["closed"] += 1
        self._store.end_session()
        self._write_summary(identity, reason=reason, partial=self._partial)
        _log.info(
            "session.closed",
            reason=reason,
            container_id=identity.container_id,
            uptime_seconds=self.uptime_seconds,
            counters=dict(self._counters),
        )
        self._opened_monotonic = None
        self._counters = _fresh_counters()

    def _write_summary(
        self,
        identity: SessionIdentity,
        *,
        reason: str,
        partial: bool,
    ) -> None:
        """**The M5 body.** Writes the durable session record. Currently writes nothing.

        M5 hands a record to :class:`~mcmanager.persistence.session_log.SessionLog`: the roster with
        per-player durations, the death list, the counters, the reason, the ``partial`` flag, and
        the log4j2 archive filename correlated by observation (list the directory at session start,
        diff it at the next one, and the new file belongs to the session that just ended).
        """
        _log.info(
            "session.summary_not_implemented",
            reason=reason,
            partial=partial,
            container_id=identity.container_id,
            started_at=identity.started_at.isoformat(),
            counters=dict(self._counters),
            has_session_log=self._session_log is not None,
            hint="M5 writes this through persistence/session_log.py",
        )


def _fresh_counters() -> dict[str, int]:
    """A zeroed counter set. A function, not a constant, so no two sessions share a dict."""
    return {
        "joins": 0,
        "leaves": 0,
        "deaths": 0,
        "advancements": 0,
        "chat_lines": 0,
        "ready_signals": 0,
    }
