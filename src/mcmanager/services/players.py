"""The authoritative online roster.

Two inputs, deliberately unequal:

- **The log stream is primary.** It is exact, totally ordered, and it carries the reason somebody
  left. Every roster mutation that matters comes from here.
- **The SLP sample is corroborating, and only when it is complete.** ``players.sample`` is
  protocol-capped (this server advertises a sample count of 12) and Paper settings or an
  anti-scrape plugin may randomise or omit it entirely. So reconciliation is gated on
  :attr:`~mcmanager.games.base.ProbeResult.sample_is_complete`, which is
  ``players_online == len(sample)``. Without that gate, the first time the roster exceeds the
  sample cap the daemon emits a storm of ``PlayerLeft`` events for people who are standing right
  there. With ``max-players = 5`` that cannot happen today; the gate exists because ``max-players``
  is a line in a file somebody will change.

**A failed probe is UNKNOWN, never zero players.** :meth:`PlayerRoster.reconcile` on an
unreachable probe is a no-op that says so in its result. That failure mode - unreachable read as
empty, idle timer armed, populated server stopped - is the one that loses somebody's build
session, and it is designed out here rather than guarded against downstream.

Sessions are keyed on **name**, not UUID. This server runs ``online-mode=false``, so the UUIDs in
the logs are offline v3 UUIDs: derived from ``OfflinePlayer:<name>``, stable per name, perfectly
good as a local key, and *not* Mojang UUIDs. They are carried on
:class:`~mcmanager.core.types.PlayerRef` when we have them and never sent anywhere.

This module holds state and does no I/O. It has no bus, no clock beyond the injected one, and no
knowledge of Docker, Discord or Minecraft. Its callers - :mod:`mcmanager.services.log_pipeline`
and :mod:`mcmanager.services.status_poller` - turn its return values into events, which is what
keeps it testable as a table of calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mcmanager.clock import Clock
    from mcmanager.core.types import LeaveReason
    from mcmanager.games.base import ProbeResult

__all__ = ["PlayerRoster", "PlayerSession", "RosterDelta"]


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayerSession:
    """One player's current stay on the server.

    Attributes:
        name: The sanitised account name. The session key.
        uuid: Offline v3 UUID when a ``UUID of player X is ...`` line preceded the join. Never send
            it to a Mojang API.
        joined_at: Wall-clock join time, tz-aware UTC. What a summary displays.
        joined_monotonic: The monotonic reading at join, which is what durations are measured
            from. Wall clocks jump; NTP stepping the clock mid-session must not be able to produce
            a negative ``session_seconds`` in a Discord message.
        address: ``ip:port`` from the preceding ``logged in`` line. **Never relayed.** It is kept
            because idle diagnostics and abuse investigation want it and it cannot be recovered
            afterwards; every presenter drops it.
        first_seen: True when this name had no prior record at join time.
        source: How we learned they were online - ``"log"`` or ``"probe"``. A probe-derived session
            has no address and only an approximate ``joined_at``, and saying so is better than
            pretending otherwise in a session summary.
    """

    name: str
    uuid: str | None = None
    joined_at: datetime
    joined_monotonic: float
    address: str | None = None
    first_seen: bool = False
    source: str = "log"

    def seconds_at(self, monotonic: float) -> float:
        """How long this session has lasted, measured monotonically. Never negative."""
        return max(0.0, monotonic - self.joined_monotonic)


@dataclass(frozen=True, slots=True, kw_only=True)
class RosterDelta:
    """What one :meth:`PlayerRoster.reconcile` decided.

    Sessions rather than bare names, because the caller's next act is to publish a
    ``PlayerLeft``, and that event carries ``session_seconds`` - which can only be computed from a
    session the reconcile has already removed. Handing back the name alone would force the poller
    to read the duration before calling reconcile, which is the kind of ordering constraint that
    holds for exactly as long as nobody edits either side.

    Attributes:
        applied: Whether the roster was changed at all. False means the probe was not trustworthy
            enough to reconcile against, and :attr:`reason` says why. **False is a common, healthy
            outcome**, not an error.
        joined: Sessions opened because the probe knew about somebody the roster did not.
        left: Sessions closed because the roster held somebody the probe did not. Always empty when
            :attr:`applied` is False, which is the entire point of the sample-completeness gate.
        reason: Why reconciliation was skipped, or ``"reconciled"`` when it ran.
        online_count: Roster size after the delta was applied.
    """

    applied: bool
    joined: tuple[PlayerSession, ...] = ()
    left: tuple[PlayerSession, ...] = ()
    reason: str = "reconciled"
    online_count: int = 0

    @property
    def changed(self) -> bool:
        """True when the roster actually moved."""
        return bool(self.joined or self.left)

    @property
    def joined_names(self) -> tuple[str, ...]:
        """Names of the sessions in :attr:`joined`."""
        return tuple(session.name for session in self.joined)

    @property
    def left_names(self) -> tuple[str, ...]:
        """Names of the sessions in :attr:`left`."""
        return tuple(session.name for session in self.left)


@final
class PlayerRoster:
    """Who is online, since when, and whether we have ever seen them before.

    Every mutating method returns the facts a caller needs to build an event, so no consumer has to
    re-derive ``online_count`` or a session duration by reaching into the roster's internals.
    """

    __slots__ = ("_clock", "_online", "_seen", "_stats")

    def __init__(self, *, clock: Clock, known_names: Iterable[str] = ()) -> None:
        """Create a roster.

        Args:
            clock: Injected time. Wall clock for display, monotonic for durations.
            known_names: Names with a prior session record, loaded from persistence. Seeding this
                is what makes ``PlayerJoined.first_seen`` mean "first time on this server" rather
                than "first time since the daemon restarted".
        """
        self._clock = clock
        self._online: dict[str, PlayerSession] = {}
        self._seen: set[str] = set(known_names)
        self._stats: dict[str, int] = {
            "joins": 0,
            "leaves": 0,
            "duplicate_joins": 0,
            "unknown_leaves": 0,
            "probe_reconciles": 0,
            "probe_skipped": 0,
            "probe_joins": 0,
            "probe_leaves": 0,
        }

    # ------------------------------------------------------------------------- introspection

    @property
    def online(self) -> tuple[str, ...]:
        """Names currently online, in join order."""
        return tuple(self._online)

    @property
    def count(self) -> int:
        """How many players are online."""
        return len(self._online)

    @property
    def sessions(self) -> Mapping[str, PlayerSession]:
        """A snapshot of the live sessions, keyed by name."""
        return dict(self._online)

    @property
    def known_names(self) -> frozenset[str]:
        """Every name this roster has ever seen. Persisted so ``first_seen`` survives a restart."""
        return frozenset(self._seen)

    @property
    def stats(self) -> Mapping[str, int]:
        """Counters, polled rather than pushed - the same rule the bus follows.

        ``probe_skipped`` climbing while ``probe_reconciles`` stays flat is the visible symptom of
        a server whose sample is being truncated or randomised, which is exactly the condition the
        completeness gate exists for.
        """
        return dict(self._stats)

    def is_online(self, name: str) -> bool:
        """Is this name currently on the server?

        The tier-2 death fallback in the log pipeline asks this about the first word of every
        unrecognised line, so it is on the hot path and is a plain dict lookup.
        """
        return name in self._online

    def session(self, name: str) -> PlayerSession | None:
        """The live session for ``name``, or ``None`` if they are not online."""
        return self._online.get(name)

    def session_seconds(self, name: str) -> float | None:
        """How long ``name`` has been on, or ``None`` if they are not online."""
        found = self._online.get(name)
        if found is None:
            return None
        return found.seconds_at(self._clock.monotonic())

    def has_seen(self, name: str) -> bool:
        """Has this name ever been recorded before?"""
        return name in self._seen

    # ----------------------------------------------------------------------------- log-driven

    def join(
        self,
        name: str,
        *,
        at: datetime | None = None,
        uuid: str | None = None,
        address: str | None = None,
        source: str = "log",
    ) -> PlayerSession:
        """Record that ``name`` is now online. Idempotent.

        A repeated join for somebody already online keeps the *original* session start, because a
        second join line is a replayed duplicate - Docker's ``since`` is second-granularity and
        repeats the whole of that second on every reattach - far more often than it is a genuine
        reconnect whose leave we missed. It does fill in a uuid or an address that arrived late.

        Args:
            name: The player.
            at: Join time. Defaults to the clock's now; the pipeline passes the event's timestamp,
                which came from Docker, so a backfilled join is dated correctly.
            uuid: Offline v3 UUID, if the authenticator line was seen.
            address: ``ip:port``. **Never relay this.**
            source: ``"log"`` or ``"probe"``.

        Returns:
            The live session, whether it was created now or already existed.
        """
        when = at if at is not None else self._clock.now()
        existing = self._online.get(name)
        if existing is not None:
            self._stats["duplicate_joins"] += 1
            merged = _merge(existing, uuid=uuid, address=address)
            self._online[name] = merged
            return merged

        session = PlayerSession(
            name=name,
            uuid=uuid,
            joined_at=when,
            joined_monotonic=self._clock.monotonic(),
            address=address,
            first_seen=name not in self._seen,
            source=source,
        )
        self._online[name] = session
        self._seen.add(name)
        self._stats["joins"] += 1
        return session

    def leave(self, name: str, *, reason: LeaveReason | None = None) -> PlayerSession | None:
        """Record that ``name`` is no longer online.

        Args:
            name: The player.
            reason: Stamped onto the event by the caller; accepted here so the call site reads the
                way it means and so a future roster-level policy has somewhere to live.

        Returns:
            The session that ended, or ``None`` when we never saw them join. ``None`` is a
            legitimate outcome and means ``PlayerLeft.session_seconds`` must stay ``None``: the
            daemon started mid-session, or the join was lost in an outage, or - as the archives
            show - a port scanner authenticated and hung up without ever joining. Inventing a
            duration there would put a fabricated number in a session summary.
        """
        del reason
        session = self._online.pop(name, None)
        if session is None:
            self._stats["unknown_leaves"] += 1
            return None
        self._stats["leaves"] += 1
        return session

    def clear(self, *, remember: bool = True) -> tuple[PlayerSession, ...]:
        """Empty the roster and return the sessions that were open.

        Called when the server stops or starts: whoever was online is not any more, and carrying a
        roster across a server restart is how a phantom player keeps the idle timer disarmed
        forever.

        Args:
            remember: Keep the names in the "seen before" set. Almost always what you want;
                ``False`` exists for tests that need a genuinely fresh roster.
        """
        sessions = tuple(self._online.values())
        self._online.clear()
        if not remember:
            self._seen.clear()
        return sessions

    def note_known(self, name: str) -> None:
        """Record that ``name`` has been seen before, without them being online.

        Used when seeding from prior session records at boot.
        """
        self._seen.add(name)

    # --------------------------------------------------------------------------- probe-driven

    def reconcile(self, probe: ProbeResult, *, trust_partial_sample: bool = False) -> RosterDelta:
        """Reconcile the roster against one status probe.

        The rules, in the order they are checked, each one present because breaking it has a
        specific bad outcome:

        1. **An unreachable probe changes nothing.** Unreachable is UNKNOWN. Reading it as "zero
           players" arms the idle timer against a server full of people.
        2. **A probe with no player count changes nothing.** Same reasoning.
        3. **An incomplete sample changes nothing** unless ``trust_partial_sample`` is set. The
           sample is capped at 12 entries and may be randomised or omitted, so
           ``players_online != len(sample)`` means the names we hold are a subset - and every
           roster member absent from that subset would otherwise be declared gone. Note that a
           genuinely empty server passes this test trivially (``0 == len(())``), so the case that
           matters most for idle shutdown is never gated out.

        Args:
            probe: The result of one out-of-band status query.
            trust_partial_sample: Override the completeness gate. Comes from
                ``probe.trust_partial_sample``, defaults false, and is documented as "leave this
                false".

        Returns:
            A :class:`RosterDelta`. ``applied=False`` with a reason is common and healthy.
        """
        if not probe.reachable:
            return self._skip("probe_unreachable")
        if probe.players_online is None:
            return self._skip("probe_reported_no_count")
        if not probe.sample_is_complete and not trust_partial_sample:
            return self._skip("sample_incomplete")

        # dict.fromkeys de-duplicates while keeping order: a sample is a list and nothing in the
        # protocol forbids it repeating a name.
        sampled = tuple(dict.fromkeys(probe.sample))
        arriving = tuple(name for name in sampled if name not in self._online)
        departing = tuple(name for name in self._online if name not in sampled)

        joined: list[PlayerSession] = []
        for name in arriving:
            joined.append(self.join(name, at=probe.probed_at, source="probe"))
            self._stats["probe_joins"] += 1
        left: list[PlayerSession] = []
        for name in departing:
            ended = self.leave(name)
            if ended is not None:
                left.append(ended)
            self._stats["probe_leaves"] += 1

        self._stats["probe_reconciles"] += 1
        return RosterDelta(
            applied=True,
            joined=tuple(joined),
            left=tuple(left),
            reason="reconciled",
            online_count=len(self._online),
        )

    def _skip(self, reason: str) -> RosterDelta:
        self._stats["probe_skipped"] += 1
        return RosterDelta(applied=False, reason=reason, online_count=len(self._online))


def _merge(session: PlayerSession, *, uuid: str | None, address: str | None) -> PlayerSession:
    """Fill in a uuid or address that arrived after the join, keeping the original start time."""
    if uuid is None and address is None:
        return session
    return PlayerSession(
        name=session.name,
        uuid=session.uuid if session.uuid is not None else uuid,
        joined_at=session.joined_at,
        joined_monotonic=session.joined_monotonic,
        address=session.address if session.address is not None else address,
        first_seen=session.first_seen,
        source=session.source,
    )
