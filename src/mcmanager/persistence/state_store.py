"""Small durable state: the resume marker, the session identity, the idle deadline.

One JSON file under ``state.dir``, written **atomically**: serialise to a sibling temp file, fsync
it, then :meth:`pathlib.Path.replace` it over the target. ``replace`` is ``rename(2)`` on POSIX and
``MoveFileEx(MOVEFILE_REPLACE_EXISTING)`` on Windows, both of which are atomic within a filesystem,
so a reader never sees a half-written file. Without that, a power cut mid-write leaves a truncated
JSON document and the daemon refuses to boot - which is a strictly worse outcome than losing the
last sixty seconds of bookkeeping.

**No database.** The data is a handful of keys. sqlite would add a file format, a lock file, a
migration story and a WAL to reason about, in exchange for nothing this problem needs.

What lives here and why each field is load-bearing:

- ``last_log_ts`` - the ``since`` cursor for reattaching the log stream after a daemon restart.
  Docker's json-file driver is a ~30MB ring, so the backfill window is real but finite; missing the
  cursor means either re-processing everything in the ring or silently losing the gap.
- ``session`` - ``(container_id, started_at)``, the session identity. Stable across a daemon
  restart, changed by a ``compose down/up``. On boot, a matching identity resumes the record; a
  mismatch closes the stale one as ``daemon_missed_shutdown`` and opens a fresh one marked partial,
  so a summary says honestly that its counts are incomplete.
- ``idle_deadline`` - persisted for observability, and **deliberately restarted from now on boot**
  by the idle manager rather than resumed. A crash-looping manager that resumed an expired deadline
  would insta-stop the server on every restart, which is the single most dangerous bug available in
  this design.

**Reading is forgiving, writing is strict.** A corrupt or unreadable state file is renamed aside
and treated as empty, with a warning - a daemon that cannot start because of its own scratch file
is worse than one that forgets where it was. A write that fails raises, because silently not
persisting is how you find out at the next restart.

``pathlib`` throughout; ``os.path`` is banned by ruff's PTH rules. ``os.fsync`` is used directly
because there is no ``Path`` equivalent, and durability is the entire point of the exercise.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Self, cast, final

from mcmanager.logging_setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from mcmanager.clock import Clock

__all__ = [
    "DEFAULT_STATE_FILENAME",
    "STATE_SCHEMA_VERSION",
    "DaemonState",
    "SessionIdentity",
    "StateStore",
]

_log: Final = get_logger("mcmanager.state_store")

STATE_SCHEMA_VERSION: Final = 1
"""Bumped on a backwards-incompatible change. A file from the future is treated as unreadable
rather than guessed at, exactly like the config loader's ``schema_version``."""

DEFAULT_STATE_FILENAME: Final = "state.json"

_STATE_FILE_SUFFIX: Final = ".json"
"""How :meth:`StateStore.__init__` tells "this is the state file" from "this is ``state.dir``".

A suffix test rather than a filesystem test, because the answer must not change depending on
whether the directory has been created yet."""

_TEMP_SUFFIX: Final = ".tmp"
"""Sibling temp file, so the rename is within one filesystem and therefore atomic. A temp file in
``/tmp`` would make the final step a cross-device copy, which is not atomic at all."""


def _empty_counters() -> dict[str, int]:
    """Default factory for :attr:`DaemonState.counters`.

    A named function rather than ``default_factory=dict``, which pyright strict infers as
    ``dict[Unknown, Unknown]`` and which then makes the whole field partially unknown.
    """
    return {}


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionIdentity:
    """Which run of the game server a state file is talking about.

    A session spans the *server's* lifetime, not the daemon's, so the identity has to survive a
    daemon restart and has to change when the container is recreated. ``(container_id, started_at)``
    does both: a ``compose down/up`` produces a new id, and a plain restart produces a new
    ``StartedAt``.

    Attributes:
        container_id: The container's Docker id at the time the session opened.
        started_at: ``State.StartedAt``, tz-aware UTC.
    """

    container_id: str
    started_at: datetime

    def matches(self, *, container_id: str | None, started_at: datetime | None) -> bool:
        """Is this the same run of the server?

        A ``None`` on either side is not a match. "We could not determine the identity" must never
        resolve to "yes, resume the old session and attribute the next four hours to it".
        """
        if container_id is None or started_at is None:
            return False
        return self.container_id == container_id and self.started_at == started_at

    def to_json(self) -> dict[str, Any]:
        """The wire form."""
        return {"container_id": self.container_id, "started_at": self.started_at.isoformat()}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Self | None:
        """Parse the wire form, or ``None`` if it is not a usable identity."""
        container_id = data.get("container_id")
        started_at = _parse_datetime(data.get("started_at"))
        if not isinstance(container_id, str) or not container_id or started_at is None:
            return None
        return cls(container_id=container_id, started_at=started_at)


@dataclass(frozen=True, slots=True, kw_only=True)
class DaemonState:
    """Everything the daemon needs to pick up where it left off.

    Frozen, like every other value object in this project: the store owns the current instance and
    replaces it wholesale, so nothing can mutate what a caller is holding.

    Attributes:
        last_log_ts: The timestamp of the last log line processed. The ``since`` for the next
            reattach. ``None`` means "start from now", never "start from the beginning".
        session: The identity of the session that was open, if one was.
        session_open: True when the daemon was shut down with a session still running. On boot,
            ``session_open`` with a matching :attr:`session` is a resume; with a mismatched one it
            is a ``daemon_missed_shutdown``.
        idle_deadline: When the idle countdown would have fired. Persisted for observability and
            for a "we were 30 seconds from stopping" line in the log. **The idle manager restarts
            the countdown from now on boot rather than honouring this**, because a crash-looping
            manager honouring an expired deadline would stop the server on every restart.
        idle_empty_since: When the server last became empty.
        known_players: Every player name ever recorded, so ``PlayerJoined.first_seen`` means "first
            time on this server" rather than "first time since the last restart".
        counters: Free-form integer counters for whatever needs to survive a restart.
        saved_at: When this state was written. Diagnostic; a state file hours older than the
            process is the first clue that saves are failing.
    """

    schema_version: int = STATE_SCHEMA_VERSION
    last_log_ts: datetime | None = None
    session: SessionIdentity | None = None
    session_open: bool = False
    idle_deadline: datetime | None = None
    idle_empty_since: datetime | None = None
    known_players: tuple[str, ...] = ()
    counters: Mapping[str, int] = field(default_factory=_empty_counters)
    saved_at: datetime | None = None

    def to_json(self) -> dict[str, Any]:
        """The wire form: plain JSON types, ISO-8601 datetimes, sorted player names."""
        return {
            "schema_version": self.schema_version,
            "last_log_ts": _format_datetime(self.last_log_ts),
            "session": self.session.to_json() if self.session is not None else None,
            "session_open": self.session_open,
            "idle_deadline": _format_datetime(self.idle_deadline),
            "idle_empty_since": _format_datetime(self.idle_empty_since),
            "known_players": sorted(self.known_players),
            "counters": dict(self.counters),
            "saved_at": _format_datetime(self.saved_at),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Self:
        """Parse the wire form leniently.

        Every field falls back to its default rather than raising: this file is the daemon's own
        scratch space, and refusing to boot because one key went missing would be a self-inflicted
        outage. A *structurally* broken file is caught in :meth:`StateStore.load` instead.
        """
        session_data = data.get("session")
        session = (
            SessionIdentity.from_json(cast("Mapping[str, Any]", session_data))
            if isinstance(session_data, dict)
            else None
        )
        counters_data = data.get("counters")
        counters: dict[str, int] = {}
        if isinstance(counters_data, dict):
            for key, value in counters_data.items():  # pyright: ignore[reportUnknownVariableType]
                if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool):
                    counters[key] = value
        return cls(
            schema_version=_as_int(data.get("schema_version"), STATE_SCHEMA_VERSION),
            last_log_ts=_parse_datetime(data.get("last_log_ts")),
            session=session,
            session_open=bool(data.get("session_open", False)),
            idle_deadline=_parse_datetime(data.get("idle_deadline")),
            idle_empty_since=_parse_datetime(data.get("idle_empty_since")),
            known_players=_as_str_tuple(data.get("known_players")),
            counters=counters,
            saved_at=_parse_datetime(data.get("saved_at")),
        )


@final
class StateStore:
    """Loads, mutates and atomically persists one :class:`DaemonState`.

    The store holds the current state and hands out immutable snapshots. Mutators mark it dirty;
    :meth:`save` writes only when there is something to write, so the checkpoint timer can run every
    sixty seconds without touching the disk on an idle daemon.

    Every method is synchronous. The payload is a few hundred bytes, the write happens on a
    checkpoint timer and at shutdown rather than per event, and pushing it to a thread would buy
    microseconds in exchange for an ordering problem between a checkpoint and the final flush.
    """

    __slots__ = ("_clock", "_dirty", "_path", "_state", "_writes")

    def __init__(self, *, path: Path, clock: Clock) -> None:
        """Create a store.

        Args:
            path: The state file, or the directory to put :data:`DEFAULT_STATE_FILENAME` in. A
                directory is accepted because ``state.dir`` is what the config calls the setting,
                and making every caller remember to append the filename is how two of them end up
                appending different ones.
            clock: Injected time, used only for ``saved_at``.
        """
        # Decided by the *shape* of the path, never by whether it happens to exist yet. An
        # earlier version asked `path.is_dir()`, which is False for a directory that has not been
        # created - i.e. on every first boot, where `state.dir` points into an empty named volume.
        # The store then treated the directory path as the state file, and once the daemon created
        # the real directory a moment later every checkpoint died in `replace()` trying to rename a
        # file over a directory. Session history is unreconstructable after the fact (the docker
        # log ring is 30MB), so a silently non-persisting store is the expensive kind of bug.
        looks_like_a_file = path.suffix == _STATE_FILE_SUFFIX or path.is_file()
        self._path = path if looks_like_a_file else path / DEFAULT_STATE_FILENAME
        self._clock = clock
        self._state = DaemonState()
        self._dirty = False
        self._writes = 0

    # -------------------------------------------------------------------------- introspection

    @property
    def path(self) -> Path:
        """The file this store reads and writes."""
        return self._path

    @property
    def state(self) -> DaemonState:
        """The current state. Immutable; mutate through the methods below."""
        return self._state

    @property
    def dirty(self) -> bool:
        """True when there are changes :meth:`save` has not written yet."""
        return self._dirty

    @property
    def writes(self) -> int:
        """How many times the file has actually been rewritten. For tests and diagnostics."""
        return self._writes

    # -------------------------------------------------------------------------------- loading

    def load(self) -> DaemonState:
        """Read the state file, tolerating everything except a file we should not overwrite.

        A missing file is the normal first boot and returns an empty state silently. A file that is
        unreadable, is not JSON, is not an object, or claims a schema version this build does not
        understand is **renamed aside** with a warning and treated as empty. Renaming rather than
        deleting it matters: it is the only evidence of whatever went wrong, and it costs a few
        hundred bytes to keep.
        """
        if not self._path.is_file():
            _log.debug("state_store.absent", path=str(self._path))
            self._state = DaemonState()
            self._dirty = False
            return self._state

        try:
            raw = self._path.read_text(encoding="utf-8")
            decoded: object = json.loads(raw)
        except (OSError, ValueError) as exc:
            self._quarantine("unreadable", str(exc))
            self._state = DaemonState()
            self._dirty = False
            return self._state

        if not isinstance(decoded, dict):
            self._quarantine("not_an_object", f"top level is {type(decoded).__name__}")
            self._state = DaemonState()
            self._dirty = False
            return self._state

        data = cast("dict[str, Any]", decoded)
        version = _as_int(data.get("schema_version"), STATE_SCHEMA_VERSION)
        if version != STATE_SCHEMA_VERSION:
            self._quarantine(
                "schema_mismatch",
                f"file is schema {version}, this build understands {STATE_SCHEMA_VERSION}",
            )
            self._state = DaemonState()
            self._dirty = False
            return self._state

        self._state = DaemonState.from_json(data)
        self._dirty = False
        _log.info(
            "state_store.loaded",
            path=str(self._path),
            last_log_ts=_format_datetime(self._state.last_log_ts),
            session_open=self._state.session_open,
            known_players=len(self._state.known_players),
        )
        return self._state

    def _quarantine(self, reason: str, detail: str) -> None:
        """Move a bad state file aside so the next boot starts clean but the evidence survives."""
        target = self._path.with_suffix(f"{self._path.suffix}.corrupt")
        try:
            self._path.replace(target)
        except OSError as exc:
            _log.warning(
                "state_store.quarantine_failed",
                path=str(self._path),
                reason=reason,
                detail=detail,
                error=str(exc),
            )
            return
        _log.warning(
            "state_store.quarantined",
            path=str(self._path),
            moved_to=str(target),
            reason=reason,
            detail=detail,
            hint="starting from an empty state; the old file is kept for inspection",
        )

    # -------------------------------------------------------------------------------- mutation

    def replace_state(self, updated: DaemonState) -> DaemonState:
        """Adopt ``updated`` as the current state, marking the store dirty if it differs.

        The comparison is what makes a periodic checkpoint cheap: a reconcile that found nothing
        new leaves the store clean and :meth:`save` does no I/O at all.
        """
        if updated == self._state:
            return self._state
        self._state = updated
        self._dirty = True
        return self._state

    def record_log_ts(self, ts: datetime | None) -> None:
        """Advance the log cursor. Never moves it backwards.

        Monotonicity is the point: lines arrive in order but a reattach replays the whole of the
        last second, and letting a replayed line rewind the cursor would make every reconnect
        re-read a little more than the one before.
        """
        if ts is None:
            return
        current = self._state.last_log_ts
        if current is not None and ts <= current:
            return
        self.replace_state(replace(self._state, last_log_ts=ts))

    def begin_session(self, identity: SessionIdentity) -> None:
        """Record that a session for ``identity`` is open."""
        self.replace_state(replace(self._state, session=identity, session_open=True))

    def end_session(self) -> None:
        """Record that the open session has been closed cleanly.

        The identity is kept: on the next boot it is what distinguishes "the same server run is
        still going" from "this is a different container now".
        """
        self.replace_state(replace(self._state, session_open=False))

    def set_idle_deadline(
        self,
        deadline: datetime | None,
        *,
        empty_since: datetime | None = None,
    ) -> None:
        """Persist (or clear) the idle countdown.

        Recorded for observability only. The idle manager restarts its countdown from now on boot;
        see the module docstring for why honouring a persisted deadline is the most dangerous thing
        this file could be used for.
        """
        self.replace_state(
            replace(self._state, idle_deadline=deadline, idle_empty_since=empty_since)
        )

    def remember_players(self, names: Iterable[str]) -> None:
        """Merge ``names`` into the known-players set, which is what backs ``first_seen``."""
        merged = set(self._state.known_players) | set(names)
        if merged == set(self._state.known_players):
            return
        self.replace_state(replace(self._state, known_players=tuple(sorted(merged))))

    def set_counter(self, name: str, value: int) -> None:
        """Set one durable counter."""
        if self._state.counters.get(name) == value:
            return
        counters = dict(self._state.counters)
        counters[name] = value
        self.replace_state(replace(self._state, counters=counters))

    # --------------------------------------------------------------------------------- saving

    def save(self, *, force: bool = False) -> bool:
        """Write the state file atomically. Returns True if a write happened.

        Args:
            force: Write even when nothing changed. Used by the shutdown flush, where "nothing
                changed" and "we never got as far as changing anything" are worth distinguishing on
                disk.

        Raises:
            OSError: If the file could not be written. Deliberately propagated: silently failing to
                persist is how a daemon discovers at its next restart that it has been forgetting
                everything for a week.
        """
        if not self._dirty and not force:
            return False
        state = replace(self._state, saved_at=self._clock.now())
        payload = json.dumps(state.to_json(), indent=2, sort_keys=True) + "\n"

        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._path.with_name(self._path.name + _TEMP_SUFFIX)
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(self._path)
        _fsync_directory(self._path.parent)

        self._state = state
        self._dirty = False
        self._writes += 1
        _log.debug("state_store.saved", path=str(self._path), bytes=len(payload))
        return True


# ------------------------------------------------------------------------------------ helpers


def _fsync_directory(directory: Path) -> None:
    """Flush the directory entry so the rename itself survives a power cut.

    POSIX only, and best-effort. Windows has no directory file descriptor to open, and several
    filesystems refuse the call; on those, the rename's durability is the filesystem's problem and
    there is nothing useful to do about it here.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        _log.debug("state_store.dir_fsync_unsupported", path=str(directory), error=str(exc))
    finally:
        os.close(fd)


def _format_datetime(value: datetime | None) -> str | None:
    """ISO-8601 with an explicit offset, or ``None``."""
    return value.isoformat() if value is not None else None


def _parse_datetime(value: object) -> datetime | None:
    """Parse an ISO-8601 string into tz-aware UTC. Anything unparsable becomes ``None``.

    A naive value is read as UTC rather than rejected: this store only ever writes tz-aware UTC, so
    a naive timestamp means somebody edited the file by hand, and the intent is unambiguous.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _as_int(value: object, default: int) -> int:
    """An int, or ``default``. ``bool`` is rejected: ``True`` is not schema version 1."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _as_str_tuple(value: object) -> tuple[str, ...]:
    """A tuple of the strings in ``value``, or empty."""
    if not isinstance(value, list):
        return ()
    entries: list[object] = list(value)  # pyright: ignore[reportUnknownArgumentType]
    return tuple(entry for entry in entries if isinstance(entry, str))
