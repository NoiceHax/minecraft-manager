"""Archived session records, and the log-archive gap filler. **Stub: body in M5.**

The server's own log4j2 already does ~85% of the archiving job: ``RollingRandomAccessFile`` with
``OnStartupTriggeringPolicy`` plus ``TimeBasedTriggeringPolicy``, gzipping to
``logs/%d{yyyy-MM-dd}-%i.log.gz`` and retaining 1000 files. **The daemon must not create or delete
anything in that directory**; it is mounted read-only.

The one real gap: rollover happens at the next *start*, not at shutdown, so the most recent
session sits uncompressed in ``latest.log`` until someone starts the server again. If the server
stays down for three days, that session has no archive. Filling that gap is the only archiving
this daemon does: on ``STOPPING -> STOPPED``, gzip ``latest.log`` to
``session-<start_utc>-<id>.log.gz`` under ``logs.archive_dir``, pruned by ``retain_archives``.

Correlating a session to its log4j2 archive is done **by observation, not prediction**: the ``%i``
index depends on how many rolls happened that day, so list the directory at session start, diff it
at the next session start, and the new filename belongs to the previous session.

Access has a primary and a fallback behind one :class:`LogSource` protocol: the read-only bind
mount (verified ``drwxrwxr-x minty:minty``, so uid 1000 reads it directly), and reading the file
over the Docker socket - which also works on a **stopped** container, and means a forgotten mount
degrades with a warning instead of failing.

**Open decision, inherited and flagged rather than silently resolved.** The socket fallback needs a
way to read a file out of a container. Plan section 4 fixes ``ContainerRuntime`` at exactly nine
methods and section 12 then calls ``runtime.get_file(...)``; those contradict, and the nine-method
ABC is what was built. So :class:`MountedLogSource` (the primary, and the one the deployment
actually uses) is implemented here, and the socket-backed ``LogSource`` is not: adding it means
either a tenth ABC method or an implementation in ``containers/`` that reaches Docker another way.
That is a decision for whoever writes M5, not something to invent in a stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, final, runtime_checkable

import structlog

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime
    from pathlib import Path

    from mcmanager.clock import Clock
    from mcmanager.persistence.state_store import SessionIdentity

__all__ = ["ARCHIVE_SUFFIX", "LATEST_LOG_NAME", "LogSource", "MountedLogSource", "SessionLog"]

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.session_log")

LATEST_LOG_NAME = "latest.log"
"""The uncompressed current log. The only file in the server's log directory this daemon reads by
name, and it is never written to or deleted."""

ARCHIVE_SUFFIX = ".log.gz"
"""What both log4j2's archives and our gap-filling archives end in."""


@runtime_checkable
class LogSource(Protocol):
    """Read-only access to the game server's log directory.

    Two implementations are intended: the ``:ro`` bind mount (:class:`MountedLogSource`), and a
    socket-backed one that also works on a stopped container. Both are read-only by contract, and
    the protocol has no ``write`` or ``delete`` for exactly that reason: the directory belongs to
    log4j2, and a daemon that *can* delete an archive will eventually delete one.
    """

    @property
    def available(self) -> bool:
        """Whether this source can be read at all right now.

        False is a warning, never a failure: a forgotten mount must degrade, not crash the daemon.
        """
        ...

    async def list_archives(self) -> tuple[str, ...]:
        """Archive file names, sorted. Empty when the source is unavailable."""
        ...

    async def read_bytes(self, name: str) -> bytes | None:
        """Read one file from the log directory, or ``None`` if it is not readable.

        ``name`` is a bare file name and is rejected if it contains a path separator: a session
        record must never be able to talk this into reading ``../../etc/shadow``.
        """
        ...


@final
class MountedLogSource:
    """The primary :class:`LogSource`: the server's log directory, bind-mounted read-only.

    Verified on the homelab as ``drwxrwxr-x minty:minty`` with ``-rw-rw-r--`` files, so the
    container's uid 1000 reads them directly with no chmod games.
    """

    def __init__(self, directory: Path) -> None:
        """Point at ``logs.server_log_dir``, which may legitimately not exist."""
        self._directory = directory

    @property
    def directory(self) -> Path:
        """The directory being read."""
        return self._directory

    @property
    def available(self) -> bool:
        """True when the directory exists and is a directory."""
        return self._directory.is_dir()

    async def list_archives(self) -> tuple[str, ...]:
        """Every ``*.log.gz`` in the directory, sorted. Empty, with a warning, if unavailable."""
        if not self.available:
            _log.warning("session_log.source_unavailable", directory=str(self._directory))
            return ()
        try:
            names = sorted(
                entry.name
                for entry in self._directory.iterdir()
                if entry.is_file() and entry.name.endswith(ARCHIVE_SUFFIX)
            )
        except OSError as exc:
            _log.warning(
                "session_log.list_failed",
                directory=str(self._directory),
                error=str(exc),
            )
            return ()
        return tuple(names)

    async def read_bytes(self, name: str) -> bytes | None:
        """Read one file by bare name. Returns ``None`` rather than raising."""
        if not _is_bare_name(name):
            _log.error("session_log.rejected_path", name=name)
            return None
        path = self._directory / name
        try:
            return path.read_bytes()
        except OSError as exc:
            _log.warning("session_log.read_failed", path=str(path), error=str(exc))
            return None


@final
class SessionLog:
    """Durable session records, plus the ``latest.log`` gap filler. **Bodies land in M5.**

    Everything here is deliberately inert. What exists now is the interface ``SessionManager`` will
    call, the directory conventions, and the guarantee the module docstring makes: this class writes
    only under ``logs.archive_dir``, and never touches the server's own log directory except to read
    it.
    """

    def __init__(
        self,
        *,
        archive_dir: Path,
        clock: Clock,
        source: LogSource | None = None,
        retain_archives: int = 50,
        archive_on_stop: bool = True,
    ) -> None:
        """Wire the log.

        Args:
            archive_dir: ``logs.archive_dir``. The only directory this class ever writes to.
            clock: Injected time, for the timestamps in record and archive names.
            source: Read access to the server's log directory. ``None`` means the mount was not
                configured, which degrades archiving to "not done" plus a warning.
            retain_archives: How many of our own gap-filling archives to keep.
            archive_on_stop: ``logs.archive_on_stop``.
        """
        self._archive_dir = archive_dir
        self._clock = clock
        self._source = source
        self._retain = retain_archives
        self._archive_on_stop = archive_on_stop
        self._known_archives: tuple[str, ...] = ()
        self._stats: dict[str, int] = {
            "records_written": 0,
            "archives_written": 0,
            "archives_pruned": 0,
            "correlations": 0,
        }

    @property
    def archive_dir(self) -> Path:
        """Where our own archives and records go."""
        return self._archive_dir

    @property
    def stats(self) -> Mapping[str, int]:
        """Counters, for ``/readyz`` and the tests."""
        return dict(self._stats)

    async def snapshot_archives(self) -> tuple[str, ...]:
        """Record the archive listing at session start, for later correlation by observation.

        Implemented, because it is cheap, read-only, and useless if it is not done at the *right*
        moment - which is a wiring property, not an M5 one.
        """
        if self._source is None:
            return ()
        self._known_archives = await self._source.list_archives()
        return self._known_archives

    async def correlate_archive(self) -> str | None:
        """Which log4j2 archive belongs to the session that just ended, by diffing the listing.

        By observation, never by prediction: the ``%i`` in ``%d{yyyy-MM-dd}-%i.log.gz`` depends on
        how many rolls happened that day, so the only honest answer comes from the difference
        between two listings.
        """
        if self._source is None:
            return None
        current = await self._source.list_archives()
        added = tuple(name for name in current if name not in set(self._known_archives))
        self._known_archives = current
        if not added:
            return None
        self._stats["correlations"] += 1
        return added[-1]

    async def archive_latest(
        self,
        identity: SessionIdentity,
        *,
        finished_at: datetime | None = None,
    ) -> Path | None:
        """**M5 body.** Gzip ``latest.log`` into ``archive_dir``. Currently archives nothing.

        The gap this fills, and the only one: log4j2 rolls at the next *start*, so a server that
        stays down for three days leaves its last session sitting uncompressed in ``latest.log``.
        The target name is ``session-<start_utc>-<container_id[:12]>.log.gz``.
        """
        _log.info(
            "session_log.archive_not_implemented",
            container_id=identity.container_id,
            started_at=identity.started_at.isoformat(),
            finished_at=None if finished_at is None else finished_at.isoformat(),
            archive_on_stop=self._archive_on_stop,
            archive_dir=str(self._archive_dir),
            source_available=self._source is not None and self._source.available,
            hint="M5 writes the gzip; nothing is written today",
        )
        return None

    async def write_record(self, record: Mapping[str, object]) -> Path | None:
        """**M5 body.** Persist one session summary as JSON. Currently writes nothing.

        The record is what ``mcmanager sessions`` lists and what Discord's end-of-session summary
        renders, and it is load-bearing rather than a nicety: the docker log driver is a 30MB ring,
        so a long session genuinely cannot be reconstructed after the fact.
        """
        _log.info(
            "session_log.record_not_implemented",
            keys=sorted(record),
            archive_dir=str(self._archive_dir),
            hint="M5 writes this file",
        )
        return None

    async def records(self, *, limit: int | None = None) -> Sequence[Mapping[str, object]]:
        """**M5 body.** Read archived session records back, newest first. Currently empty."""
        _log.debug("session_log.records_not_implemented", limit=limit)
        return ()

    async def prune(self) -> tuple[str, ...]:
        """**M5 body.** Delete our oldest archives past ``retain_archives``. Currently a no-op.

        Only ever inside ``archive_dir``. The server's own directory is read-only and stays that
        way; deleting a log4j2 archive would destroy history this daemon did not create.
        """
        _log.debug(
            "session_log.prune_not_implemented",
            retain=self._retain,
            archive_dir=str(self._archive_dir),
        )
        return ()


def _is_bare_name(name: str) -> bool:
    """Reject anything that is not a plain file name in the log directory."""
    return bool(name) and "/" not in name and "\\" not in name and name not in {".", ".."}
