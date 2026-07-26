"""Tests for the read-only log source and the archive-correlation-by-observation logic.

Only two things in this module have real bodies today, and both are here because they are the kind
of thing that is wrong silently: the bare-name guard on ``read_bytes`` (a session record must never
be able to talk this into reading outside the log directory), and the listing diff that correlates
a session to its log4j2 archive.
"""

from __future__ import annotations

import gzip
from typing import TYPE_CHECKING

import pytest

from mcmanager.clock import ManualClock
from mcmanager.persistence.session_log import (
    ARCHIVE_SUFFIX,
    LATEST_LOG_NAME,
    LogSource,
    MountedLogSource,
    SessionLog,
)
from mcmanager.persistence.state_store import SessionIdentity

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "mc-logs"
    directory.mkdir()
    # Bytes, not text: write_text() would translate the newline on Windows and the assertion
    # below is about exactly what came back off disk.
    (directory / LATEST_LOG_NAME).write_bytes(b"hello\n")
    (directory / f"2026-07-25-1{ARCHIVE_SUFFIX}").write_bytes(gzip.compress(b"old\n"))
    return directory


# ------------------------------------------------------------------------------ MountedLogSource


async def test_it_lists_only_archives(log_dir: Path) -> None:
    source = MountedLogSource(log_dir)
    assert source.available
    assert await source.list_archives() == ("2026-07-25-1.log.gz",)


async def test_a_missing_directory_degrades_rather_than_raising(tmp_path: Path) -> None:
    """A forgotten `:ro` mount must be a warning. Refusing to boot over it would be worse."""
    source = MountedLogSource(tmp_path / "nope")
    assert not source.available
    assert await source.list_archives() == ()
    assert await source.read_bytes(LATEST_LOG_NAME) is None


async def test_it_reads_a_file_by_bare_name(log_dir: Path) -> None:
    source = MountedLogSource(log_dir)
    assert await source.read_bytes(LATEST_LOG_NAME) == b"hello\n"


@pytest.mark.parametrize(
    "name",
    ["../secrets", "..", ".", "", "sub/latest.log", "sub\\latest.log", "/etc/shadow"],
)
async def test_it_refuses_anything_that_is_not_a_bare_name(log_dir: Path, name: str) -> None:
    """The session record is derived from log content, so this is attacker-adjacent input."""
    source = MountedLogSource(log_dir)
    assert await source.read_bytes(name) is None


async def test_a_mounted_source_satisfies_the_protocol(log_dir: Path) -> None:
    assert isinstance(MountedLogSource(log_dir), LogSource)


# ------------------------------------------------------------------------------------ SessionLog


async def test_correlation_finds_the_file_that_appeared(log_dir: Path, clock: ManualClock) -> None:
    """By observation, never by prediction: the ``%i`` index depends on the day's roll count."""
    log = SessionLog(
        archive_dir=log_dir.parent / "archives", clock=clock, source=MountedLogSource(log_dir)
    )
    await log.snapshot_archives()

    (log_dir / f"2026-07-26-1{ARCHIVE_SUFFIX}").write_bytes(gzip.compress(b"new\n"))

    assert await log.correlate_archive() == f"2026-07-26-1{ARCHIVE_SUFFIX}"
    assert log.stats["correlations"] == 1


async def test_correlation_returns_none_when_nothing_rolled(
    log_dir: Path,
    clock: ManualClock,
) -> None:
    log = SessionLog(
        archive_dir=log_dir.parent / "archives", clock=clock, source=MountedLogSource(log_dir)
    )
    await log.snapshot_archives()
    assert await log.correlate_archive() is None


async def test_the_m5_bodies_are_inert_and_write_nothing(
    log_dir: Path,
    tmp_path: Path,
    clock: ManualClock,
) -> None:
    """The stub must not half-implement archiving: it writes nothing and says so.

    Named because the failure mode of a half-written archiver is deleting somebody's history.
    """
    archives = tmp_path / "archives"
    log = SessionLog(archive_dir=archives, clock=clock, source=MountedLogSource(log_dir))
    identity = SessionIdentity(container_id="c0ffee", started_at=clock.now())

    assert await log.archive_latest(identity) is None
    assert await log.write_record({"anything": 1}) is None
    assert await log.records() == ()
    assert await log.prune() == ()
    assert not archives.exists()
    assert _names(log_dir) == [f"2026-07-25-1{ARCHIVE_SUFFIX}", LATEST_LOG_NAME]


def _names(directory: Path) -> list[str]:
    """Sorted directory listing. A sync helper so ruff's ASYNC240 stays happy."""
    return sorted(entry.name for entry in directory.iterdir())
