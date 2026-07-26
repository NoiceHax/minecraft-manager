"""Tests for session identity and the resume/missed-shutdown decision.

The property that matters: **a session spans the game server's lifetime, not the daemon's.** A
daemon restart must resume the same session; a ``compose down/up`` must start a new one; and a
daemon that vanished without closing its record must not be allowed to claim counts it never saw.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from mcmanager.clock import ManualClock
from mcmanager.containers.dto import ContainerSnapshot, ContainerState
from mcmanager.core.events import (
    ChatMessage,
    Event,
    PlayerJoined,
    ServerCrashed,
    ServerStopped,
)
from mcmanager.core.types import PlayerRef, ServerId, Source
from mcmanager.persistence.state_store import SessionIdentity, StateStore
from mcmanager.services.players import PlayerRoster
from mcmanager.services.session import DAEMON_MISSED_SHUTDOWN, SessionManager

if TYPE_CHECKING:
    from pathlib import Path

SERVER: ServerId = "minecraft"
CONTAINER_ID = "c0ffee0000000000000000000000000000000000000000000000000000000000"
STARTED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def store(tmp_path: Path, clock: ManualClock) -> StateStore:
    store = StateStore(path=tmp_path, clock=clock)
    store.load()
    return store


@pytest.fixture
def roster(clock: ManualClock) -> PlayerRoster:
    return PlayerRoster(clock=clock)


def build(*, clock: ManualClock, store: StateStore, roster: PlayerRoster) -> SessionManager:
    return SessionManager(
        sink=RecordingSink(),
        clock=clock,
        server_id=SERVER,
        roster=roster,
        store=store,
    )


def snapshot(
    *,
    container_id: str = CONTAINER_ID,
    started_at: datetime = STARTED_AT,
    running: bool = True,
) -> ContainerSnapshot:
    return ContainerSnapshot(
        name="minecraft",
        id=container_id,
        exists=True,
        state=ContainerState.RUNNING if running else ContainerState.EXITED,
        running=running,
        started_at=started_at if running else None,
        observed_at=STARTED_AT,
    )


# ------------------------------------------------------------------------------------ identity


def test_a_running_snapshot_opens_a_session(
    clock: ManualClock,
    store: StateStore,
    roster: PlayerRoster,
) -> None:
    manager = build(clock=clock, store=store, roster=roster)
    manager.on_snapshot(snapshot())
    assert manager.is_open
    assert manager.identity == SessionIdentity(container_id=CONTAINER_ID, started_at=STARTED_AT)
    assert store.state.session_open is True


def test_the_same_snapshot_twice_does_not_reopen(
    clock: ManualClock,
    store: StateStore,
    roster: PlayerRoster,
) -> None:
    """The reconcile inspect fires every 60 seconds; it must not restart the session each time."""
    manager = build(clock=clock, store=store, roster=roster)
    manager.on_snapshot(snapshot())
    manager.on_snapshot(snapshot())
    assert manager.stats["opened"] == 1


def test_a_stopped_snapshot_opens_nothing(
    clock: ManualClock,
    store: StateStore,
    roster: PlayerRoster,
) -> None:
    manager = build(clock=clock, store=store, roster=roster)
    manager.on_snapshot(snapshot(running=False))
    assert not manager.is_open


def test_a_new_container_id_is_a_new_session(
    clock: ManualClock,
    store: StateStore,
    roster: PlayerRoster,
) -> None:
    """``compose down/up`` changes the id, and that genuinely is a different run of the server."""
    manager = build(clock=clock, store=store, roster=roster)
    manager.on_snapshot(snapshot())
    manager.on_snapshot(snapshot(container_id="dead" + "0" * 60))
    assert manager.stats["opened"] == 2
    assert manager.identity is not None
    assert manager.identity.container_id.startswith("dead")


def test_a_new_started_at_is_a_new_session(
    clock: ManualClock,
    store: StateStore,
    roster: PlayerRoster,
) -> None:
    """A plain ``docker restart`` keeps the id and changes ``StartedAt``. Also a new session."""
    manager = build(clock=clock, store=store, roster=roster)
    manager.on_snapshot(snapshot())
    manager.on_snapshot(snapshot(started_at=STARTED_AT + timedelta(hours=1)))
    assert manager.stats["opened"] == 2


# --------------------------------------------------------------------- resume vs missed shutdown


def test_a_matching_identity_resumes_and_is_marked_partial(
    clock: ManualClock,
    store: StateStore,
    roster: PlayerRoster,
) -> None:
    """The daemon restarted; the server did not. Counts from before this process are not ours."""
    store.begin_session(SessionIdentity(container_id=CONTAINER_ID, started_at=STARTED_AT))

    manager = build(clock=clock, store=store, roster=roster)
    manager.on_snapshot(snapshot())

    assert manager.stats["resumed"] == 1
    assert manager.partial is True
    assert manager.stats["missed_shutdowns"] == 0


def test_a_mismatched_open_record_is_closed_as_a_missed_shutdown(
    clock: ManualClock,
    store: StateStore,
    roster: PlayerRoster,
) -> None:
    """A stale open record plus a different container means we missed the whole stop."""
    store.begin_session(
        SessionIdentity(container_id="0" * 64, started_at=STARTED_AT - timedelta(days=1))
    )

    manager = build(clock=clock, store=store, roster=roster)
    manager.on_snapshot(snapshot())

    assert manager.stats["missed_shutdowns"] == 1
    assert manager.partial is False
    assert DAEMON_MISSED_SHUTDOWN == "daemon_missed_shutdown"


def test_a_cleanly_closed_record_is_not_a_missed_shutdown(
    clock: ManualClock,
    store: StateStore,
    roster: PlayerRoster,
) -> None:
    store.begin_session(SessionIdentity(container_id=CONTAINER_ID, started_at=STARTED_AT))
    store.end_session()

    manager = build(clock=clock, store=store, roster=roster)
    manager.on_snapshot(snapshot())

    assert manager.stats["missed_shutdowns"] == 0
    assert manager.stats["resumed"] == 0


# -------------------------------------------------------------------------------- closing


async def test_a_server_stop_closes_the_session(
    clock: ManualClock,
    store: StateStore,
    roster: PlayerRoster,
) -> None:
    manager = build(clock=clock, store=store, roster=roster)
    manager.on_snapshot(snapshot())
    await manager.on_server_event(
        ServerStopped(ts=clock.now(), server_id=SERVER, source=Source.RUNTIME, clean=True)
    )
    assert not manager.is_open
    assert store.state.session_open is False


async def test_a_crash_closes_the_session(
    clock: ManualClock,
    store: StateStore,
    roster: PlayerRoster,
) -> None:
    manager = build(clock=clock, store=store, roster=roster)
    manager.on_snapshot(snapshot())
    await manager.on_server_event(
        ServerCrashed(ts=clock.now(), server_id=SERVER, source=Source.RUNTIME, exit_code=1)
    )
    assert not manager.is_open
    assert manager.stats["closed"] == 1


async def test_daemon_shutdown_leaves_the_session_open(
    clock: ManualClock,
    store: StateStore,
    roster: PlayerRoster,
) -> None:
    """**The load-bearing one.** The daemon going away does not end the server's run.

    Writing ``open=false`` here would make the next boot believe it had seen the whole session and
    report counts it never observed.
    """
    manager = build(clock=clock, store=store, roster=roster)
    manager.on_snapshot(snapshot())

    await manager.aclose()

    assert store.state.session_open is True
    assert store.state.session is not None
    assert store.state.session.container_id == CONTAINER_ID


async def test_aclose_is_idempotent(
    clock: ManualClock,
    store: StateStore,
    roster: PlayerRoster,
) -> None:
    manager = build(clock=clock, store=store, roster=roster)
    manager.on_snapshot(snapshot())
    await manager.aclose()
    await manager.aclose()
    assert manager.stats["checkpoints"] == 1


# ------------------------------------------------------------------------------ counters


async def test_counters_track_the_session_and_reset_when_it_closes(
    clock: ManualClock,
    store: StateStore,
    roster: PlayerRoster,
) -> None:
    manager = build(clock=clock, store=store, roster=roster)
    manager.on_snapshot(snapshot())
    await manager.on_player_event(
        PlayerJoined(
            ts=clock.now(), server_id=SERVER, source=Source.LOG, player=PlayerRef(name="Steve")
        )
    )
    await manager.on_chat(
        ChatMessage(
            ts=clock.now(),
            server_id=SERVER,
            source=Source.LOG,
            player=PlayerRef(name="Steve"),
            message="hello",
        )
    )
    assert manager.counters["joins"] == 1
    assert manager.counters["chat_lines"] == 1

    await manager.on_server_event(
        ServerStopped(ts=clock.now(), server_id=SERVER, source=Source.RUNTIME, clean=True)
    )
    assert manager.counters["joins"] == 0


async def test_a_checkpoint_remembers_players_for_first_seen(
    clock: ManualClock,
    store: StateStore,
    roster: PlayerRoster,
) -> None:
    """``first_seen`` must mean "first time on this server", not "since the last restart"."""
    roster.join("Steve")
    manager = build(clock=clock, store=store, roster=roster)
    manager.on_snapshot(snapshot())

    await manager.checkpoint()

    assert "Steve" in store.state.known_players
