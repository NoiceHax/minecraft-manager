"""Tests for :mod:`mcmanager.persistence.state_store`.

Not in the original assignment for this agent; added because a state store that silently loses the
log cursor, or that refuses to boot because of its own scratch file, fails in ways nothing else in
the suite would notice.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from mcmanager.persistence.state_store import (
    DEFAULT_STATE_FILENAME,
    STATE_SCHEMA_VERSION,
    DaemonState,
    SessionIdentity,
    StateStore,
)

if TYPE_CHECKING:
    from pathlib import Path

    from mcmanager.clock import ManualClock

STARTED_AT = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path, manual_clock: ManualClock) -> StateStore:
    return StateStore(path=tmp_path / DEFAULT_STATE_FILENAME, clock=manual_clock)


# ---------------------------------------------------------------------------------- placement


def test_a_directory_path_gets_the_default_filename(
    tmp_path: Path,
    manual_clock: ManualClock,
) -> None:
    # `state.dir` is what the config calls the setting; making every caller remember to append the
    # filename is how two of them end up appending different ones.
    store = StateStore(path=tmp_path, clock=manual_clock)

    assert store.path == tmp_path / DEFAULT_STATE_FILENAME


def test_a_state_dir_that_does_not_exist_yet_is_still_a_directory(
    tmp_path: Path,
    manual_clock: ManualClock,
) -> None:
    """The first-boot regression.

    ``state.dir`` points inside a named volume, which arrives empty, so on the very first boot the
    directory does not exist when ``Application.__init__`` builds the store. Deciding
    file-vs-directory with ``is_dir()`` answered "file" here and put the state at
    ``/var/lib/mcmanager/state`` instead of ``.../state/state.json``.
    """
    state_dir = tmp_path / "volume" / "state"
    assert not state_dir.exists()

    store = StateStore(path=state_dir, clock=manual_clock)

    assert store.path == state_dir / DEFAULT_STATE_FILENAME


def test_checkpointing_survives_the_state_dir_being_created_after_the_store(
    tmp_path: Path,
    manual_clock: ManualClock,
) -> None:
    """The consequence of the bug above, which is what actually broke.

    ``Application.__init__`` builds the store, then ``_ensure_writable_dirs()`` creates
    ``state.dir`` a moment later. With the path already mis-resolved to the directory itself,
    every checkpoint died in ``replace()`` trying to rename a file over a directory - and it is
    silent, because ``save()`` is called from a background task.
    """
    state_dir = tmp_path / "volume" / "state"
    store = StateStore(path=state_dir, clock=manual_clock)

    state_dir.mkdir(parents=True)  # what _ensure_writable_dirs() does, after the store exists

    store.record_log_ts(STARTED_AT)
    assert store.save() is True
    assert store.path.is_file()
    assert state_dir.is_dir()
    # No half-written temp file left lying next to the volume.
    assert not (state_dir.parent / "state.tmp").exists()


def test_a_missing_parent_directory_is_created_on_save(
    tmp_path: Path,
    manual_clock: ManualClock,
) -> None:
    store = StateStore(path=tmp_path / "nested" / "deeper" / "state.json", clock=manual_clock)
    store.record_log_ts(STARTED_AT)

    assert store.save() is True
    assert store.path.is_file()


# ------------------------------------------------------------------------------- round trips


def test_an_absent_file_loads_as_empty_state(store: StateStore) -> None:
    state = store.load()

    assert state == DaemonState()
    assert store.dirty is False


def test_a_full_round_trip_preserves_every_field(
    store: StateStore,
    manual_clock: ManualClock,
) -> None:
    identity = SessionIdentity(container_id="abc123", started_at=STARTED_AT)
    store.begin_session(identity)
    store.record_log_ts(STARTED_AT + timedelta(minutes=5))
    store.set_idle_deadline(STARTED_AT + timedelta(minutes=20), empty_since=STARTED_AT)
    store.remember_players(["Steve", "Alex"])
    store.set_counter("sessions", 7)
    store.save()

    reloaded = StateStore(path=store.path, clock=manual_clock)
    state = reloaded.load()

    assert state.session == identity
    assert state.session_open is True
    assert state.last_log_ts == STARTED_AT + timedelta(minutes=5)
    assert state.idle_deadline == STARTED_AT + timedelta(minutes=20)
    assert state.idle_empty_since == STARTED_AT
    assert state.known_players == ("Alex", "Steve")
    assert state.counters == {"sessions": 7}
    assert state.saved_at == manual_clock.now()


def test_the_file_is_readable_json(store: StateStore) -> None:
    # `cat state.json` during an incident should not require a tool.
    store.begin_session(SessionIdentity(container_id="abc123", started_at=STARTED_AT))
    store.save()

    decoded = json.loads(store.path.read_text(encoding="utf-8"))
    assert decoded["schema_version"] == STATE_SCHEMA_VERSION
    assert decoded["session"]["container_id"] == "abc123"


def test_no_temp_file_survives_a_save(store: StateStore) -> None:
    store.record_log_ts(STARTED_AT)
    store.save()

    leftovers = [path.name for path in store.path.parent.iterdir() if path.name.endswith(".tmp")]
    assert leftovers == []


# ------------------------------------------------------------------------------- dirtiness


def test_a_clean_store_does_not_write(store: StateStore) -> None:
    assert store.save() is False
    assert store.writes == 0
    assert not store.path.exists()


def test_force_writes_even_when_clean(store: StateStore) -> None:
    assert store.save(force=True) is True
    assert store.path.is_file()


def test_a_no_op_mutation_leaves_the_store_clean(store: StateStore) -> None:
    store.record_log_ts(STARTED_AT)
    store.save()
    assert store.dirty is False

    store.remember_players(["Steve"])
    store.save()
    store.remember_players(["Steve"])

    assert store.dirty is False
    assert store.writes == 2


def test_replace_state_marks_dirty_only_on_a_real_change(store: StateStore) -> None:
    store.replace_state(DaemonState())
    assert store.dirty is False

    store.replace_state(DaemonState(session_open=True))
    assert store.dirty is True


# --------------------------------------------------------------------------- the log cursor


def test_the_log_cursor_never_moves_backwards(store: StateStore) -> None:
    """A reattach replays the whole of the last second.

    Letting a replayed line rewind the cursor would make every reconnect re-read a little more
    than the one before.
    """
    store.record_log_ts(STARTED_AT + timedelta(seconds=30))
    store.record_log_ts(STARTED_AT)

    assert store.state.last_log_ts == STARTED_AT + timedelta(seconds=30)


def test_a_none_log_cursor_is_ignored(store: StateStore) -> None:
    store.record_log_ts(STARTED_AT)
    store.record_log_ts(None)

    assert store.state.last_log_ts == STARTED_AT


# ------------------------------------------------------------------------- session identity


def test_a_session_identity_matches_only_on_both_halves() -> None:
    identity = SessionIdentity(container_id="abc123", started_at=STARTED_AT)

    assert identity.matches(container_id="abc123", started_at=STARTED_AT)
    assert not identity.matches(container_id="def456", started_at=STARTED_AT)
    assert not identity.matches(container_id="abc123", started_at=STARTED_AT + timedelta(1))


def test_an_unknown_identity_never_matches() -> None:
    # "We could not determine the identity" must never resolve to "yes, resume the old session and
    # attribute the next four hours to it".
    identity = SessionIdentity(container_id="abc123", started_at=STARTED_AT)

    assert not identity.matches(container_id=None, started_at=STARTED_AT)
    assert not identity.matches(container_id="abc123", started_at=None)


def test_ending_a_session_keeps_the_identity(store: StateStore) -> None:
    # On the next boot the identity is what distinguishes "same server run" from "new container".
    identity = SessionIdentity(container_id="abc123", started_at=STARTED_AT)
    store.begin_session(identity)
    store.end_session()

    assert store.state.session == identity
    assert store.state.session_open is False


# -------------------------------------------------------------------------------- corruption


def test_a_torn_file_is_quarantined_rather_than_fatal(store: StateStore) -> None:
    """A daemon that will not start because of its own scratch file is worse than one that forgets.

    The file is renamed rather than deleted: it is the only evidence of whatever went wrong.
    """
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text('{"last_log_ts": "2026-01-0', encoding="utf-8")

    state = store.load()

    assert state == DaemonState()
    assert not store.path.exists()
    assert store.path.with_suffix(".json.corrupt").is_file()


def test_a_json_document_that_is_not_an_object_is_quarantined(store: StateStore) -> None:
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("[1, 2, 3]", encoding="utf-8")

    assert store.load() == DaemonState()
    assert store.path.with_suffix(".json.corrupt").is_file()


def test_a_file_from_a_future_schema_is_refused_not_guessed_at(store: StateStore) -> None:
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps({"schema_version": STATE_SCHEMA_VERSION + 1, "session_open": True}),
        encoding="utf-8",
    )

    state = store.load()

    assert state.session_open is False
    assert store.path.with_suffix(".json.corrupt").is_file()


def test_a_missing_key_falls_back_to_its_default(store: StateStore) -> None:
    # This is the daemon's own scratch space. Refusing to boot over one absent key would be a
    # self-inflicted outage.
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(json.dumps({"schema_version": STATE_SCHEMA_VERSION}), encoding="utf-8")

    state = store.load()

    assert state.last_log_ts is None
    assert state.known_players == ()
    assert not store.path.with_suffix(".json.corrupt").exists()


@pytest.mark.parametrize(
    "payload",
    [
        {"last_log_ts": "not a date"},
        {"last_log_ts": 12345},
        {"session": {"container_id": "abc", "started_at": "nonsense"}},
        {"session": {"started_at": "2026-01-01T00:00:00+00:00"}},
        {"known_players": "Steve"},
        {"known_players": ["Steve", 7, None]},
        {"counters": {"sessions": "many", "ok": 3}},
    ],
)
def test_junk_values_degrade_to_defaults(store: StateStore, payload: dict[str, object]) -> None:
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps({"schema_version": STATE_SCHEMA_VERSION, **payload}),
        encoding="utf-8",
    )

    state = store.load()

    assert state.last_log_ts is None or state.last_log_ts.tzinfo is not None
    assert all(isinstance(name, str) for name in state.known_players)
    assert all(isinstance(value, int) for value in state.counters.values())


def test_a_naive_timestamp_in_a_hand_edited_file_is_read_as_utc(store: StateStore) -> None:
    # This store only ever writes tz-aware UTC, so a naive value means somebody edited the file and
    # the intent is unambiguous.
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps({"schema_version": STATE_SCHEMA_VERSION, "last_log_ts": "2026-01-01T10:00:00"}),
        encoding="utf-8",
    )

    state = store.load()

    assert state.last_log_ts == STARTED_AT


def test_an_offset_timestamp_is_normalised_to_utc(store: StateStore) -> None:
    # The homelab runs Asia/Kolkata, so +05:30 is exactly what a hand-written value looks like.
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps(
            {"schema_version": STATE_SCHEMA_VERSION, "last_log_ts": "2026-01-01T15:30:00+05:30"}
        ),
        encoding="utf-8",
    )

    state = store.load()

    assert state.last_log_ts == STARTED_AT
    assert state.last_log_ts is not None
    assert state.last_log_ts.tzinfo is UTC


def test_loading_over_a_dirty_store_resets_it(store: StateStore) -> None:
    store.record_log_ts(STARTED_AT)
    assert store.dirty is True

    store.load()

    assert store.dirty is False
    assert store.state.last_log_ts is None


# ------------------------------------------------------------------------------ idle deadline


def test_the_idle_deadline_is_persisted_for_observability(store: StateStore) -> None:
    """Persisted, but the idle manager restarts its countdown from now on boot.

    A crash-looping manager that honoured an expired deadline would stop the server on every
    restart, which is the single most dangerous bug available in this design. This store records
    the number; deciding what to do with it is somebody else's job, deliberately.
    """
    deadline = STARTED_AT + timedelta(minutes=15)
    store.set_idle_deadline(deadline, empty_since=STARTED_AT)

    assert store.state.idle_deadline == deadline

    store.set_idle_deadline(None)
    assert store.state.idle_deadline is None
    assert store.state.idle_empty_since is None


def test_setting_a_counter_to_its_current_value_is_a_no_op(store: StateStore) -> None:
    store.set_counter("sessions", 3)
    store.save()
    store.set_counter("sessions", 3)

    assert store.dirty is False


def test_a_boolean_schema_version_degrades_to_the_default(store: StateStore) -> None:
    # `isinstance(True, int)` is True in Python, so a naive check would take `"schema_version":
    # true` as a genuine integer. It is rejected as a *value* and falls back to the current schema,
    # which is the same lenient rule every other field follows: this is the daemon's own scratch
    # file, and one hand-edited key must not be a refusal to boot.
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps({"schema_version": True, "session_open": True}),
        encoding="utf-8",
    )

    state = store.load()

    assert state.schema_version == STATE_SCHEMA_VERSION
    assert state.session_open is True
    assert not store.path.with_suffix(".json.corrupt").exists()
