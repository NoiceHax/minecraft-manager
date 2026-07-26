"""Tests for :mod:`mcmanager.services.players`.

The one that matters most is
:func:`test_an_incomplete_probe_sample_produces_zero_phantom_leaves`. Everything else in this file
is bookkeeping; that one is the difference between an idle manager that works and one that stops a
server people are building on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from mcmanager.core.types import LeaveReason
from mcmanager.games.base import ProbeResult
from mcmanager.services.players import PlayerRoster

if TYPE_CHECKING:
    from mcmanager.clock import ManualClock

PROBED_AT = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def probe(
    *,
    reachable: bool = True,
    online: int | None = 0,
    sample: tuple[str, ...] = (),
    error: str | None = None,
) -> ProbeResult:
    """Build a probe result with the shape a test cares about."""
    return ProbeResult(
        reachable=reachable,
        players_online=online,
        players_max=5,
        sample=sample,
        error=error,
        probed_at=PROBED_AT,
    )


@pytest.fixture
def roster(manual_clock: ManualClock) -> PlayerRoster:
    return PlayerRoster(clock=manual_clock)


# --------------------------------------------------------------------------------- log-driven


def test_a_join_puts_a_player_online(roster: PlayerRoster) -> None:
    session = roster.join("Steve")

    assert roster.online == ("Steve",)
    assert roster.count == 1
    assert roster.is_online("Steve")
    assert session.name == "Steve"
    assert session.source == "log"


def test_the_first_join_of_a_name_is_flagged_first_seen(roster: PlayerRoster) -> None:
    assert roster.join("Steve").first_seen is True
    roster.leave("Steve")
    assert roster.join("Steve").first_seen is False


def test_known_names_seed_first_seen_across_a_daemon_restart(manual_clock: ManualClock) -> None:
    # first_seen has to mean "first time on this server", not "first time since we rebooted",
    # or every restart re-announces the regulars.
    roster = PlayerRoster(clock=manual_clock, known_names=("Steve",))

    assert roster.join("Steve").first_seen is False
    assert roster.join("Alex").first_seen is True


def test_a_duplicate_join_keeps_the_original_session_start(roster: PlayerRoster) -> None:
    first = roster.join("Steve")
    roster.join("Steve")

    assert roster.count == 1
    assert roster.session("Steve") is not None
    assert roster.sessions["Steve"].joined_monotonic == first.joined_monotonic
    assert roster.stats["duplicate_joins"] == 1
    assert roster.stats["joins"] == 1


def test_a_duplicate_join_fills_in_a_late_uuid_and_address(roster: PlayerRoster) -> None:
    roster.join("Steve")
    merged = roster.join("Steve", uuid="u-1", address="1.2.3.4:1")

    assert merged.uuid == "u-1"
    assert merged.address == "1.2.3.4:1"


def test_a_leave_returns_the_session_and_removes_the_player(roster: PlayerRoster) -> None:
    roster.join("Steve")
    session = roster.leave("Steve", reason=LeaveReason.QUIT)

    assert session is not None
    assert session.name == "Steve"
    assert roster.count == 0
    assert not roster.is_online("Steve")


def test_a_leave_for_somebody_we_never_saw_join_returns_none(roster: PlayerRoster) -> None:
    # The daemon started mid-session, or - as the real archives show - a port scanner
    # authenticated and hung up without ever joining. Inventing a duration here would put a
    # fabricated number in a session summary.
    assert roster.leave("KittyScan") is None
    assert roster.stats["unknown_leaves"] == 1


async def test_session_seconds_are_measured_on_the_monotonic_clock(
    roster: PlayerRoster,
    manual_clock: ManualClock,
) -> None:
    roster.join("Steve")
    await manual_clock.advance(125.0)

    assert roster.session_seconds("Steve") == pytest.approx(125.0)


def test_session_seconds_is_none_for_somebody_offline(roster: PlayerRoster) -> None:
    assert roster.session_seconds("Steve") is None


def test_clear_empties_the_roster_but_remembers_the_names(roster: PlayerRoster) -> None:
    roster.join("Steve")
    roster.join("Alex")

    ended = roster.clear()

    assert {session.name for session in ended} == {"Steve", "Alex"}
    assert roster.count == 0
    assert roster.has_seen("Steve")


def test_clear_without_remember_forgets_the_names(roster: PlayerRoster) -> None:
    roster.join("Steve")
    roster.clear(remember=False)

    assert roster.known_names == frozenset()


def test_note_known_records_a_name_without_bringing_them_online(roster: PlayerRoster) -> None:
    roster.note_known("Steve")

    assert roster.has_seen("Steve")
    assert roster.count == 0


# ------------------------------------------------------------------------------- probe-driven


def test_an_incomplete_probe_sample_produces_zero_phantom_leaves(roster: PlayerRoster) -> None:
    """The single most important test in this module.

    ``players.sample`` is protocol-capped at 12 and Paper settings or an anti-scrape plugin may
    randomise or truncate it. A roster that reconciles against a truncated sample declares every
    player missing from it gone - a storm of ``PlayerLeft`` events for people standing in the
    world, an idle timer that arms, and eventually a stopped server.
    """
    roster.join("Steve")
    roster.join("Alex")
    roster.join("Notch")

    # The server says three people are online but only named one of them.
    delta = roster.reconcile(probe(online=3, sample=("Steve",)))

    assert delta.applied is False
    assert delta.reason == "sample_incomplete"
    assert delta.left == ()
    assert delta.joined == ()
    assert roster.online == ("Steve", "Alex", "Notch")
    assert roster.stats["probe_leaves"] == 0
    assert roster.stats["probe_skipped"] == 1


def test_an_unreachable_probe_is_unknown_and_never_empty(roster: PlayerRoster) -> None:
    # Unreachable read as "zero players" is the failure mode that loses somebody's build session.
    roster.join("Steve")

    result = probe(reachable=False, online=None, error="ConnectionRefusedError")
    delta = roster.reconcile(result)

    assert result.is_empty is False
    assert delta.applied is False
    assert delta.reason == "probe_unreachable"
    assert roster.online == ("Steve",)


def test_a_probe_with_no_player_count_changes_nothing(roster: PlayerRoster) -> None:
    roster.join("Steve")

    delta = roster.reconcile(probe(online=None))

    assert delta.applied is False
    assert delta.reason == "probe_reported_no_count"
    assert roster.online == ("Steve",)


def test_a_complete_sample_reconciles_in_both_directions(roster: PlayerRoster) -> None:
    roster.join("Steve")
    roster.join("Alex")

    delta = roster.reconcile(probe(online=2, sample=("Steve", "Notch")))

    assert delta.applied is True
    assert delta.joined_names == ("Notch",)
    assert delta.left_names == ("Alex",)
    assert delta.online_count == 2
    assert set(roster.online) == {"Steve", "Notch"}
    assert delta.changed is True


def test_a_reachable_empty_server_clears_the_roster(roster: PlayerRoster) -> None:
    # This is the case idle shutdown depends on, and it passes the completeness gate trivially
    # because 0 == len(()). If the gate ever blocked it, idle shutdown would never fire.
    roster.join("Steve")

    delta = roster.reconcile(probe(online=0, sample=()))

    assert delta.applied is True
    assert delta.left_names == ("Steve",)
    assert roster.count == 0


def test_a_matching_complete_sample_is_a_no_op(roster: PlayerRoster) -> None:
    roster.join("Steve")

    delta = roster.reconcile(probe(online=1, sample=("Steve",)))

    assert delta.applied is True
    assert delta.changed is False
    assert roster.online == ("Steve",)


def test_trust_partial_sample_overrides_the_gate(roster: PlayerRoster) -> None:
    roster.join("Steve")
    roster.join("Alex")

    delta = roster.reconcile(probe(online=3, sample=("Steve",)), trust_partial_sample=True)

    assert delta.applied is True
    assert delta.left_names == ("Alex",)


def test_a_repeated_name_in_the_sample_is_de_duplicated(roster: PlayerRoster) -> None:
    delta = roster.reconcile(probe(online=2, sample=("Steve", "Steve")))

    assert delta.applied is True
    assert delta.joined_names == ("Steve",)
    assert roster.count == 1


async def test_a_probe_derived_leave_still_carries_its_duration(
    roster: PlayerRoster,
    manual_clock: ManualClock,
) -> None:
    # The delta hands back sessions rather than names precisely so the poller can put a real
    # session_seconds on the PlayerLeft it publishes.
    roster.join("Steve")
    await manual_clock.advance(90.0)

    delta = roster.reconcile(probe(online=0, sample=()))

    assert len(delta.left) == 1
    assert delta.left[0].seconds_at(manual_clock.monotonic()) == pytest.approx(90.0)


def test_a_probe_derived_join_is_labelled_as_such(roster: PlayerRoster) -> None:
    roster.reconcile(probe(online=1, sample=("Steve",)))

    session = roster.session("Steve")
    assert session is not None
    assert session.source == "probe"
    assert session.address is None
    assert session.joined_at == PROBED_AT


def test_stats_distinguish_reconciles_from_skips(roster: PlayerRoster) -> None:
    roster.join("Steve")
    roster.reconcile(probe(online=4, sample=("Steve",)))
    roster.reconcile(probe(reachable=False, online=None))
    roster.reconcile(probe(online=1, sample=("Steve",)))

    stats = roster.stats
    assert stats["probe_skipped"] == 2
    assert stats["probe_reconciles"] == 1


def test_seconds_at_never_goes_negative(roster: PlayerRoster) -> None:
    session = roster.join("Steve")

    # A wall clock can jump; the monotonic reading a caller passes should never be able to produce
    # "-4 seconds online" in a Discord message.
    assert session.seconds_at(session.joined_monotonic - 10.0) == 0.0
