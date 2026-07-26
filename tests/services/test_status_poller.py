"""Tests for :mod:`mcmanager.services.status_poller`.

Not in the original assignment for this agent; added because the poller is the component that
turns a failed probe into a decision, and shipping it untested would leave the plan's single
loudest warning - "a failed probe is UNKNOWN, never zero players" - resting on a docstring.

Everything here runs on :class:`~mcmanager.clock.ManualClock`. A 45-second polling interval is
exercised in about a millisecond, and nothing sleeps for real.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, final

import pytest

from mcmanager.core.events import Event, PlayerJoined, PlayerLeft
from mcmanager.core.types import LeaveReason, Source
from mcmanager.games.base import ProbeResult
from mcmanager.services.players import PlayerRoster
from mcmanager.services.status_poller import StatusPoller

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcmanager.clock import Clock, ManualClock
    from mcmanager.core.types import ReadySignal, ServerId, Stream

SERVER_ID = "minecraft"


@final
class RecordingSink:
    """Structurally an ``EventSink``."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)

    def of[E: Event](self, kind: type[E]) -> list[E]:
        return [event for event in self.events if isinstance(event, kind)]


@final
class ScriptedAdapter:
    """A ``GameAdapter`` whose ``probe`` returns queued results, then repeats the last one."""

    def __init__(self, *, clock: Clock, results: Sequence[ProbeResult] = ()) -> None:
        self._clock = clock
        self.results = list(results)
        self.calls: list[tuple[str, int, float]] = []
        self.raise_next = False

    def queue(self, result: ProbeResult) -> None:
        self.results.append(result)

    @property
    def game_id(self) -> str:
        return "scripted"

    @property
    def default_port(self) -> int:
        return 25565

    @property
    def default_stop_timeout(self) -> int:
        return 90

    def parse_line(
        self,
        raw: str,
        *,
        ts: datetime,
        server_id: ServerId,
        stream: Stream,
    ) -> Event:
        del ts, server_id, stream
        msg = f"ScriptedAdapter does not parse: {raw!r}"
        raise NotImplementedError(msg)

    def ready_signal(self, event: Event) -> ReadySignal | None:
        del event
        return None

    def stop_signal(self, event: Event) -> bool:
        del event
        return False

    async def probe(
        self,
        host: str,
        port: int,
        *,
        timeout: float,  # noqa: ASYNC109 - the GameAdapter contract; handed to the query library
    ) -> ProbeResult:
        self.calls.append((host, port, timeout))
        if self.raise_next:
            self.raise_next = False
            msg = "mcstatus broke its own contract"
            raise RuntimeError(msg)
        if len(self.results) > 1:
            return self.results.pop(0)
        if self.results:
            return self.results[0]
        return ProbeResult(reachable=False, error="no result queued", probed_at=self._clock.now())


def ok(
    clock: Clock,
    *,
    online: int = 0,
    sample: tuple[str, ...] = (),
) -> ProbeResult:
    return ProbeResult(
        reachable=True,
        players_online=online,
        players_max=5,
        sample=sample,
        version="26.2",
        latency_ms=3.2,
        probed_at=clock.now(),
    )


def down(clock: Clock, error: str = "ConnectionRefusedError") -> ProbeResult:
    return ProbeResult(reachable=False, error=error, probed_at=clock.now())


def build(
    clock: Clock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
    **kwargs: object,
) -> StatusPoller:
    return StatusPoller(
        adapter=adapter,
        host="minecraft",
        port=25565,
        clock=clock,
        sink=sink,
        roster=roster,
        server_id=SERVER_ID,
        **kwargs,  # pyright: ignore[reportArgumentType]
    )


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


@pytest.fixture
def roster(manual_clock: ManualClock) -> PlayerRoster:
    return PlayerRoster(clock=manual_clock)


@pytest.fixture
def adapter(manual_clock: ManualClock) -> ScriptedAdapter:
    return ScriptedAdapter(clock=manual_clock)


# -------------------------------------------------------------------- failure means UNKNOWN


async def test_a_failed_probe_is_unknown_and_never_zero_players(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    """The failure mode the plan calls out as the one that loses somebody's build session."""
    roster.join("Steve")
    adapter.queue(down(manual_clock))
    poller = build(manual_clock, adapter, sink, roster)

    await poller.poll_once()

    assert poller.reachable is False
    assert poller.players_online is None
    assert roster.online == ("Steve",)
    assert sink.events == []
    assert poller.stats["skipped_unreachable"] == 1


async def test_an_adapter_that_raises_is_absorbed_as_a_failed_probe(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    # GameAdapter.probe promises never to raise. An adapter is the most likely place for a
    # third-party library to break that promise, and one bad probe must not kill the task.
    adapter.raise_next = True
    poller = build(manual_clock, adapter, sink, roster)

    result = await poller.poll_once()

    assert result.reachable is False
    assert result.error is not None
    assert "RuntimeError" in result.error
    assert poller.consecutive_failures == 1


async def test_failures_are_counted_and_reset_by_a_success(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    adapter.queue(down(manual_clock))
    adapter.queue(down(manual_clock))
    adapter.queue(ok(manual_clock))
    poller = build(manual_clock, adapter, sink, roster)

    await poller.poll_once()
    await poller.poll_once()
    assert poller.consecutive_failures == 2

    await poller.poll_once()
    assert poller.consecutive_failures == 0
    assert poller.reachable is True
    assert poller.last_success_at is not None


# ---------------------------------------------------------------------- the completeness gate


async def test_an_incomplete_sample_publishes_no_phantom_leaves(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    roster.join("Steve")
    roster.join("Alex")
    adapter.queue(ok(manual_clock, online=4, sample=("Steve",)))
    poller = build(manual_clock, adapter, sink, roster)

    await poller.poll_once()

    assert sink.of(PlayerLeft) == []
    assert set(roster.online) == {"Steve", "Alex"}
    assert poller.stats["skipped_incomplete"] == 1
    assert poller.stats["probe_leaves"] == 0


async def test_trust_partial_sample_is_honoured_when_set(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    roster.join("Steve")
    roster.join("Alex")
    adapter.queue(ok(manual_clock, online=4, sample=("Steve",)))
    poller = build(manual_clock, adapter, sink, roster, trust_partial_sample=True)

    await poller.poll_once()

    assert [event.player.name for event in sink.of(PlayerLeft)] == ["Alex"]


# ------------------------------------------------------------------------- probe-derived events


async def test_a_complete_sample_publishes_probe_sourced_joins(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    # A join that happened while the log stream was detached is otherwise invisible forever.
    adapter.queue(ok(manual_clock, online=1, sample=("Steve",)))
    poller = build(manual_clock, adapter, sink, roster)

    await poller.poll_once()

    joined = sink.of(PlayerJoined)
    assert len(joined) == 1
    assert joined[0].source is Source.PROBE
    assert joined[0].player.name == "Steve"
    assert joined[0].online_count == 1
    assert joined[0].first_seen is True


async def test_a_probe_derived_leave_carries_a_real_duration_and_no_guessed_reason(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    roster.join("Steve")
    await manual_clock.advance(300.0)
    adapter.queue(ok(manual_clock, online=0, sample=()))
    poller = build(manual_clock, adapter, sink, roster)

    await poller.poll_once()

    left = sink.of(PlayerLeft)[0]
    assert left.source is Source.PROBE
    # The probe cannot know why somebody left, and a guessed "quit" would be a fabricated fact in
    # a session summary.
    assert left.reason is LeaveReason.UNKNOWN
    assert left.session_seconds == pytest.approx(300.0)
    assert left.online_count == 0


async def test_a_matching_roster_publishes_nothing(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    roster.join("Steve")
    adapter.queue(ok(manual_clock, online=1, sample=("Steve",)))
    poller = build(manual_clock, adapter, sink, roster)

    await poller.poll_once()

    assert sink.events == []
    assert poller.stats["reconciled"] == 1


# ------------------------------------------------------------------------------ the observer


async def test_every_result_reaches_the_observer(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    # This is lifecycle's hook: it wraps the result in a ProbeSignal. Failures matter to it as
    # much as successes, so both are delivered.
    seen: list[ProbeResult] = []
    adapter.queue(down(manual_clock))
    adapter.queue(ok(manual_clock, online=0))
    poller = build(manual_clock, adapter, sink, roster, on_probe=seen.append)

    await poller.poll_once()
    await poller.poll_once()

    assert [result.reachable for result in seen] == [False, True]


async def test_a_raising_observer_does_not_stop_the_poller(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    def explode(result: ProbeResult) -> None:
        del result
        msg = "lifecycle is broken"
        raise RuntimeError(msg)

    adapter.queue(ok(manual_clock, online=1, sample=("Steve",)))
    poller = build(manual_clock, adapter, sink, roster, on_probe=explode)

    await poller.poll_once()

    assert poller.stats["observer_errors"] == 1
    # Reconciliation still happened: a broken observer is not a reason to stop tracking players.
    assert sink.of(PlayerJoined) != []


# -------------------------------------------------------------------------------- the loop


async def test_run_probes_immediately_then_on_the_interval(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    adapter.queue(ok(manual_clock, online=0))
    poller = build(manual_clock, adapter, sink, roster, interval=45.0)
    task = asyncio.create_task(poller.run())
    try:
        # ManualClock.advance only fires timers already registered, so the loop is given a chance
        # to reach its first sleep before time moves.
        await manual_clock.tick()
        assert len(adapter.calls) == 1, "the first probe must not wait 45 seconds"

        await manual_clock.advance(45.0)
        assert len(adapter.calls) == 2

        await manual_clock.advance(90.0)
        assert len(adapter.calls) == 4
    finally:
        await poller.aclose()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_aclose_stops_the_loop_at_the_next_interval(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    adapter.queue(ok(manual_clock, online=0))
    poller = build(manual_clock, adapter, sink, roster, interval=45.0)
    task = asyncio.create_task(poller.run())
    await manual_clock.tick()

    await poller.aclose()
    await manual_clock.advance(45.0)
    await manual_clock.tick()

    assert task.done()
    assert len(adapter.calls) == 1


async def test_a_disabled_poller_never_probes(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    poller = build(manual_clock, adapter, sink, roster, enabled=False)

    await poller.run()

    assert adapter.calls == []


async def test_the_configured_timeout_reaches_the_adapter(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    adapter.queue(ok(manual_clock, online=0))
    poller = build(manual_clock, adapter, sink, roster, timeout=7.5)

    await poller.poll_once()

    assert adapter.calls == [("minecraft", 25565, 7.5)]


async def test_last_result_is_kept_for_the_status_command(
    manual_clock: ManualClock,
    adapter: ScriptedAdapter,
    sink: RecordingSink,
    roster: PlayerRoster,
) -> None:
    adapter.queue(ok(manual_clock, online=2, sample=("Steve", "Alex")))
    poller = build(manual_clock, adapter, sink, roster)

    await poller.poll_once()

    assert poller.last_result is not None
    assert poller.last_result.version == "26.2"
    assert poller.players_online == 2
    assert poller.last_success_at == datetime(2026, 1, 1, tzinfo=UTC)
