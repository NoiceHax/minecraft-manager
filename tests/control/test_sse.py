"""The SSE fan-out: filtering, framing, and the lossy per-client queue.

The property this file exists to protect is the one in the plan: **a slow ``curl`` must never stall
the daemon**. Everything else - type filters, ``since``, keepalives - is convenience on top of a
bounded, drop-oldest, count-it-out-loud queue.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from mcmanager.control.sse import (
    EventFilter,
    SseChannel,
    SseClient,
    known_type_names,
    parse_since,
    render_frame,
    resolve_types,
)
from mcmanager.core.bus import EventBus
from mcmanager.core.events import (
    ChatMessage,
    ConsoleLog,
    Event,
    PlayerEvent,
    PlayerJoined,
    PlayerLeft,
    ServerEvent,
    ServerStopping,
)
from mcmanager.core.types import PlayerRef, Source

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mcmanager.clock import ManualClock

NOW = datetime(2026, 7, 25, 22, 58, 20, tzinfo=UTC)


def joined(name: str = "Steve", *, seq: int = 0, ts: datetime = NOW) -> PlayerJoined:
    return PlayerJoined(
        ts=ts,
        server_id="mc",
        source=Source.LOG,
        raw=f"{name} joined the game",
        seq=seq,
        player=PlayerRef(name=name),
        online_count=1,
    )


def console(message: str = "Preparing spawn area: 2%", *, seq: int = 0) -> ConsoleLog:
    return ConsoleLog(
        ts=NOW,
        server_id="mc",
        source=Source.LOG,
        raw=message,
        seq=seq,
        message=message,
        level="INFO",
        thread="Server thread",
    )


async def drain(client: SseClient, *, expected: int) -> list[str]:
    """Pull exactly ``expected`` frames, or fail rather than hang the suite."""
    frames: list[str] = []
    iterator: AsyncIterator[str] = client.__aiter__()
    for _ in range(expected):
        async with asyncio.timeout(1.0):
            frames.append(await iterator.__anext__())
    return frames


class TestTypeResolution:
    def test_a_concrete_type_resolves(self) -> None:
        assert resolve_types(["PlayerJoined"]) == frozenset({PlayerJoined})

    def test_a_category_base_resolves(self) -> None:
        # EVENT_BY_NAME deliberately holds only concrete types; the category bases live here.
        assert resolve_types(["PlayerEvent"]) == frozenset({PlayerEvent})

    def test_several_names_resolve_together(self) -> None:
        assert resolve_types(["PlayerEvent", "ServerEvent"]) == frozenset(
            {PlayerEvent, ServerEvent}
        )

    def test_blank_entries_are_ignored(self) -> None:
        assert resolve_types([" ", "", "PlayerJoined"]) == frozenset({PlayerJoined})

    def test_an_unknown_name_raises_and_lists_the_accepted_ones(self) -> None:
        # A silently ignored filter looks exactly like a broken daemon.
        with pytest.raises(ValueError, match="unknown event type 'PlayerJoin'") as caught:
            resolve_types(["PlayerJoin"])
        assert "PlayerJoined" in str(caught.value)

    def test_the_known_names_include_both_concrete_types_and_categories(self) -> None:
        names = known_type_names()
        assert "PlayerJoined" in names
        assert "PlayerEvent" in names
        assert "Event" in names


class TestParseSince:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [("30s", 30), ("15m", 900), ("2h", 7200), ("1d", 86400), ("0s", 0)],
    )
    def test_a_relative_age_is_subtracted_from_now(self, text: str, seconds: int) -> None:
        assert parse_since(text, now=NOW) == NOW - timedelta(seconds=seconds)

    def test_an_rfc3339_timestamp_with_a_z_is_accepted(self) -> None:
        assert parse_since("2026-07-25T22:58:20Z", now=NOW) == NOW

    def test_an_offset_timestamp_is_converted_to_utc(self) -> None:
        assert parse_since("2026-07-26T04:28:20+05:30", now=NOW) == NOW

    def test_a_naive_timestamp_is_refused(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            parse_since("2026-07-25T22:58:20", now=NOW)

    def test_nonsense_is_refused_with_both_accepted_forms_named(self) -> None:
        with pytest.raises(ValueError, match="relative age") as caught:
            parse_since("soon", now=NOW)
        assert "RFC3339" in str(caught.value)

    def test_an_empty_value_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_since("   ", now=NOW)


class TestEventFilter:
    def test_an_empty_filter_matches_everything(self) -> None:
        assert EventFilter().matches(joined())

    def test_a_category_filter_matches_subclasses(self) -> None:
        assert EventFilter(types=frozenset({PlayerEvent})).matches(joined())

    def test_a_category_filter_does_not_match_a_sibling(self) -> None:
        # ChatMessage is deliberately NOT a PlayerEvent - see core/events.py.
        chat = ChatMessage(
            ts=NOW, server_id="mc", source=Source.LOG, player=PlayerRef(name="Steve"), message="hi"
        )
        assert not EventFilter(types=frozenset({PlayerEvent})).matches(chat)

    def test_since_excludes_older_events(self) -> None:
        cutoff = NOW - timedelta(minutes=5)
        assert not EventFilter(since=cutoff).matches(joined(ts=NOW - timedelta(minutes=10)))
        assert EventFilter(since=cutoff).matches(joined(ts=NOW))

    def test_since_seq_is_strictly_greater(self) -> None:
        assert not EventFilter(since_seq=10).matches(joined(seq=10))
        assert EventFilter(since_seq=10).matches(joined(seq=11))


class TestFraming:
    def test_an_events_frame_carries_the_serde_payload_and_the_seq_as_the_id(self) -> None:
        frame = render_frame(joined(seq=41), "events")
        assert frame is not None
        assert frame.startswith("id: 41\nevent: PlayerJoined\ndata: {")
        assert frame.endswith("\n\n")
        payload = json.loads(frame.split("data: ", 1)[1].strip())
        assert payload["type"] == "PlayerJoined"
        assert payload["player"]["name"] == "Steve"

    def test_a_frame_is_always_a_single_data_line(self) -> None:
        # A newline inside the JSON would end the frame early and truncate the event.
        chatty = ChatMessage(
            ts=NOW,
            server_id="mc",
            source=Source.LOG,
            player=PlayerRef(name="Steve"),
            message="line one\nline two",
        )
        frame = render_frame(chatty, "events")
        assert frame is not None
        assert len([line for line in frame.splitlines() if line.startswith("data:")]) == 1

    def test_a_logs_frame_projects_a_console_line(self) -> None:
        frame = render_frame(console("Done"), "logs")
        assert frame is not None
        payload = json.loads(frame.split("data: ", 1)[1].strip())
        assert payload["message"] == "Done"
        assert payload["thread"] == "Server thread"

    def test_a_logs_frame_falls_back_to_raw_for_a_recognised_event(self) -> None:
        # Recognised lines emit no ConsoleLog, so a /logs view built only from ConsoleLog would
        # have holes exactly where the joins were.
        frame = render_frame(joined(), "logs")
        assert frame is not None
        payload = json.loads(frame.split("data: ", 1)[1].strip())
        assert payload["message"] == "Steve joined the game"
        assert payload["type"] == "PlayerJoined"

    def test_an_event_with_no_raw_contributes_nothing_to_the_log_stream(self) -> None:
        bare = ServerStopping(ts=NOW, server_id="mc", source=Source.INTERNAL, raw=None)
        assert render_frame(bare, "logs") is None
        assert render_frame(bare, "events") is not None


class TestClientQueue:
    async def test_frames_arrive_in_order(self) -> None:
        client = SseClient(wake=asyncio.Event(), event_filter=EventFilter(), maxsize=10)
        client.offer(joined("Steve", seq=1))
        client.offer(joined("Alex", seq=2))
        frames = await drain(client, expected=2)
        assert "Steve" in frames[0]
        assert "Alex" in frames[1]

    async def test_a_full_queue_drops_the_oldest_and_counts_it(self) -> None:
        client = SseClient(wake=asyncio.Event(), event_filter=EventFilter(), maxsize=2)
        for index in range(5):
            client.offer(joined(f"p{index}", seq=index))
        assert client.dropped == 3
        assert client.queued == 2

    async def test_a_drop_is_announced_in_band_before_the_next_frame(self) -> None:
        # The counter is not silent: somebody watching `curl -N` sees the gap.
        client = SseClient(wake=asyncio.Event(), event_filter=EventFilter(), maxsize=1)
        client.offer(joined("first", seq=1))
        client.offer(joined("second", seq=2))
        frames = await drain(client, expected=2)
        assert frames[0].startswith(": dropped 1 frame(s)")
        assert "second" in frames[1]

    async def test_offering_never_raises_on_a_closed_client(self) -> None:
        client = SseClient(wake=asyncio.Event(), event_filter=EventFilter())
        client.close()
        client.offer(joined())
        assert client.queued == 0

    async def test_filtered_events_never_enter_the_queue(self) -> None:
        client = SseClient(
            wake=asyncio.Event(),
            event_filter=EventFilter(types=frozenset({PlayerLeft})),
        )
        client.offer(joined())
        assert client.queued == 0

    async def test_the_iterator_finishes_when_the_client_is_closed(self) -> None:
        client = SseClient(wake=asyncio.Event(), event_filter=EventFilter())
        collected: list[str] = []

        async def reader() -> None:
            async for frame in client:
                collected.append(frame)

        task = asyncio.create_task(reader())
        await asyncio.sleep(0)
        client.offer(joined())
        await asyncio.sleep(0)
        client.close()
        async with asyncio.timeout(1.0):
            await task
        assert len(collected) == 1

    def test_a_zero_sized_queue_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            SseClient(wake=asyncio.Event(), event_filter=EventFilter(), maxsize=0)


class TestChannel:
    async def _bus(self) -> tuple[EventBus, asyncio.Task[None]]:
        bus = EventBus()
        task = asyncio.create_task(bus.run())
        await asyncio.sleep(0)
        return bus, task

    async def test_the_channel_fans_a_published_event_out_to_every_client(
        self, clock: ManualClock
    ) -> None:
        bus, task = await self._bus()
        channel = SseChannel(clock=clock)
        channel.attach(bus)
        first = channel.open()
        second = channel.open()

        bus.publish(joined())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert first.queued == 1
        assert second.queued == 1
        await channel.aclose()
        await bus.aclose()
        await task

    async def test_a_client_only_receives_what_it_asked_for(self, clock: ManualClock) -> None:
        bus, task = await self._bus()
        channel = SseChannel(clock=clock)
        channel.attach(bus)
        players = channel.open(event_filter=EventFilter(types=frozenset({PlayerEvent})))

        bus.publish(joined())
        bus.publish(console())
        for _ in range(4):
            await asyncio.sleep(0)

        assert players.queued == 1
        await channel.aclose()
        await bus.aclose()
        await task

    async def test_history_is_replayed_only_when_a_since_was_asked_for(
        self, clock: ManualClock
    ) -> None:
        bus, task = await self._bus()
        channel = SseChannel(clock=clock)
        channel.attach(bus)
        bus.publish(joined(seq=1))
        for _ in range(3):
            await asyncio.sleep(0)

        plain = channel.open()
        assert plain.queued == 0, "a bare --follow wants what happens next, not a dump"

        resuming = channel.open(event_filter=EventFilter(since_seq=0))
        assert resuming.queued == 1

        await channel.aclose()
        await bus.aclose()
        await task

    async def test_history_is_bounded(self, clock: ManualClock) -> None:
        bus, task = await self._bus()
        channel = SseChannel(clock=clock, history=3)
        channel.attach(bus)
        for index in range(10):
            bus.publish(joined(seq=index))
        for _ in range(24):
            await asyncio.sleep(0)

        assert len(channel.history()) == 3
        await channel.aclose()
        await bus.aclose()
        await task

    async def test_a_keepalive_reaches_every_client_on_the_injected_clock(
        self, clock: ManualClock
    ) -> None:
        channel = SseChannel(clock=clock, keepalive_seconds=15.0)
        channel.attach(EventBus())
        client = channel.open()

        runner = asyncio.create_task(channel.run_keepalive())
        await clock.tick()
        await clock.advance(15.0)

        assert client.queued == 1
        frames = await drain(client, expected=1)
        assert frames[0].startswith(": keepalive")

        await channel.aclose()
        runner.cancel()

    async def test_closing_the_channel_ends_every_stream(self, clock: ManualClock) -> None:
        channel = SseChannel(clock=clock)
        channel.attach(EventBus())
        client = channel.open()
        await channel.aclose()
        assert client.closed
        assert channel.stats["clients"] == 0

    async def test_closing_a_client_twice_is_harmless(self, clock: ManualClock) -> None:
        channel = SseChannel(clock=clock)
        channel.attach(EventBus())
        client = channel.open()
        channel.close_client(client)
        channel.close_client(client)
        assert channel.stats["clients"] == 0

    async def test_the_channel_unsubscribes_from_the_bus_on_close(self, clock: ManualClock) -> None:
        bus = EventBus()
        channel = SseChannel(clock=clock)
        subscription = channel.attach(bus)
        assert subscription.active
        await channel.aclose()
        assert not subscription.active

    async def test_a_slow_client_never_blocks_the_bus(self, clock: ManualClock) -> None:
        """The whole reason this module is shaped the way it is.

        A reader that never reads gets a bounded buffer and a drop counter; the bus keeps
        dispatching, and every other subscriber sees every event.
        """
        bus, task = await self._bus()
        channel = SseChannel(clock=clock, queue_max=4)
        channel.attach(bus)
        stalled = channel.open()

        seen: list[Event] = []

        async def attentive(event: Event) -> None:
            seen.append(event)

        bus.subscribe_all(attentive, name="attentive")

        for index in range(50):
            bus.publish(joined(seq=index))
        for _ in range(160):
            await asyncio.sleep(0)

        assert len(seen) == 50, "the bus kept dispatching to everyone else"
        assert stalled.queued == 4
        assert stalled.dropped == 46
        assert channel.stats["dropped"] == 46

        await channel.aclose()
        await bus.aclose()
        await task
