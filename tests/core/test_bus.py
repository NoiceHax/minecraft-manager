"""The event bus.

Everything downstream of the bus is a finite state machine that is only correct if it observes
events once, in order, and keeps observing them after some *other* subscriber has broken. So the
tests here are mostly about failure: a handler that raises, a handler that hangs, a queue that
fills, a shutdown that arrives mid-dispatch.

The bus deliberately takes no ``Clock`` - its only deadlines are ``asyncio.timeout`` windows - so
the timeout tests use genuinely short real timeouts (tens of milliseconds) rather than a
:class:`~mcmanager.clock.ManualClock`. That is the one place in this suite where a test waits on
the wall clock, and it is bounded by the timeouts the tests themselves set.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Never

import pytest

from mcmanager.core.bus import BusStats, EventBus, Subscription
from mcmanager.core.events import (
    ChatMessage,
    ConsoleLog,
    Event,
    PlayerAdvancement,
    PlayerDeath,
    PlayerEvent,
    PlayerJoined,
    PlayerLeft,
    ServerEvent,
    ServerStarting,
)
from mcmanager.core.types import ChatKind, DispatchMode, PlayerRef, Source

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine

TS = datetime(2026, 7, 25, 22, 58, 20, tzinfo=UTC)


# ------------------------------------------------------------------------------- fixtures


def joined(name: str = "Hypixelite") -> PlayerJoined:
    return PlayerJoined(ts=TS, server_id="mc", source=Source.LOG, player=PlayerRef(name=name))


def left(name: str = "Hypixelite") -> PlayerLeft:
    return PlayerLeft(ts=TS, server_id="mc", source=Source.LOG, player=PlayerRef(name=name))


def death(name: str = "Hypixelite") -> PlayerDeath:
    return PlayerDeath(
        ts=TS,
        server_id="mc",
        source=Source.LOG,
        player=PlayerRef(name=name),
        message=f"{name} was blown up by Creeper",
    )


def advancement(name: str = "Hypixelite") -> PlayerAdvancement:
    return PlayerAdvancement(
        ts=TS,
        server_id="mc",
        source=Source.LOG,
        player=PlayerRef(name=name),
        title="Stone Age",
    )


def chat(message: str = "hello") -> ChatMessage:
    return ChatMessage(
        ts=TS,
        server_id="mc",
        source=Source.LOG,
        player=PlayerRef(name="Hypixelite"),
        message=message,
        kind=ChatKind.CHAT,
    )


def console(message: str = "noise") -> ConsoleLog:
    return ConsoleLog(ts=TS, server_id="mc", source=Source.LOG, message=message)


def starting() -> ServerStarting:
    return ServerStarting(ts=TS, server_id="mc", source=Source.RUNTIME, version="26.2")


class Recorder:
    """A handler that remembers what it saw, and optionally misbehaves."""

    def __init__(
        self,
        *,
        raises: BaseException | None = None,
        hangs: bool = False,
        before: Callable[[Event], None] | None = None,
    ) -> None:
        self.seen: list[Event] = []
        self.calls = 0
        self._raises = raises
        self._hangs = hangs
        self._before = before

    async def __call__(self, event: Event) -> None:
        self.calls += 1
        if self._before is not None:
            self._before(event)
        if self._raises is not None:
            raise self._raises
        if self._hangs:
            await asyncio.Event().wait()
        self.seen.append(event)

    @property
    def names(self) -> list[str]:
        return [type(event).__name__ for event in self.seen]


@pytest.fixture
async def running_bus() -> AsyncIterator[tuple[EventBus, asyncio.Task[None]]]:
    """A bus with its dispatch loop already scheduled, torn down cleanly afterwards."""
    bus = EventBus(handler_timeout=0.5)
    task = asyncio.create_task(bus.run(), name="test-bus")
    await settle()
    try:
        yield bus, task
    finally:
        await bus.aclose(drain_timeout=0.5)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def settle(passes: int = 5) -> None:
    """Give the loop enough passes for the dispatch loop to drain what is queued."""
    for _ in range(passes):
        await asyncio.sleep(0)


async def drain(bus: EventBus, task: asyncio.Task[None]) -> None:
    """Wait until the bus has dispatched everything queued, or the loop task died."""
    for _ in range(200):
        if bus.stats.queued == 0:
            await settle()
            if bus.stats.queued == 0:
                return
        if task.done():
            return
        await asyncio.sleep(0)
    msg = "bus did not drain"
    raise AssertionError(msg)


# ------------------------------------------------------------------------------- delivery


async def test_handlers_are_called_in_registration_order(
    running_bus: tuple[EventBus, asyncio.Task[None]],
) -> None:
    bus, task = running_bus
    order: list[str] = []

    def note(name: str) -> Callable[[Event], Coroutine[None, None, None]]:
        async def handler(event: Event) -> None:
            order.append(name)

        return handler

    bus.subscribe(PlayerJoined, note("first"), name="first")
    bus.subscribe(Event, note("second"), name="second")
    bus.subscribe(PlayerEvent, note("third"), name="third")

    bus.publish(joined())
    await drain(bus, task)

    assert order == ["first", "second", "third"]


async def test_events_are_delivered_in_total_fifo_order(
    running_bus: tuple[EventBus, asyncio.Task[None]],
) -> None:
    """``PlayerJoined -> PlayerLeft`` is a causal chain, not two independent observations."""
    bus, task = running_bus
    recorder = Recorder()
    bus.subscribe_all(recorder, name="all")

    for index in range(20):
        bus.publish(joined(f"p{index}") if index % 2 == 0 else left(f"p{index}"))
    await drain(bus, task)

    assert [event.seq for event in recorder.seen] == list(range(1, 21))


async def test_seq_is_stamped_by_the_bus_not_the_producer(
    running_bus: tuple[EventBus, asyncio.Task[None]],
) -> None:
    bus, task = running_bus
    recorder = Recorder()
    bus.subscribe_all(recorder, name="all")

    original = joined()
    bus.publish(original)
    await drain(bus, task)

    assert original.seq == 0, "publish must not mutate the caller's event"
    assert recorder.seen[0].seq == 1


# ------------------------------------------------------------------------------- matching


@pytest.mark.parametrize(
    ("subscribed_to", "expected"),
    [
        (PlayerEvent, ["PlayerJoined", "PlayerLeft", "PlayerDeath", "PlayerAdvancement"]),
        (PlayerJoined, ["PlayerJoined"]),
        (ServerEvent, ["ServerStarting"]),
        (
            Event,
            [
                "PlayerJoined",
                "PlayerLeft",
                "PlayerDeath",
                "PlayerAdvancement",
                "ChatMessage",
                "ConsoleLog",
                "ServerStarting",
            ],
        ),
    ],
    ids=["player-base", "concrete", "server-base", "everything"],
)
async def test_subscription_matches_subclasses(
    running_bus: tuple[EventBus, asyncio.Task[None]],
    subscribed_to: type[Event],
    expected: list[str],
) -> None:
    bus, task = running_bus
    recorder = Recorder()
    bus.subscribe(subscribed_to, recorder, name="under-test")

    for event in (joined(), left(), death(), advancement(), chat(), console(), starting()):
        bus.publish(event)
    await drain(bus, task)

    assert recorder.names == expected


async def test_chat_and_console_are_not_player_events(
    running_bus: tuple[EventBus, asyncio.Task[None]],
) -> None:
    """A subscriber asking "what did players do" must not be handed the chat firehose."""
    bus, task = running_bus
    recorder = Recorder()
    bus.subscribe(PlayerEvent, recorder, name="players")

    bus.publish(chat())
    bus.publish(console())
    await drain(bus, task)

    assert recorder.seen == []


async def test_the_match_cache_survives_a_later_subscribe(
    running_bus: tuple[EventBus, asyncio.Task[None]],
) -> None:
    """The MRO cache is invalidated on subscribe, or a late subscriber never receives anything."""
    bus, task = running_bus
    first = Recorder()
    bus.subscribe(PlayerJoined, first, name="first")
    bus.publish(joined())
    await drain(bus, task)

    second = Recorder()
    bus.subscribe(PlayerJoined, second, name="second")
    bus.publish(joined())
    await drain(bus, task)

    assert first.calls == 2
    assert second.calls == 1


# ------------------------------------------------------------------------- error isolation


async def test_a_raising_handler_does_not_kill_the_bus_or_block_its_siblings(
    running_bus: tuple[EventBus, asyncio.Task[None]],
) -> None:
    bus, task = running_bus
    before = Recorder()
    broken = Recorder(raises=RuntimeError("subscriber is broken"))
    after = Recorder()
    bus.subscribe_all(before, name="before")
    broken_sub = bus.subscribe_all(broken, name="broken")
    bus.subscribe_all(after, name="after")

    bus.publish(joined())
    bus.publish(left())
    await drain(bus, task)

    assert not task.done(), "the dispatch loop must survive a broken subscriber"
    assert before.calls == 2
    assert after.calls == 2, "a raising handler must not stop the ones registered after it"
    assert broken_sub.failures == 2
    assert bus.stats.handler_errors == 2


async def test_a_hanging_handler_is_timed_out_and_counted(
    running_bus: tuple[EventBus, asyncio.Task[None]],
) -> None:
    """Without a per-handler timeout, one hung SEQUENTIAL subscriber stalls the bus forever."""
    bus, task = running_bus
    hung = Recorder(hangs=True)
    downstream = Recorder()
    hung_sub = bus.subscribe_all(hung, name="hung")
    bus.subscribe_all(downstream, name="downstream")

    bus.publish(joined())
    await asyncio.wait_for(_until(lambda: hung_sub.failures == 1), timeout=5.0)
    await drain(bus, task)

    assert hung_sub.failures == 1
    assert downstream.calls == 1, "the bus must keep going after cancelling the hung handler"
    assert bus.stats.handler_errors == 1
    assert not task.done()


async def test_an_outer_cancellation_is_not_mistaken_for_a_handler_timeout() -> None:
    """The fiddly bit: ``asyncio.timeout`` must re-raise a cancellation it did not cause.

    If this is wrong, every shutdown that lands mid-handler looks like a subscriber fault: the
    subscriber gets a failure it did not earn, and eventually gets paused for it.
    """
    bus = EventBus(handler_timeout=60.0)
    hung = Recorder(hangs=True)
    subscription = bus.subscribe_all(hung, name="hung")
    task = asyncio.create_task(bus.run())
    bus.publish(joined())
    await asyncio.wait_for(_until(lambda: hung.calls == 1), timeout=5.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert subscription.failures == 0, "an outer cancellation is not the subscriber's fault"
    assert bus.stats.handler_errors == 0


async def test_a_subscriber_is_paused_after_repeated_failures() -> None:
    bus = EventBus(handler_timeout=0.5, max_consecutive_failures=3)
    broken = Recorder(raises=RuntimeError("nope"))
    healthy = Recorder()
    broken_sub = bus.subscribe_all(broken, name="broken")
    bus.subscribe_all(healthy, name="healthy")
    task = asyncio.create_task(bus.run())

    for _ in range(6):
        bus.publish(joined())
    await drain(bus, task)

    assert broken_sub.paused
    assert broken.calls == 3, "a paused subscriber stops being called"
    assert healthy.calls == 6, "pausing one subscriber must not affect the others"
    assert bus.stats.paused_subscribers == 1

    broken_sub.resume()
    assert not broken_sub.paused
    assert broken_sub.consecutive_failures == 0

    await bus.aclose(drain_timeout=0.2)
    await asyncio.gather(task, return_exceptions=True)


async def test_a_success_resets_the_consecutive_failure_count() -> None:
    bus = EventBus(handler_timeout=0.5, max_consecutive_failures=3)
    state = {"fail": True}

    async def flaky(event: Event) -> None:
        if state["fail"]:
            msg = "transient"
            raise RuntimeError(msg)

    subscription = bus.subscribe_all(flaky, name="flaky")
    task = asyncio.create_task(bus.run())

    bus.publish(joined())
    bus.publish(joined())
    await drain(bus, task)
    assert subscription.consecutive_failures == 2

    state["fail"] = False
    bus.publish(joined())
    await drain(bus, task)

    assert subscription.consecutive_failures == 0
    assert subscription.failures == 2, "the lifetime counter does not reset"
    assert not subscription.paused

    await bus.aclose(drain_timeout=0.2)
    await asyncio.gather(task, return_exceptions=True)


async def test_a_raising_predicate_is_a_failure_and_skips_only_that_subscriber(
    running_bus: tuple[EventBus, asyncio.Task[None]],
) -> None:
    bus, task = running_bus

    def explode(event: Event) -> bool:
        msg = "predicate is broken"
        raise RuntimeError(msg)

    recorder = Recorder()
    sibling = Recorder()
    subscription = bus.subscribe_all(recorder, name="filtered", predicate=explode)
    bus.subscribe_all(sibling, name="sibling")

    bus.publish(joined())
    await drain(bus, task)

    assert recorder.calls == 0
    assert sibling.calls == 1
    assert subscription.failures == 1


async def test_a_predicate_filters_without_counting_a_failure(
    running_bus: tuple[EventBus, asyncio.Task[None]],
) -> None:
    bus, task = running_bus
    recorder = Recorder()
    subscription = bus.subscribe(
        PlayerJoined,
        recorder,
        name="only-steve",
        predicate=lambda event: event.player.name == "Steve",
    )

    bus.publish(joined("Steve"))
    bus.publish(joined("Alex"))
    await drain(bus, task)

    assert [event.player.name for event in recorder.seen if isinstance(event, PlayerJoined)] == [
        "Steve"
    ]
    assert subscription.failures == 0


class ExplodingEvent(Event):
    """An event that cannot be stamped. Stands in for any internal bug on the publish path."""

    __slots__ = ()

    def with_seq(self, seq: int) -> Never:
        msg = "with_seq is broken"
        raise RuntimeError(msg)


async def test_publish_never_raises_even_when_the_bus_is_broken() -> None:
    """``publish`` is called from the parser hot path; producers must never have to guard it."""
    bus = EventBus()

    bus.publish(ExplodingEvent(ts=TS, server_id="mc", source=Source.INTERNAL))

    assert bus.stats.handler_errors == 1
    assert bus.stats.queued == 0


# ---------------------------------------------------------------------------- unsubscribe


async def test_unsubscribe_stops_delivery(
    running_bus: tuple[EventBus, asyncio.Task[None]],
) -> None:
    bus, task = running_bus
    recorder = Recorder()
    subscription = bus.subscribe_all(recorder, name="temporary")

    bus.publish(joined())
    await drain(bus, task)
    subscription.unsubscribe()
    bus.publish(joined())
    await drain(bus, task)

    assert recorder.calls == 1
    assert bus.stats.subscribers == 0


async def test_unsubscribe_is_idempotent(
    running_bus: tuple[EventBus, asyncio.Task[None]],
) -> None:
    bus, _ = running_bus
    subscription = bus.subscribe_all(Recorder(), name="temporary")
    subscription.unsubscribe()
    subscription.unsubscribe()
    assert bus.stats.subscribers == 0


async def test_unsubscribing_a_later_subscriber_mid_dispatch_takes_effect_immediately(
    running_bus: tuple[EventBus, asyncio.Task[None]],
) -> None:
    """The registry is snapshotted per event, so liveness is re-checked per subscriber."""
    bus, task = running_bus
    victim = Recorder()
    holder: dict[str, Subscription] = {}

    def cancel_the_next_one(event: Event) -> None:
        holder["victim"].unsubscribe()

    assassin = Recorder(before=cancel_the_next_one)
    bus.subscribe_all(assassin, name="assassin")
    holder["victim"] = bus.subscribe_all(victim, name="victim")

    bus.publish(joined())
    await drain(bus, task)

    assert assassin.calls == 1
    assert victim.calls == 0, "unsubscribe must apply to the event already in flight"


async def test_a_handler_may_unsubscribe_itself(
    running_bus: tuple[EventBus, asyncio.Task[None]],
) -> None:
    bus, task = running_bus
    holder: dict[str, Subscription] = {}

    def suicide(event: Event) -> None:
        holder["self"].unsubscribe()

    recorder = Recorder(before=suicide)
    holder["self"] = bus.subscribe_all(recorder, name="one-shot")

    bus.publish(joined())
    bus.publish(joined())
    await drain(bus, task)

    assert recorder.calls == 1


# --------------------------------------------------------------------------- backpressure


async def test_a_full_queue_drops_an_incoming_console_log() -> None:
    bus = EventBus(maxsize=3)
    for index in range(3):
        bus.publish(console(f"line {index}"))

    bus.publish(console("overflow"))

    assert bus.stats.queued == 3
    assert bus.stats.dropped_console == 1
    assert bus.stats.dropped_critical == 0


async def test_a_full_queue_evicts_the_oldest_console_log_for_a_real_event() -> None:
    bus = EventBus(maxsize=3)
    bus.publish(console("oldest"))
    bus.publish(console("newer"))
    bus.publish(joined("Steve"))

    bus.publish(left("Steve"))

    delivered = Recorder()
    bus.subscribe_all(delivered, name="all")
    task = asyncio.create_task(bus.run())
    await drain(bus, task)

    assert delivered.names == ["ConsoleLog", "PlayerJoined", "PlayerLeft"]
    surviving = [e.message for e in delivered.seen if isinstance(e, ConsoleLog)]
    assert surviving == ["newer"], "the *oldest* console log is the one evicted"
    assert bus.stats.dropped_console == 1
    assert bus.stats.dropped_critical == 0

    await bus.aclose(drain_timeout=0.2)
    await asyncio.gather(task, return_exceptions=True)


async def test_a_full_queue_with_nothing_evictable_drops_and_counts_critical() -> None:
    """``dropped_critical`` being non-zero means the design is wrong, so it is surfaced loudly."""
    bus = EventBus(maxsize=2)
    bus.publish(joined("a"))
    bus.publish(joined("b"))

    bus.publish(joined("c"))
    bus.publish(joined("d"))

    assert bus.stats.queued == 2
    assert bus.stats.dropped_console == 0
    assert bus.stats.dropped_critical == 2


async def test_a_dropped_event_still_consumed_its_sequence_number() -> None:
    """A gap in ``seq`` is the record that something was lost; renumbering would hide it."""
    bus = EventBus(maxsize=1)
    bus.publish(joined("a"))
    bus.publish(joined("b"))  # dropped: nothing evictable
    bus.publish(console("c"))  # dropped: incoming ConsoleLog

    recorder = Recorder()
    bus.subscribe_all(recorder, name="all")
    task = asyncio.create_task(bus.run())
    await drain(bus, task)

    bus.publish(joined("d"))
    await drain(bus, task)

    assert [event.seq for event in recorder.seen] == [1, 4]

    await bus.aclose(drain_timeout=0.2)
    await asyncio.gather(task, return_exceptions=True)


# -------------------------------------------------------------------------------- shutdown


async def test_aclose_drains_what_is_queued() -> None:
    bus = EventBus()
    recorder = Recorder()
    bus.subscribe_all(recorder, name="all")
    task = asyncio.create_task(bus.run())
    await settle()

    for index in range(50):
        bus.publish(joined(f"p{index}"))
    await bus.aclose(drain_timeout=5.0)

    assert recorder.calls == 50
    assert bus.stats.queued == 0
    await asyncio.wait_for(task, timeout=1.0)
    assert task.done()
    assert not task.cancelled()


async def test_publish_after_close_is_a_counted_no_op_not_an_exception() -> None:
    """Teardown paths publish; making that an error would mean guarding every one of them."""
    bus = EventBus()
    task = asyncio.create_task(bus.run())
    await settle()
    await bus.aclose(drain_timeout=0.2)

    bus.publish(joined())
    bus.publish(console())

    assert bus.stats.suppressed_after_close == 2
    assert bus.stats.queued == 0
    assert bus.closed
    await asyncio.wait_for(task, timeout=1.0)


async def test_aclose_is_idempotent() -> None:
    bus = EventBus()
    task = asyncio.create_task(bus.run())
    await settle()
    await bus.aclose(drain_timeout=0.2)
    await bus.aclose(drain_timeout=0.2)
    await asyncio.wait_for(task, timeout=1.0)


async def test_a_never_returning_concurrent_handler_is_cancelled_at_shutdown() -> None:
    bus = EventBus(handler_timeout=30.0)
    hung = Recorder(hangs=True)
    bus.subscribe_all(hung, name="hung-relay", mode=DispatchMode.CONCURRENT)
    task = asyncio.create_task(bus.run())
    await settle()

    bus.publish(joined())
    await asyncio.wait_for(_until(lambda: hung.calls == 1), timeout=5.0)
    assert bus.stats.concurrent_tasks == 1

    await bus.aclose(drain_timeout=0.05)

    assert bus.stats.concurrent_tasks == 0
    await asyncio.wait_for(task, timeout=1.0)


async def test_a_slow_drain_is_bounded_by_the_timeout() -> None:
    bus = EventBus(handler_timeout=30.0)
    bus.subscribe_all(Recorder(hangs=True), name="hung")
    task = asyncio.create_task(bus.run())
    await settle()
    bus.publish(joined())
    bus.publish(joined())
    await settle()

    await bus.aclose(drain_timeout=0.05)

    assert bus.stats.queued >= 1, "the undelivered remainder is left in place, not silently lost"
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


# --------------------------------------------------------------------------- concurrency


async def test_concurrent_handlers_do_not_block_the_dispatch_loop() -> None:
    bus = EventBus(handler_timeout=5.0)
    release = asyncio.Event()
    slow_calls = 0

    async def slow(event: Event) -> None:
        nonlocal slow_calls
        slow_calls += 1
        await release.wait()

    fast = Recorder()
    bus.subscribe_all(slow, name="slow-io", mode=DispatchMode.CONCURRENT)
    bus.subscribe_all(fast, name="fast")
    task = asyncio.create_task(bus.run())
    await settle()

    for _ in range(3):
        bus.publish(joined())
    await drain(bus, task)

    assert fast.calls == 3, "a blocked CONCURRENT subscriber must not stall the queue"
    assert slow_calls == 3
    release.set()
    await bus.aclose(drain_timeout=1.0)
    await asyncio.wait_for(task, timeout=1.0)


async def test_run_refuses_to_start_twice() -> None:
    bus = EventBus()
    first = asyncio.create_task(bus.run())
    await settle()
    await asyncio.wait_for(bus.run(), timeout=1.0)

    await bus.aclose(drain_timeout=0.2)
    await asyncio.wait_for(first, timeout=1.0)


# ------------------------------------------------------------------------------- stats


def test_stats_is_both_a_mapping_and_a_record() -> None:
    stats = EventBus().stats
    assert isinstance(stats, BusStats)
    assert stats["queued"] == 0
    assert dict(stats)["dropped_critical"] == 0
    assert set(stats) >= {"queued", "dropped_console", "dropped_critical", "handler_errors"}
    with pytest.raises(KeyError):
        _ = stats["nonexistent"]


async def test_stats_counts_subscribers_and_dispatches(
    running_bus: tuple[EventBus, asyncio.Task[None]],
) -> None:
    bus, task = running_bus
    bus.subscribe_all(Recorder(), name="a")
    bus.subscribe(PlayerJoined, Recorder(), name="b")

    bus.publish(joined())
    bus.publish(console())
    await drain(bus, task)

    stats = bus.stats
    assert stats.subscribers == 2
    assert stats.published == 2
    assert stats.dispatched == 2
    assert stats.queued == 0


def test_a_bus_needs_room_for_at_least_one_event() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        EventBus(maxsize=0)


def test_subscription_repr_names_the_subscriber() -> None:
    bus = EventBus()
    subscription = bus.subscribe(PlayerEvent, Recorder(), name="discord-relay")
    assert "discord-relay" in repr(subscription)
    assert "PlayerEvent" in repr(subscription)


# ------------------------------------------------------------------------------- helpers


async def _until(condition: Callable[[], bool]) -> None:
    while not condition():  # noqa: ASYNC110 - polling a plain predicate, not an event
        await asyncio.sleep(0.001)
