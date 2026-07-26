"""The pure parts of the docker stream plumbing.

Nothing here touches Docker. The three things being tested are the three things that are easy to
get subtly wrong and impossible to notice in production:

1. the line splitter, because the stream is multiplexed and chunk boundaries are arbitrary;
2. the timestamp parser, because every event's ``ts`` comes from it;
3. the thread-to-loop handoff, whose defining property is that its callback **cannot raise**.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from mcmanager.containers.errors import LogStreamError
from mcmanager.containers.streams import (
    LOG_STREAM_PATH,
    LineSplitter,
    StreamHandoff,
    parse_docker_datetime,
    split_docker_timestamp,
)

# --------------------------------------------------------------------------------- splitting


def test_splitter_returns_whole_lines_from_one_chunk() -> None:
    splitter = LineSplitter()
    assert splitter.feed(b"one\ntwo\nthree\n") == ["one", "two", "three"]


def test_splitter_buffers_a_line_across_chunk_boundaries() -> None:
    """The property the multiplexed stream makes mandatory.

    docker-py hands us frame payloads, and a frame boundary has nothing to do with a line
    boundary. A splitter that treated each chunk independently would emit ``"[Server "`` and
    ``"thread/INFO]: Steve joined the game"`` as two lines and the parser would recognise neither.
    """
    splitter = LineSplitter()
    assert splitter.feed(b"[13:24:37] [Server ") == []
    assert splitter.feed(b"thread/INFO]: Steve joi") == []
    assert splitter.feed(b"ned the game\n") == [
        "[13:24:37] [Server thread/INFO]: Steve joined the game"
    ]


def test_splitter_handles_crlf_and_lf_in_the_same_stream() -> None:
    splitter = LineSplitter()
    assert splitter.feed(b"windows\r\nunix\n") == ["windows", "unix"]


def test_splitter_flush_returns_a_trailing_line_with_no_newline() -> None:
    """A container that dies mid-line still has a last line worth keeping."""
    splitter = LineSplitter()
    assert splitter.feed(b"partial") == []
    assert splitter.flush() == ["partial"]
    assert splitter.flush() == []


def test_splitter_force_emits_a_pathologically_long_line_instead_of_buffering_forever() -> None:
    """A 10MB line with no newline must not become unbounded memory in the pump thread."""
    splitter = LineSplitter(max_line_bytes=64)
    emitted = splitter.feed(b"x" * 100)
    assert len(emitted) == 1
    assert emitted[0] == "x" * 100
    assert splitter.flush() == []


def test_splitter_replaces_undecodable_bytes_rather_than_raising() -> None:
    """One corrupt byte must never take the pump down and stop the whole log stream."""
    splitter = LineSplitter()
    lines = splitter.feed(b"caf\xff\n")
    assert len(lines) == 1
    assert lines[0].startswith("caf")


def test_splitter_preserves_a_leading_tab_so_continuations_survive() -> None:
    """``\\tat net.minecraft...`` is how a stack frame is recognised. Stripping it loses that."""
    splitter = LineSplitter()
    assert splitter.feed(b"\tat net.minecraft.Foo.bar(Foo.java:1)\n") == [
        "\tat net.minecraft.Foo.bar(Foo.java:1)"
    ]


# ------------------------------------------------------------------------------- timestamps


def test_parses_rfc3339_nano_which_python_truncates_to_microseconds() -> None:
    parsed = parse_docker_datetime("2026-07-25T17:28:20.809123456Z")
    assert parsed == datetime(2026, 7, 25, 17, 28, 20, 809123, tzinfo=UTC)
    assert parsed is not None
    assert parsed.tzinfo is not None


def test_parses_an_offset_timestamp_into_utc() -> None:
    """The host is ``Asia/Kolkata``. Anything that comes back with an offset is normalised."""
    parsed = parse_docker_datetime("2026-07-25T22:58:20.809+05:30")
    assert parsed == datetime(2026, 7, 25, 17, 28, 20, 809000, tzinfo=UTC)


def test_go_zero_time_means_never_happened_not_year_one() -> None:
    """``StartedAt`` on a container that never ran. Treating it as a real date makes uptime
    calculations report two thousand years."""
    assert parse_docker_datetime("0001-01-01T00:00:00Z") is None


@pytest.mark.parametrize("value", [None, "", "not-a-date", "Up 3 hours"])
def test_unparsable_timestamps_are_none_not_exceptions(value: str | None) -> None:
    assert parse_docker_datetime(value) is None


def test_splits_the_docker_prefix_off_a_log_line() -> None:
    ts, text = split_docker_timestamp(
        "2026-07-25T17:28:20.809123456Z [13:24:37] [Server thread/INFO]: Steve joined the game"
    )
    assert ts == datetime(2026, 7, 25, 17, 28, 20, 809123, tzinfo=UTC)
    assert text == "[13:24:37] [Server thread/INFO]: Steve joined the game"


def test_a_line_with_no_prefix_keeps_its_whole_text() -> None:
    """Degrade to "no timestamp", never to a dropped or truncated line."""
    ts, text = split_docker_timestamp("no timestamp here")
    assert ts is None
    assert text == "no timestamp here"


def test_a_line_whose_first_word_is_not_a_timestamp_is_left_alone() -> None:
    ts, text = split_docker_timestamp("[13:24:37] [Server thread/INFO]: hello")
    assert ts is None
    assert text == "[13:24:37] [Server thread/INFO]: hello"


def test_the_wrapper_grammar_timestamp_also_parses() -> None:
    """``mc-server-runner`` lines still arrive with Docker's prefix; the in-line one is ignored."""
    ts, text = split_docker_timestamp(
        "2026-07-25T17:28:20.809123456Z 2026-07-25T22:58:20.809+0530\tINFO\tmc-server-runner\tDone"
    )
    assert ts == datetime(2026, 7, 25, 17, 28, 20, 809123, tzinfo=UTC)
    assert text.startswith("2026-07-25T22:58:20.809+0530\tINFO")


def test_every_parsed_timestamp_is_utc_not_merely_aware() -> None:
    parsed = parse_docker_datetime("2026-07-25T22:58:20+05:30")
    assert parsed is not None
    assert parsed.utcoffset() == timedelta(0)


# --------------------------------------------------------------------------------- handoff


async def test_handoff_delivers_items_offered_from_a_real_thread() -> None:
    """The whole point: a blocking thread hands work to the loop without either one blocking."""
    handoff: StreamHandoff[str] = StreamHandoff(maxlen=16)

    def pump() -> None:
        for index in range(5):
            handoff.offer(f"line-{index}")
        handoff.finish()

    threading.Thread(target=pump, daemon=True).start()
    received = [item async for item in handoff]
    assert received == [f"line-{index}" for index in range(5)]


async def test_handoff_drops_oldest_and_counts_instead_of_raising_when_full() -> None:
    """**Correction 4.** This is the test that names the bug it exists to prevent.

    The obvious implementation, ``loop.call_soon_threadsafe(queue.put_nowait, item)``, raises
    ``QueueFull`` inside the loop's callback runner once a slow consumer lets the queue fill.
    Nothing can handle an exception raised there, and the item is lost regardless. So the append
    is required to be total: it evicts, it counts, and it cannot raise.
    """
    handoff: StreamHandoff[int] = StreamHandoff(maxlen=3)
    for value in range(6):
        handoff.offer(value)
    handoff.finish()

    assert [item async for item in handoff] == [3, 4, 5]
    assert handoff.dropped == 3


async def test_handoff_yields_everything_buffered_before_reporting_a_fault() -> None:
    """The last few lines before a crash are the interesting ones; they must not be discarded."""
    handoff: StreamHandoff[str] = StreamHandoff()
    handoff.offer("penultimate")
    handoff.offer("last")
    handoff.finish(OSError("connection reset"))

    received: list[str] = []

    async def drain() -> None:
        async for item in handoff:
            received.append(item)

    with pytest.raises(LogStreamError, match="connection reset"):
        await drain()
    assert received == ["penultimate", "last"]


async def test_handoff_clean_finish_ends_iteration_without_raising() -> None:
    """A clean EOF means the container stopped. It is a signal, not a fault."""
    handoff: StreamHandoff[str] = StreamHandoff()
    handoff.offer("only")
    handoff.finish()
    assert [item async for item in handoff] == ["only"]
    assert handoff.closed


# ------------------------------------------------------------------------- feature detection


def test_the_private_log_stream_path_is_the_one_in_use() -> None:
    """If this ever flips to ``"fallback"``, docker-py moved its internals and ``close()`` on a
    log stream may have quietly become a no-op again. That is worth a red test, not a shrug."""
    assert LOG_STREAM_PATH in {"private", "fallback"}
    assert LOG_STREAM_PATH == "private"
