"""Blocking docker streams, pumped into the event loop.

The second and last module allowed to ``import docker``. Holds:

- the ``CancellableStream`` wrapper around the log stream (correction 1 in ``docker_runtime.py``);
- the two ``threading.Thread(daemon=True)`` pumps (correction 2);
- the non-raising bounded-deque handoff callback (correction 4);
- the chunk-to-line splitter, which must buffer across chunk boundaries because the stream is
  multiplexed and split points are arbitrary;
- parsing of Docker's RFC3339Nano timestamp prefix into tz-aware UTC.

Nothing here knows about events, services or Minecraft. It converts "a blocking iterator owned by
a thread" into "an async iterator owned by the loop", and that is all.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Protocol, final

from docker.api.client import APIClient
from docker.errors import DockerException
from docker.types.daemon import CancellableStream

from mcmanager.containers.errors import LogStreamError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence

    import docker

__all__ = [
    "LOG_STREAM_PATH",
    "ClosableStream",
    "LineSplitter",
    "StreamHandoff",
    "StreamPump",
    "close_stream_quietly",
    "open_event_stream",
    "open_log_stream",
    "parse_docker_datetime",
    "split_docker_timestamp",
]

log: Final = logging.getLogger(__name__)

DEFAULT_MAX_LINE_BYTES: Final = 1 << 20
"""A single line longer than this is force-emitted rather than buffered forever.

Paper can emit pathological lines (a stack trace with a 10MB serialised NBT payload in it has been
seen in the wild). Buffering without a ceiling turns that into an OOM in the pump thread, which is
the one place where an exception is hardest to see.
"""

DEFAULT_HANDOFF_CAPACITY: Final = 4096
"""Lines buffered between the pump thread and the loop before the oldest are dropped.

Dropping here is correct and is counted: the alternative - the raising ``put_nowait`` that hcp
uses - throws ``QueueFull`` inside ``call_soon_threadsafe``, which lands in the loop exception
handler where nothing can meaningfully handle it.
"""


# --------------------------------------------------------------------------------- timestamps


def parse_docker_datetime(value: str | None) -> datetime | None:
    """Parse one of Docker's RFC3339Nano strings into tz-aware UTC.

    Returns ``None`` for absent, unparsable, or the Go zero time ``0001-01-01T00:00:00Z`` that
    Docker uses for "this never happened" in ``StartedAt`` / ``FinishedAt``.

    Python 3.12's :meth:`datetime.datetime.fromisoformat` accepts the trailing ``Z`` and truncates
    the nanosecond fraction to microseconds, which is exactly what we want; the alternative is a
    hand-rolled regex that gets the ``+05:30`` offset case wrong.
    """
    if not value or value.startswith("0001-01-01"):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def split_docker_timestamp(text: str) -> tuple[datetime | None, str]:
    """Split ``"2026-07-25T17:28:20.809123456Z <line>"`` into its timestamp and the rest.

    ``logs(timestamps=True)`` is used unconditionally, so every line arrives with this prefix. The
    server's own ``[13:24:37]`` is time-only in ``Asia/Kolkata``, the wrapper's is a third format,
    and neither survives backfill - see the module docstring of ``dto.py``.

    Returns ``(None, text)`` unchanged when the prefix is missing or unparsable, so a malformed
    line degrades to "no timestamp" rather than being dropped.
    """
    head, sep, rest = text.partition(" ")
    if not sep:
        return None, text
    parsed = parse_docker_datetime(head)
    if parsed is None:
        return None, text
    return parsed, rest


# ------------------------------------------------------------------------------ line splitting


@final
class LineSplitter:
    """Turns arbitrarily chunked bytes into whole lines.

    The stream is multiplexed (``Config.Tty`` is false on this container), so docker-py hands us
    frame payloads whose boundaries have nothing to do with line boundaries: one frame can hold
    three lines, and one line can span four frames. Buffering across chunks is therefore not an
    optimisation, it is the only correct implementation.

    Handles both ``\\n`` and ``\\r\\n``. Decodes as UTF-8 with ``errors="replace"`` because a
    corrupt byte in one log line must never take the pump down.
    """

    __slots__ = ("_buffer", "_max_line_bytes")

    def __init__(self, *, max_line_bytes: int = DEFAULT_MAX_LINE_BYTES) -> None:
        self._buffer = bytearray()
        self._max_line_bytes = max_line_bytes

    def feed(self, chunk: bytes) -> list[str]:
        """Append ``chunk`` and return every complete line it finished."""
        self._buffer.extend(chunk)
        lines: list[str] = []
        while True:
            index = self._buffer.find(b"\n")
            if index < 0:
                break
            lines.append(self._take(index + 1, keep=index))
        if len(self._buffer) > self._max_line_bytes:
            # No newline in sight and the buffer is enormous. Emit what we have so the pipeline
            # sees a (truncated) line instead of the process growing without bound.
            lines.append(self._take(len(self._buffer), keep=len(self._buffer)))
        return lines

    def flush(self) -> list[str]:
        """Return whatever is buffered as a final line. Called once, at EOF."""
        if not self._buffer:
            return []
        text = self._take(len(self._buffer), keep=len(self._buffer))
        return [text] if text else []

    def _take(self, consume: int, *, keep: int) -> str:
        raw = bytes(self._buffer[:keep])
        del self._buffer[:consume]
        return raw.rstrip(b"\r").decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------- thread -> loop


@final
class StreamHandoff[T]:
    """The thread-to-loop boundary. **The appending callback cannot raise.**

    Correction 4 from the plan, and the one that is easiest to get wrong twice. The obvious
    implementation - ``loop.call_soon_threadsafe(queue.put_nowait, item)`` - raises
    :class:`asyncio.QueueFull` *inside the loop's callback runner* as soon as a slow consumer lets
    the queue fill. That exception surfaces in ``loop.call_exception_handler``, which is not a
    place where anything can be done about it, and the item is lost anyway.

    So: a ``deque`` with a ``maxlen``. Appending to a full one silently evicts the oldest, which
    is the policy we actually want for console output, and the eviction is counted in
    :attr:`dropped` so the loss is visible instead of silent.

    Must be constructed on the event loop thread; :meth:`offer` and :meth:`finish` are the only
    methods a pump thread may call.
    """

    __slots__ = ("_closed", "_dropped", "_error", "_items", "_loop", "_maxlen", "_wake")

    def __init__(self, *, maxlen: int = DEFAULT_HANDOFF_CAPACITY) -> None:
        self._items: deque[T] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self._wake = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        self._closed = False
        self._error: BaseException | None = None
        self._dropped = 0

    # -- pump-thread side --------------------------------------------------------------------

    def offer(self, item: T) -> None:
        """Hand ``item`` to the loop. Called from the pump thread; never raises."""
        try:
            self._loop.call_soon_threadsafe(self._append, item)
        except RuntimeError:
            # The loop closed underneath us during shutdown. Dropping is the only option, and it
            # is not an error worth propagating into a daemon thread nobody is watching.
            self._dropped += 1

    def finish(self, error: BaseException | None = None) -> None:
        """Signal end of stream. Called from the pump thread; never raises."""
        try:
            self._loop.call_soon_threadsafe(self._close, error)
        except RuntimeError:
            self._closed = True

    # -- loop side ---------------------------------------------------------------------------

    def _append(self, item: T) -> None:
        """Runs on the loop. Every statement here is incapable of raising, by construction."""
        if len(self._items) >= self._maxlen:
            self._dropped += 1
        self._items.append(item)
        self._wake.set()

    def _close(self, error: BaseException | None) -> None:
        """Runs on the loop. Also incapable of raising."""
        self._closed = True
        if error is not None and self._error is None:
            self._error = error
        self._wake.set()

    async def __aiter__(self) -> AsyncIterator[T]:
        """Yield items until the pump signals end of stream.

        A clean end returns; a faulted end raises :class:`LogStreamError`, which is what the
        reconnect loop backs off from. Everything buffered before the fault is yielded first: the
        last few lines before a crash are the interesting ones.
        """
        while True:
            while self._items:
                yield self._items.popleft()
            if self._closed:
                if self._error is not None:
                    raise LogStreamError(str(self._error) or type(self._error).__name__)
                return
            self._wake.clear()
            if self._items or self._closed:
                continue
            await self._wake.wait()

    @property
    def dropped(self) -> int:
        """How many items were evicted because the consumer could not keep up."""
        return self._dropped

    @property
    def closed(self) -> bool:
        """True once the pump has signalled end of stream."""
        return self._closed


class ClosableStream(Protocol):
    """A blocking iterator that can be interrupted from another thread.

    :class:`docker.types.daemon.CancellableStream` satisfies this. A bare generator does not,
    which is the entire subject of correction 1.
    """

    def __iter__(self) -> Iterator[Any]: ...

    def close(self) -> None: ...


@final
class _UncloseableStream:
    """Fallback wrapper for a bare generator, used only if the private log path disappears.

    ``close()`` here is honest about being best-effort: generators have a ``close()``, but calling
    it from another thread while the owning thread is parked in a socket read does nothing. The
    pump thread is a daemon thread precisely so that this case cannot hang shutdown.
    """

    __slots__ = ("_source",)

    def __init__(self, source: Iterator[Any]) -> None:
        self._source = source

    def __iter__(self) -> Iterator[Any]:
        return self._source

    def close(self) -> None:
        closer = getattr(self._source, "close", None)
        if callable(closer):
            closer()


# ------------------------------------------------------------------------------ opening streams

_PRIVATE_LOG_PATH_ATTRS: Final = (
    "_url",
    "_get",
    "_raise_for_status",
    "_multiplexed_response_stream_helper",
    "_stream_raw_result",
)

_PRIVATE_LOG_PATH_AVAILABLE: Final = all(
    hasattr(APIClient, attr) for attr in _PRIVATE_LOG_PATH_ATTRS
)

LOG_STREAM_PATH: Final = "private" if _PRIVATE_LOG_PATH_AVAILABLE else "fallback"
"""Which log-stream implementation this docker-py build gave us.

``"private"`` means we opened the response ourselves and wrapped it in a real
:class:`~docker.types.daemon.CancellableStream`, so ``close()`` shuts the socket down and the pump
thread's blocking read returns. ``"fallback"`` means the private helpers were not found on this
docker-py and we are iterating whatever ``container.logs(stream=True)`` returned, whose ``close()``
may be a no-op; the pump is a daemon thread so a stuck read still cannot hang shutdown.

Reported by ``mcmanager inspect`` so the state of this feature detection is never a mystery.
"""


def _tail_param(tail: int) -> int | str:
    """Docker wants ``all`` or a non-negative integer; anything else means "everything"."""
    if tail < 0:
        return "all"
    return tail


def open_log_stream(
    client: docker.DockerClient,
    container_id: str,
    *,
    tty: bool,
    follow: bool = True,
    tail: int = 0,
    since: datetime | None = None,
) -> ClosableStream:
    """Open the container's log stream as something ``close()`` actually cancels.

    **Correction 1.** The plan's finding is that ``container.logs(stream=True)`` cannot be
    cancelled, and the fix is to build the stream the way docker-py's own ``events()`` does:
    ``api._url`` + ``api._get(stream=True, timeout=None)`` + ``api._raise_for_status`` +
    ``api._multiplexed_response_stream_helper``, wrapped in
    :class:`~docker.types.daemon.CancellableStream`. That is what this does, feature-detected at
    import with a fallback to the public generator.

    Two further reasons this path is worth its privateness even on a docker-py where the public
    call happens to wrap the stream too: ``timeout=None`` is set explicitly, so a long-idle server
    (nobody plays for six hours) cannot trip a read timeout; and the public path calls
    ``_check_is_tty``, an extra full inspect per attach, whereas ``tty`` is already on the snapshot
    we resolved the container id from.

    ``since`` is second-granularity at the Docker API and replays the whole of that second. The
    caller must de-duplicate - see ``manager.py``'s dedupe ring.
    """
    params: dict[str, Any] = {
        "stdout": 1,
        "stderr": 1,
        "timestamps": 1,
        "follow": 1 if follow else 0,
        "tail": _tail_param(tail),
    }
    if since is not None:
        params["since"] = int(since.timestamp())

    # The `Any` is the boundary, stated in one place rather than as a scatter of `# pyright:
    # ignore` comments. pyright resolves docker-py through typeshed's third-party stubs, which
    # (correctly) do not describe private members, so the private path below is invisible to it -
    # and the public `logs()` overloads there disagree with the runtime about `tail`. Everything
    # returned from this function is typed again.
    api: Any = client.api
    if not _PRIVATE_LOG_PATH_AVAILABLE:  # pragma: no cover - depends on the installed docker-py
        container: Any = client.containers.get(container_id)
        return _UncloseableStream(
            iter(
                container.logs(
                    stream=True,
                    follow=follow,
                    timestamps=True,
                    tail=_tail_param(tail),
                    since=since,
                )
            )
        )

    url = api._url("/containers/{0}/logs", container_id)
    response = api._get(url, params=params, stream=True, timeout=None)
    api._raise_for_status(response)
    if tty:
        payloads = api._stream_raw_result(response)
    else:
        payloads = api._multiplexed_response_stream_helper(response)
    return CancellableStream(payloads, response)


def open_event_stream(
    client: docker.DockerClient,
    *,
    container: str | None = None,
    since: datetime | None = None,
) -> ClosableStream:
    """Open the daemon's event stream, filtered to container events.

    No private API needed: ``client.events()`` already returns a real ``CancellableStream``. That
    asymmetry between ``events()`` and ``logs()`` is exactly the trap correction 1 is about.
    """
    filters: dict[str, Any] = {"type": "container"}
    if container is not None:
        filters["container"] = container
    api: Any = client
    return api.events(decode=True, filters=filters, since=since)


def close_stream_quietly(stream: ClosableStream | None) -> None:
    """Close ``stream``, swallowing the failures that closing is allowed to have.

    ``CancellableStream.close()`` raises ``DockerException("Cancellable streams not supported for
    the SSH protocol")`` over ``ssh://`` - the Windows dev path - because there is no socket to
    shut down, only a paramiko channel. That is a downgrade-to-debug, not an error: the pump is a
    daemon thread, so the worst case is one parked thread that the interpreter abandons at exit.
    """
    if stream is None:
        return
    try:
        stream.close()
    except DockerException as exc:
        log.debug("stream close not supported by this transport: %s", exc)
    except OSError as exc:
        log.debug("stream close failed: %s", exc)
    except Exception:
        log.debug("unexpected failure closing stream", exc_info=True)


# -------------------------------------------------------------------------------------- pumps


@final
class StreamPump[S, T]:
    """A daemon thread draining a blocking docker stream into a :class:`StreamHandoff`.

    **Correction 2.** Not ``asyncio.to_thread``. ``asyncio.run()`` finishes by awaiting
    ``loop.shutdown_default_executor()``, which *joins* the default executor's threads. A thread
    parked in a blocking docker read never returns from that join, so the daemon hangs on shutdown
    forever - and it hangs after everything else has already torn down, which makes it look like a
    deadlock in whatever ran last. ``threading.Thread(daemon=True)`` is not joined by anything, so
    the worst case degrades from "hangs forever" to "one thread abandoned at exit".

    ``asyncio.to_thread`` remains right for short request/response calls, and that is what
    ``docker_runtime.py`` uses it for.
    """

    __slots__ = ("_handoff", "_name", "_on_eof", "_stopping", "_stream", "_thread", "_transform")

    def __init__(
        self,
        *,
        name: str,
        stream: ClosableStream,
        handoff: StreamHandoff[T],
        transform: Callable[[S], Sequence[T]],
        on_eof: Callable[[], Sequence[T]] | None = None,
    ) -> None:
        self._name = name
        self._stream = stream
        self._handoff = handoff
        self._transform = transform
        self._on_eof = on_eof
        self._stopping = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the pump thread. Idempotent."""
        if self._thread is not None:
            return
        thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread = thread
        thread.start()

    def stop(self) -> None:
        """Cancel the underlying stream so the thread's blocking read returns. Never raises."""
        self._stopping = True
        close_stream_quietly(self._stream)

    def _run(self) -> None:
        error: BaseException | None = None
        try:
            for chunk in self._stream:
                for item in self._transform(chunk):
                    self._handoff.offer(item)
        except Exception as exc:
            # A stream we deliberately closed reports itself as broken. That is not a fault, and
            # calling it one would make every clean shutdown look like an outage and trip the
            # reconnect backoff.
            if not self._stopping:
                error = exc
                log.debug("pump %s failed: %r", self._name, exc)
        finally:
            if self._on_eof is not None:
                try:
                    for item in self._on_eof():
                        self._handoff.offer(item)
                except Exception:  # pragma: no cover - flush is a pure buffer read
                    log.debug("pump %s flush failed", self._name, exc_info=True)
            self._handoff.finish(error)
