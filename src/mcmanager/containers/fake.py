"""A scriptable in-memory ``ContainerRuntime``.

Every service test runs against this, and ``runtime = "fake"`` boots the entire daemon offline on
a machine with no Docker at all.

**Fake this interface, never the ``docker`` library.** Mocking ``docker.DockerClient`` encodes
your assumptions about docker-py into the test suite, so a wrong assumption produces a *passing*
test. Faking our own ABC means exactly one file can be wrong about docker-py, and that file is
covered by the live tests.

Three behaviours here are deliberate reproductions of Docker's, not conveniences:

- ``follow_logs(since=...)`` floors ``since`` to the second and replays the whole of that second,
  because that is what the real API does and it is why ``manager.py`` needs a dedupe ring. A fake
  that replayed exactly from the microsecond would let the dedupe bug ship.
- ``follow_logs`` on a container that is **not running** yields the history and ends immediately.
  Blocking there instead would hide the reconnect loop treating that instant EOF as a reason to
  reattach a second later, forever, while the server is simply off.
- ``State.Health.Status`` is not cleared when the container stops. ``set_state`` leaves
  :attr:`health_raw` alone unless told otherwise, so a test that stops a healthy container sees
  the stale ``"unhealthy"`` string and proves ``ContainerSnapshot.health`` covers for it.

**Not the default.** A homelab daemon silently running against a fake in production is worse than
a startup failure, so ``runtime`` has no default value that reaches this class by accident.

Defaults mirror the verified homelab container - ``interval=30s, start_period=120s, retries=2``,
so the 210-second health guard window is exercised without every test restating it - and
``Config.Tty`` is false, matching the real one.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final, final

from mcmanager.containers.base import ContainerRuntime
from mcmanager.containers.dto import (
    ContainerSnapshot,
    ContainerState,
    ExecResult,
    HealthTiming,
    LogLine,
    MountInfo,
    NetworkAttachment,
    RuntimeEvent,
)
from mcmanager.containers.errors import (
    ContainerNotFoundError,
    LogStreamError,
    RuntimeUnavailableError,
)
from mcmanager.core.types import Stream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence

    from mcmanager.clock import Clock, TimerHandle

__all__ = ["FakeRuntime"]

DEFAULT_HEALTH_TIMING: Final = HealthTiming(
    interval=timedelta(seconds=30),
    timeout=timedelta(seconds=5),
    start_period=timedelta(seconds=120),
    retries=2,
)
"""The homelab container's real healthcheck config: a 210-second guard window."""

_EPOCH_DEFAULT: Final = datetime(2025, 12, 25, 0, 0, 0, tzinfo=UTC)


@final
class _FakeStream[T]:
    """One attached consumer. Buffers what it was pushed and wakes its reader.

    Per-consumer buffers rather than one shared queue, so that two followers (the daemon and a
    ``mcmanager logs --follow``) each see every line, and closing one does not close the other.
    """

    __slots__ = ("buffer", "closed", "error", "wake")

    def __init__(self) -> None:
        self.buffer: deque[T] = deque()
        self.wake = asyncio.Event()
        self.closed = False
        self.error: BaseException | None = None

    def push(self, item: T) -> None:
        self.buffer.append(item)
        self.wake.set()

    def finish(self, error: BaseException | None = None) -> None:
        self.closed = True
        if error is not None:
            self.error = error
        self.wake.set()

    async def __aiter__(self) -> AsyncIterator[T]:
        while True:
            while self.buffer:
                yield self.buffer.popleft()
            if self.error is not None:
                error = self.error
                self.error = None
                self.closed = True
                raise error
            if self.closed:
                return
            self.wake.clear()
            if self.buffer or self.closed or self.error is not None:
                continue
            await self.wake.wait()


@final
class FakeRuntime(ContainerRuntime):
    """An in-memory container runtime a test can drive line by line and event by event."""

    def __init__(
        self,
        *,
        clock: Clock,
        name: str = "minecraft",
        container_id: str = "fake00000000000000000000000000000000000000000000000000000000",
        exists: bool = True,
        auto_transition: bool = True,
    ) -> None:
        """Create a fake.

        Args:
            clock: The injected clock. Scheduled lines fire on it, so a
                :class:`~mcmanager.clock.ManualClock` gives a test exact control over log timing.
            name: The container name this fake answers to. Any other name inspects as absent.
            container_id: Initial id. :meth:`recreate` changes it, which is how a
                ``compose down/up`` is simulated.
            exists: Whether the container exists at all.
            auto_transition: When true, :meth:`start` and :meth:`stop` also move the snapshot and
                emit the matching Docker events, so a lifecycle test does not have to narrate what
                Docker would obviously have done. Turn it off to script the transitions by hand.
        """
        self._clock = clock
        self._name = name
        self._auto_transition = auto_transition

        self._exists = exists
        self._container_id = container_id
        self._state = ContainerState.EXITED
        self._running = False
        self._health_raw: str | None = None
        self._health_failing_streak: int | None = None
        self._health_timing = DEFAULT_HEALTH_TIMING
        self._tty = False
        self._exit_code: int | None = 0
        self._oom_killed = False
        self._restart_count = 0
        self._image = "itzg/minecraft-server:latest"
        self._created_at: datetime | None = _EPOCH_DEFAULT
        self._started_at: datetime | None = None
        self._finished_at: datetime | None = None
        self._labels: dict[str, str] = {"com.docker.compose.project": "minecraft"}
        self._networks: tuple[NetworkAttachment, ...] = (
            NetworkAttachment(name="homelab", ip_address="172.20.0.9", aliases=(name,)),
        )
        self._mounts: tuple[MountInfo, ...] = (
            MountInfo(
                source="/home/minty/homelab/data/minecraft",
                destination="/data",
                mode="rw",
                rw=True,
                kind="bind",
            ),
        )
        self._log_driver = "json-file"
        self._log_options: dict[str, str] = {"max-size": "10m", "max-file": "3"}
        self._restart_policy = "no"
        self._stop_signal = "SIGTERM"

        self._reachable = True
        self._history: list[LogLine] = []
        self._event_history: list[RuntimeEvent] = []
        self._log_streams: set[_FakeStream[LogLine]] = set()
        self._event_streams: set[_FakeStream[RuntimeEvent]] = set()
        self._scheduled: list[TimerHandle] = []
        self._failures: dict[str, deque[BaseException]] = {}
        self._exec_results: deque[ExecResult] = deque()
        self._exec_handler: Callable[[tuple[str, ...]], ExecResult] | None = None
        self._closed = False

        self.default_exec_result = ExecResult(exit_code=0, stdout="")
        """Returned by :meth:`exec` when nothing more specific was queued."""

        self.calls: list[str] = []
        """Every runtime operation, in order. ``["inspect", "start", "follow_logs", ...]``."""

        self.start_calls: list[str] = []
        self.stop_calls: list[tuple[str, int]] = []
        self.exec_calls: list[tuple[str, tuple[str, ...]]] = []
        self.follow_calls: list[tuple[datetime | None, int]] = []
        """``(since, tail)`` for every ``follow_logs`` attach. The reconnect tests read this."""

        self.watch_calls: list[tuple[str | None, datetime | None]] = []

    # ------------------------------------------------------------------ scripting: container

    @property
    def endpoint(self) -> str:
        """Matches ``DockerRuntime.endpoint`` so error messages read the same either way."""
        return "fake://in-memory"

    @property
    def container_id(self) -> str:
        return self._container_id

    def set_reachable(self, reachable: bool) -> None:
        """Simulate the Docker daemon going away and coming back.

        While unreachable, every operation except :meth:`ping` raises
        :class:`~mcmanager.containers.errors.RuntimeUnavailableError` and :meth:`ping` returns
        False. Attached streams are faulted, which is what drives the reconnect loop.
        """
        self._reachable = reachable
        if not reachable:
            self._fault_streams(RuntimeUnavailableError("fake runtime is unreachable"))

    def set_state(
        self,
        state: ContainerState,
        *,
        running: bool | None = None,
        exit_code: int | None = None,
        health_raw: str | None = None,
        failing_streak: int | None = None,
        oom_killed: bool | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        restart_count: int | None = None,
    ) -> None:
        """Move the container to ``state``.

        ``running`` defaults to whatever ``state`` implies. ``health_raw`` is left untouched when
        not passed, on purpose: Docker does not clear it on stop, and neither does this.
        """
        self._state = state
        self._exists = state is not ContainerState.ABSENT
        self._running = running if running is not None else state is ContainerState.RUNNING
        if exit_code is not None:
            self._exit_code = exit_code
        if health_raw is not None:
            self._health_raw = health_raw
        if failing_streak is not None:
            self._health_failing_streak = failing_streak
        if oom_killed is not None:
            self._oom_killed = oom_killed
        if restart_count is not None:
            self._restart_count = restart_count
        if self._running:
            self._started_at = started_at if started_at is not None else self._clock.now()
            self._finished_at = None
        else:
            self._finished_at = finished_at if finished_at is not None else self._clock.now()

    def set_health(self, raw: str | None, *, failing_streak: int | None = None) -> None:
        """Set the raw ``State.Health.Status`` string, stale values included."""
        self._health_raw = raw
        self._health_failing_streak = failing_streak

    def set_health_timing(self, timing: HealthTiming) -> None:
        """Override the healthcheck config, e.g. to test a container that declares none."""
        self._health_timing = timing

    def set_absent(self) -> None:
        """There is no such container. A legitimate state, not an error."""
        self._exists = False
        self._running = False
        self._state = ContainerState.ABSENT

    def recreate(self, container_id: str) -> None:
        """Simulate ``compose down && compose up``: same name, brand new id.

        This is the scenario a cached container id fails silently: the old id keeps resolving to
        nothing while the daemon waits forever for logs from a container that no longer exists.
        """
        self._container_id = container_id
        self._exists = True
        self._history.clear()
        self._fault_streams(LogStreamError("container was recreated"))

    # ---------------------------------------------------------------------- scripting: logs

    def script_lines(
        self,
        lines: Iterable[tuple[float, str] | str],
        *,
        stream: Stream = Stream.STDOUT,
    ) -> None:
        """Schedule log lines at second offsets from now.

        ``script_lines([(0.0, "..."), (32.5, 'Done (32.521s)! For help, type "help"')])`` with a
        :class:`~mcmanager.clock.ManualClock` means ``await clock.advance(33)`` produces exactly
        the timing a real startup has, in about a millisecond.

        A bare string is shorthand for offset ``0.0``. Lines are emitted by the clock, so they are
        recorded in history whether or not anything is following - a follower that attaches later
        still gets them from the backfill.
        """
        for entry in lines:
            offset, text = (0.0, entry) if isinstance(entry, str) else entry
            if offset <= 0:
                self.emit_line(text, stream=stream)
                continue
            handle = self._clock.call_later(
                offset,
                lambda text=text, stream=stream: self._emit_scheduled(text, stream),
            )
            self._scheduled.append(handle)

    def emit_line(
        self,
        text: str,
        *,
        stream: Stream = Stream.STDOUT,
        ts: datetime | None = None,
    ) -> LogLine:
        """Emit one line right now: record it in history and push it to every follower."""
        now = self._clock.now()
        line = LogLine(
            text=text,
            ts=ts if ts is not None else now,
            received_at=now,
            stream=stream,
        )
        self._history.append(line)
        for consumer in self._log_streams:
            consumer.push(line)
        return line

    def emit_lines(self, texts: Iterable[str], *, stream: Stream = Stream.STDOUT) -> None:
        """Emit several lines at the current instant. All share one timestamp, which is what
        makes them a useful test of the second-granularity replay."""
        for text in texts:
            self.emit_line(text, stream=stream)

    def close_log_stream(self) -> None:
        """End every attached log stream cleanly.

        A clean EOF means the container stopped. It corroborates the ``die`` event and must not be
        treated as a fault or backed off from, which is what the reconnect tests assert.
        """
        for consumer in list(self._log_streams):
            consumer.finish()

    def break_log_stream(self, error: BaseException | None = None) -> None:
        """Fault every attached log stream. This *is* an error and does trip the backoff."""
        fault = error if error is not None else LogStreamError("fake stream broke")
        for consumer in list(self._log_streams):
            consumer.finish(fault)

    # --------------------------------------------------------------------- scripting: events

    def emit_event(self, action: str, **attributes: str) -> RuntimeEvent:
        """Emit a Docker event, e.g. ``emit_event("die", exitCode="1")``.

        Every attribute value is a string, exactly as Docker sends them. That is not pedantry:
        ``attrs["exitCode"] == 0`` is silently always false, which is why
        :attr:`~mcmanager.containers.dto.RuntimeEvent.exit_code` exists.
        """
        event = RuntimeEvent(
            action=action,
            container_id=self._container_id,
            container_name=self._name,
            ts=self._clock.now(),
            attributes={"name": self._name, **attributes},
        )
        self._event_history.append(event)
        for consumer in self._event_streams:
            consumer.push(event)
        return event

    def close_event_stream(self) -> None:
        """End every attached event stream cleanly."""
        for consumer in list(self._event_streams):
            consumer.finish()

    def break_event_stream(self, error: BaseException | None = None) -> None:
        """Fault every attached event stream."""
        fault = error if error is not None else LogStreamError("fake event stream broke")
        for consumer in list(self._event_streams):
            consumer.finish(fault)

    # -------------------------------------------------------------------- scripting: failures

    def fail_next(self, operation: str, error: BaseException) -> None:
        """Make the next call to ``operation`` raise ``error``.

        ``operation`` is a method name: ``"inspect"``, ``"start"``, ``"stop"``, ``"logs_tail"``,
        ``"follow_logs"``, ``"watch_events"``, ``"exec"``, ``"ping"``. Calls queue, so
        ``fail_next`` three times makes three attempts fail and the fourth succeed - which is how
        the backoff sequence is tested.

        For the two streaming methods the failure surfaces on the first iteration, because the
        call itself is defined not to block.
        """
        self._failures.setdefault(operation, deque()).append(error)

    def queue_exec_result(self, result: ExecResult) -> None:
        """Queue one :meth:`exec` result. Consumed in order, then :attr:`default_exec_result`."""
        self._exec_results.append(result)

    def set_exec_handler(self, handler: Callable[[tuple[str, ...]], ExecResult] | None) -> None:
        """Answer :meth:`exec` from a function of the argv. Beats queueing for ``rcon-cli list``."""
        self._exec_handler = handler

    # ------------------------------------------------------------------------- assertions

    def assert_stopped_with_timeout(self, timeout: int) -> None:
        """Assert some ``stop()`` was issued with exactly ``timeout`` seconds of grace.

        90 seconds is how long a Minecraft world save can take, and it used to be a bare ``-t 90``
        in a bash script with no explanation. This is the test that keeps it from drifting.
        """
        if not self.stop_calls:
            msg = f"expected stop(timeout={timeout}), but stop() was never called"
            raise AssertionError(msg)
        seen = [value for _, value in self.stop_calls]
        if timeout not in seen:
            msg = f"expected stop(timeout={timeout}), saw timeouts {seen}"
            raise AssertionError(msg)

    def assert_never_stopped(self) -> None:
        """Assert nothing ever stopped the server. The idle dry-run soak asserts this."""
        if self.stop_calls:
            msg = f"expected no stop(), but saw {self.stop_calls}"
            raise AssertionError(msg)

    @property
    def history(self) -> Sequence[LogLine]:
        """Every line ever emitted, in order."""
        return tuple(self._history)

    @property
    def event_history(self) -> Sequence[RuntimeEvent]:
        return tuple(self._event_history)

    @property
    def attached_log_streams(self) -> int:
        """How many followers are attached. Zero after a clean teardown, or the manager leaks."""
        return len(self._log_streams)

    @property
    def attached_event_streams(self) -> int:
        return len(self._event_streams)

    # ------------------------------------------------------------------------ ContainerRuntime

    async def ping(self) -> bool:
        self.calls.append("ping")
        if self._pop_failure("ping") is not None:
            return False
        return self._reachable and not self._closed

    async def inspect(self, name: str) -> ContainerSnapshot:
        self.calls.append("inspect")
        self._raise_queued("inspect")
        self._require_reachable("inspect")
        observed_at = self._clock.now()
        if name != self._name or not self._exists:
            return ContainerSnapshot.missing(name, observed_at=observed_at)
        return self._snapshot(observed_at)

    async def start(self, name: str) -> None:
        self.calls.append("start")
        self.start_calls.append(name)
        self._raise_queued("start")
        self._require_reachable("start")
        self._require_exists(name)
        if not self._auto_transition or self._running:
            return
        self.set_state(ContainerState.RUNNING, exit_code=None, health_raw="starting")
        self.emit_event("start")

    async def stop(self, name: str, *, timeout: int) -> None:  # noqa: ASYNC109
        self.calls.append("stop")
        self.stop_calls.append((name, timeout))
        self._raise_queued("stop")
        self._require_reachable("stop")
        self._require_exists(name)
        if not self._auto_transition or not self._running:
            return
        # Verified on the real container: mc-server-runner traps SIGTERM, writes "stop" to the
        # console and exits 0. A graceful stop is exit code 0 here, not 143.
        self.set_state(ContainerState.EXITED, exit_code=0)
        self.emit_event("die", exitCode="0")
        self.emit_event("stop")
        self.close_log_stream()

    async def logs_tail(
        self,
        name: str,
        *,
        lines: int = 100,
        since: datetime | None = None,
    ) -> list[LogLine]:
        self.calls.append("logs_tail")
        self._raise_queued("logs_tail")
        self._require_reachable("logs_tail")
        self._require_exists(name)
        return self._backfill(since=since, tail=lines)

    def follow_logs(
        self,
        name: str,
        *,
        since: datetime | None = None,
        tail: int = 0,
    ) -> AsyncIterator[LogLine]:
        return self._iter_logs(name, since=since, tail=tail)

    async def _iter_logs(
        self,
        name: str,
        *,
        since: datetime | None,
        tail: int,
    ) -> AsyncIterator[LogLine]:
        self.calls.append("follow_logs")
        self.follow_calls.append((since, tail))
        self._raise_queued("follow_logs")
        self._require_reachable("follow_logs")
        self._require_exists(name)

        history = self._backfill(since=since, tail=tail)
        if not self._running:
            # Docker's third behaviour worth reproducing: a follow on a container that is not
            # running delivers the history and ends *immediately* rather than waiting for output
            # that can never come. A fake that blocked here would hide the reconnect loop spinning
            # on that instant EOF, which is exactly the bug this reproduces.
            for line in history:
                yield line
            return
        consumer: _FakeStream[LogLine] = _FakeStream()
        for line in history:
            consumer.push(line)
        self._log_streams.add(consumer)
        try:
            async for line in consumer:
                yield line
        finally:
            self._log_streams.discard(consumer)

    def watch_events(
        self,
        name: str | None = None,
        *,
        since: datetime | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        return self._iter_events(name, since=since)

    async def _iter_events(
        self,
        name: str | None,
        *,
        since: datetime | None,
    ) -> AsyncIterator[RuntimeEvent]:
        self.calls.append("watch_events")
        self.watch_calls.append((name, since))
        self._raise_queued("watch_events")
        self._require_reachable("watch_events")

        consumer: _FakeStream[RuntimeEvent] = _FakeStream()
        if since is not None:
            floor = since.replace(microsecond=0)
            for event in self._event_history:
                if event.ts >= floor:
                    consumer.push(event)
        self._event_streams.add(consumer)
        try:
            async for event in consumer:
                yield event
        finally:
            self._event_streams.discard(consumer)

    async def exec(
        self,
        name: str,
        cmd: Sequence[str],
        *,
        timeout: float | None = None,  # noqa: ASYNC109, ARG002
    ) -> ExecResult:
        argv = tuple(cmd)
        self.calls.append("exec")
        self.exec_calls.append((name, argv))
        self._raise_queued("exec")
        self._require_reachable("exec")
        self._require_exists(name)
        if self._exec_handler is not None:
            return self._exec_handler(argv)
        if self._exec_results:
            return self._exec_results.popleft()
        return self.default_exec_result

    async def aclose(self) -> None:
        """Detach everything. Safe to call twice, never raises."""
        self._closed = True
        for consumer in list(self._log_streams):
            consumer.finish()
        for consumer in list(self._event_streams):
            consumer.finish()
        self._log_streams.clear()
        self._event_streams.clear()
        for handle in self._scheduled:
            handle.cancel()
        self._scheduled.clear()

    # -------------------------------------------------------------------------------- internals

    def _emit_scheduled(self, text: str, stream: Stream) -> None:
        if self._closed:
            return
        self.emit_line(text, stream=stream)

    def _snapshot(self, observed_at: datetime) -> ContainerSnapshot:
        return ContainerSnapshot(
            name=self._name,
            id=self._container_id,
            exists=True,
            state=self._state,
            status_text=self._state.value,
            running=self._running,
            _health_raw=self._health_raw,
            health_failing_streak=self._health_failing_streak,
            health_timing=self._health_timing,
            tty=self._tty,
            exit_code=self._exit_code,
            oom_killed=self._oom_killed,
            restart_count=self._restart_count,
            image=self._image,
            created_at=self._created_at,
            started_at=self._started_at,
            finished_at=self._finished_at,
            labels=dict(self._labels),
            networks=self._networks,
            mounts=self._mounts,
            log_driver=self._log_driver,
            log_options=dict(self._log_options),
            restart_policy=self._restart_policy,
            stop_signal=self._stop_signal,
            observed_at=observed_at,
        )

    def _backfill(self, *, since: datetime | None, tail: int) -> list[LogLine]:
        """History selection with Docker's own semantics.

        ``since`` is floored to the second and is inclusive, so every line from that second comes
        back again - the replay the dedupe ring exists for. ``tail`` is then applied to what is
        left, and ``tail=0`` means "no history at all", which is why the reconnect path asks for
        ``tail=-1`` whenever it passes a ``since``.
        """
        selected = list(self._history)
        if since is not None:
            floor = since.replace(microsecond=0)
            selected = [line for line in selected if line.ts is None or line.ts >= floor]
        if tail >= 0:
            selected = selected[-tail:] if tail else []
        return selected

    def _fault_streams(self, error: BaseException) -> None:
        for consumer in list(self._log_streams):
            consumer.finish(error)
        for consumer in list(self._event_streams):
            consumer.finish(error)

    def _pop_failure(self, operation: str) -> BaseException | None:
        queued = self._failures.get(operation)
        if not queued:
            return None
        return queued.popleft()

    def _raise_queued(self, operation: str) -> None:
        error = self._pop_failure(operation)
        if error is not None:
            raise error

    def _require_reachable(self, operation: str) -> None:
        if self._reachable and not self._closed:
            return
        msg = f"{operation}: fake runtime is unreachable"
        raise RuntimeUnavailableError(msg, endpoint=self.endpoint)

    def _require_exists(self, name: str) -> None:
        if name != self._name or not self._exists:
            raise ContainerNotFoundError(name)

    def describe(self) -> Mapping[str, object]:
        """A dict of the current scripted state. For assertion failure messages."""
        return {
            "name": self._name,
            "id": self._container_id,
            "state": self._state.value,
            "running": self._running,
            "health_raw": self._health_raw,
            "reachable": self._reachable,
            "lines": len(self._history),
            "events": len(self._event_history),
        }
