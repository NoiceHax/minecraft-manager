"""Long-lived task supervision.

Deliberately **not** ``asyncio.TaskGroup``: a TaskGroup cancels every sibling on the first
exception, which is exactly wrong here. The Discord gateway hiccupping must not take down log
streaming, and log streaming reconnecting must not take down Discord.

- A non-critical task that dies is logged, backed off (1 -> 2 -> 4 ... -> 30s, plus or minus 20%
  jitter) and restarted; its backoff resets after 60 seconds of health.
- A ``critical=True`` task that dies (the bus dispatch loop) triggers application shutdown. It is
  never restarted: the bus is a singleton with subscribers already bound to it, and a second
  dispatch loop over the same queue would deliver each event to a coin-flip of the two.
- :meth:`Supervisor.shutdown` cancels in **reverse spawn order**, with a per-task timeout, so
  producers stop before the consumers they feed.

Signal handling belongs here too, because "SIGTERM arrives" and "every task must stop" are the
same event: ``loop.add_signal_handler`` is Unix-only, so the Windows dev path falls back to
``signal.signal`` plus ``call_soon_threadsafe``. Both paths only ever *request* shutdown - they set
:attr:`Supervisor.shutdown_requested` and return - because doing real work inside a signal handler
is how you get a half-saved world.

Every wait in this module goes through the injected :class:`~mcmanager.clock.Clock`, so the backoff
schedule is asserted exactly, in about a millisecond, by advancing a
:class:`~mcmanager.clock.ManualClock`.
"""

from __future__ import annotations

import asyncio
import random
import signal
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, final

import structlog

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import FrameType

    from mcmanager.clock import Clock

__all__ = ["Supervisor", "TaskPolicy"]

type _SignalHandler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None
"""What ``signal.getsignal`` returns and ``signal.signal`` accepts, so the previous handler can be
put back exactly as it was found."""

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.supervisor")

DEFAULT_BACKOFF_INITIAL = 1.0
DEFAULT_BACKOFF_MAX = 30.0
DEFAULT_JITTER_RATIO = 0.2
DEFAULT_HEALTHY_SECONDS = 60.0
"""How long a task must run before its backoff is considered earned back."""


class TaskPolicy:
    """What to do when a supervised task returns or raises.

    ``RESTART`` for pumps and loops, ``ONCE`` for a task that legitimately finishes.

    A plain string namespace rather than an enum because it crosses no serialisation boundary and
    ``policy=TaskPolicy.ONCE`` reads the same either way.
    """

    RESTART: Final = "restart"
    ONCE: Final = "once"


@final
@dataclass(slots=True)
class _TaskRecord:
    """One supervised task and everything known about how it has behaved."""

    name: str
    factory: Callable[[], Coroutine[Any, Any, None]]
    policy: str
    backoff_initial: float
    backoff_max: float
    critical: bool
    task: asyncio.Task[None] | None = field(default=None)
    restarts: int = 0
    failures: int = 0


@final
class Supervisor:
    """Owns every long-lived task in the daemon."""

    def __init__(
        self,
        clock: Clock,
        *,
        on_critical_failure: Callable[[str, BaseException | None], None] | None = None,
        healthy_after: float = DEFAULT_HEALTHY_SECONDS,
        jitter_ratio: float = DEFAULT_JITTER_RATIO,
        rng: random.Random | None = None,
    ) -> None:
        """Create a supervisor.

        Args:
            clock: Injected time. Every backoff wait goes through ``clock.sleep``, which is what
                makes "1s, 2s, 4s, ... capped at 30s" an exact assertion rather than a slow test.
            on_critical_failure: Called with ``(task_name, error)`` when a ``critical`` task dies.
                Optional: :attr:`shutdown_requested` is always set as well, so a caller can await
                it instead of supplying a callback.
            healthy_after: Seconds a task must survive before its backoff resets to the initial
                value. Without this, a task that crashes once an hour would eventually be waiting
                30 seconds to restart from a single old failure.
            jitter_ratio: Fractional jitter applied to each backoff, plus or minus. Set to 0 for
                a deterministic schedule.
            rng: Random source, injectable so a test can pin the jitter.
        """
        self._clock = clock
        self._on_critical_failure = on_critical_failure
        self._healthy_after = healthy_after
        self._jitter_ratio = jitter_ratio
        self._rng = rng if rng is not None else random.Random()  # noqa: S311
        self._records: list[_TaskRecord] = []
        self._shutting_down = False
        self._shutdown_requested = asyncio.Event()
        self._loop_signals: list[signal.Signals] = []
        self._raw_signals: list[tuple[signal.Signals, _SignalHandler]] = []

    # -- spawning ---------------------------------------------------------------------------

    def spawn(
        self,
        name: str,
        factory: Callable[[], Coroutine[Any, Any, None]],
        *,
        policy: str = TaskPolicy.RESTART,
        backoff_initial: float = DEFAULT_BACKOFF_INITIAL,
        backoff_max: float = DEFAULT_BACKOFF_MAX,
        critical: bool = False,
    ) -> None:
        """Start ``factory()`` under supervision. ``factory`` is re-invoked on restart.

        ``factory`` is a zero-argument callable returning a fresh coroutine, not a coroutine
        object: a coroutine can only be awaited once, so a restart needs a way to make a new one.

        Args:
            name: Identifies the task in every log line, and in the shutdown order.
            factory: Builds the coroutine to run.
            policy: ``TaskPolicy.RESTART`` (default) or ``TaskPolicy.ONCE``.
            backoff_initial: First restart delay, in seconds.
            backoff_max: Ceiling for the doubling backoff.
            critical: A critical task dying requests application shutdown instead of restarting.
        """
        if self._shutting_down:
            _log.warning("supervisor.spawn_after_shutdown", task=name)
            return
        record = _TaskRecord(
            name=name,
            factory=factory,
            policy=policy,
            backoff_initial=backoff_initial,
            backoff_max=backoff_max,
            critical=critical,
        )
        record.task = asyncio.create_task(self._run(record), name=f"supervisor:{name}")
        self._records.append(record)
        _log.debug("supervisor.spawned", task=name, policy=policy, critical=critical)

    async def _run(self, record: _TaskRecord) -> None:
        """Run, watch, back off, restart. One of these per supervised task, forever."""
        delay = record.backoff_initial
        while True:
            started = self._clock.monotonic()
            error: BaseException | None = None
            try:
                await record.factory()
            except asyncio.CancelledError:
                _log.debug("supervisor.task_cancelled", task=record.name)
                raise
            except Exception as exc:
                error = exc
                record.failures += 1

            ran = self._clock.monotonic() - started
            if error is None:
                _log.info("supervisor.task_returned", task=record.name, ran_seconds=ran)
            else:
                _log.error(
                    "supervisor.task_crashed",
                    task=record.name,
                    ran_seconds=ran,
                    error=str(error),
                    exc_info=error,
                )

            if record.critical:
                self._request_shutdown(record.name, error)
                return
            if self._shutting_down or record.policy != TaskPolicy.RESTART:
                return

            if ran >= self._healthy_after:
                delay = record.backoff_initial
            wait = self._with_jitter(min(delay, record.backoff_max))
            record.restarts += 1
            _log.warning(
                "supervisor.restarting",
                task=record.name,
                delay_seconds=wait,
                restarts=record.restarts,
            )
            await self._clock.sleep(wait)
            delay = min(delay * 2.0, record.backoff_max)

    def _with_jitter(self, delay: float) -> float:
        """Apply plus-or-minus ``jitter_ratio`` so restarts do not synchronise into a thundering
        herd after a Docker daemon restart takes every pump down at the same instant."""
        if self._jitter_ratio <= 0.0:
            return delay
        factor = 1.0 + self._rng.uniform(-self._jitter_ratio, self._jitter_ratio)
        return max(delay * factor, 0.0)

    # -- shutdown ---------------------------------------------------------------------------

    @property
    def shutdown_requested(self) -> asyncio.Event:
        """Set when a critical task dies or a signal arrives. ``app.py`` awaits this."""
        return self._shutdown_requested

    def request_shutdown(self, reason: str) -> None:
        """Ask the application to shut down. Idempotent, and safe from a signal handler."""
        self._request_shutdown(reason, None)

    def _request_shutdown(self, reason: str, error: BaseException | None) -> None:
        already = self._shutdown_requested.is_set()
        self._shutdown_requested.set()
        if already:
            return
        _log.critical(
            "supervisor.shutdown_requested",
            reason=reason,
            error=None if error is None else str(error),
        )
        callback = self._on_critical_failure
        if callback is None:
            return
        try:
            callback(reason, error)
        except Exception:
            # A failing shutdown callback must not stop the shutdown it was told about.
            _log.exception("supervisor.shutdown_callback_failed", reason=reason)

    async def shutdown(self, *, task_timeout: float = 5.0) -> None:
        """Cancel every task in reverse spawn order and wait for it, bounded by ``task_timeout``.

        Reverse order because tasks are spawned dependencies-first: the log pump is spawned after
        the bus it publishes into, so cancelling in reverse stops producers before consumers and
        no event is generated with nothing left to receive it.

        Idempotent, and never raises: a task that refuses to die is logged and left behind rather
        than allowed to block the rest of the shutdown sequence.
        """
        if self._shutting_down:
            return
        self._shutting_down = True
        self.remove_signal_handlers()
        for record in reversed(self._records):
            task = record.task
            if task is None or task.done():
                continue
            task.cancel()
            _, pending = await asyncio.wait({task}, timeout=task_timeout)
            if pending:
                _log.error(
                    "supervisor.task_would_not_stop",
                    task=record.name,
                    task_timeout=task_timeout,
                    hint="task swallowed CancelledError; it is abandoned, not awaited",
                )
            else:
                _log.debug("supervisor.task_stopped", task=record.name)

    # -- signals ----------------------------------------------------------------------------

    def install_signal_handlers(self, *signals: signal.Signals) -> None:
        """Route SIGINT/SIGTERM to :attr:`shutdown_requested`.

        ``loop.add_signal_handler`` is the correct mechanism and is Unix-only; on Windows it
        raises ``NotImplementedError`` and the fallback is ``signal.signal`` plus
        ``call_soon_threadsafe``, because a C-level signal handler runs between bytecodes on the
        main thread and must not touch loop state directly.
        """
        wanted = signals if signals else (signal.SIGINT, signal.SIGTERM)
        loop = asyncio.get_running_loop()
        for sig in wanted:
            try:
                loop.add_signal_handler(sig, self.request_shutdown, f"signal:{sig.name}")
            except (NotImplementedError, RuntimeError, ValueError):
                previous = signal.getsignal(sig)
                signal.signal(sig, self._make_thread_signal_handler(loop, sig))
                self._raw_signals.append((sig, previous))
                _log.debug("supervisor.signal_handler_installed", signal=sig.name, via="signal")
            else:
                self._loop_signals.append(sig)
                _log.debug("supervisor.signal_handler_installed", signal=sig.name, via="loop")

    def _make_thread_signal_handler(
        self,
        loop: asyncio.AbstractEventLoop,
        sig: signal.Signals,
    ) -> Callable[[int, FrameType | None], None]:
        def _handler(signum: int, frame: FrameType | None) -> None:  # noqa: ARG001
            # Runs between bytecodes on the main thread, so it does nothing but hand the fact
            # back to the loop. Anything else here risks re-entering half-updated state.
            loop.call_soon_threadsafe(self.request_shutdown, f"signal:{sig.name}")

        return _handler

    def remove_signal_handlers(self) -> None:
        """Undo :meth:`install_signal_handlers`. Idempotent, and never raises."""
        if self._loop_signals:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                for sig in self._loop_signals:
                    try:
                        loop.remove_signal_handler(sig)
                    except (NotImplementedError, RuntimeError, ValueError):
                        _log.debug("supervisor.signal_handler_removal_failed", signal=sig.name)
            self._loop_signals.clear()
        for sig, previous in self._raw_signals:
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError, TypeError):
                _log.debug("supervisor.signal_handler_restore_failed", signal=sig.name)
        self._raw_signals.clear()

    # -- introspection ----------------------------------------------------------------------

    @property
    def task_names(self) -> tuple[str, ...]:
        """Supervised task names, in spawn order. Shutdown walks this backwards."""
        return tuple(record.name for record in self._records)

    @property
    def restart_counts(self) -> Mapping[str, int]:
        """How many times each task has been restarted. Surfaced by ``/readyz``."""
        return {record.name: record.restarts for record in self._records}

    @property
    def failure_counts(self) -> Mapping[str, int]:
        """How many times each task has died with an exception rather than returning."""
        return {record.name: record.failures for record in self._records}
