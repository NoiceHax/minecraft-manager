"""Idle auto-shutdown. **Stub: real interface and wiring now, decision body in M4.**

Replaces ``minecraft-idle-stop.sh``, which polled ``docker exec ... rcon-cli list``, kept its state
in ``/tmp/minecraft_idle_since``, and - verifiably - **was not scheduled at all**: empty crontab,
no systemd timer, nothing running. Idle shutdown has been dead. "Matches the old script" is
therefore a worthless acceptance bar; this is built to spec.

Rules that are not negotiable:

- **Disabled by default**, and ``dry_run`` on top of that for the cutover soak.
- **Nothing arms unless the server is actually up.** An empty roster is not evidence of an idle
  server when nothing is running - it is evidence that nothing is running. Without this the
  safety-net poll re-arms after every ``ServerStopped``, publishes ``IdleStarted``,
  ``IdleWarning`` and ``IdleStopTriggered`` against a container that is already down, and repeats
  for as long as the daemon lives. It also breaks the plan's phase-4 gate ("every 'would stop'
  matches a genuinely empty server") outright, and would have M4 issue a real stop against a
  stopped container every timeout period, since neither ``min_uptime`` nor probe freshness
  excludes that.
- **``min_uptime_minutes``**: never idle-stop a server that just booted. "Start, nobody joined in
  15 minutes, instant shutdown" reads as broken to everyone watching.
- **A failed probe is UNKNOWN, not empty.** That failure mode is the one that loses somebody's
  build session.
- **On daemon shutdown: cancel the timer, publish ``IdleCancelled(reason="daemon_shutdown")``, and
  explicitly do not trigger a stop.** An idle timer firing during teardown and killing the server
  because the manager restarted would be the single most dangerous bug in this design.
- The deadline is persisted but **restarted from now** on boot, so a crash-looping manager cannot
  repeatedly insta-stop the server.

**What is real here and what is not.** The countdown itself - arm, warn, cancel, and above all
:meth:`IdleManager.aclose` - is fully implemented, because those are exactly the paths that make the
shutdown ordering in ``app.py`` provable today rather than in M4. The one deliberately unwritten
body is :meth:`IdleManager._fire`: when the deadline expires it publishes
:class:`~mcmanager.core.events.IdleStopTriggered`, logs, and **never calls the controller**. Filling
that in - together with the ``min_uptime`` and probe-freshness gates it must consult first - is M4.
Until then this module is structurally incapable of stopping anything, which is the correct state
for a subsystem whose worst failure mode is stopping a populated server.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Protocol, final

import structlog

from mcmanager.core.events import (
    Event,
    IdleCancelled,
    IdleStarted,
    IdleStopTriggered,
    IdleWarning,
    PlayerEvent,
    PlayerJoined,
    PlayerLeft,
    ServerCrashed,
    ServerEvent,
    ServerReady,
    ServerStopped,
    ServerStopping,
)
from mcmanager.core.types import LifecycleState, Source

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from mcmanager.clock import Clock, TimerHandle
    from mcmanager.core.bus import EventBus, Subscription
    from mcmanager.core.types import ServerId
    from mcmanager.persistence.state_store import StateStore
    from mcmanager.services.controller import ServerController
    from mcmanager.services.players import PlayerRoster

__all__ = [
    "ARMABLE_STATES",
    "DAEMON_SHUTDOWN",
    "SERVER_NOT_RUNNING",
    "EventSink",
    "IdleManager",
]

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.idle")

DAEMON_SHUTDOWN = "daemon_shutdown"
"""The one :class:`~mcmanager.core.events.IdleCancelled` reason that means "we are going away, the
server is not". Named rather than spelled inline, because ``app.py`` and its test both assert on it.
"""

SERVER_NOT_RUNNING = "server_not_running"
"""``IdleCancelled`` reason for a countdown disarmed because the server is not up any more."""

ARMABLE_STATES: frozenset[LifecycleState] = frozenset(
    {LifecycleState.READY, LifecycleState.DEGRADED}
)
"""The only states in which an empty roster means "nobody is playing".

``READY`` is the obvious one. ``DEGRADED`` is included because the container is running and the
server may well be answering - a run that overran its start deadline, or one whose healthcheck is
flapping, is still a server somebody could be building in, and it is still one worth stopping when
it is genuinely empty.

Everything else is excluded for a reason: ``STARTING`` is what ``min_uptime_minutes`` exists for
and arming there would count the boot against the idle timeout; ``STOPPING`` is already going
away; ``BLIND`` means we cannot see, and unknown is never empty - the same rule that governs a
failed probe; ``STOPPED``, ``CRASHED``, ``ABSENT`` and ``UNKNOWN`` have nothing to stop.
"""


class EventSink(Protocol):
    """The narrow slice of :class:`~mcmanager.core.bus.EventBus` this module needs.

    Same shape as the sinks in ``containers/manager.py`` and ``services/log_pipeline.py``: one
    synchronous ``publish``. Narrow on purpose, so the idle tests need no dispatch loop.
    """

    def publish(self, event: Event) -> None: ...


@final
class IdleManager:
    """Watches the roster, arms a countdown when it empties, and disarms it when it does not.

    Subscribes to :class:`~mcmanager.core.events.PlayerEvent` and
    :class:`~mcmanager.core.events.ServerEvent` in ``SEQUENTIAL`` mode, because the causal chain
    ``PlayerJoined -> IdleCancelled -> PlayerLeft -> IdleStarted`` is only correct when it is
    observed in order, and neither handler touches the network.
    """

    def __init__(
        self,
        *,
        sink: EventSink,
        clock: Clock,
        server_id: ServerId,
        roster: PlayerRoster,
        controller: ServerController,
        store: StateStore | None = None,
        enabled: bool = False,
        dry_run: bool = True,
        timeout_seconds: float = 900.0,
        warn_seconds: float = 120.0,
        poll_interval_seconds: float = 60.0,
        min_uptime_seconds: float = 1200.0,
        treat_probe_failure_as_empty: bool = False,
    ) -> None:
        """Wire the manager.

        Args:
            sink: Where the four idle events go. ``EventBus`` satisfies this structurally.
            clock: Injected time. Both timers are ``clock.call_later`` handles, which is what makes
                a fifteen-minute countdown a one-millisecond test.
            server_id: Stamped onto every event.
            roster: The authoritative online set. Emptiness is read from here and never from a
                probe result directly: the poller has already reconciled probes into the roster,
                and has already refused to treat an unreachable server as empty.
            controller: How a stop would be issued in M4. Held, and deliberately never called.
            store: Where the deadline is persisted for observability. Optional, so the unit tests
                need no filesystem.
            enabled: ``idle.enabled``. False by default; nothing arms while it is false.
            dry_run: ``idle.dry_run``. Stamped onto every event, so the soak is auditable.
            timeout_seconds: ``idle.timeout_minutes``, in seconds.
            warn_seconds: ``idle.warn_minutes``, in seconds. Zero disables the warning.
            poll_interval_seconds: How often :meth:`run` re-examines the roster. The timers do the
                real work; this loop is the safety net for a lost player event.
            min_uptime_seconds: ``idle.min_uptime_minutes``, in seconds. **An M4 gate**, recorded
                here so the configured value is visible in the logs from day one.
            treat_probe_failure_as_empty: ``idle.treat_probe_failure_as_empty``. **An M4 gate**,
                and documented as "leave this false".
        """
        self._sink = sink
        self._clock = clock
        self._server_id = server_id
        self._roster = roster
        self._controller = controller
        self._store = store

        self._enabled = enabled
        self._dry_run = dry_run
        self._timeout = timeout_seconds
        self._warn = warn_seconds
        self._poll_interval = poll_interval_seconds
        self._min_uptime = min_uptime_seconds
        self._probe_failure_is_empty = treat_probe_failure_as_empty

        self._deadline: datetime | None = None
        self._empty_since: datetime | None = None
        self._armed_monotonic: float | None = None
        self._timer: TimerHandle | None = None
        self._warn_timer: TimerHandle | None = None
        self._subscriptions: tuple[Subscription, ...] = ()
        self._closing = False

        self._stats: dict[str, int] = {
            "armed": 0,
            "cancelled": 0,
            "warned": 0,
            "triggered": 0,
            "stops_issued": 0,
            "polls": 0,
        }

    # -------------------------------------------------------------------------- introspection

    @property
    def enabled(self) -> bool:
        """``idle.enabled``. Nothing arms while this is false."""
        return self._enabled

    @property
    def dry_run(self) -> bool:
        """True while the countdown is computed, published, and never acted on."""
        return self._dry_run

    @property
    def armed(self) -> bool:
        """True while a countdown is running."""
        return self._timer is not None

    @property
    def deadline(self) -> datetime | None:
        """When the stop would happen, tz-aware UTC, or ``None`` while disarmed."""
        return self._deadline

    @property
    def empty_since(self) -> datetime | None:
        """When the roster last emptied."""
        return self._empty_since

    @property
    def stats(self) -> Mapping[str, int]:
        """Counters, surfaced by ``/readyz`` and ``mcmanager status``.

        ``stops_issued`` is the one to watch during the soak: it must be zero for the whole dry-run
        phase, and this module currently has no code path that can increment it.
        """
        return dict(self._stats)

    def describe(self) -> dict[str, object]:
        """A JSON-safe summary, for ``mcmanager status`` and ``/status``."""
        return {
            "enabled": self._enabled,
            "dry_run": self._dry_run,
            "armed": self.armed,
            "deadline": None if self._deadline is None else self._deadline.isoformat(),
            "empty_since": None if self._empty_since is None else self._empty_since.isoformat(),
            "timeout_seconds": self._timeout,
            "min_uptime_seconds": self._min_uptime,
            "online": self._roster.count,
        }

    # --------------------------------------------------------------------------------- wiring

    def subscribe(self, bus: EventBus) -> tuple[Subscription, ...]:
        """Register the two subscriptions this manager needs, and return them for the tests."""
        self._subscriptions = (
            bus.subscribe(PlayerEvent, self.on_player_event, name="idle.players"),
            bus.subscribe(ServerEvent, self.on_server_event, name="idle.server"),
        )
        return self._subscriptions

    async def start(self) -> None:
        """Adopt the persisted state, then arm if the server is already empty.

        The persisted deadline is read, logged, and **discarded**: the countdown always restarts
        from now. Honouring a stored deadline would mean a crash-looping manager stops the server
        on every boot, which is the failure this project exists not to have.
        """
        stored = None if self._store is None else self._store.state.idle_deadline
        if stored is not None:
            _log.info(
                "idle.persisted_deadline_ignored",
                deadline=stored.isoformat(),
                hint="the countdown always restarts from now; see the module docstring",
            )
        if self._store is not None:
            self._store.set_idle_deadline(None)
        _log.info(
            "idle.started",
            enabled=self._enabled,
            dry_run=self._dry_run,
            timeout_seconds=self._timeout,
            warn_seconds=self._warn,
            min_uptime_seconds=self._min_uptime,
            online=self._roster.count,
        )

    async def run(self) -> None:
        """The safety-net loop, spawned as a non-critical supervised task.

        The two ``call_later`` timers do the real work; this only re-examines the roster in case a
        player event was lost, and it is where M4's probe-freshness gate will live. It returns once
        :meth:`aclose` has been called.
        """
        while not self._closing:
            await self._clock.sleep(self._poll_interval)
            if self._closing:
                return
            self._stats["polls"] += 1
            self._reconsider("poll")

    async def aclose(self) -> None:
        """**The dangerous path, and the reason this stub exists at all.**

        Cancels both timers and publishes ``IdleCancelled(reason="daemon_shutdown")``. It does not,
        under any circumstance, stop the server: an idle timer firing during teardown and killing
        the Minecraft server because the *manager* was restarting would be the single most
        dangerous bug in this design. ``_closing`` is set first, so a timer that fires in the same
        tick finds the door already shut.

        Safe to call twice, and never raises.
        """
        if self._closing:
            return
        self._closing = True
        was_armed = self.armed
        # Read the elapsed time before the timers go, because cancelling clears the mark.
        idle_seconds = self._idle_seconds()
        self._cancel_timers()
        if was_armed:
            self._publish_cancelled(DAEMON_SHUTDOWN, idle_seconds=idle_seconds)
        else:
            _log.debug("idle.shutdown_not_armed")
        self._deadline = None
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions = ()

    # ------------------------------------------------------------------------------- handlers

    async def on_player_event(self, event: PlayerEvent) -> None:
        """Bus handler. A join disarms; a leave that empties the roster arms."""
        if self._closing:
            return
        if isinstance(event, PlayerJoined):
            self._cancel("player_joined")
            return
        if isinstance(event, PlayerLeft):
            self._reconsider("player_left", empty_since=event.ts)

    async def on_server_event(self, event: ServerEvent) -> None:
        """Bus handler. A stop or a crash disarms; a ready with an empty roster arms.

        ``ServerCrashed`` is in the disarming set for the same reason ``ServerStopped`` is: the
        server is gone, and a countdown left running against it would fire at nothing.
        """
        if self._closing:
            return
        if isinstance(event, ServerStopping | ServerStopped | ServerCrashed):
            self._cancel("server_stopped")
            return
        if isinstance(event, ServerReady):
            self._reconsider("server_ready", empty_since=event.ts)

    # ------------------------------------------------------------------------------ countdown

    def _reconsider(self, reason: str, *, empty_since: datetime | None = None) -> None:
        """Arm or disarm so the countdown matches reality. The one place that is decided.

        Two questions, in this order, because they answer different things. *Is there a server?*
        comes first: an empty roster against a stopped container is not an idle server, and
        arming on it is how the safety-net poll ends up publishing a stop trigger every fifteen
        minutes at something that has been off since Tuesday. Only then, *is anybody on it?*
        """
        if not self._server_is_up():
            self._cancel(SERVER_NOT_RUNNING)
            return
        if self._roster.count > 0:
            self._cancel("player_online")
            return
        self._arm(reason, empty_since=empty_since)

    def _server_is_up(self) -> bool:
        """Is the server in a state where "nobody is online" means "idle"?

        See :data:`ARMABLE_STATES`.
        """
        return self._controller.state in ARMABLE_STATES

    def _arm(self, reason: str, *, empty_since: datetime | None = None) -> None:
        if not self._enabled:
            _log.debug("idle.disabled", reason=reason, online=self._roster.count)
            return
        if self.armed:
            return

        now = self._clock.now()
        self._empty_since = empty_since if empty_since is not None else now
        self._armed_monotonic = self._clock.monotonic()
        self._deadline = now + timedelta(seconds=self._timeout)
        self._timer = self._clock.call_later(self._timeout, self._fire)
        if 0.0 < self._warn < self._timeout:
            self._warn_timer = self._clock.call_later(self._timeout - self._warn, self._warned)
        self._stats["armed"] += 1

        if self._store is not None:
            self._store.set_idle_deadline(self._deadline, empty_since=self._empty_since)

        _log.info(
            "idle.armed",
            reason=reason,
            deadline=self._deadline.isoformat(),
            timeout_seconds=self._timeout,
            dry_run=self._dry_run,
        )
        self._sink.publish(
            IdleStarted(
                ts=now,
                server_id=self._server_id,
                source=Source.INTERNAL,
                raw=f"idle countdown armed ({reason})",
                deadline=self._deadline,
                empty_since=self._empty_since,
                timeout_seconds=self._timeout,
                dry_run=self._dry_run,
            )
        )

    def _cancel(self, reason: str) -> None:
        if not self.armed:
            return
        idle_seconds = self._idle_seconds()
        self._cancel_timers()
        self._publish_cancelled(reason, idle_seconds=idle_seconds)
        self._deadline = None
        self._empty_since = None
        if self._store is not None:
            self._store.set_idle_deadline(None)

    def _publish_cancelled(self, reason: str, *, idle_seconds: float | None) -> None:
        self._stats["cancelled"] += 1
        _log.info("idle.cancelled", reason=reason, idle_seconds=idle_seconds)
        self._sink.publish(
            IdleCancelled(
                ts=self._clock.now(),
                server_id=self._server_id,
                source=Source.INTERNAL,
                raw=f"idle countdown cancelled ({reason})",
                reason=reason,
                idle_seconds=idle_seconds,
            )
        )

    def _cancel_timers(self) -> None:
        for handle in (self._timer, self._warn_timer):
            if handle is not None:
                handle.cancel()
        self._timer = None
        self._warn_timer = None
        self._armed_monotonic = None

    def _warned(self) -> None:
        """The warning timer fired. Announced so a player about to reconnect has a chance."""
        self._warn_timer = None
        if self._closing or self._deadline is None:
            return
        self._stats["warned"] += 1
        _log.info("idle.warning", remaining_seconds=self._warn, deadline=self._deadline.isoformat())
        self._sink.publish(
            IdleWarning(
                ts=self._clock.now(),
                server_id=self._server_id,
                source=Source.TIMER,
                raw=f"idle stop in {self._warn:g}s",
                remaining_seconds=self._warn,
                deadline=self._deadline,
                dry_run=self._dry_run,
            )
        )

    def _fire(self) -> None:
        """The deadline expired. **This is the M4 body, and it is deliberately inert.**

        What M4 adds, in this order, on top of the running-server gate already applied below: the
        ``min_uptime_seconds`` gate; a check that the last probe was fresh and reachable (an
        unreachable server is UNKNOWN, never empty, unless ``treat_probe_failure_as_empty`` is
        set, which it must not be); and only then, when ``dry_run`` is false,
        ``await self._controller.stop(actor="idle-manager", ...)``.

        Today it publishes the event and stops nothing at all. That is the right behaviour for a
        subsystem whose worst failure is stopping a populated server, and it means the shutdown
        ordering in ``app.py`` is provable before the dangerous code exists.
        """
        self._timer = None
        if self._closing:
            _log.warning("idle.fire_during_shutdown_ignored")
            return
        if not self._server_is_up():
            # Belt and braces to `_reconsider`'s gate: the deadline can only have been armed while
            # the server was up, but fifteen minutes is long enough for it to have gone away since,
            # and M4 replaces this method's body with a real `controller.stop(...)`.
            _log.info(
                "idle.fire_ignored_server_not_running",
                state=self._controller.state.value,
                online=self._roster.count,
            )
            self._cancel_timers()
            self._deadline = None
            self._empty_since = None
            if self._store is not None:
                self._store.set_idle_deadline(None)
            return

        idle_seconds = self._idle_seconds()
        self._stats["triggered"] += 1
        _log.warning(
            "idle.stop_not_implemented",
            idle_seconds=idle_seconds,
            dry_run=self._dry_run,
            online=self._roster.count,
            controller_state=self._controller.state.value,
            min_uptime_seconds=self._min_uptime,
            treat_probe_failure_as_empty=self._probe_failure_is_empty,
            hint="M4 fills this in; no stop is issued, and none can be",
        )
        self._sink.publish(
            IdleStopTriggered(
                ts=self._clock.now(),
                server_id=self._server_id,
                source=Source.TIMER,
                raw="idle deadline reached",
                idle_seconds=idle_seconds if idle_seconds is not None else 0.0,
                dry_run=True,
            )
        )
        self._deadline = None
        self._empty_since = None
        if self._store is not None:
            self._store.set_idle_deadline(None)

    def _idle_seconds(self) -> float | None:
        if self._armed_monotonic is None:
            return None
        return max(self._clock.monotonic() - self._armed_monotonic, 0.0)
