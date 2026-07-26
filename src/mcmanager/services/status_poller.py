"""Periodic out-of-band status probe.

Runs :meth:`~mcmanager.games.base.GameAdapter.probe` on a clock-injected interval (45s), hands the
result to the lifecycle machine and reconciles the roster against it.

Two rules it must not break, both of which are one bad line away from stopping a server people are
playing on:

- **A failed probe is UNKNOWN, not zero players.** It never touches the roster, never publishes a
  leave, and never claims the server is empty. ``ProbeResult.is_empty`` is
  ``reachable and players_online == 0`` for exactly this reason, and this module reads that
  property rather than testing ``not players_online``.
- **Reconciliation is gated on** :attr:`~mcmanager.games.base.ProbeResult.sample_is_complete`. The
  player sample is protocol-capped at 12 entries and can be randomised or omitted by Paper settings
  and anti-scrape plugins. Reconciling against a truncated sample produces a storm of phantom
  ``PlayerLeft`` events the moment the server is busier than the cap. The gate lives in
  :meth:`~mcmanager.services.players.PlayerRoster.reconcile`; this module honours its verdict and
  counts the skips.

**Why lifecycle arrives as a callback rather than an import.** The poller hands every result to an
injected ``on_probe`` callable. Lifecycle wraps it in its own ``ProbeSignal`` and reduces it; the
poller neither knows nor cares. That is what keeps this module testable with no state machine, and
what keeps ``lifecycle.py`` free of any knowledge of ``mcstatus``.

**The events this module publishes are probe-derived** and carry
:attr:`~mcmanager.core.types.Source.PROBE`, so a consumer can tell "we watched them join" from "we
noticed they were there". They exist because a join that happened while the log stream was detached
- a Docker outage, a daemon restart - is otherwise invisible forever, and a roster that is wrong in
that direction keeps the idle timer disarmed indefinitely.

Two SLP queries now exist against this server: ``mc-health`` already runs one every 30 seconds from
inside the container, and this poller adds one every 45. Harmless, but that is the explanation if
SLP lag ever appears.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, final

from mcmanager.core.events import PlayerJoined, PlayerLeft
from mcmanager.core.types import LeaveReason, PlayerRef, Source
from mcmanager.games.base import ProbeResult
from mcmanager.logging_setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from datetime import datetime

    from mcmanager.clock import Clock
    from mcmanager.core.types import ServerId
    from mcmanager.games.base import GameAdapter
    from mcmanager.services.log_pipeline import EventSink
    from mcmanager.services.players import PlayerRoster, RosterDelta

__all__ = ["DEFAULT_INTERVAL", "DEFAULT_TIMEOUT", "StatusPoller"]

_log: Final = get_logger("mcmanager.status_poller")

DEFAULT_INTERVAL: Final = 45.0
"""Seconds between probes. Matches ``probe.interval_seconds``."""

DEFAULT_TIMEOUT: Final = 5.0
"""Protocol-level deadline handed to the query library, not a client-side ``asyncio.timeout``."""


@final
class StatusPoller:
    """Polls the game's status endpoint and feeds lifecycle and the roster.

    Spawn :meth:`run` as a non-critical supervised task. It never raises and never exits on its
    own; the supervisor cancels it during shutdown.
    """

    __slots__ = (
        "_adapter",
        "_clock",
        "_closing",
        "_consecutive_failures",
        "_enabled",
        "_host",
        "_interval",
        "_last_result",
        "_last_success_at",
        "_on_probe",
        "_port",
        "_reachable",
        "_roster",
        "_server_id",
        "_sink",
        "_stats",
        "_timeout",
        "_trust_partial_sample",
    )

    def __init__(
        self,
        *,
        adapter: GameAdapter,
        host: str,
        port: int,
        clock: Clock,
        sink: EventSink,
        roster: PlayerRoster,
        server_id: ServerId,
        interval: float = DEFAULT_INTERVAL,
        timeout: float = DEFAULT_TIMEOUT,
        trust_partial_sample: bool = False,
        on_probe: Callable[[ProbeResult], None] | None = None,
        enabled: bool = True,
    ) -> None:
        """Wire the poller.

        Args:
            adapter: The game adapter whose ``probe`` performs the query.
            host: Hostname to probe. ``"minecraft"`` in production, resolved over the shared
                ``homelab`` bridge network. **Never ``localhost``**: inside our own container that
                is our loopback, where nothing listens, and a probe that always fails combined with
                idle shutdown is how a populated server gets stopped every fifteen minutes.
            port: SLP port.
            clock: Injected time. Every interval and every duration is measured on it.
            sink: Where probe-derived player events go.
            roster: The authoritative online set to reconcile against.
            server_id: Stamped onto published events.
            interval: Seconds between probes.
            timeout: Per-probe deadline handed to the query library.
            trust_partial_sample: Override the sample-completeness gate. Defaults false and is
                documented as "leave this false".
            on_probe: Called with every result, success or failure, before reconciliation. This is
                the lifecycle machine's hook; it turns the result into a ``ProbeSignal``. A raising
                callback is caught and counted, because a broken observer must not stop the poller.
            enabled: When false, :meth:`run` returns immediately. ``probe.enabled = false`` is a
                legitimate configuration on a deployment with no reachable status port.
        """
        self._adapter = adapter
        self._host = host
        self._port = port
        self._clock = clock
        self._sink = sink
        self._roster = roster
        self._server_id = server_id
        self._interval = interval
        self._timeout = timeout
        self._trust_partial_sample = trust_partial_sample
        self._on_probe = on_probe
        self._enabled = enabled

        self._last_result: ProbeResult | None = None
        self._last_success_at: datetime | None = None
        self._consecutive_failures = 0
        self._reachable = False
        self._closing = False
        self._stats: dict[str, int] = {
            "probes": 0,
            "successes": 0,
            "failures": 0,
            "reconciled": 0,
            "skipped_incomplete": 0,
            "skipped_unreachable": 0,
            "probe_joins": 0,
            "probe_leaves": 0,
            "observer_errors": 0,
        }

    # -------------------------------------------------------------------------- introspection

    @property
    def last_result(self) -> ProbeResult | None:
        """The most recent probe, successful or not. ``None`` before the first one."""
        return self._last_result

    @property
    def last_success_at(self) -> datetime | None:
        """When the server last answered. What ``mcmanager status`` shows as the probe age."""
        return self._last_success_at

    @property
    def consecutive_failures(self) -> int:
        """Failed probes since the last success. Reset by any success."""
        return self._consecutive_failures

    @property
    def reachable(self) -> bool:
        """Did the last probe succeed?

        Explicitly **not** "is the server empty": a False here means we do not know how many
        players are online, and every consumer must treat it that way.
        """
        return self._reachable

    @property
    def players_online(self) -> int | None:
        """The count the server last reported, or ``None`` when the last probe failed.

        ``None`` rather than ``0``. The type is the guard: a caller that wants a number has to
        decide what to do about not having one.
        """
        result = self._last_result
        if result is None or not result.reachable:
            return None
        return result.players_online

    @property
    def stats(self) -> Mapping[str, int]:
        """Counters, polled rather than pushed.

        ``skipped_incomplete`` climbing while ``reconciled`` stays flat means the sample is being
        truncated or randomised - the condition the completeness gate exists for, made visible
        rather than silent.
        """
        return dict(self._stats)

    # -------------------------------------------------------------------------------- the loop

    async def run(self) -> None:
        """Probe forever, on the interval. Spawned as a supervised task.

        Probes **first**, then sleeps, so the daemon has a reading within milliseconds of boot
        rather than 45 seconds later - which matters because the first thing anybody asks a freshly
        started daemon is whether the server is up.
        """
        if not self._enabled:
            _log.info("status_poller.disabled", host=self._host, port=self._port)
            return
        _log.info(
            "status_poller.started",
            host=self._host,
            port=self._port,
            interval_seconds=self._interval,
        )
        while not self._closing:
            await self.poll_once()
            if self._closing:
                return
            await self._clock.sleep(self._interval)

    async def aclose(self) -> None:
        """Ask the loop to stop at its next opportunity. Idempotent, never raises.

        Deliberately does not cancel anything: the supervisor cancels tasks in reverse spawn order
        with per-task timeouts, and a poller that cancelled itself would race that.
        """
        self._closing = True

    # ------------------------------------------------------------------------------ one probe

    async def poll_once(self) -> ProbeResult:
        """Run one probe, notify the observer, and reconcile the roster. Never raises."""
        result = await self._probe()
        self._stats["probes"] += 1
        self._last_result = result
        self._record(result)
        self._notify(result)
        self._reconcile(result)
        return result

    async def _probe(self) -> ProbeResult:
        """Call the adapter. The adapter is contractually total; this is belt and braces.

        ``GameAdapter.probe`` promises never to raise, but an adapter is the most likely place for
        a third-party library to break that promise, and one bad probe must not kill the poller
        task and start the supervisor's restart backoff.
        """
        try:
            return await self._adapter.probe(self._host, self._port, timeout=self._timeout)
        except Exception as exc:
            _log.exception("status_poller.probe_raised", host=self._host, port=self._port)
            return ProbeResult(
                reachable=False,
                error=f"{type(exc).__name__}: {exc}",
                probed_at=self._clock.now(),
            )

    def _record(self, result: ProbeResult) -> None:
        """Update the counters, and log the edges rather than every poll.

        Edge-triggered on purpose: a server that is down for six hours at a 45-second interval
        would otherwise write 480 identical warnings, and the useful facts are "it stopped
        answering" and "it started again, after this long".
        """
        if result.reachable:
            self._stats["successes"] += 1
            self._last_success_at = result.probed_at
            if not self._reachable:
                _log.info(
                    "status_poller.reachable",
                    host=self._host,
                    port=self._port,
                    after_failures=self._consecutive_failures,
                    players_online=result.players_online,
                )
            self._reachable = True
            self._consecutive_failures = 0
            return

        self._stats["failures"] += 1
        self._consecutive_failures += 1
        if self._reachable or self._consecutive_failures == 1:
            _log.warning(
                "status_poller.unreachable",
                host=self._host,
                port=self._port,
                error=result.error,
                hint="unreachable means UNKNOWN, never zero players",
            )
        self._reachable = False

    def _notify(self, result: ProbeResult) -> None:
        if self._on_probe is None:
            return
        try:
            self._on_probe(result)
        except Exception:
            # A broken observer is the lifecycle machine's problem, not a reason to stop probing.
            self._stats["observer_errors"] += 1
            _log.exception("status_poller.observer_failed")

    # ---------------------------------------------------------------------------- the roster

    def _reconcile(self, result: ProbeResult) -> RosterDelta:
        """Reconcile the roster and publish the delta, subject to the completeness gate."""
        delta = self._roster.reconcile(result, trust_partial_sample=self._trust_partial_sample)
        if not delta.applied:
            key = (
                "skipped_unreachable"
                if delta.reason == "probe_unreachable"
                else "skipped_incomplete"
            )
            self._stats[key] += 1
            if delta.reason == "sample_incomplete":
                _log.debug(
                    "status_poller.sample_incomplete",
                    players_online=result.players_online,
                    sample_size=len(result.sample),
                    online=self._roster.count,
                    hint="sample is capped at 12 and may be randomised; not reconciling",
                )
            return delta

        self._stats["reconciled"] += 1
        if delta.changed:
            _log.info(
                "status_poller.roster_reconciled",
                joined=list(delta.joined_names),
                left=list(delta.left_names),
                online=delta.online_count,
            )
        for session in delta.joined:
            self._stats["probe_joins"] += 1
            self._sink.publish(
                PlayerJoined(
                    ts=result.probed_at,
                    server_id=self._server_id,
                    source=Source.PROBE,
                    raw=f"status probe reported {session.name} online",
                    player=PlayerRef(name=session.name, uuid=session.uuid),
                    online_count=delta.online_count,
                    first_seen=session.first_seen,
                )
            )
        for session in delta.left:
            self._stats["probe_leaves"] += 1
            self._sink.publish(
                PlayerLeft(
                    ts=result.probed_at,
                    server_id=self._server_id,
                    source=Source.PROBE,
                    raw=f"status probe no longer reports {session.name} online",
                    player=PlayerRef(name=session.name, uuid=session.uuid),
                    # The probe cannot know *why* somebody left, and guessing "quit" would put a
                    # fabricated reason in a session summary. UNKNOWN is the honest answer.
                    reason=LeaveReason.UNKNOWN,
                    session_seconds=session.seconds_at(self._clock.monotonic()),
                    online_count=delta.online_count,
                )
            )
        return delta
