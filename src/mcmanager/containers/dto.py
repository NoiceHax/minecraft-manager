"""Data transfer objects at the container boundary.

Everything above this layer - services, games, control, discordbot - sees only these types and
:class:`~mcmanager.containers.base.ContainerRuntime`. ``import docker`` exists in exactly two
modules (``docker_runtime.py`` and ``streams.py``), and their job is to turn ``Any``-shaped
inspect dictionaries into the structures below. docker-py ships no type stubs, so that boundary is
also where the typing suppressions live.

The single most important thing in this module is :attr:`ContainerSnapshot.health`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Self

from mcmanager.core.types import Stream

__all__ = [
    "ContainerSnapshot",
    "ContainerState",
    "ExecResult",
    "HealthState",
    "HealthTiming",
    "LogLine",
    "MountInfo",
    "NetworkAttachment",
    "RuntimeEvent",
]


def _empty_str_map() -> dict[str, str]:
    """Default factory for the string maps below.

    A named function rather than a bare ``default_factory=dict``, which infers
    ``dict[Unknown, Unknown]`` under pyright strict and makes the whole field partially unknown.
    """
    return {}


class ContainerState(StrEnum):
    """``State.Status`` from a container inspect, plus two states Docker cannot report."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    REMOVING = "removing"
    EXITED = "exited"
    DEAD = "dead"

    ABSENT = "absent"
    """No container with that name. A legitimate state - warn and keep running - not an error."""

    UNKNOWN = "unknown"
    """Docker answered with a status string we do not recognise. Never guessed at."""


class HealthState(StrEnum):
    """Healthcheck result. Values match Docker's own strings so ``HealthState(raw)`` just works."""

    NONE = "none"
    """The image defines no healthcheck."""

    STARTING = "starting"
    """Inside ``start_period`` - 120 seconds on this container. Not a failure."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"

    UNKNOWN = "unknown"
    """We refuse to have an opinion: the container is not running, or Docker reported a value we
    do not recognise."""


@dataclass(frozen=True, slots=True, kw_only=True)
class HealthTiming:
    """The container's own healthcheck configuration, read from ``Config.Healthcheck``.

    Carried on every snapshot so the lifecycle state machine reads its timing **from the
    container** instead of hardcoding this deployment's numbers. On the homelab that is
    ``interval=30s, retries=2, start_period=120s`` and therefore a 210-second guard window.
    """

    interval: timedelta | None = None
    timeout: timedelta | None = None
    start_period: timedelta | None = None
    retries: int | None = None

    @property
    def guard_window(self) -> timedelta | None:
        """``start_period + interval * (retries + 1)``: how long an ``unhealthy`` may be ignored.

        Returns ``None`` when the container declares no healthcheck, which callers must read as
        "no opinion", never as zero.
        """
        if self.start_period is None or self.interval is None:
            return None
        retries = self.retries if self.retries is not None else 0
        return self.start_period + self.interval * (retries + 1)


@dataclass(frozen=True, slots=True, kw_only=True)
class MountInfo:
    """One entry of ``Mounts``. Surfaced by ``mcmanager inspect`` because a forgotten ``:ro`` bind
    of the server's log directory is a silent degradation otherwise."""

    source: str
    destination: str
    mode: str = ""
    rw: bool = True
    kind: str = "bind"


@dataclass(frozen=True, slots=True, kw_only=True)
class NetworkAttachment:
    """One entry of ``NetworkSettings.Networks``.

    ``mcmanager inspect`` prints these because "is the server on the ``homelab`` network?" is the
    first question when ``minecraft:25565`` stops resolving, and the answer is here.
    """

    name: str
    ip_address: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ContainerSnapshot:
    """One inspect, normalised.

    Attributes:
        name: The name we asked about. Always populated, even when the container is absent -
            resolution is by name on every reconnect, never by a cached id, because a
            ``compose down/up`` changes the id and a cached id silently follows a dead container
            forever.
        _health_raw: The raw ``State.Health.Status`` string, or ``None``. **Do not read this.**
            Read :attr:`health`. It is stored under a private name precisely so that reaching past
            the derived property is a visible act rather than an accident.
        tty: ``Config.Tty``. False on this container, which means the log stream is multiplexed
            frames with 8-byte headers and arbitrary split points, not raw bytes. The line splitter
            must buffer across chunks either way.
        oom_killed: ``State.OOMKilled``. Decisive: an OOM is never a clean stop.
        observed_at: When this inspect happened, tz-aware UTC. Snapshots are compared by age, so
            this is not optional.
    """

    name: str
    id: str | None = None
    exists: bool = True
    state: ContainerState = ContainerState.UNKNOWN
    status_text: str | None = None
    running: bool = False
    _health_raw: str | None = None
    health_failing_streak: int | None = None
    health_timing: HealthTiming = field(default_factory=HealthTiming)
    tty: bool = False
    exit_code: int | None = None
    oom_killed: bool = False
    restart_count: int = 0
    image: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    labels: Mapping[str, str] = field(default_factory=_empty_str_map)
    networks: tuple[NetworkAttachment, ...] = ()
    mounts: tuple[MountInfo, ...] = ()
    log_driver: str | None = None
    log_options: Mapping[str, str] = field(default_factory=_empty_str_map)
    restart_policy: str | None = None
    stop_signal: str | None = None
    observed_at: datetime

    # -- the stale-health fix ----------------------------------------------------------------

    @property
    def health(self) -> HealthState:
        """The **only** sanctioned way to ask about health.

        Observed on the real homelab container: while ``minecraft`` was *exited*,
        ``State.Health.Status`` still read ``"unhealthy"`` - with ``FailingStreak`` at 0 and the
        last five probes having exited 0. Docker simply never clears the field on stop. Any
        readiness or alerting logic that trusts that string reports a permanently unhealthy server
        the moment it is turned off.

        So the rule is encoded once, here, as a derived property rather than a field a constructor
        might forget to normalise: **if it is not running, we have no opinion.**
        """
        if not self.running:
            return HealthState.UNKNOWN
        if self._health_raw is None:
            return HealthState.NONE
        try:
            return HealthState(self._health_raw)
        except ValueError:
            return HealthState.UNKNOWN

    # -- derived conveniences ----------------------------------------------------------------

    @property
    def health_reported_raw(self) -> str | None:
        """What Docker literally said, for ``mcmanager inspect`` and for bug reports.

        Named so that reading it is obviously a diagnostic act. Never make a decision on it.
        """
        return self._health_raw

    @property
    def absent(self) -> bool:
        """True when there is no such container."""
        return not self.exists or self.state is ContainerState.ABSENT

    @property
    def exited(self) -> bool:
        """True when the container existed and has stopped."""
        return self.exists and self.state in (ContainerState.EXITED, ContainerState.DEAD)

    @property
    def guard_window(self) -> timedelta | None:
        """Shorthand for ``health_timing.guard_window``."""
        return self.health_timing.guard_window

    def uptime(self, now: datetime) -> timedelta | None:
        """How long the container has been running, or ``None`` if it is not.

        Takes ``now`` rather than calling a clock: this module has no clock, by design.
        """
        if not self.running or self.started_at is None:
            return None
        return now - self.started_at

    def alias_for(self, network: str) -> tuple[str, ...]:
        """DNS aliases the container answers to on ``network``. Empty if it is not attached."""
        for attachment in self.networks:
            if attachment.name == network:
                return attachment.aliases
        return ()

    @classmethod
    def missing(cls, name: str, *, observed_at: datetime) -> Self:
        """The snapshot for "there is no such container"."""
        return cls(
            name=name,
            exists=False,
            state=ContainerState.ABSENT,
            running=False,
            observed_at=observed_at,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LogLine:
    """One line of container output.

    Attributes:
        text: The line with its trailing newline removed. ANSI is **not** stripped here - that is
            the parser's first step, and the raw bytes are what ``mcmanager logs --raw`` shows.
        ts: The timestamp Docker prefixed onto the line. ``logs(timestamps=True)`` is used
            unconditionally: the server's own ``[13:24:37]`` is time-only in ``Asia/Kolkata``, the
            wrapper's is a third format entirely, and neither survives backfill, where lines may be
            hours old. Docker's RFC3339Nano prefix is uniform across all three grammars and is UTC.
            ``None`` only if the prefix was missing or unparsable.
        received_at: When the pump handed us the line, from the injected clock.
        stream: stdout or stderr.
    """

    text: str
    ts: datetime | None
    received_at: datetime
    stream: Stream = Stream.STDOUT

    @property
    def event_ts(self) -> datetime:
        """The timestamp an event built from this line should carry: ``ts`` or ``received_at``."""
        return self.ts if self.ts is not None else self.received_at


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeEvent:
    """A Docker daemon event about a container, normalised.

    Attributes:
        action: ``"start"``, ``"die"``, ``"health_status: healthy"``, ... Kept verbatim; the
            useful bits are exposed as properties below.
        attributes: ``Actor.Attributes``. Every value is a **string**, including ``exitCode`` -
            a landmine worth naming, since ``attrs["exitCode"] == 0`` is silently always false.
    """

    action: str
    container_id: str | None = None
    container_name: str | None = None
    ts: datetime
    attributes: Mapping[str, str] = field(default_factory=_empty_str_map)

    @property
    def exit_code(self) -> int | None:
        """``exitCode``, parsed. ``None`` when absent or not an integer."""
        raw = self.attributes.get("exitCode")
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    @property
    def health_status(self) -> HealthState | None:
        """For ``health_status: healthy`` style actions, the state; otherwise ``None``.

        These are **edge-triggered only** - Docker emits one when the status *changes* - so they
        are useless as a heartbeat and must never be used as one.
        """
        prefix = "health_status: "
        if not self.action.startswith(prefix):
            return None
        try:
            return HealthState(self.action[len(prefix) :].strip())
        except ValueError:
            return HealthState.UNKNOWN

    @property
    def is_lifecycle(self) -> bool:
        """True for the actions that move the lifecycle machine."""
        return self.action in ("create", "start", "restart", "stop", "kill", "die", "destroy")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecResult:
    """The outcome of a command run inside a container.

    This is how RCON works by default: ``docker exec minecraft rcon-cli <cmd>``. The itzg image
    configures ``rcon-cli`` from the container's own environment, so mcmanager never needs the RCON
    password at all - nothing to store, rotate or leak.
    """

    exit_code: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def output(self) -> str:
        """stdout, falling back to stderr when the command was silent on stdout."""
        return self.stdout if self.stdout else self.stderr
