"""Shared vocabulary: the enums and small value objects every layer agrees on.

Nothing in this module imports anything from the rest of the project, and nothing here knows about
Docker, Discord or Minecraft. It is the bottom of the dependency graph on purpose - an enum defined
next to its consumer inevitably gets duplicated by the next consumer.

Every enum is a :class:`enum.StrEnum` so that serialisation is `value`, config files can spell them
as plain strings, and log lines read as words rather than ``<Source.LOG: 1>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

__all__ = [
    "AdvancementKind",
    "ChatKind",
    "ContainerName",
    "ControlAction",
    "DispatchMode",
    "LeaveReason",
    "LifecycleState",
    "LineOrigin",
    "PlayerRef",
    "ReadySignal",
    "RuntimeKind",
    "ServerId",
    "Source",
    "Stream",
]

# Semantic aliases. Deliberately plain `str` rather than NewType: these cross the pydantic,
# JSON and docker-py boundaries constantly and NewType buys friction, not safety, at those seams.
type ServerId = str
type ContainerName = str

# Which container runtime `containers.factory.build_runtime` should build. "fake" is never the
# default: a homelab daemon silently running against a fake in production is worse than a startup
# failure.
type RuntimeKind = Literal["real", "fake"]


class Source(StrEnum):
    """Where a fact came from. Stamped onto every :class:`~mcmanager.core.events.Event`.

    This is provenance, not causation: it answers "how do we know this?" so a surprising event can
    be traced back to the subsystem that produced it.
    """

    LOG = "log"
    """Parsed out of the container's log stream."""

    RUNTIME = "runtime"
    """A Docker daemon event or a container inspect."""

    PROBE = "probe"
    """A Server List Ping / status query."""

    RCON = "rcon"
    """A command channel round trip."""

    TIMER = "timer"
    """A scheduled deadline fired (idle timeout, start deadline)."""

    CLI = "cli"
    """A local `mcmanager` subcommand."""

    DISCORD = "discord"
    """A slash command or gateway interaction."""

    WEB = "web"
    """The aiohttp control surface."""

    INTERNAL = "internal"
    """The daemon reasoning about itself: reconciliation, supervision, shutdown."""


class Stream(StrEnum):
    """Which of the container's two output streams a log line arrived on."""

    STDOUT = "stdout"
    STDERR = "stderr"


class LineOrigin(StrEnum):
    """Which of the interleaved grammars a raw log line belongs to.

    The container multiplexes three unrelated formats onto one stream, and telling them apart is
    step two of parsing (step one is stripping ANSI). Only :attr:`SERVER` lines are ever candidates
    for player or chat events.
    """

    SERVER = "server"
    """``[13:24:37] [Server thread/INFO]: ...`` - log4j2, time-only, optionally ANSI-coloured,
    optionally carrying a second bracket group for the plugin name."""

    WRAPPER = "wrapper"
    """``2026-07-25T22:58:20.809+0530\\tINFO\\tmc-server-runner\\tDone`` - the entrypoint
    supervisor. Tab separated, full date, only ever present in the docker stream."""

    CONTINUATION = "continuation"
    """``\\tat net.minecraft...`` - a stack frame or wrapped message belonging to the line above."""

    RAW = "raw"
    """Matched no known grammar. Becomes a ConsoleLog and feeds the unrecognised-line ratio, which
    is the canary for a Paper upgrade silently breaking the patterns."""


class ReadySignal(StrEnum):
    """Which independent signal promoted the server to READY.

    All three assert the identical fact - ``mc-health`` is itself an SLP query - but they arrive up
    to 90 seconds apart. Recording which one fired keeps the "first of three" readiness policy
    auditable, and reverting to strict health-only is then a config change.
    """

    LOG_DONE = "log"
    """The ``Done (32.521s)! For help, type "help"`` line. Earliest, and carries the duration."""

    HEALTHCHECK = "health"
    """A docker ``health_status: healthy`` event. Authoritative, but gated behind a 120s
    start_period plus interval, so it lands late."""

    PROBE = "probe"
    """Our own SLP query succeeded."""


class AdvancementKind(StrEnum):
    """Minecraft distinguishes three tiers, each with its own message template and colour."""

    ADVANCEMENT = "advancement"
    """``has made the advancement [Stone Age]``"""

    GOAL = "goal"
    """``has reached the goal [Adventuring Time]``"""

    CHALLENGE = "challenge"
    """``has completed the challenge [Beaconator]``"""


class ChatKind(StrEnum):
    """How a chat line was produced. Presenters escape all of them identically."""

    CHAT = "chat"
    """``<Steve> hello`` - a normal player message."""

    EMOTE = "emote"
    """``* Steve waves`` - /me."""

    SAY = "say"
    """``[Steve] hello`` - /say, which an operator or the console can issue."""

    RCON = "rcon"
    """``[Rcon] ...`` - issued over the command channel, i.e. quite possibly by us."""


class LeaveReason(StrEnum):
    """Why a player stopped being online.

    :attr:`SERVER_CLOSED` is the load-bearing one: ``bharath_720 lost connection: Server closed``
    is a shutdown, not a voluntary leave, and counting it as one corrupts every session summary.
    """

    QUIT = "quit"
    """Disconnected on purpose."""

    TIMED_OUT = "timed_out"
    """Connection dropped."""

    KICKED = "kicked"
    """Removed by an operator or a plugin."""

    SERVER_CLOSED = "server_closed"
    """The server shut down underneath them."""

    UNKNOWN = "unknown"
    """A disconnect reason we have no mapping for. Never guess."""


class LifecycleState(StrEnum):
    """States of the lifecycle reducer.

    Defined here rather than in ``services/lifecycle.py`` because the CLI, the control surface and
    the Discord presenters all render it, and none of them should import the state machine to name
    a state.
    """

    UNKNOWN = "unknown"
    """Before the first successful inspect."""

    ABSENT = "absent"
    """No such container. A legitimate state: warn, keep running."""

    STOPPED = "stopped"
    """Exited cleanly, or never started."""

    CRASHED = "crashed"
    """Exited unexpectedly. Distinct from STOPPED so Discord can shout about it."""

    STARTING = "starting"
    """Container running, server not yet answering."""

    READY = "ready"
    """Answering SLP / healthy / logged ``Done``."""

    DEGRADED = "degraded"
    """Running but unhealthy past the start-period guard, or past the start deadline."""

    STOPPING = "stopping"
    """A stop is in flight. Leaves during this window are SERVER_CLOSED, not quits."""

    BLIND = "blind"
    """The Docker daemon went away. Last-known state is retained, and no phantom ServerStopped is
    emitted, which is the entire reason RuntimeUnavailable exists."""


class DispatchMode(StrEnum):
    """How the bus invokes one subscriber."""

    SEQUENTIAL = "sequential"
    """One event at a time, in total FIFO order, awaited before the next is dispatched. The default,
    because the stateful consumers are finite state machines that are only correct if they observe
    ``PlayerJoined -> IdleCancelled -> PlayerLeft -> IdleStarted`` in that order. Contract: a
    SEQUENTIAL handler must not await network I/O."""

    CONCURRENT = "concurrent"
    """Fire and forget into a task. For network I/O - Discord, SSE fan-out - where ordering is
    nice-to-have and blocking the bus is not acceptable."""


class ControlAction(StrEnum):
    """A mutating command someone asked for, recorded by ``CommandIssued`` for the audit trail."""

    START = "start"
    STOP = "stop"
    RESTART = "restart"


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayerRef:
    """Identity of a player, as far as we can honestly know it.

    ``online-mode=false`` on this server, so ``uuid`` - when the logs give us one - is an offline v3
    UUID: stable per name, but **not** a Mojang UUID. Never send it to a Mojang API. Session stats
    key on :attr:`name`.

    ``name`` is the sanitised name: ANSI already stripped, prefix already removed. If a name here
    ever contains ``[`` or a timestamp, the parser has regressed to the old bridge script's
    unanchored-greedy-capture bug.
    """

    name: str
    uuid: str | None = None

    def __str__(self) -> str:
        return self.name
