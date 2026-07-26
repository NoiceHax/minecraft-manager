"""The extensibility seam: everything game-specific lives behind this Protocol.

Adding Valheim is one package under ``games/``, one line in ``games/registry.py``, and
``game = "valheim"`` in the TOML. Nothing in ``core``, ``containers``, ``services``,
``persistence``, ``control`` or ``discordbot`` changes.

Two things would falsify that claim, and both are explicitly avoided:

1. **Minecraft fields leaking onto base events.** They do not: ``AdvancementKind`` and ``ChatKind``
   are generic enough to be shared, and anything genuinely Minecraft-shaped stays in the message
   text.
2. **``services/lifecycle.py`` importing ``mcstatus``.** It does not: it consumes a
   :class:`ProbeResult`, which is what this module defines.

A ``Protocol`` rather than an ABC, because an adapter is a bundle of pure functions plus one async
probe; there is no shared state to inherit and no constructor to agree on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from mcmanager.core.events import Event
    from mcmanager.core.types import ReadySignal, ServerId, Stream

__all__ = ["GameAdapter", "ProbeResult"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbeResult:
    """The outcome of one out-of-band status query (Server List Ping, for Minecraft).

    Attributes:
        reachable: Did the server answer at all. **A failed probe means UNKNOWN, never "zero
            players".** Treating it as zero is the bug that stops a populated server, which is why
            ``idle.treat_probe_failure_as_empty`` exists, defaults to false, and is documented as
            "leave this false".
        players_online: The count the server reported.
        players_max: ``max-players``; 5 on this deployment.
        sample: The player-name sample. Protocol-capped at 12 entries, and Paper settings or
            anti-scrape plugins may randomise or omit it. See :attr:`sample_is_complete`.
        version: Reported server version string.
        motd: Reported description, already flattened to plain text.
        latency_ms: Round-trip time, useful as a lag signal.
        error: Why it failed, when it failed. Never a traceback.
        probed_at: tz-aware UTC, from the caller's injected clock.
    """

    reachable: bool
    players_online: int | None = None
    players_max: int | None = None
    sample: tuple[str, ...] = ()
    version: str | None = None
    motd: str | None = None
    latency_ms: float | None = None
    error: str | None = None
    probed_at: datetime

    @property
    def sample_is_complete(self) -> bool:
        """Whether :attr:`sample` may be trusted as the *whole* roster.

        Derived rather than stored, for the same reason
        :attr:`~mcmanager.containers.dto.ContainerSnapshot.health` is: a caller that reconciles the
        online roster against a truncated sample generates a storm of fake ``PlayerLeft`` events
        the moment the server is busier than the sample cap. Making this a settable field would
        mean one probe implementation could get it wrong; making it derived means none can.

        True only when the probe succeeded and the sample accounts for every reported player.
        """
        return (
            self.reachable
            and self.players_online is not None
            and self.players_online == len(self.sample)
        )

    @property
    def is_empty(self) -> bool:
        """True only when the server answered and said zero. Unreachable is not empty."""
        return self.reachable and self.players_online == 0


@runtime_checkable
class GameAdapter(Protocol):
    """Everything the daemon needs to know about one kind of game server."""

    @property
    def game_id(self) -> str:
        """Registry key, matching ``server.game`` in the config. e.g. ``"minecraft"``."""
        ...

    @property
    def default_port(self) -> int:
        """Port used for :meth:`probe` when the config does not override it."""
        ...

    @property
    def default_stop_timeout(self) -> int:
        """Seconds the server should be given to shut down cleanly.

        90 for Minecraft, because that is how long a world save can take, and it was previously a
        bare ``-t 90`` in a bash script with no explanation attached.
        """
        ...

    def parse_line(
        self,
        raw: str,
        *,
        ts: datetime,
        server_id: ServerId,
        stream: Stream,
    ) -> Event:
        """Turn one raw log line into exactly one event.

        The purity contract, which the whole parser design rests on:

        - **Total.** Always returns an ``Event``, never ``None``, never raises. Unrecognised input
          becomes a ``ConsoleLog``. A parser that can throw takes the log pipeline down with it.
        - **Pure.** No clock, no config, no bus, no I/O. ``ts`` is injected. Module state is
          compiled regexes and nothing else.
        - **Stateless.** Anything needing memory - stitching a login IP onto the following join,
          a disconnect reason onto the following leave, knowing who is currently online - is the
          log pipeline's job, not this one's. That is what makes ``mcmanager replay`` able to run
          a 36-file archive through the parser with no daemon in sight, and what makes the parser
          testable as a table of strings.
        """
        ...

    def ready_signal(self, event: Event) -> ReadySignal | None:
        """Does this event prove the server is up, and by which signal?

        Returns ``None`` for everything else. Lets the lifecycle machine stay game-agnostic while
        still recognising ``Done (32.521s)!``.
        """
        ...

    def stop_signal(self, event: Event) -> bool:
        """Does this event mean a shutdown has begun?

        For Minecraft: ``[Rcon: Stopping the server]`` and friends. Matters because a
        ``lost connection: Server closed`` arriving *after* this is a shutdown casualty, not a
        voluntary leave.
        """
        ...

    async def probe(
        self,
        host: str,
        port: int,
        *,
        timeout: float,  # noqa: ASYNC109
    ) -> ProbeResult:
        """Query the server out of band. Never raises: failures come back as
        ``ProbeResult(reachable=False, error=...)``.

        ``timeout`` is a protocol-level deadline passed to the query library, not a client-side
        ``asyncio.timeout`` - hence the ``ASYNC109`` suppression. A cancelled probe and a probe
        that timed out are the same fact here, and both mean UNKNOWN rather than empty.
        """
        ...
