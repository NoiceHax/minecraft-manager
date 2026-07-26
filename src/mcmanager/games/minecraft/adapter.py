"""The Minecraft :class:`~mcmanager.games.base.GameAdapter`.

Binds :func:`mcmanager.games.minecraft.parser.parse`,
:func:`mcmanager.games.minecraft.probe.query_status` and the readiness/stop signal predicates into
the Protocol from ``games/base.py``. It contains no logic of its own beyond that wiring, which is
the point: the seam has to be thin enough that a second game is genuinely a new package rather
than a refactor of this one.

The one piece of state it holds is a :class:`~mcmanager.clock.Clock`, and only because
:attr:`~mcmanager.games.base.ProbeResult.probed_at` has to be tz-aware UTC and no module outside
``mcmanager.clock`` may call ``datetime.now``. Parsing takes its timestamp as an argument and
touches no clock at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from mcmanager.clock import SystemClock
from mcmanager.core.events import ServerReady, ServerStopping
from mcmanager.games.minecraft import probe as slp
from mcmanager.games.minecraft.parser import parse

if TYPE_CHECKING:
    from datetime import datetime

    from mcmanager.clock import Clock
    from mcmanager.core.events import Event
    from mcmanager.core.types import ReadySignal, ServerId, Stream
    from mcmanager.games.base import ProbeResult

__all__ = ["GAME_ID", "MinecraftAdapter"]

GAME_ID = "minecraft"
"""Registry key, matching ``server.game`` in the TOML."""

_DEFAULT_STOP_TIMEOUT = 90
"""Seconds the JVM gets to save the world before Docker sends SIGKILL.

Ninety, because that is how long a world save can take on this host, and because it was
previously a bare ``-t 90`` in ``~/homelab/scripts/minecraft-idle-stop.sh`` with no explanation
attached to it. Verified consequence: a graceful stop exits **0** here, since ``mc-server-runner``
traps SIGTERM, writes ``stop`` and exits cleanly. Exit 137 with a stop intent means this number
was too small.
"""


@final
class MinecraftAdapter:
    """Everything the daemon needs to know about a Minecraft server.

    Structurally a :class:`~mcmanager.games.base.GameAdapter`; the Protocol is
    ``runtime_checkable`` so the registry can assert the conformance rather than assume it.
    """

    __slots__ = ("_clock",)

    def __init__(self, *, clock: Clock | None = None) -> None:
        """Build an adapter.

        Args:
            clock: Used only to timestamp probe results. Defaults to
                :class:`~mcmanager.clock.SystemClock` so that ``mcmanager replay``, which never
                probes, can build an adapter without wiring one.
        """
        self._clock: Clock = clock if clock is not None else SystemClock()

    @property
    def game_id(self) -> str:
        return GAME_ID

    @property
    def default_port(self) -> int:
        return slp.DEFAULT_PORT

    @property
    def default_stop_timeout(self) -> int:
        return _DEFAULT_STOP_TIMEOUT

    def parse_line(
        self,
        raw: str,
        *,
        ts: datetime,
        server_id: ServerId,
        stream: Stream,
    ) -> Event:
        """Delegate to the pure parser. Total; never raises."""
        return parse(raw, ts=ts, server_id=server_id, stream=stream)

    def ready_signal(self, event: Event) -> ReadySignal | None:
        """``Done (32.521s)! For help, type "help"`` is the log-side readiness proof.

        Returns the signal the *event itself* recorded rather than a hardcoded
        :attr:`~mcmanager.core.types.ReadySignal.LOG_DONE`, so that a ``ServerReady`` synthesised
        by the health watcher or the status poller reports itself honestly when it passes through
        here.
        """
        if isinstance(event, ServerReady):
            return event.detected_by
        return None

    def stop_signal(self, event: Event) -> bool:
        """Has a shutdown begun?

        Matters beyond the obvious: a ``lost connection: Server closed`` arriving after this is a
        shutdown casualty, not a voluntary leave, and counting it as one corrupts every session
        summary.
        """
        return isinstance(event, ServerStopping)

    async def probe(
        self,
        host: str,
        port: int,
        *,
        timeout: float,  # noqa: ASYNC109 - handed to mcstatus, not a client-side deadline
    ) -> ProbeResult:
        """Server List Ping. Never raises; failure comes back as ``reachable=False``."""
        return await slp.query_status(host, port, timeout=timeout, clock=self._clock)
