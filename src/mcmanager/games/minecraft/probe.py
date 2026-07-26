"""Server List Ping, via ``mcstatus``.

Returns a :class:`~mcmanager.games.base.ProbeResult` and **never raises**: a failed probe means
UNKNOWN, not zero players. Treating an unreachable server as empty, combined with idle shutdown,
is precisely how you stop a populated server - which is why
:attr:`~mcmanager.games.base.ProbeResult.is_empty` is ``reachable and players_online == 0`` and
never just ``not players_online``.

``host`` comes from config and is ``"minecraft"``, not ``"localhost"``: from inside our container
localhost is our own loopback, where nothing listens. It resolves because both containers are
attached to the external ``homelab`` bridge network, which is also why container-to-container
traffic never touches the host's ufw INPUT chain.

Two SLP queries now exist against this server: ``mc-health`` already runs one every 30 seconds
from inside the container, and this poller adds one every 45. Harmless, but that is the
explanation if SLP lag ever appears.

**The sample is not the roster.** ``players.sample`` is protocol-capped (this server advertises
``Server Ping Player Sample Count: 12``) and Paper settings or anti-scrape plugins may randomise
or omit it entirely. :attr:`~mcmanager.games.base.ProbeResult.sample_is_complete` is derived as
``players_online == len(sample)`` and reconciliation must gate on it, or a busy server produces a
storm of fake ``PlayerLeft`` events.

**The UUIDs are offline v3 UUIDs.** ``online-mode=false`` here, so ``sample[i].id`` is derived
from ``OfflinePlayer:<name>`` - stable per name, but *not* a Mojang UUID. Never send one to a
Mojang API, and never treat two servers' UUIDs for the same name as authoritative identity. They
are dropped rather than returned, because ``ProbeResult`` has nowhere honest to put them and
session statistics key on the name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcstatus import JavaServer

from mcmanager.games.base import ProbeResult

if TYPE_CHECKING:
    from datetime import datetime

    from mcstatus.status_response import JavaStatusResponse

    from mcmanager.clock import Clock

__all__ = ["DEFAULT_PORT", "query_status"]

DEFAULT_PORT = 25565
"""The Java Edition default. Published on this host; RCON's 25575 deliberately is not."""


async def query_status(
    host: str,
    port: int,
    *,
    timeout: float,  # noqa: ASYNC109 - handed to mcstatus, not a client-side deadline
    clock: Clock,
) -> ProbeResult:
    """Query the server out of band. Never raises.

    Args:
        host: DNS name or address. ``"minecraft"`` in production, resolved on the ``homelab``
            bridge network.
        port: SLP port, normally :data:`DEFAULT_PORT`.
        timeout: Protocol-level deadline handed to ``mcstatus``, not a client-side
            ``asyncio.timeout``. A probe that timed out and a probe that was cancelled assert the
            same thing here, and that thing is UNKNOWN.
        clock: Injected, and the only reason this function needs one:
            :attr:`~mcmanager.games.base.ProbeResult.probed_at` must be tz-aware UTC and no module
            outside ``mcmanager.clock`` may call ``datetime.now``.

    Returns:
        A populated :class:`~mcmanager.games.base.ProbeResult` on success, or one with
        ``reachable=False`` and ``error`` set. Every failure mode - DNS, refused connection,
        timeout, a malformed status payload - lands in the same place, because none of them is
        evidence about how many people are playing.
    """
    probed_at = clock.now()
    try:
        server = await JavaServer.async_lookup(f"{host}:{port}", timeout=timeout)
        # mcstatus declares `async_status(self, **kwargs)` with untyped kwargs, so pyright strict
        # calls the member partially unknown. The return type is precise; only the parameters are
        # not, and we pass none.
        status: JavaStatusResponse = await server.async_status()  # pyright: ignore[reportUnknownMemberType]
    except Exception as exc:
        # Deliberately broad. mcstatus raises OSError subclasses, asyncio.TimeoutError, its own
        # protocol errors, and - on a malformed payload - whatever the parser felt like. The
        # caller's only correct response to any of them is identical: we do not know.
        return ProbeResult(
            reachable=False,
            error=_describe(exc),
            probed_at=probed_at,
        )

    return _to_result(status, probed_at=probed_at)


def _to_result(status: JavaStatusResponse, *, probed_at: datetime) -> ProbeResult:
    """Normalise an mcstatus response into the DTO. Pure."""
    sample = tuple(player.name for player in status.players.sample or ())
    return ProbeResult(
        reachable=True,
        players_online=status.players.online,
        players_max=status.players.max,
        sample=sample,
        version=status.version.name,
        motd=status.motd.to_plain(),
        latency_ms=status.latency,
        probed_at=probed_at,
    )


def _describe(exc: BaseException) -> str:
    """A one-line reason, never a traceback.

    The exception type is included because ``ConnectionRefusedError`` and ``TimeoutError`` mean
    very different things operationally - the first says the container is down, the second says
    the JVM is wedged or still loading chunks - and several of these carry an empty message.
    """
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
