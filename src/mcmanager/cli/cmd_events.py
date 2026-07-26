"""``mcmanager events`` - a live tail of the event bus over SSE. **Requires the daemon.**

``--follow``, ``--since``, ``--type PlayerEvent`` (resolved with the bus's subclass semantics, so
category names work), ``--json``.

**It refuses to degrade.** Unlike ``status`` there is no standalone approximation of "what is
happening right now" - events are what the daemon *decided*, not what Docker said - so with no
daemon this exits **69** naming the URL it tried. A plausible-looking output would be a lie.

This is the command the 48-hour shadow run is verified with: watch ``events --follow`` while
somebody plays, then ``mcmanager replay`` over the same session's archive, and the two event
streams must agree. Every fact comes from the same :mod:`mcmanager.core.serde` encoder that writes
session records and feeds ``/events``, so ``--json`` output round-trips exactly.

``--type`` is validated **before** connecting, against the same table the server uses, so a typo
is an immediate error listing the accepted names rather than a stream that silently matches
nothing - which looks identical to a broken daemon.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from mcmanager.cli import render
from mcmanager.control.sse import resolve_types
from mcmanager.core.serde import event_to_json
from mcmanager.errors import EXIT_CONFIG, EXIT_OK

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from mcmanager.cli.main import CliContext
    from mcmanager.core.events import Event

__all__ = ["run"]

_DEFAULT_SINCE = "15m"
"""What a non-following ``events`` asks for when nobody said.

Without a ``since`` the server streams only what happens next, so a bare ``mcmanager events``
would print nothing and exit - which reads as "no events" rather than "you asked for the future".
"""


async def run(
    ctx: CliContext,
    *,
    follow: bool = False,
    types: Sequence[str] = (),
    since: str | None = None,
    seq: int | None = None,
    limit: int | None = None,
    idle_timeout: float = 1.5,
) -> int:
    """Stream events until interrupted, ``--limit`` is reached, or the tail goes quiet.

    Without ``--follow`` the command reads the daemon's replay ring and stops as soon as the
    stream has been idle for ``idle_timeout`` seconds. The alternative - a second "give me the
    history" endpoint - would mean a second wire format for the same events, which is exactly what
    ``core/serde.py`` exists to prevent.
    """
    from mcmanager.cli.client import DaemonClient

    wanted = [name.strip() for entry in types for name in entry.split(",") if name.strip()]
    try:
        resolve_types(wanted)
    except ValueError as exc:
        render.emit_error(str(exc))
        return EXIT_CONFIG

    effective_since = since
    if not follow and since is None and seq is None:
        effective_since = _DEFAULT_SINCE

    printed = 0
    async with DaemonClient(ctx.url, token=ctx.token) as client:
        # Before streaming, not during: an unreachable daemon otherwise looks like a stream that
        # simply never produced a frame, which the idle timeout below would report as "no events".
        await client.ensure_reachable()
        stream = client.stream_events(types=wanted, since=effective_since, seq=seq)
        try:
            async for event in _bounded(stream, follow=follow, idle_timeout=idle_timeout):
                render.emit(_format(event, ctx), flush=True)
                printed += 1
                if limit is not None and printed >= limit:
                    break
        finally:
            # Breaking out of an `async for` does not close the generator, and leaving it to the
            # loop's finalisation means the HTTP response outlives the session that owns it.
            await _close(stream)

    if printed == 0 and not follow:
        render.emit_error(
            f"no events in the daemon's replay ring for since={effective_since}. "
            "The ring is bounded and holds the recent past only; use --follow to watch live."
        )
    return EXIT_OK


def _format(event: Event, ctx: CliContext) -> str:
    if ctx.json_output:
        return event_to_json(event)
    return render.render_event(event, palette=ctx.palette, timestamp=True)


async def _bounded(
    stream: AsyncIterator[Event],
    *,
    follow: bool,
    idle_timeout: float,
) -> AsyncIterator[Event]:
    """Yield from ``stream``, stopping on silence when not following.

    ``asyncio.timeout`` rather than the injected clock, deliberately: this is a client-side
    deadline on a real socket in a foreground process, the same case the bus uses it for. Nothing
    here schedules work on virtual time, so a ``ManualClock`` would have nothing to control.
    """
    if follow:
        async for event in stream:
            yield event
        return

    iterator = stream.__aiter__()
    while True:
        try:
            async with asyncio.timeout(idle_timeout):
                event = await iterator.__anext__()
        except (TimeoutError, StopAsyncIteration):
            break
        yield event
    await _close(iterator)


async def _close(iterator: AsyncIterator[Event]) -> None:
    """Close an async generator, if that is what this is. Tolerates anything else."""
    closer: object = getattr(iterator, "aclose", None)
    if not callable(closer):
        return
    result: object = closer()
    if asyncio.iscoroutine(result):
        await result
