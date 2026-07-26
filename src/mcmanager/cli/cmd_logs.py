"""``mcmanager logs`` - the server console.

``--follow`` attaches to the daemon's ``/logs`` SSE and therefore **requires** it (exit 69 with the
URL). Without ``--follow`` this reads Docker directly and needs no daemon at all, which is the
combination that matters when the daemon is the broken thing.

``--raw`` bypasses the parser and prints exactly what came off the wire, escape bytes included.
That is the right tool for "why did the parser not match this line", and it is the only place in
the CLI where unsanitised bytes reach a terminal - deliberately, and only on request.

The non-following path parses with the **game adapter directly**, not the log pipeline. The
pipeline is stateful by design (it stitches a login address onto the next join, a disconnect
reason onto the next leave, and it knows who is online); running a one-shot tail through it would
produce enrichment based on a roster reconstructed from a hundred lines, which is worse than no
enrichment because it looks the same. What this prints is the parser's honest, stateless view, and
``mcmanager events`` is where the enriched stream lives.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcmanager.cli import render
from mcmanager.containers.errors import RuntimeUnavailableError
from mcmanager.containers.factory import build_runtime
from mcmanager.control.sse import parse_since
from mcmanager.core.events import ConsoleLog
from mcmanager.core.serde import event_to_json
from mcmanager.errors import EXIT_CONFIG, EXIT_OK, EXIT_UNAVAILABLE

if TYPE_CHECKING:
    from datetime import datetime

    from mcmanager.cli.client import LogRecord
    from mcmanager.cli.main import CliContext

__all__ = ["run"]


async def run(
    ctx: CliContext,
    *,
    follow: bool = False,
    tail: int = 100,
    since: str | None = None,
    raw: bool = False,
    container: str | None = None,
) -> int:
    """Print the console, from the daemon when following and from Docker otherwise."""
    if follow:
        return await _follow(ctx, since=since, raw=raw)
    return await _tail(ctx, tail=tail, since=since, raw=raw, container=container)


# --------------------------------------------------------------------------------- standalone


async def _tail(
    ctx: CliContext,
    *,
    tail: int,
    since: str | None,
    raw: bool,
    container: str | None,
) -> int:
    """Read recent output straight from Docker. Works on a **stopped** container.

    That last property is what makes this the first command to run after a crash: the container is
    gone, the daemon may be confused, and the last forty lines are still sitting in the json-file
    driver's ring.
    """
    from mcmanager.config import runtime_unreachable_message

    name = container or ctx.settings.server.container
    cutoff: datetime | None = None
    if since is not None:
        try:
            cutoff = parse_since(since, now=ctx.clock.now())
        except ValueError as exc:
            render.emit_error(str(exc))
            return EXIT_CONFIG

    runtime = build_runtime(
        ctx.settings.runtime,
        clock=ctx.clock,
        docker_host=ctx.settings.docker.host,
    )
    try:
        # tail=-1 means "everything since the cutoff": Docker applies `tail` *after* `since`, so
        # asking for both a since and a line count silently returns the wrong window.
        lines = await runtime.logs_tail(name, lines=-1 if cutoff else tail, since=cutoff)
    except RuntimeUnavailableError as exc:
        render.emit_error(runtime_unreachable_message(endpoint=None, error=str(exc)))
        return EXIT_UNAVAILABLE
    finally:
        await runtime.aclose()

    if raw:
        for line in lines:
            render.emit(line.text)
        return EXIT_OK

    from mcmanager.games.registry import get_adapter

    adapter = get_adapter(ctx.settings.server.game, clock=ctx.clock)
    for line in lines:
        event = adapter.parse_line(
            line.text,
            ts=line.event_ts,
            server_id=ctx.settings.server.id,
            stream=line.stream,
        )
        if ctx.json_output:
            render.emit(event_to_json(event))
        elif isinstance(event, ConsoleLog):
            render.emit(
                render.render_log_line(
                    ts=event.ts,
                    message=event.message,
                    level=event.level,
                    thread=event.thread,
                    palette=ctx.palette,
                )
            )
        else:
            # A recognised line has no ConsoleLog of its own - that is the spec - so rendering it
            # as an event is what keeps this view from having holes exactly where the joins were.
            render.emit(render.render_event(event, palette=ctx.palette, timestamp=True))
    return EXIT_OK


# ------------------------------------------------------------------------------------- follow


async def _follow(ctx: CliContext, *, since: str | None, raw: bool) -> int:
    """Attach to the daemon's ``/logs`` stream.

    ``--raw`` still means "no rendering", but the daemon has already sanitised: what crosses the
    wire is ``ConsoleLog.message`` or ``Event.raw``, both of which are ANSI-stripped. Genuinely
    raw bytes are only available from the non-following path, and the message below says so
    rather than letting somebody believe they are debugging the parser when they are not.
    """
    from mcmanager.cli.client import DaemonClient

    if raw:
        render.emit_error(
            "note: --raw with --follow shows the daemon's already-sanitised text. For the "
            "original bytes, drop --follow: `mcmanager logs --raw -n 200`."
        )
    async with DaemonClient(ctx.url, token=ctx.token) as client:
        await client.ensure_reachable()
        async for record in client.stream_logs(since=since):
            if ctx.json_output:
                render.emit(json.dumps(_record_to_dict(record), sort_keys=True), flush=True)
            else:
                render.emit(
                    render.render_log_line(
                        ts=record.ts,
                        message=record.message,
                        level=record.level,
                        thread=record.thread,
                        palette=ctx.palette,
                    ),
                    flush=True,
                )
    return EXIT_OK


def _record_to_dict(record: LogRecord) -> dict[str, object]:
    """The ``--json`` form of one streamed console line. Mirrors the SSE payload exactly."""
    return {
        "ts": None if record.ts is None else record.ts.isoformat(),
        "seq": record.seq,
        "type": record.type,
        "message": record.message,
        "level": record.level,
        "thread": record.thread,
        "origin": record.origin,
        "stream": record.stream,
    }
