"""``mcmanager start|stop|restart``.

Posts to ``/control/*`` with the bearer token from ``web.token``. **Requires the daemon** and exits
69 without it - starting the server from a CLI process that is about to exit would leave nothing
watching the container it just started, no session record, and no Discord notification.

These are literally the same :class:`~mcmanager.services.controller.ServerController` calls the
Discord slash commands make. That is the whole point of building the CLI first: by the time a slash
command exists there is nowhere for a rule to hide inside its handler, because the handler has
nothing to do but call this and render the result.

Exit codes distinguish the three outcomes, because a script needs to:

- **0** - accepted and done (or recorded, under ``idle.dry_run``);
- **1** - the daemon considered it and refused, e.g. "the server is already ready". Not an error in
  the daemon; an error in the request;
- **69** - the daemon could not be reached, or it accepted the command and the runtime then failed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcmanager.cli import render
from mcmanager.errors import EXIT_OK, EXIT_UNAVAILABLE

if TYPE_CHECKING:
    from mcmanager.cli.main import CliContext

__all__ = ["run"]

_REFUSED = 1
"""The daemon said no. A plain 1: the request was wrong, not the world."""


async def run(ctx: CliContext, *, action: str, reason: str | None = None) -> int:
    """Ask the daemon to start, stop or restart the server.

    No confirmation prompt, deliberately. This command is run from a script, from an SSH one-liner
    and from ``docker exec`` far more often than interactively, and a prompt that only sometimes
    appears is worse than none. The audit trail is the safety net: every attempt publishes a
    :class:`~mcmanager.core.events.CommandIssued` naming the actor, including the refused ones.
    """
    from mcmanager.cli.client import DaemonClient

    if ctx.token is None:
        render.emit_error(
            "no control token: POST /control/* requires web.token. Set it in secrets_dir/"
            "web_token, MCM_WEB__TOKEN, or pass --token. Reads (status, players, events) need "
            "none, so only the mutating commands are affected."
        )
        return EXIT_UNAVAILABLE

    async with DaemonClient(ctx.url, token=ctx.token) as client:
        result = await client.control(action, actor=ctx.actor, reason=reason)

    if ctx.json_output:
        render.emit(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        render.emit(render.render_control_result(result, palette=ctx.palette))

    if not result.accepted:
        return _REFUSED
    if result.error is not None:
        return EXIT_UNAVAILABLE
    return EXIT_OK
