"""``mcmanager`` entry point and subcommand dispatch.

``argparse``, not click or typer: the dispatch is flat, and it is one fewer dependency in the
runtime image.

Commands, and whether they need the daemon::

    run                  -    the daemon itself
    check-config         no   validate + resolve + print the redacted config; exit 0 or 78
    inspect              no   raw truth from Docker: state, DERIVED health, exit code, OOM flag,
                              Tty, healthcheck timings, network aliases, mounts, log driver caps
    status               prefers - falls back to a standalone view with a `daemon: offline` banner
    players              prefers - standalone falls back to the status probe's sample
    events               YES  live tail of the bus; --follow --since --type --json
    logs                 either - --follow attaches to the daemon; without it, reads Docker
    replay <file>        no   run a .log or .log.gz through the parser offline
    sessions             no   list/show archived session records from state.dir
    start|stop|restart   YES  the same ServerController calls the slash commands make

Endpoint resolution: ``--url`` -> ``MCMANAGER_URL`` -> ``web.url`` -> ``http://127.0.0.1:8787``.

**Commands that require the daemon exit 69** naming the URL they tried and suggesting
``docker exec``, rather than silently degrading. A ``status`` that quietly shows stale standalone
data while the daemon is wedged is worse than an error, because it looks like an answer.

**Every command module is imported lazily, inside its branch.** That is not a startup-time
micro-optimisation: it is what makes ``mcmanager replay`` provably free of the daemon. Replay pulls
in the parser and nothing else - no aiohttp, no docker, no pydantic, no Discord - and
``tests/cli/test_main.py`` asserts exactly that by inspecting ``sys.modules`` afterwards. If that
test ever fails, the parser's purity contract has been broken somewhere upstream.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast, final

from mcmanager.cli import render
from mcmanager.cli.client import DEFAULT_URL, TOKEN_ENV, resolve_url
from mcmanager.clock import Clock, SystemClock
from mcmanager.config import Settings
from mcmanager.errors import EXIT_INTERNAL, EXIT_OK, McManagerError
from mcmanager.logging_setup import configure_logging

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["CliContext", "build_parser", "main"]

_INTERRUPTED: Final = 130
"""128 + SIGINT. What a shell expects from a Ctrl-C, and what stops a script retrying forever."""


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CliContext:
    """Everything a command needs that did not come from its own flags.

    Built once by :func:`main` and passed down, so no command module reads ``sys.argv``, the
    environment, or a global. That is what lets every command be called directly from a test with
    a fake settings object and a :class:`~mcmanager.clock.ManualClock`.
    """

    settings: Settings
    url: str
    token: str | None
    json_output: bool
    palette: render.Palette
    clock: Clock
    actor: str

    @property
    def color(self) -> bool:
        return self.palette.enabled


# ------------------------------------------------------------------------------------- parser


def _add_global_options(parser: argparse.ArgumentParser) -> None:
    """Options accepted both before and after the subcommand.

    Every default is :data:`argparse.SUPPRESS`. Without that, a subparser that repeats an option
    would overwrite the top-level parser's value with its own default, and ``mcmanager --json
    status`` would silently print a table. The real defaults are set once, on the root parser.
    """
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="config file (default: $MCM_CONFIG_FILE, then config/mcmanager.toml)",
    )
    parser.add_argument(
        "--url",
        metavar="URL",
        default=argparse.SUPPRESS,
        help=f"control server (default: $MCMANAGER_URL, then web.url, then {DEFAULT_URL})",
    )
    parser.add_argument(
        "--token",
        metavar="TOKEN",
        default=argparse.SUPPRESS,
        help=f"bearer token for the control endpoints (default: ${TOKEN_ENV}, then web.token)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="emit JSON instead of a rendered view; skips the renderer entirely",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=argparse.SUPPRESS,
        help="never emit ANSI colour (also honoured: NO_COLOR, and any non-tty stdout)",
    )


def build_parser() -> argparse.ArgumentParser:
    """The whole command line, in one readable block."""
    common = argparse.ArgumentParser(add_help=False)
    _add_global_options(common)

    parser = argparse.ArgumentParser(
        prog="mcmanager",
        description="Event-driven Docker-native manager for game servers, starting with Minecraft.",
        epilog=(
            "Standalone commands (inspect, replay, sessions, logs without --follow) need no "
            "daemon: set DOCKER_HOST=ssh://user@host to read a remote box directly."
        ),
    )
    _add_global_options(parser)
    parser.set_defaults(config=None, url=None, token=None, json=False, no_color=False)
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    subparsers.add_parser("run", parents=[common], help="run the daemon")
    subparsers.add_parser(
        "check-config",
        parents=[common],
        help="validate and print the resolved config; exit 0 or 78",
    )

    inspect_cmd = subparsers.add_parser(
        "inspect",
        parents=[common],
        help="raw container truth from Docker, with health derived (no daemon needed)",
    )
    inspect_cmd.add_argument(
        "container", nargs="?", help="container name (default: server.container)"
    )

    status_cmd = subparsers.add_parser(
        "status", parents=[common], help="server state, players, session, idle countdown"
    )
    status_cmd.add_argument(
        "--standalone",
        action="store_true",
        help="skip the daemon and build the view from an inspect plus a status probe",
    )

    players_cmd = subparsers.add_parser("players", parents=[common], help="who is online")
    players_cmd.add_argument("--standalone", action="store_true", help="skip the daemon")

    events_cmd = subparsers.add_parser(
        "events", parents=[common], help="live tail of the event bus (needs the daemon)"
    )
    events_cmd.add_argument("-f", "--follow", action="store_true", help="keep streaming")
    events_cmd.add_argument(
        "--type",
        action="append",
        default=[],
        metavar="NAME",
        help="filter by event type; repeatable, comma separated, accepts PlayerEvent",
    )
    events_cmd.add_argument("--since", metavar="WHEN", help="RFC3339 timestamp or an age like 15m")
    events_cmd.add_argument("--seq", type=int, metavar="N", help="only events after this seq")
    events_cmd.add_argument("--limit", type=int, metavar="N", help="stop after N events")
    events_cmd.add_argument(
        "--idle-timeout",
        type=float,
        default=1.5,
        metavar="SECONDS",
        help="without --follow, stop once the stream has been quiet this long",
    )

    logs_cmd = subparsers.add_parser("logs", parents=[common], help="the server console")
    logs_cmd.add_argument("-f", "--follow", action="store_true", help="attach to the daemon")
    logs_cmd.add_argument("-n", "--tail", type=int, default=100, metavar="N", help="lines to show")
    logs_cmd.add_argument("--since", metavar="WHEN", help="RFC3339 timestamp or an age like 15m")
    logs_cmd.add_argument(
        "--raw",
        action="store_true",
        help="bypass the parser and print the original line, ANSI and all",
    )
    logs_cmd.add_argument("--container", help="container name (default: server.container)")

    replay_cmd = subparsers.add_parser(
        "replay", parents=[common], help="run a .log or .log.gz through the parser offline"
    )
    replay_cmd.add_argument("file", help="a .log or .log.gz file, or - for stdin")
    replay_cmd.add_argument("--quiet", action="store_true", help="summary only, no event stream")
    replay_cmd.add_argument("--no-summary", action="store_true", help="event stream only")
    replay_cmd.add_argument(
        "--json-array",
        action="store_true",
        help="emit one JSON array instead of one object per line (fixture generation)",
    )
    replay_cmd.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        metavar="RATIO",
        help="unrecognised-line ratio considered acceptable (default: 0.01)",
    )
    replay_cmd.add_argument(
        "--strict", action="store_true", help="exit non-zero when the ratio exceeds --threshold"
    )
    replay_cmd.add_argument("--server-id", default="replay", help="server_id stamped on events")

    sessions_cmd = subparsers.add_parser(
        "sessions", parents=[common], help="archived session records from state.dir"
    )
    sessions_cmd.add_argument("--id", dest="session_id", help="show one session in full")
    sessions_cmd.add_argument("--limit", type=int, default=20, metavar="N")
    sessions_cmd.add_argument("--all", action="store_true", help="no limit")

    for action in ("start", "stop", "restart"):
        control_cmd = subparsers.add_parser(
            action, parents=[common], help=f"{action} the server (needs the daemon)"
        )
        control_cmd.add_argument("--actor", help="who to record in the audit trail")
        control_cmd.add_argument("--reason", help="free text recorded with the command")

    return parser


# ------------------------------------------------------------------------------------ helpers


def _flag(args: argparse.Namespace, name: str, default: object = None) -> Any:  # noqa: ANN401
    """Read one parsed option.

    ``argparse.Namespace`` is dynamically typed by construction, so this is the single place the
    project accepts an ``Any``: every call site immediately narrows it with ``bool()``, ``int()``
    or a ``cast``, and confining the looseness to one three-line function is better than sprinkling
    ``getattr`` through eight command branches.
    """
    return cast("Any", getattr(args, name, default))


def _default_actor() -> str:
    """Who to attribute a control command to when ``--actor`` was not given."""
    for key in ("MCMANAGER_ACTOR", "USER", "USERNAME"):
        value = os.environ.get(key)
        if value:
            return f"{value} (cli)"
    return "cli"


def _build_context(args: argparse.Namespace) -> CliContext:
    """Load settings and resolve the endpoint. Raises ``ConfigError`` (exit 78) on a bad config."""
    from mcmanager.config import load_settings

    settings = load_settings(config_file=_flag(args, "config"))
    token = _flag(args, "token") or os.environ.get(TOKEN_ENV)
    if not token and settings.web.token is not None:
        token = settings.web.token.get_secret_value()
    return CliContext(
        settings=settings,
        url=resolve_url(
            flag=_flag(args, "url"),
            env=os.environ,
            configured=settings.web.url,
        ),
        token=token or None,
        json_output=bool(_flag(args, "json", default=False)),
        palette=render.Palette.for_stream(no_color=bool(_flag(args, "no_color", default=False))),
        clock=SystemClock(),
        actor=_flag(args, "actor") or _default_actor(),
    )


def _warn_once(settings: Settings) -> None:
    """Print the config's non-fatal warnings to stderr, so ``--json`` output stays clean."""
    for warning in settings.warnings:
        render.emit_error(f"warning: {warning}")


def _quiet_logging(ctx: CliContext) -> None:
    """Send every log line to **stderr** and raise the floor to WARNING.

    Two reasons, and the first is the one that bites:

    - structlog's unconfigured default writes to **stdout**. One ``bus.subscribed`` debug line in
      the middle of ``mcmanager events --json | jq`` corrupts the stream, and the failure looks
      like a serialisation bug rather than a logging one.
    - A CLI is not a daemon. ``mcmanager status`` should print a status, not a boot log; anything
      the user needs to know about is already rendered.

    ``mcmanager run`` does **not** come through here: the daemon configures logging from
    ``[logs]`` itself, where the level and the format are the operator's to choose.
    """
    configure_logging(level="WARNING", fmt="console", stream=sys.stderr, clock=ctx.clock)


# ----------------------------------------------------------------------------------- dispatch


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (default ``sys.argv[1:]``), run the subcommand, return an exit code.

    Every deliberate failure in this project descends from
    :class:`~mcmanager.errors.McManagerError` and carries its own exit code, so this function is
    the single place those become a process status. Anything else propagates with a traceback,
    because an unexpected exception is a bug and hiding it behind ``exit 1`` helps nobody.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    command = cast("str | None", args.command)
    if command is None:
        parser.print_help()
        return EXIT_OK

    try:
        return _dispatch(command, args)
    except McManagerError as exc:
        render.emit_error(str(exc))
        return exc.exit_code
    except KeyboardInterrupt:
        render.emit_error("interrupted")
        return _INTERRUPTED


def _dispatch(command: str, args: argparse.Namespace) -> int:
    """Route to one command module, importing it only now. See the module docstring."""
    if command == "replay":
        # The purity check: no config, no clock, no context. Only the parser.
        from mcmanager.cli import cmd_replay

        return cmd_replay.run(
            source=_flag(args, "file"),
            json_output=bool(_flag(args, "json", default=False)),
            json_array=bool(_flag(args, "json_array", default=False)),
            quiet=bool(_flag(args, "quiet", default=False)),
            summary=not bool(_flag(args, "no_summary", default=False)),
            threshold=float(_flag(args, "threshold", default=0.01)),
            strict=bool(_flag(args, "strict", default=False)),
            server_id=str(_flag(args, "server_id", default="replay")),
            palette=render.Palette.for_stream(
                no_color=bool(_flag(args, "no_color", default=False))
            ),
        )

    ctx = _build_context(args)

    if command == "run":
        return _run_daemon(ctx)

    _quiet_logging(ctx)
    if command == "check-config":
        return _check_config(ctx)

    _warn_once(ctx.settings)

    if command == "inspect":
        from mcmanager.cli import cmd_inspect

        return asyncio.run(cmd_inspect.run(ctx, container=_flag(args, "container")))
    if command in ("status", "players"):
        from mcmanager.cli import cmd_status

        return asyncio.run(
            cmd_status.run(
                ctx,
                players_only=command == "players",
                standalone=bool(_flag(args, "standalone", default=False)),
            )
        )
    if command == "events":
        from mcmanager.cli import cmd_events

        return asyncio.run(
            cmd_events.run(
                ctx,
                follow=bool(_flag(args, "follow", default=False)),
                types=list(cast("list[str]", _flag(args, "type", default=[]))),
                since=_flag(args, "since"),
                seq=_flag(args, "seq"),
                limit=_flag(args, "limit"),
                idle_timeout=float(_flag(args, "idle_timeout", default=1.5)),
            )
        )
    if command == "logs":
        from mcmanager.cli import cmd_logs

        return asyncio.run(
            cmd_logs.run(
                ctx,
                follow=bool(_flag(args, "follow", default=False)),
                tail=int(_flag(args, "tail", default=100)),
                since=_flag(args, "since"),
                raw=bool(_flag(args, "raw", default=False)),
                container=_flag(args, "container"),
            )
        )
    if command == "sessions":
        from mcmanager.cli import cmd_sessions

        limit = None if bool(_flag(args, "all", default=False)) else int(_flag(args, "limit", 20))
        return cmd_sessions.run(ctx, session_id=_flag(args, "session_id"), limit=limit)
    if command in ("start", "stop", "restart"):
        from mcmanager.cli import cmd_control

        return asyncio.run(cmd_control.run(ctx, action=command, reason=_flag(args, "reason")))

    render.emit_error(f"unknown command {command!r}")  # pragma: no cover - argparse rejects first
    return EXIT_INTERNAL


# ------------------------------------------------------------------------- inline subcommands


def _check_config(ctx: CliContext) -> int:
    """``mcmanager check-config``: validate, resolve, print, exit 0.

    A bad config never reaches here - :func:`~mcmanager.config.load_settings` raises
    :class:`~mcmanager.errors.ConfigError`, which :func:`main` turns into exit 78 after printing
    *every* validation failure. This is the first line of the deploy runbook for that reason.
    """
    from mcmanager.config import redact, startup_banner

    if ctx.json_output:
        render.emit(json.dumps(redact(ctx.settings), indent=2, sort_keys=True, default=str))
    else:
        render.emit(startup_banner(ctx.settings))
    return EXIT_OK


def _run_daemon(ctx: CliContext) -> int:
    """``mcmanager run``: hand off to the composition root in :mod:`mcmanager.app`.

    Imported inside the branch rather than at module scope, like every other command module:
    ``app`` pulls in aiohttp, docker and the whole service graph, and ``mcmanager replay`` has to
    stay provably free of all of it (see the module docstring and ``TestReplayIsStandalone``).

    The symbol is bound directly rather than fished out with ``getattr``, so pyright type-checks
    this seam. ``app.run`` owns the only ``asyncio.run`` call in the daemon and returns the
    process exit code already translated from :class:`~mcmanager.errors.McManagerError`.
    """
    from mcmanager.app import run as run_daemon

    return run_daemon(ctx.settings)
