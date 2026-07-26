"""``mcmanager replay <file>`` - run a log through the parser offline.

Accepts ``.log`` and ``.log.gz`` (and ``-`` for stdin), needs no daemon, no Docker, no network and
no config. Prints the resulting event stream and reports the unrecognised-line ratio.

Three jobs in one command:

- it is how the parser is developed at all;
- it is the fixture generator: ``mcmanager replay --json-array run.log > expected/run.json``;
- it is the corroboration step - replay all 36 real archives, check the event distribution is
  plausible, the unrecognised ratio is under threshold, and nothing raised.

**Its existence is the proof of the parser's purity contract.** If ``replay`` ever needs a clock, a
bus or a config, the contract in :mod:`mcmanager.games.minecraft.parser` has been broken.
``main.py`` imports this module inside its own dispatch branch, and ``tests/cli/test_main.py``
asserts afterwards that ``sys.modules`` holds no ``docker``, no ``aiohttp``, no ``discord`` and no
``mcmanager.config`` - which turns the claim into a test rather than a comment.

The ratio is the canary for a Paper upgrade breaking the patterns. A parser that stops matching
does not crash; it goes quiet. ``--strict`` turns that from a printed number into a non-zero exit,
which is how it belongs in CI.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from mcmanager.cli import render
from mcmanager.errors import EXIT_CONFIG, EXIT_OK

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mcmanager.games.minecraft.parser import ParseStats

__all__ = ["REPLAY_TS", "read_lines", "run"]

REPLAY_TS: Final = datetime(1970, 1, 1, tzinfo=UTC)
"""The timestamp every replayed event carries.

The parser takes ``ts`` as an argument and never reads a clock - that is the contract - and a
replay measures *classification*, not chronology. A fixed epoch keeps the output reproducible, so
a golden file regenerated on another machine is byte-identical. The real timestamps come from
Docker's log prefix, which a bare ``.log`` archive does not have in the first place.
"""

_OVER_THRESHOLD: Final = 1
"""Exit code for ``--strict`` when the ratio is too high. Plain 1: this is a CI assertion failing,
not a sysexits condition, and a build system reads any non-zero the same way."""


def run(
    *,
    source: str,
    json_output: bool = False,
    json_array: bool = False,
    quiet: bool = False,
    summary: bool = True,
    threshold: float = 0.01,
    strict: bool = False,
    server_id: str = "replay",
    palette: render.Palette | None = None,
) -> int:
    """Parse ``source``, print the events, then the summary.

    Synchronous on purpose. There is no I/O here beyond reading one file, and wrapping it in
    ``asyncio.run`` would imply a concurrency this command neither has nor wants.
    """
    from mcmanager.core.serde import event_to_dict, event_to_json
    from mcmanager.core.types import Stream
    from mcmanager.games.minecraft import parser

    pal = palette if palette is not None else render.Palette.plain()
    try:
        lines = list(read_lines(source))
    except OSError as exc:
        render.emit_error(f"could not read {source}: {exc}")
        return EXIT_CONFIG

    machine_readable = json_output or json_array
    if not quiet:
        collected: list[dict[str, Any]] = []
        for raw in lines:
            event = parser.parse(raw, ts=REPLAY_TS, server_id=server_id, stream=Stream.STDOUT)
            if json_array:
                collected.append(event_to_dict(event))
            elif json_output:
                render.emit(event_to_json(event))
            else:
                render.emit(render.render_event(event, palette=pal))
        if json_array:
            render.emit(json.dumps(collected, indent=2, sort_keys=True, ensure_ascii=False))

    stats = parser.scan(lines, server_id=server_id)
    if summary:
        if machine_readable:
            # The block form would corrupt a redirected fixture file, so the JSON paths get the
            # summary as JSON on **stderr** and the stream stays clean on stdout.
            render.emit_error(json.dumps(_stats_to_dict(stats, threshold), sort_keys=True))
        else:
            if not quiet:
                render.emit("")
            render.emit(
                render.render_replay_summary(stats, source=source, palette=pal, threshold=threshold)
            )

    if strict and stats.unrecognised_ratio > threshold:
        render.emit_error(
            f"unrecognised-line ratio {stats.unrecognised_ratio:.4f} exceeds {threshold:.4f}. "
            "Paper's log format has probably drifted: read the samples above, then update "
            "games/minecraft/patterns.py."
        )
        return _OVER_THRESHOLD
    return EXIT_OK


def read_lines(source: str) -> Iterator[str]:
    """Yield lines from a ``.log``, a ``.log.gz``, or stdin when ``source`` is ``-``.

    Decoded with ``errors="replace"``: real archives contain bytes that are not valid UTF-8 - a
    multibyte sequence truncated at a rollover boundary, or a plugin writing latin-1 - and a
    replay tool that dies on one bad byte is useless exactly when it is needed. The parser is
    total anyway, so a replacement character becomes a ``ConsoleLog`` and the run continues.
    """
    if source == "-":
        yield from (line.rstrip("\n") for line in sys.stdin)
        return

    path = Path(source)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                yield line.rstrip("\n")
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line.rstrip("\n")


def _stats_to_dict(stats: ParseStats, threshold: float) -> dict[str, Any]:
    """The machine-readable summary, for a CI job that wants the number rather than the block."""
    return {
        "total": stats.total,
        "server_lines": stats.server_lines,
        "console_lines": stats.console_lines,
        "known_players": sorted(stats.known_players),
        "player_lines": stats.player_lines,
        "unrecognised_player_lines": stats.unrecognised_player_lines,
        "unrecognised_ratio": stats.unrecognised_ratio,
        "console_ratio": stats.console_ratio,
        "threshold": threshold,
        "over_threshold": stats.unrecognised_ratio > threshold,
        "by_event": dict(stats.by_event),
        "samples": list(stats.samples),
    }
