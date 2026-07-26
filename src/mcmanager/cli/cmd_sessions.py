"""``mcmanager sessions`` - list and show archived session records.

Reads ``state.dir`` directly, so it works with the daemon stopped, which is exactly when somebody
wants to know what the last session looked like. Records marked ``partial`` are labelled: a summary
whose counts are incomplete must say so rather than quietly under-reporting somebody's playtime.
The docker log driver is a 30MB ring, so a long session genuinely **cannot** be reconstructed after
the fact - which is why these records are load-bearing and why honesty about their gaps matters.

**On-disk layout, which the session manager (M5) owns and this module only reads:**

    <state.dir>/sessions/*.json      one object per file
    <state.dir>/sessions.jsonl       one object per line, appended

Both are accepted, because the writer has not landed yet and the reader should not be the thing
that constrains it. Decoding is lenient about missing keys and strict about wrong types, mirroring
:class:`~mcmanager.persistence.state_store.StateStore`'s rule that reading is forgiving and writing
is not: a reader that refuses a record it half understands loses history nothing can regenerate.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from mcmanager.cli import render
from mcmanager.control.views import SessionView
from mcmanager.core.serde import SerdeError
from mcmanager.errors import EXIT_OK, EXIT_UNAVAILABLE

if TYPE_CHECKING:
    from pathlib import Path

    from mcmanager.cli.main import CliContext

__all__ = ["load_sessions", "run"]

_SESSIONS_DIR = "sessions"
_SESSIONS_JSONL = "sessions.jsonl"


def run(ctx: CliContext, *, session_id: str | None = None, limit: int | None = 20) -> int:
    """List sessions, or show one in full.

    Synchronous: this reads files and nothing else. Returns 69 only when a specific ``--id`` was
    asked for and is not there - an empty list is not an error, it is a daemon that has not run.
    """
    directory = ctx.settings.state.dir
    sessions = load_sessions(directory)

    if session_id is not None:
        found = next((item for item in sessions if item.id == session_id), None)
        if found is None:
            render.emit_error(
                f"no session {session_id!r} under {directory}. "
                "Run `mcmanager sessions` to list what is there."
            )
            return EXIT_UNAVAILABLE
        if ctx.json_output:
            render.emit(json.dumps(found.to_dict(), indent=2, sort_keys=True))
        else:
            render.emit(render.render_session(found, now=ctx.clock.now(), palette=ctx.palette))
        return EXIT_OK

    shown = sessions if limit is None else sessions[:limit]
    if ctx.json_output:
        render.emit(
            json.dumps(
                {"count": len(shown), "sessions": [item.to_dict() for item in shown]},
                indent=2,
                sort_keys=True,
            )
        )
        return EXIT_OK

    if not sessions and not directory.exists():
        render.emit(
            f"no session records: {directory} does not exist yet. "
            "The daemon creates it on its first run."
        )
        return EXIT_OK
    render.emit(render.render_sessions(shown, now=ctx.clock.now(), palette=ctx.palette))
    return EXIT_OK


def load_sessions(state_dir: Path) -> list[SessionView]:
    """Read every session record under ``state_dir``, newest first.

    A record that cannot be decoded is reported on stderr and skipped, never fatal. One corrupt
    file - a torn write during a power cut - must not hide the other forty.
    """
    found: list[SessionView] = []

    directory = state_dir / _SESSIONS_DIR
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            payload = _read_object(path)
            decoded = None if payload is None else _decode(payload, source=str(path))
            if decoded is not None:
                found.append(decoded)

    stream = state_dir / _SESSIONS_JSONL
    if stream.is_file():
        for number, line in enumerate(_read_lines(stream), start=1):
            if not line.strip():
                continue
            where = f"{stream}:{number}"
            payload = _decode_line(line, source=where)
            decoded = None if payload is None else _decode(payload, source=where)
            if decoded is not None:
                found.append(decoded)

    # Newest first. Records with no start time sort last rather than breaking the comparison.
    found.sort(key=lambda item: (item.started_at is not None, item.started_at), reverse=True)
    return found


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        render.emit_error(f"skipping {path}: {exc}")
        return []


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        render.emit_error(f"skipping {path}: {exc}")
        return None
    return _decode_line(text, source=str(path))


def _decode_line(text: str, *, source: str) -> dict[str, Any] | None:
    try:
        decoded: object = json.loads(text)
    except ValueError as exc:
        render.emit_error(f"skipping {source}: not valid JSON ({exc})")
        return None
    if not isinstance(decoded, dict):
        render.emit_error(f"skipping {source}: expected a JSON object")
        return None
    return cast("dict[str, Any]", decoded)


def _decode(payload: dict[str, Any], *, source: str) -> SessionView | None:
    try:
        return SessionView.from_dict(payload)
    except SerdeError as exc:
        render.emit_error(f"skipping {source}: {exc}")
        return None
