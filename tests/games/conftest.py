"""Fixtures for the Minecraft parser tests, and the golden-file machinery.

**Provenance of everything under ``tests/fixtures/logs/``.** Pulled read-only from the live
homelab on 2026-07-26, and committed with their **raw ANSI bytes intact** - that is the whole
point of them, since one of the three defects this project replaces is a parser that never
stripped those bytes and put them into Discord messages.

``archives/*.log``
    The 36 per-run log4j2 archives from ``~/homelab/data/minecraft/logs/*.log.gz`` plus the
    in-progress ``latest.log``, decompressed. Paper's grammar only: time-only timestamps, no
    date. 3,419 lines total.

``docker_stream.log``
    ``docker logs --timestamps minecraft``. Committed **as well as** the archives, not instead of
    them, because it is the only place two of the three grammars appear: the ``mc-server-runner``
    wrapper lines (``2026-07-25T22:58:20.809+0530\\tINFO\\tmc-server-runner\\tDone``) and the
    ``[init]`` / ``[mc-image-helper]`` entrypoint lines. A fixture set built from the ``.gz``
    files alone would test one grammar out of three and look complete.

``run_normal_session.log``, ``run_deaths_session.log``
    Two archives promoted to stable names because the goldens point at them: one with joins,
    chat, advancements and a clean shutdown, one with five real deaths.

Goldens are plain sorted JSON, compared byte for byte. No ``syrupy``: a snapshot library that
writes its own format is one more thing to be wrong about, and ``git diff`` on a JSON file is
already the review tool.

Regenerate with::

    MCMANAGER_UPDATE_GOLDENS=1 uv run pytest tests/games

An environment variable rather than a ``--update-goldens`` flag because ``pytest_addoption`` may
only be defined in the initial conftest, and ``tests/conftest.py`` belongs to another module's
owner.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from mcmanager.core.types import Stream
from mcmanager.games.minecraft.parser import parse

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mcmanager.core.events import Event

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "logs"
ARCHIVE_DIR = FIXTURE_DIR / "archives"
GOLDEN_DIR = FIXTURE_DIR / "expected"

GOLDEN_SOURCES = ("run_normal_session.log", "run_deaths_session.log", "docker_stream.log")
"""The fixtures that have committed goldens. The other 36 archives feed the ratio test."""

UPDATE_GOLDENS = os.environ.get("MCMANAGER_UPDATE_GOLDENS") == "1"

_DOCKER_TS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z) (?P<text>.*)$", re.DOTALL
)

_FIXED_TS = datetime(2026, 1, 1, tzinfo=UTC)
"""The timestamp goldens are generated with.

Fixed rather than taken from the line, because ``parse`` takes ``ts`` as an argument by contract
and a golden that embedded a real timestamp would be asserting something the parser does not do.
"""


def strip_docker_timestamp(line: str) -> str:
    """Remove Docker's RFC3339Nano prefix from a ``docker logs --timestamps`` line.

    In production this happens at the container boundary: ``LogLine.text`` never carries the
    prefix and ``LogLine.ts`` carries the parsed value. Reproduced here so the committed fixture
    can stay byte-identical to what ``docker logs --timestamps`` actually emitted, prefix and all.
    """
    found = _DOCKER_TS_RE.match(line)
    return found["text"] if found is not None else line


def read_lines(path: Path) -> list[str]:
    """Read a fixture, keeping every byte except the line terminator.

    ``errors="replace"`` because a log file is not guaranteed to be valid UTF-8 and a fixture
    loader that raises would hide the very robustness this suite is meant to prove.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if path.name == "docker_stream.log":
        return [strip_docker_timestamp(line) for line in lines]
    return lines


def archive_lines() -> list[str]:
    """Every line of every real archive, in filename order."""
    lines: list[str] = []
    for path in sorted(ARCHIVE_DIR.glob("*.log")):
        lines.extend(read_lines(path))
    return lines


def _jsonable(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return {str(k): _jsonable(v) for k, v in mapping.items()}
    if isinstance(value, list | tuple):
        sequence = cast("Sequence[object]", value)
        return [_jsonable(v) for v in sequence]
    return value


def event_to_dict(event: Event) -> dict[str, Any]:
    """A stable dict for one event.

    ``core/serde.py`` is the production serialiser and would be the right thing to use, but it is
    still a stub at this milestone. This is deliberately dumb and local: when serde lands, the
    goldens are what proves the two agree.
    """
    payload = {k: _jsonable(v) for k, v in dataclasses.asdict(event).items()}
    payload["type"] = event.name
    return payload


def render_golden(lines: list[str]) -> str:
    """Parse every line and render the canonical golden text."""
    events = [
        event_to_dict(parse(line, ts=_FIXED_TS, server_id="fixture", stream=Stream.STDOUT))
        for line in lines
    ]
    return json.dumps(events, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def assert_matches_golden(name: str, actual: str) -> None:
    """Compare against ``expected/<name>.json``, or rewrite it when updating."""
    golden = GOLDEN_DIR / f"{name}.json"
    if UPDATE_GOLDENS:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(actual, encoding="utf-8", newline="\n")
        return
    if not golden.exists():
        pytest.fail(
            f"missing golden {golden}. Regenerate with MCMANAGER_UPDATE_GOLDENS=1 pytest "
            f"tests/games"
        )
    expected = golden.read_text(encoding="utf-8")
    if expected != actual:
        pytest.fail(
            f"golden {golden.name} does not match.\n"
            f"If Paper changed its log format this is the signal; if the change is intended, "
            f"regenerate with MCMANAGER_UPDATE_GOLDENS=1 pytest tests/games and review the diff."
        )


@pytest.fixture(scope="session")
def all_archive_lines() -> list[str]:
    """Every line of the 36 committed archives. Session-scoped; it is read once."""
    return archive_lines()


@pytest.fixture(scope="session")
def load_fixture() -> Callable[[str], list[str]]:
    """``load_fixture("docker_stream.log")`` -> its lines, docker prefix already removed.

    Handed out as a fixture rather than imported, because a cross-module import between test files
    depends on pytest's import mode and the presence of ``__init__.py``, and that is a footgun
    nobody should have to think about to read a log file.
    """

    def _load(name: str) -> list[str]:
        return read_lines(FIXTURE_DIR / name)

    return _load


@pytest.fixture(scope="session")
def check_golden() -> Callable[[str, list[str]], None]:
    """``check_golden("docker_stream", lines)`` - parse, render, compare byte for byte."""

    def _check(name: str, lines: list[str]) -> None:
        assert_matches_golden(name, render_golden(lines))

    return _check
