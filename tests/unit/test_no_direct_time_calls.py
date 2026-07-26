"""Belt and braces for the Clock injection rule.

``pyproject.toml``'s ``flake8-tidy-imports.banned-api`` block already refuses ``asyncio.sleep``,
``time.sleep``, ``time.monotonic`` and ``datetime.now`` outside ``clock.py``. This test asserts the
same property from the other direction, by walking the AST of every module under ``src/mcmanager``.

Two reasons it is worth having both:

1. **Ruff's rule is import-shaped.** ``banned-api`` matches the qualified name a symbol was
   imported under. ``from datetime import datetime as dt`` then ``dt.now()``, or
   ``getattr(asyncio, "sleep")``, or a ``# noqa: TID251`` somebody added to unblock a merge, all
   slip past it. The AST walk matches the call site.
2. **The failure mode is expensive and silent.** One stray ``await asyncio.sleep(30)`` in a poll
   loop does not break anything visibly; it turns a millisecond test into a thirty-second one, and
   then somebody marks the test ``slow`` and the whole time-injection design quietly dies.

The rule is exactly the one in the plan: nothing outside ``src/mcmanager/clock.py`` may call
``asyncio.sleep``, ``time.sleep``, ``time.monotonic``, ``datetime.now`` or ``datetime.utcnow``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mcmanager"

CLOCK_MODULE = SRC_ROOT / "clock.py"
"""The one sanctioned home of every time primitive."""

BANNED_ATTRIBUTES: dict[str, str] = {
    "sleep": "Clock.sleep",
    "monotonic": "Clock.monotonic",
    "monotonic_ns": "Clock.monotonic",
    "now": "Clock.now",
    "utcnow": "Clock.now (which returns tz-aware UTC)",
    "today": "Clock.now",
    "time": "Clock.monotonic",
    "perf_counter": "Clock.monotonic",
}
"""Attribute names that are a time primitive when reached through ``asyncio``/``time``/``datetime``.

Keyed on the attribute rather than the full dotted path so that ``dt.now()``, ``t.monotonic()`` and
``asyncio.sleep()`` are all caught regardless of how the module was aliased.
"""

TIME_MODULE_ROOTS = frozenset({"asyncio", "time", "datetime", "dt", "aio"})
"""Names that, used as the receiver of one of the attributes above, mean the real thing.

Kept deliberately small: an over-broad list would flag ``clock.now()`` and ``self._clock.sleep()``,
which are exactly the calls this whole design wants people to make.
"""

BARE_BANNED_NAMES = frozenset({"sleep", "monotonic", "utcnow", "perf_counter"})
"""``from time import sleep`` then ``sleep(30)``. The ``from`` import is caught separately, but the
call site is what actually costs thirty seconds, so it is checked too."""

ALLOWED_RECEIVER_HINTS = ("clock", "_clock", "self.clock", "self._clock")


class Violation(NamedTuple):
    """One banned call, with enough context to fix it without opening the file."""

    path: Path
    line: int
    call: str
    use_instead: str

    def __str__(self) -> str:
        relative = self.path.relative_to(SRC_ROOT.parent.parent)
        return f"{relative}:{self.line}: {self.call}() - use {self.use_instead} instead"


def _source_files() -> list[Path]:
    return sorted(path for path in SRC_ROOT.rglob("*.py") if path != CLOCK_MODULE)


def _dotted_name(node: ast.expr) -> str:
    """Render ``a.b.c`` from an attribute chain. Anything else renders as ``<expr>``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    return "<expr>"


def _root_name(node: ast.expr) -> str:
    return _dotted_name(node).split(".")[0]


def _iter_violations(path: Path) -> Iterator[Violation]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    # `from time import sleep` style imports: record what was pulled in bare, so the call-site
    # check below can tell `sleep(1)` (banned) from a locally defined `sleep` helper (not our
    # problem, and not something that exists in this codebase).
    bare_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"asyncio", "time", "datetime"}:
            for alias in node.names:
                if alias.name in BARE_BANNED_NAMES:
                    bare_imports.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        if isinstance(func, ast.Attribute):
            attribute = func.attr
            if attribute not in BANNED_ATTRIBUTES:
                continue
            receiver = _dotted_name(func.value)
            if any(hint in receiver for hint in ALLOWED_RECEIVER_HINTS):
                continue
            if _root_name(func.value) not in TIME_MODULE_ROOTS:
                continue
            yield Violation(
                path=path,
                line=node.lineno,
                call=f"{receiver}.{attribute}",
                use_instead=BANNED_ATTRIBUTES[attribute],
            )

        elif isinstance(func, ast.Name) and func.id in bare_imports:
            yield Violation(
                path=path,
                line=node.lineno,
                call=func.id,
                use_instead=BANNED_ATTRIBUTES.get(func.id, "the injected Clock"),
            )


def test_the_source_tree_is_where_we_think_it_is() -> None:
    """A path typo would make every assertion below vacuously true."""
    assert SRC_ROOT.is_dir(), SRC_ROOT
    assert CLOCK_MODULE.is_file(), CLOCK_MODULE
    assert len(_source_files()) > 20


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.name))
def test_module_makes_no_direct_time_calls(path: Path) -> None:
    """Parametrised per module so a failure names the file in the test id."""
    violations = list(_iter_violations(path))
    assert not violations, "\n".join(str(violation) for violation in violations)


def test_no_module_in_the_whole_tree_calls_a_time_primitive() -> None:
    """The same assertion as one report, which is what you want when several files drift at once."""
    violations = [violation for path in _source_files() for violation in _iter_violations(path)]
    assert not violations, (
        "these calls must go through the injected Clock "
        "(mcmanager.clock.Clock), which is the only reason a 15-minute idle "
        "timeout is testable in a millisecond:\n"
        + "\n".join(str(violation) for violation in violations)
    )


def test_clock_module_itself_does_call_them() -> None:
    """The detector has to actually detect something, or the whole file is theatre.

    ``clock.py`` is the one module allowed to make these calls, so it doubles as the fixture that
    proves the AST walk is not silently matching nothing.
    """
    found = list(_iter_violations(CLOCK_MODULE))
    names = {violation.call for violation in found}

    assert "asyncio.sleep" in names
    assert "time.monotonic" in names
    assert "datetime.now" in names


@pytest.mark.parametrize(
    "snippet",
    [
        "import asyncio\nasync def f():\n    await asyncio.sleep(30)\n",
        "import time\ndef f():\n    return time.monotonic()\n",
        "import time\ndef f():\n    time.sleep(1)\n",
        "from datetime import datetime\ndef f():\n    return datetime.now()\n",
        "from datetime import datetime\ndef f():\n    return datetime.utcnow()\n",
        "from datetime import datetime as dt\ndef f():\n    return dt.now()\n",
        "from time import sleep\ndef f():\n    sleep(5)\n",
        "from asyncio import sleep\nasync def f():\n    await sleep(5)\n",
    ],
)
def test_detector_catches_every_spelling(snippet: str, tmp_path: Path) -> None:
    """Including the aliased ones that ruff's import-shaped rule cannot see."""
    path = tmp_path / "sample.py"
    path.write_text(snippet, encoding="utf-8")
    assert list(_iter_violations(path)), snippet


@pytest.mark.parametrize(
    "snippet",
    [
        "async def f(clock):\n    await clock.sleep(30)\n",
        "class A:\n    async def f(self):\n        await self._clock.sleep(30)\n",
        "class A:\n    def f(self):\n        return self.clock.now()\n",
        "def f(manual_clock):\n    return manual_clock.monotonic()\n",
        "def f(snapshot, now):\n    return snapshot.uptime(now)\n",
    ],
)
def test_detector_does_not_flag_injected_clock_use(snippet: str, tmp_path: Path) -> None:
    """The false-positive half. A detector that flags ``clock.sleep()`` gets switched off."""
    path = tmp_path / "sample.py"
    path.write_text(snippet, encoding="utf-8")
    assert not list(_iter_violations(path)), snippet
