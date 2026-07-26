"""Fixtures for the CLI tests.

Every fixture here is deliberately hermetic: a config file written into ``tmp_path``, the fake
container runtime, a :class:`~mcmanager.clock.ManualClock`, and a URL pointing at a port nothing
listens on. Nothing in ``tests/cli`` should depend on the developer's environment, and in
particular nothing should find ``config/mcmanager.toml`` by accident.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import structlog

from mcmanager.cli.main import CliContext
from mcmanager.cli.render import Palette
from mcmanager.config import get_settings, load_settings
from mcmanager.logging_setup import configure_logging

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from mcmanager.clock import ManualClock
    from mcmanager.config import Settings

UNREACHABLE_URL = "http://127.0.0.1:1"
"""Port 1 is privileged and nothing binds it, so a connection there fails immediately.

Loopback is what the autouse network guard allows, and a refused connection is exactly the
condition every "the daemon is not there" test needs.
"""


@pytest.fixture(autouse=True)
def isolate_settings_cache() -> Iterator[None]:
    """``get_settings`` is ``lru_cache``d process-wide; a stale entry would leak between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def logs_to_stderr() -> Iterator[None]:
    """Do what ``main()`` does: send log output to stderr.

    structlog's unconfigured default writes to **stdout**, and these tests assert on stdout - one
    ``bus.subscribed`` debug line in the middle of ``--json`` output is exactly the corruption
    :func:`mcmanager.cli.main._quiet_logging` exists to prevent. The command functions are called
    directly here rather than through ``main()``, so the same configuration is applied here.

    Reset afterwards so this cannot leak into ``tests/unit/test_logging_setup.py``.
    """
    configure_logging(level="WARNING", fmt="console", stream=sys.stderr)
    try:
        yield
    finally:
        structlog.reset_defaults()


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "state"
    directory.mkdir()
    return directory


@pytest.fixture
def config_file(tmp_path: Path, state_dir: Path) -> Path:
    """A minimal, valid config: fake runtime, everything under ``tmp_path``.

    ``runtime = "fake"`` is what lets these tests exercise the standalone commands with no Docker
    anywhere - the same switch that makes the whole daemon runnable offline on Windows.
    """
    path = tmp_path / "mcmanager.toml"
    path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'runtime = "fake"',
                "",
                "[server]",
                'id = "minecraft"',
                'container = "minecraft"',
                'host = "minecraft"',
                "",
                "[logs]",
                "archive_on_stop = false",
                f'archive_dir = "{(tmp_path / "archives").as_posix()}"',
                f'daemon_log_dir = "{(tmp_path / "logs").as_posix()}"',
                "",
                "[state]",
                f'dir = "{state_dir.as_posix()}"',
                "",
                "[web]",
                f'url = "{UNREACHABLE_URL}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def settings(config_file: Path) -> Settings:
    return load_settings(config_file=config_file)


@pytest.fixture
def ctx(settings: Settings, clock: ManualClock) -> CliContext:
    """The context every command module takes, wired for offline tests."""
    return CliContext(
        settings=settings,
        url=UNREACHABLE_URL,
        token=None,
        json_output=False,
        palette=Palette.plain(),
        clock=clock,
        actor="tester",
    )


@pytest.fixture
def json_ctx(ctx: CliContext) -> CliContext:
    """The same context with ``--json``, so both output paths are covered by one fixture pair."""
    return replace(ctx, json_output=True)
