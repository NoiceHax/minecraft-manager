"""Shared fixtures.

Two things happen here that shape every test in the suite:

1. **:class:`~mcmanager.clock.ManualClock` is the default clock.** A test that needs to observe a
   15-minute idle timeout advances virtual time and finishes in about a millisecond.
2. **Outbound network access is blocked by an autouse fixture** unless the test is marked
   ``@pytest.mark.live``. A stray real connection is then caught on day one rather than in CI six
   weeks later, and the live tests remain explicitly opt-in.
"""

from __future__ import annotations

import asyncio
import os
import socket
from typing import TYPE_CHECKING, Any, cast

import pytest

from mcmanager.clock import ManualClock

if TYPE_CHECKING:
    from collections.abc import Iterator

_REAL_SOCKET = socket.socket
_REAL_CREATE_CONNECTION = socket.create_connection

# Loopback has to stay open. On Windows the default ProactorEventLoop builds its self-pipe with
# socket.socketpair(), which is emulated as a real TCP connect to 127.0.0.1 - so a guard that
# refuses every connection outright would make asyncio itself unusable and the entire suite would
# fail for a reason that has nothing to do with the tests. Blocking everything that leaves the
# machine is the property that actually matters here.
_ALLOWED_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "0.0.0.0", "", "<broadcast>"})  # noqa: S104


class NetworkAccessInTestError(RuntimeError):
    """A test tried to open a non-loopback connection without ``@pytest.mark.live``."""


def _check_address(address: Any) -> None:
    host: str | None = None
    if isinstance(address, tuple):
        parts = cast("tuple[object, ...]", address)
        if parts:
            host = str(parts[0])
    elif isinstance(address, str | bytes):
        host = address.decode(errors="replace") if isinstance(address, bytes) else address
    if host is not None and host in _ALLOWED_HOSTS:
        return
    msg = (
        f"blocked network access to {address!r} during a test. "
        "Fake the ContainerRuntime / GameAdapter interface instead, or mark the test "
        "@pytest.mark.live (which also requires MCMANAGER_LIVE=1)."
    )
    raise NetworkAccessInTestError(msg)


class _GuardedSocket(_REAL_SOCKET):
    """A socket that refuses to leave the machine."""

    def connect(self, address: Any, /) -> None:
        _check_address(address)
        super().connect(address)

    def connect_ex(self, address: Any, /) -> int:
        _check_address(address)
        return super().connect_ex(address)


def _guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
    _check_address(address)
    return _REAL_CREATE_CONNECTION(address, *args, **kwargs)


@pytest.fixture(autouse=True)
def block_network(request: pytest.FixtureRequest) -> Iterator[None]:
    """Refuse non-loopback connections for every test that is not marked ``live``.

    ``@pytest.mark.live`` additionally requires ``MCMANAGER_LIVE=1``; without it the test is
    skipped rather than allowed through, so a CI run can never touch the homelab by accident.
    """
    # pytest does not annotate FixtureRequest.node, so pyright strict sees it as unknown.
    node = cast("pytest.Item", request.node)  # pyright: ignore[reportUnknownMemberType]
    if node.get_closest_marker("live") is not None:
        if os.environ.get("MCMANAGER_LIVE") != "1":
            pytest.skip("live test: set MCMANAGER_LIVE=1 to run against the real homelab")
        yield
        return

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(socket, "socket", _GuardedSocket)
    monkeypatch.setattr(socket, "create_connection", _guarded_create_connection)
    try:
        yield
    finally:
        monkeypatch.undo()


@pytest.fixture
def manual_clock() -> ManualClock:
    """A clock that only moves when the test says so.

    Starts at 2026-01-01T00:00:00Z: tz-aware, stable across runs, and obviously synthetic in
    assertion output.
    """
    return ManualClock()


@pytest.fixture
def clock(manual_clock: ManualClock) -> ManualClock:
    """Alias for :func:`manual_clock`, for readability at injection sites."""
    return manual_clock


@pytest.fixture
async def loop() -> asyncio.AbstractEventLoop:
    """The loop the current test is running on.

    Provided instead of redefining pytest-asyncio's ``event_loop`` fixture, which is deprecated
    and would be an error under this project's ``filterwarnings = ["error"]``.
    """
    return asyncio.get_running_loop()
