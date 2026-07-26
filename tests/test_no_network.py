"""The network guard guards.

If this file ever goes quiet, a test somewhere is free to reach the real homelab, and the suite
stops being reproducible.
"""

from __future__ import annotations

import asyncio
import socket

import pytest


def test_an_outbound_connection_is_refused() -> None:
    with pytest.raises(RuntimeError, match="blocked network access"), socket.socket() as sock:
        sock.connect(("8.8.8.8", 53))


def test_create_connection_is_refused_too() -> None:
    with pytest.raises(RuntimeError, match="blocked network access"):
        socket.create_connection(("example.invalid", 80), timeout=0.1)


def test_the_docker_socket_is_refused() -> None:
    """A test that reaches the real Docker daemon is a live test, and must say so."""
    family = getattr(socket, "AF_UNIX", None)
    if family is None:
        pytest.skip("no AF_UNIX on this platform")
    with pytest.raises(RuntimeError, match="blocked network access"), socket.socket(family) as sock:
        sock.connect("/var/run/docker.sock")


async def test_the_event_loop_still_works_under_the_guard() -> None:
    """Loopback stays open on purpose: asyncio's own self-pipe needs it on Windows."""
    await asyncio.sleep(0)
    assert asyncio.get_running_loop() is not None
