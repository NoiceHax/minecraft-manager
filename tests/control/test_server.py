"""The runner and the ``app.py`` seam.

``build_control_surface`` is looked up **by name** from ``app.py`` rather than imported, so the two
halves can land independently. That makes the contract easy to break silently, and the tests here
are what stop that: the name exists, it is callable, and what it returns has the two methods the
``ControlSurface`` protocol asks for.
"""

from __future__ import annotations

import importlib
import socket
from typing import TYPE_CHECKING

import pytest

from mcmanager.control.routes import CONTEXT, ControlContext
from mcmanager.control.server import ControlServer, build_app, disabled_surface
from mcmanager.control.sse import SseChannel
from mcmanager.control.views import (
    LivenessView,
    SessionView,
    StatusView,
    evaluate_readiness,
)
from mcmanager.core.types import LifecycleState

if TYPE_CHECKING:
    from datetime import datetime

    from mcmanager.clock import ManualClock


def _no_sessions(limit: int) -> list[SessionView]:
    """A sessions provider with nothing to give."""
    del limit
    return []


def _context(clock: ManualClock, *, channel: SseChannel | None = None) -> ControlContext:
    now: datetime = clock.now()
    return ControlContext(
        server_id="minecraft",
        clock=clock,
        channel=channel if channel is not None else SseChannel(clock=clock),
        status=lambda: StatusView(
            server_id="minecraft",
            container="minecraft",
            state=LifecycleState.STOPPED,
            observed_at=now,
        ),
        players=lambda: (),
        sessions=_no_sessions,
        readiness=lambda: evaluate_readiness(runtime_available=True, log_stream_attached=True),
        liveness=lambda: LivenessView(),
    )


class TestBuildApp:
    def test_every_documented_route_is_mounted(self, clock: ManualClock) -> None:
        app = build_app(_context(clock))
        paths = {resource.canonical for resource in app.router.resources()}
        assert paths == {
            "/",
            "/healthz",
            "/readyz",
            "/status",
            "/players",
            "/sessions",
            "/events",
            "/logs",
            "/control/{action}",
        }

    def test_the_context_is_reachable_through_the_typed_app_key(self, clock: ManualClock) -> None:
        context = _context(clock)
        app = build_app(context)
        assert app[CONTEXT] is context

    def test_the_error_middleware_is_installed(self, clock: ManualClock) -> None:
        # Without it an unhandled handler exception answers with aiohttp's HTML error page, which
        # the CLI cannot decode into anything useful.
        middlewares = build_app(_context(clock)).middlewares
        names = {getattr(middleware, "__name__", "") for middleware in middlewares}
        assert "error_middleware" in names


class TestControlServer:
    async def test_start_binds_and_aclose_releases(self, clock: ManualClock) -> None:
        server = ControlServer(context=_context(clock), host="127.0.0.1", port=0)
        assert not server.running
        await server.start()
        try:
            assert server.running
        finally:
            await server.aclose()
        assert not server.running

    async def test_aclose_is_safe_twice(self, clock: ManualClock) -> None:
        server = ControlServer(context=_context(clock), host="127.0.0.1", port=0)
        await server.start()
        await server.aclose()
        await server.aclose()

    async def test_aclose_before_start_is_harmless(self, clock: ManualClock) -> None:
        await ControlServer(context=_context(clock), host="127.0.0.1", port=0).aclose()

    async def test_starting_twice_is_a_no_op(self, clock: ManualClock) -> None:
        server = ControlServer(context=_context(clock), host="127.0.0.1", port=0)
        await server.start()
        await server.start()
        await server.aclose()

    async def test_the_keepalive_task_is_started_and_then_cancelled(
        self, clock: ManualClock
    ) -> None:
        channel = SseChannel(clock=clock, keepalive_seconds=15.0)
        server = ControlServer(context=_context(clock, channel=channel), host="127.0.0.1", port=0)
        await server.start()
        client = channel.open()
        await clock.tick()
        await clock.advance(15.0)
        assert client.queued == 1, "the keepalive runs without the app's supervisor knowing"
        await server.aclose()
        # And it stops: after aclose nothing further is pushed, however far time moves.
        drained = client.queued
        await clock.advance(60.0)
        assert client.queued == drained

    async def test_closing_the_server_ends_every_stream(self, clock: ManualClock) -> None:
        channel = SseChannel(clock=clock)
        server = ControlServer(context=_context(clock, channel=channel), host="127.0.0.1", port=0)
        await server.start()
        client = channel.open()
        await server.aclose()
        assert client.closed

    async def test_a_taken_port_raises_rather_than_degrading(self, clock: ManualClock) -> None:
        """A daemon whose control port never came up looks healthy and is undebuggable.

        So :meth:`ControlServer.start` propagates the ``OSError`` instead of logging it and
        carrying on, and the runner it half-built is cleaned up on the way out.
        """
        port = _free_port()
        first = ControlServer(context=_context(clock), host="127.0.0.1", port=port)
        await first.start()
        second = ControlServer(context=_context(clock), host="127.0.0.1", port=port)
        try:
            with pytest.raises(OSError, match=r".*"):
                await second.start()
        finally:
            await second.aclose()
            await first.aclose()

    def test_the_url_reports_loopback_for_a_wildcard_bind(self, clock: ManualClock) -> None:
        server = ControlServer(context=_context(clock), host="0.0.0.0", port=8787)  # noqa: S104
        assert server.url == "http://127.0.0.1:8787"


def _free_port() -> int:
    """A loopback port nothing is using, released immediately so the server can claim it."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        chosen: int = probe.getsockname()[1]
    return chosen


class TestAppSeam:
    """``app.py`` resolves this factory by name; the name is the contract."""

    def test_the_factory_exists_under_the_name_app_py_looks_up(self) -> None:
        module = importlib.import_module("mcmanager.control.server")
        factory = getattr(module, "build_control_surface", None)
        assert callable(factory)

    def test_a_disabled_surface_satisfies_the_protocol_and_does_nothing(self) -> None:
        surface = disabled_surface()
        assert hasattr(surface, "start")
        assert hasattr(surface, "aclose")

    async def test_a_disabled_surface_starts_and_closes_without_binding(self) -> None:
        surface = disabled_surface()
        await surface.start()
        await surface.aclose()

    def test_importing_the_control_package_does_not_import_the_composition_root(self) -> None:
        """``AppServices`` is a type-only import; a runtime one would be a dependency cycle.

        ``app.py`` imports this module dynamically, so a module-level ``from mcmanager.app import
        ...`` here would either loop or drag Discord and the services layer into every CLI
        invocation.
        """
        import ast
        from pathlib import Path

        source = Path(importlib.import_module("mcmanager.control.server").__file__ or "")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        runtime_imports: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                runtime_imports.add(node.module)
            elif isinstance(node, ast.Import):
                runtime_imports.update(alias.name for alias in node.names)
        assert "mcmanager.app" not in runtime_imports
