"""The HTTP surface, exercised against a real aiohttp server.

Not a mock: the interesting behaviour is in status codes, headers, the auth handshake and SSE
framing, and a mock of ``web.Request`` would encode this file's assumptions about aiohttp rather
than test them. The loopback bind is allowed by the autouse network guard in ``tests/conftest.py``.

The two properties worth naming up front:

- **``/healthz`` never consults Docker.** A dead socket must not restart the daemon.
- **``/readyz`` says which subsystem is red**, in both the 200 and the 503 case.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from mcmanager.control.routes import ControlContext, outcome_to_dict
from mcmanager.control.server import build_app
from mcmanager.control.sse import SseChannel
from mcmanager.control.views import (
    LivenessView,
    PlayerView,
    SessionView,
    StatusView,
    evaluate_readiness,
)
from mcmanager.core.bus import EventBus
from mcmanager.core.events import ConsoleLog, PlayerJoined
from mcmanager.core.types import ControlAction, LifecycleState, PlayerRef, Source
from mcmanager.services.controller import ControlOutcome

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from mcmanager.clock import ManualClock

NOW = datetime(2026, 7, 25, 22, 58, 20, tzinfo=UTC)
type _Client = TestClient[web.Request, web.Application]
TOKEN = "s3cret-token"  # noqa: S105 - a literal for a test server, not a credential


class RecordingController:
    """Satisfies ``routes.Controller`` structurally, exactly as ``ServerController`` does."""

    def __init__(self, outcome: ControlOutcome | None = None) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self._outcome = outcome

    def _result(self, action: ControlAction, actor: str) -> ControlOutcome:
        if self._outcome is not None:
            return self._outcome
        return ControlOutcome(
            action=action,
            actor=actor,
            accepted=True,
            state_before=LifecycleState.STOPPED,
        )

    async def start(self, *, actor: str, via: Source, reason: str | None) -> ControlOutcome:
        self.calls.append(("start", actor, reason))
        assert via is Source.WEB
        return self._result(ControlAction.START, actor)

    async def stop(self, *, actor: str, via: Source, reason: str | None) -> ControlOutcome:
        self.calls.append(("stop", actor, reason))
        return self._result(ControlAction.STOP, actor)

    async def restart(self, *, actor: str, via: Source, reason: str | None) -> ControlOutcome:
        self.calls.append(("restart", actor, reason))
        return self._result(ControlAction.RESTART, actor)


def _status() -> StatusView:
    return StatusView(
        server_id="minecraft",
        container="minecraft",
        state=LifecycleState.READY,
        observed_at=NOW,
        running=True,
        players_online=1,
        roster=(PlayerView(name="Steve", online_since=NOW, session_seconds=60.0),),
    )


def _context(
    clock: ManualClock,
    *,
    channel: SseChannel | None = None,
    controller: RecordingController | None = None,
    token: str | None = TOKEN,
    sessions: Sequence[SessionView] = (),
    liveness: LivenessView | None = None,
    ready: bool = True,
) -> ControlContext:
    return ControlContext(
        server_id="minecraft",
        clock=clock,
        channel=channel if channel is not None else SseChannel(clock=clock),
        status=_status,
        players=lambda: _status().roster,
        sessions=lambda limit: list(sessions)[:limit],
        readiness=lambda: evaluate_readiness(
            runtime_available=ready,
            log_stream_attached=ready,
        ),
        liveness=lambda: liveness if liveness is not None else LivenessView(tasks=4),
        controller=controller,
        token=token,
    )


async def _client(context: ControlContext) -> AsyncIterator[_Client]:
    server = TestServer(build_app(context))
    client = TestClient(server)
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
async def context(clock: ManualClock) -> ControlContext:
    return _context(clock)


@pytest.fixture
async def client(
    context: ControlContext,
) -> AsyncIterator[_Client]:
    async for made in _client(context):
        yield made


class TestHealthEndpoints:
    async def test_healthz_answers_200_and_a_body(self, client: _Client) -> None:
        response = await client.get("/healthz")
        assert response.status == 200
        body: dict[str, Any] = await response.json()
        assert body["alive"] is True
        assert body["tasks"] == 4

    async def test_healthz_goes_red_only_for_the_supervisor(self, clock: ManualClock) -> None:
        # Liveness is "should I restart this process". A dead Docker socket is not that: restarting
        # fixes nothing and crash-loops past the real fault.
        context = _context(
            clock,
            liveness=LivenessView(alive=True, supervisor_ok=False, detail="bus dispatch died"),
            ready=False,
        )
        async for client in _client(context):
            response = await client.get("/healthz")
            assert response.status == 503
            body: dict[str, Any] = await response.json()
            assert body["detail"] == "bus dispatch died"

    async def test_healthz_stays_green_while_readyz_is_red(self, clock: ManualClock) -> None:
        context = _context(clock, ready=False)
        async for client in _client(context):
            assert (await client.get("/healthz")).status == 200
            assert (await client.get("/readyz")).status == 503

    async def test_readyz_names_the_failing_subsystem_in_the_body(self, clock: ManualClock) -> None:
        context = _context(clock, ready=False)
        async for client in _client(context):
            response = await client.get("/readyz")
            assert response.status == 503
            body: dict[str, Any] = await response.json()
            failing = [check["name"] for check in body["checks"] if not check["ok"]]
            assert failing == ["runtime", "log_stream"]

    async def test_readyz_returns_the_full_report_when_ready_too(self, client: _Client) -> None:
        response = await client.get("/readyz")
        assert response.status == 200
        body: dict[str, Any] = await response.json()
        assert body["ready"] is True
        assert {check["name"] for check in body["checks"]} == {
            "runtime",
            "discord",
            "log_stream",
            "bus",
        }


class TestReads:
    async def test_the_index_lists_the_endpoints(self, client: _Client) -> None:
        body: dict[str, Any] = await (await client.get("/")).json()
        assert "GET /healthz" in body["endpoints"]
        assert body["server_id"] == "minecraft"

    async def test_status_serialises_the_view(self, client: _Client) -> None:
        response = await client.get("/status")
        assert response.status == 200
        body: dict[str, Any] = await response.json()
        assert StatusView.from_dict(body) == _status()

    async def test_players_carries_the_roster_and_a_count(self, client: _Client) -> None:
        body: dict[str, Any] = await (await client.get("/players")).json()
        assert body["online"] == 1
        assert body["players"][0]["name"] == "Steve"

    async def test_players_never_carries_a_client_address(self, client: _Client) -> None:
        # PlayerJoined.address exists on the event and must not reach a presenter surface.
        body: dict[str, Any] = await (await client.get("/players")).json()
        assert "address" not in body["players"][0]

    async def test_sessions_honours_the_limit(self, clock: ManualClock) -> None:
        records = tuple(SessionView(id=f"s-{index}") for index in range(5))
        context = _context(clock, sessions=records)
        async for client in _client(context):
            body: dict[str, Any] = await (await client.get("/sessions?limit=2")).json()
            assert body["count"] == 2

    async def test_a_non_numeric_limit_is_a_400_with_the_standard_error_shape(
        self, client: _Client
    ) -> None:
        response = await client.get("/sessions?limit=lots")
        assert response.status == 400
        body: dict[str, Any] = await response.json()
        assert body["error"]["code"] == "bad_request"

    async def test_an_out_of_range_limit_is_refused(self, client: _Client) -> None:
        assert (await client.get("/sessions?limit=0")).status == 400
        assert (await client.get("/sessions?limit=99999")).status == 400

    async def test_an_unknown_path_uses_the_standard_error_shape(self, client: _Client) -> None:
        response = await client.get("/nope")
        assert response.status == 404
        body: dict[str, Any] = await response.json()
        assert body["error"]["code"] == "not_found"

    async def test_a_handler_that_raises_becomes_a_500_not_a_dead_daemon(
        self, clock: ManualClock
    ) -> None:
        def explode() -> StatusView:
            msg = "boom"
            raise RuntimeError(msg)

        context = replace(_context(clock), status=explode)
        async for client in _client(context):
            response = await client.get("/status")
            assert response.status == 500
            body: dict[str, Any] = await response.json()
            assert body["error"]["code"] == "internal_error"
            # And the server is still serving.
            assert (await client.get("/healthz")).status == 200


class TestControlAuth:
    async def test_a_valid_token_reaches_the_controller(self, clock: ManualClock) -> None:
        controller = RecordingController()
        context = _context(clock, controller=controller)
        async for client in _client(context):
            response = await client.post(
                "/control/stop",
                headers={"Authorization": f"Bearer {TOKEN}"},
                json={"actor": "kunal", "reason": "dinner"},
            )
            assert response.status == 200
            body: dict[str, Any] = await response.json()
            assert body["ok"] is True
            assert controller.calls == [("stop", "kunal", "dinner")]

    async def test_a_missing_header_is_401_with_a_challenge(self, clock: ManualClock) -> None:
        context = _context(clock, controller=RecordingController())
        async for client in _client(context):
            response = await client.post("/control/start")
            assert response.status == 401
            assert "Bearer" in response.headers["WWW-Authenticate"]

    async def test_a_wrong_token_is_403_and_never_reaches_the_controller(
        self, clock: ManualClock
    ) -> None:
        controller = RecordingController()
        context = _context(clock, controller=controller)
        async for client in _client(context):
            response = await client.post(
                "/control/start", headers={"Authorization": "Bearer wrong"}
            )
            assert response.status == 403
            assert controller.calls == []

    async def test_a_non_ascii_token_is_compared_not_crashed_on(self, clock: ManualClock) -> None:
        # hmac.compare_digest raises TypeError on str inputs with non-ASCII characters, which
        # turned a *correct* token into a 500 internal_error and gave an attacker a one-header
        # traceback generator. Nothing enforces that web.token is ASCII, so compare as bytes.
        token = "tökén-höchstens"  # noqa: S105 - a literal for a test server, not a credential
        controller = RecordingController()
        context = _context(clock, controller=controller, token=token)
        async for client in _client(context):
            good = await client.post(
                "/control/stop",
                headers={"Authorization": f"Bearer {token}"},
                json={"actor": "kunal", "reason": "unicode"},
            )
            assert good.status == 200
            assert controller.calls == [("stop", "kunal", "unicode")]

            bad = await client.post(
                "/control/start", headers={"Authorization": "Bearer tökén-anderes"}
            )
            assert bad.status == 403
            assert controller.calls == [("stop", "kunal", "unicode")]

    async def test_a_token_in_the_query_string_is_not_accepted(self, clock: ManualClock) -> None:
        # Query strings end up in access logs, shell history and proxy logs.
        context = _context(clock, controller=RecordingController())
        async for client in _client(context):
            assert (await client.post(f"/control/start?token={TOKEN}")).status == 401

    async def test_an_unset_token_refuses_with_the_config_key_named(
        self, clock: ManualClock
    ) -> None:
        context = _context(clock, controller=RecordingController(), token=None)
        async for client in _client(context):
            response = await client.post(
                "/control/start", headers={"Authorization": f"Bearer {TOKEN}"}
            )
            assert response.status == 503
            body: dict[str, Any] = await response.json()
            assert "web.token" in body["error"]["message"]

    async def test_reads_need_no_token_at_all(self, clock: ManualClock) -> None:
        # The port is exposed and never published; the access path is `docker exec`.
        context = _context(clock, token=None)
        async for client in _client(context):
            assert (await client.get("/status")).status == 200


class TestControlOutcomes:
    async def test_an_unknown_action_is_404(self, clock: ManualClock) -> None:
        context = _context(clock, controller=RecordingController())
        async for client in _client(context):
            response = await client.post(
                "/control/melt", headers={"Authorization": f"Bearer {TOKEN}"}
            )
            assert response.status == 404

    async def test_a_refusal_is_409_and_carries_the_reason(self, clock: ManualClock) -> None:
        refused = ControlOutcome(
            action=ControlAction.START,
            actor="kunal",
            accepted=False,
            rejection="the server is already ready",
            state_before=LifecycleState.READY,
        )
        context = _context(clock, controller=RecordingController(refused))
        async for client in _client(context):
            response = await client.post(
                "/control/start", headers={"Authorization": f"Bearer {TOKEN}"}
            )
            assert response.status == 409
            body: dict[str, Any] = await response.json()
            assert body["rejection"] == "the server is already ready"
            assert body["ok"] is False

    async def test_a_runtime_failure_is_502_and_distinguishable_from_a_refusal(
        self, clock: ManualClock
    ) -> None:
        failed = ControlOutcome(
            action=ControlAction.STOP,
            actor="kunal",
            accepted=True,
            error="docker said no",
        )
        context = _context(clock, controller=RecordingController(failed))
        async for client in _client(context):
            response = await client.post(
                "/control/stop", headers={"Authorization": f"Bearer {TOKEN}"}
            )
            assert response.status == 502
            body: dict[str, Any] = await response.json()
            assert body["accepted"] is True
            assert body["error"] == "docker said no"

    async def test_an_empty_body_defaults_the_actor(self, clock: ManualClock) -> None:
        controller = RecordingController()
        context = _context(clock, controller=controller)
        async for client in _client(context):
            await client.post("/control/restart", headers={"Authorization": f"Bearer {TOKEN}"})
            assert controller.calls == [("restart", "web", None)]

    async def test_a_malformed_body_is_400(self, clock: ManualClock) -> None:
        context = _context(clock, controller=RecordingController())
        async for client in _client(context):
            response = await client.post(
                "/control/stop",
                headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
                data=b"{not json",
            )
            assert response.status == 400

    async def test_no_controller_configured_refuses_with_503(self, clock: ManualClock) -> None:
        context = _context(clock, controller=None)
        async for client in _client(context):
            response = await client.post(
                "/control/start", headers={"Authorization": f"Bearer {TOKEN}"}
            )
            assert response.status == 503

    def test_the_outcome_encoder_carries_both_the_verdict_and_the_message(self) -> None:
        outcome = ControlOutcome(
            action=ControlAction.STOP, actor="kunal", accepted=True, dry_run=True
        )
        payload = outcome_to_dict(outcome)
        assert payload["dry_run"] is True
        assert payload["message"] == str(outcome)


class TestStreams:
    async def _channel_context(self, clock: ManualClock) -> tuple[ControlContext, EventBus, Any]:
        bus = EventBus()
        task = asyncio.create_task(bus.run())
        await asyncio.sleep(0)
        channel = SseChannel(clock=clock)
        channel.attach(bus)
        return _context(clock, channel=channel), bus, task

    async def test_events_streams_published_events(self, clock: ManualClock) -> None:
        context, bus, task = await self._channel_context(clock)
        async for client in _client(context):
            response = await client.get("/events")
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("text/event-stream")
            # X-Accel-Buffering: nginx buffers proxied responses by default, which turns a live
            # tail into 4KB lumps minutes late.
            assert response.headers["X-Accel-Buffering"] == "no"

            bus.publish(
                PlayerJoined(
                    ts=NOW,
                    server_id="mc",
                    source=Source.LOG,
                    raw="Steve joined the game",
                    player=PlayerRef(name="Steve"),
                )
            )
            frame = await _read_frame(response)
            assert "PlayerJoined" in frame
            response.close()
        await bus.aclose()
        await task

    async def test_events_honours_a_type_filter(self, clock: ManualClock) -> None:
        context, bus, task = await self._channel_context(clock)
        async for client in _client(context):
            response = await client.get("/events?type=PlayerEvent")
            bus.publish(ConsoleLog(ts=NOW, server_id="mc", source=Source.LOG, raw="x", message="x"))
            bus.publish(
                PlayerJoined(
                    ts=NOW,
                    server_id="mc",
                    source=Source.LOG,
                    raw="Steve joined the game",
                    player=PlayerRef(name="Steve"),
                )
            )
            frame = await _read_frame(response)
            assert "PlayerJoined" in frame
            response.close()
        await bus.aclose()
        await task

    async def test_an_unknown_type_is_a_400_before_the_stream_opens(
        self, clock: ManualClock
    ) -> None:
        context, bus, task = await self._channel_context(clock)
        async for client in _client(context):
            response = await client.get("/events?type=NoSuchEvent")
            assert response.status == 400
            body: dict[str, Any] = await response.json()
            assert "NoSuchEvent" in body["error"]["message"]
        await bus.aclose()
        await task

    async def test_an_unparsable_since_is_a_400(self, clock: ManualClock) -> None:
        context, bus, task = await self._channel_context(clock)
        async for client in _client(context):
            assert (await client.get("/events?since=soon")).status == 400
            assert (await client.get("/events?seq=soon")).status == 400
        await bus.aclose()
        await task

    async def test_logs_projects_console_lines(self, clock: ManualClock) -> None:
        context, bus, task = await self._channel_context(clock)
        async for client in _client(context):
            response = await client.get("/logs")
            bus.publish(
                ConsoleLog(
                    ts=NOW,
                    server_id="mc",
                    source=Source.LOG,
                    raw="Done (32.5s)!",
                    message="Done (32.5s)!",
                    level="INFO",
                    thread="Server thread",
                )
            )
            frame = await _read_frame(response)
            payload = json.loads(frame.split("data: ", 1)[1].strip())
            assert payload["message"] == "Done (32.5s)!"
            response.close()
        await bus.aclose()
        await task

    async def test_a_disconnecting_client_is_deregistered(self, clock: ManualClock) -> None:
        context, bus, task = await self._channel_context(clock)
        channel = context.channel
        async for client in _client(context):
            response = await client.get("/events")
            assert (await _read_any(response)).startswith(":")  # the connected preamble
            assert channel.stats["clients"] == 1
            response.close()
            for _ in range(50):
                await asyncio.sleep(0)
                if not channel.clients:
                    break
            assert channel.clients == ()
        await bus.aclose()
        await task


async def _read_frame(response: Any) -> str:
    """Read until the first **data** frame arrives, skipping the ``: connected`` preamble.

    Bounded by a timeout so a bug here fails the test rather than hanging the whole suite.
    """
    buffer = ""
    async with asyncio.timeout(2.0):
        while True:
            chunk = await response.content.read(256)
            if not chunk:
                return ""
            buffer += chunk.decode("utf-8")
            complete = buffer.split("\n\n")
            for frame in complete[:-1]:
                if frame.strip() and not frame.startswith(":"):
                    return frame


async def _read_any(response: Any) -> str:
    """Read one frame of any kind, comments included."""
    buffer = ""
    async with asyncio.timeout(2.0):
        while "\n\n" not in buffer:
            chunk = await response.content.read(256)
            if not chunk:
                break
            buffer += chunk.decode("utf-8")
    return buffer.split("\n\n")[0]
