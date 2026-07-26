"""The CLI's HTTP/SSE client, against a real aiohttp server.

Mostly end-to-end against the actual control app, because a client tested against a mock is a
client tested against this file's beliefs about aiohttp. The hand-rolled handlers below exist for
the cases the real server cannot produce on demand: a frame split across two TCP writes, a
non-JSON body, and a connection that closes mid-stream.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from mcmanager.cli.client import (
    DEFAULT_URL,
    URL_ENV,
    DaemonClient,
    DaemonRequestError,
    LogRecord,
    resolve_url,
)
from mcmanager.control.routes import ControlContext
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
from mcmanager.errors import DaemonUnreachableError
from mcmanager.services.controller import ControlOutcome

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mcmanager.clock import ManualClock
    from mcmanager.core.events import Event

NOW = datetime(2026, 7, 25, 22, 58, 20, tzinfo=UTC)
TOKEN = "s3cret-token"  # noqa: S105 - a literal for a test server
WRONG_TOKEN = "not-the-token"  # noqa: S105 - ditto


class StubController:
    def __init__(self, outcome: ControlOutcome) -> None:
        self._outcome = outcome

    async def start(self, *, actor: str, via: Source, reason: str | None) -> ControlOutcome:
        del actor, via, reason
        return self._outcome

    async def stop(self, *, actor: str, via: Source, reason: str | None) -> ControlOutcome:
        del actor, via, reason
        return self._outcome

    async def restart(self, *, actor: str, via: Source, reason: str | None) -> ControlOutcome:
        del actor, via, reason
        return self._outcome


def _no_sessions(limit: int) -> list[SessionView]:
    """A sessions provider with nothing to give. Named rather than a lambda so ruff can see the
    unused argument is deliberate."""
    del limit
    return []


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


async def _serve(app: web.Application) -> AsyncIterator[str]:
    server = TestServer(app)
    await server.start_server()
    try:
        yield str(server.make_url("")).rstrip("/")
    finally:
        await server.close()


async def _daemon(
    clock: ManualClock,
    *,
    controller: StubController | None = None,
    channel: SseChannel | None = None,
    sessions: tuple[SessionView, ...] = (),
) -> AsyncIterator[str]:
    context = ControlContext(
        server_id="minecraft",
        clock=clock,
        channel=channel if channel is not None else SseChannel(clock=clock),
        status=_status,
        players=lambda: _status().roster,
        sessions=lambda limit: list(sessions)[:limit],
        readiness=lambda: evaluate_readiness(runtime_available=True, log_stream_attached=True),
        liveness=lambda: LivenessView(tasks=3),
        controller=controller,
        token=TOKEN,
    )
    async for url in _serve(build_app(context)):
        yield url


class TestResolveUrl:
    def test_the_flag_wins(self) -> None:
        assert (
            resolve_url(
                flag="http://flag:1", env={URL_ENV: "http://env:2"}, configured="http://c:3"
            )
            == "http://flag:1"
        )

    def test_the_env_var_beats_the_config(self) -> None:
        assert resolve_url(env={URL_ENV: "http://env:2"}, configured="http://c:3") == "http://env:2"

    def test_the_config_is_used_when_nothing_else_says(self) -> None:
        assert resolve_url(env={}, configured="http://c:3") == "http://c:3"

    def test_the_default_is_loopback_8787(self) -> None:
        assert resolve_url(env={}) == DEFAULT_URL

    def test_a_trailing_slash_is_stripped(self) -> None:
        assert resolve_url(flag="http://x:1/") == "http://x:1"

    def test_an_empty_flag_falls_through(self) -> None:
        assert resolve_url(flag="", env={URL_ENV: "http://env:2"}) == "http://env:2"


class TestReads:
    async def test_status_decodes_into_the_shared_view(self, clock: ManualClock) -> None:
        async for url in _daemon(clock):
            async with DaemonClient(url) as client:
                assert await client.status() == _status()

    async def test_players_decodes_the_roster(self, clock: ManualClock) -> None:
        async for url in _daemon(clock):
            async with DaemonClient(url) as client:
                players = await client.players()
                assert [player.name for player in players] == ["Steve"]

    async def test_sessions_passes_the_limit_through(self, clock: ManualClock) -> None:
        records = tuple(SessionView(id=f"s-{index}") for index in range(5))
        async for url in _daemon(clock, sessions=records):
            async with DaemonClient(url) as client:
                assert len(await client.sessions(limit=2)) == 2

    async def test_liveness_decodes(self, clock: ManualClock) -> None:
        async for url in _daemon(clock):
            async with DaemonClient(url) as client:
                assert (await client.liveness()).tasks == 3

    async def test_a_503_readiness_is_decoded_rather_than_raised(self, clock: ManualClock) -> None:
        """The whole point of ``/readyz`` is the body, and the body arrives with the 503."""
        context = ControlContext(
            server_id="minecraft",
            clock=clock,
            channel=SseChannel(clock=clock),
            status=_status,
            players=lambda: (),
            sessions=_no_sessions,
            readiness=lambda: evaluate_readiness(
                runtime_available=False, runtime_detail="socket gone"
            ),
            liveness=lambda: LivenessView(),
            token=None,
        )
        async for url in _serve(build_app(context)):
            async with DaemonClient(url) as client:
                view = await client.readiness()
                assert not view.ready
                assert [check.name for check in view.failing()] == ["runtime", "log_stream"]


class TestControl:
    async def test_a_successful_control_call(self, clock: ManualClock) -> None:
        outcome = ControlOutcome(action=ControlAction.STOP, actor="kunal", accepted=True)
        async for url in _daemon(clock, controller=StubController(outcome)):
            async with DaemonClient(url, token=TOKEN) as client:
                result = await client.control("stop", actor="kunal", reason="dinner")
                assert result.ok
                assert result.action == "stop"

    async def test_a_refusal_comes_back_as_a_result_not_an_exception(
        self, clock: ManualClock
    ) -> None:
        outcome = ControlOutcome(
            action=ControlAction.START,
            actor="kunal",
            accepted=False,
            rejection="already ready",
        )
        async for url in _daemon(clock, controller=StubController(outcome)):
            async with DaemonClient(url, token=TOKEN) as client:
                result = await client.control("start", actor="kunal")
                assert not result.accepted
                assert result.rejection == "already ready"

    async def test_a_runtime_failure_comes_back_as_a_result_too(self, clock: ManualClock) -> None:
        outcome = ControlOutcome(
            action=ControlAction.STOP, actor="kunal", accepted=True, error="docker said no"
        )
        async for url in _daemon(clock, controller=StubController(outcome)):
            async with DaemonClient(url, token=TOKEN) as client:
                result = await client.control("stop", actor="kunal")
                assert result.accepted
                assert result.error == "docker said no"

    async def test_a_rejected_token_raises_with_exit_code_77(self, clock: ManualClock) -> None:
        # 77 rather than 69: a bad credential is a human problem, and a supervisor that retries a
        # rejected token forever gets the app rate limited.
        outcome = ControlOutcome(action=ControlAction.STOP, actor="x", accepted=True)
        async for url in _daemon(clock, controller=StubController(outcome)):
            async with DaemonClient(url, token=WRONG_TOKEN) as client:
                with pytest.raises(DaemonRequestError) as caught:
                    await client.control("stop", actor="kunal")
                assert caught.value.status == 403
                assert caught.value.exit_code == 77

    async def test_a_missing_token_raises_401(self, clock: ManualClock) -> None:
        outcome = ControlOutcome(action=ControlAction.STOP, actor="x", accepted=True)
        async for url in _daemon(clock, controller=StubController(outcome)):
            async with DaemonClient(url) as client:
                with pytest.raises(DaemonRequestError) as caught:
                    await client.control("stop", actor="kunal")
                assert caught.value.status == 401


async def _first_event(stream: AsyncIterator[Event]) -> Event:
    """The first event, as a coroutine.

    ``asyncio.create_task`` wants a coroutine, and ``__anext__()`` is a bare awaitable - so the
    wait-then-publish shape these tests need is spelled out here once.
    """
    async for event in stream:
        return event
    msg = "the stream closed before producing an event"
    raise AssertionError(msg)


async def _first_record(stream: AsyncIterator[LogRecord]) -> LogRecord:
    """The first log record, as a coroutine."""
    async for record in stream:
        return record
    msg = "the stream closed before producing a record"
    raise AssertionError(msg)


class TestStreams:
    async def test_events_are_decoded_through_serde(self, clock: ManualClock) -> None:
        bus = EventBus()
        pump = asyncio.create_task(bus.run())
        await asyncio.sleep(0)
        channel = SseChannel(clock=clock)
        channel.attach(bus)

        async for url in _daemon(clock, channel=channel):
            async with DaemonClient(url) as client:
                stream = client.stream_events()
                waiting = asyncio.create_task(_first_event(stream))
                for _ in range(20):
                    await asyncio.sleep(0)
                bus.publish(
                    PlayerJoined(
                        ts=NOW,
                        server_id="mc",
                        source=Source.LOG,
                        raw="Steve joined the game",
                        player=PlayerRef(name="Steve"),
                        online_count=1,
                    )
                )
                async with asyncio.timeout(2.0):
                    event = await waiting
                assert isinstance(event, PlayerJoined)
                assert event.player.name == "Steve"
                await stream.aclose()
        await bus.aclose()
        await pump

    async def test_log_records_are_decoded(self, clock: ManualClock) -> None:
        bus = EventBus()
        pump = asyncio.create_task(bus.run())
        await asyncio.sleep(0)
        channel = SseChannel(clock=clock)
        channel.attach(bus)

        async for url in _daemon(clock, channel=channel):
            async with DaemonClient(url) as client:
                stream = client.stream_logs()
                waiting = asyncio.create_task(_first_record(stream))
                for _ in range(20):
                    await asyncio.sleep(0)
                bus.publish(
                    ConsoleLog(
                        ts=NOW,
                        server_id="mc",
                        source=Source.LOG,
                        raw="Done",
                        message="Done",
                        level="INFO",
                        thread="Server thread",
                    )
                )
                async with asyncio.timeout(2.0):
                    record = await waiting
                assert record.message == "Done"
                assert record.thread == "Server thread"
                await stream.aclose()
        await bus.aclose()
        await pump

    async def test_a_frame_split_across_two_writes_is_reassembled(self) -> None:
        """The SSE parser buffers on ``\\n\\n``, not on whatever the socket happened to deliver."""

        async def handler(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse()
            response.content_type = "text/event-stream"
            await response.prepare(request)
            await response.write(b'event: PlayerJoined\ndata: {"type": "Play')
            await asyncio.sleep(0)
            await response.write(
                b'erJoined", "ts": "2026-07-25T22:58:20Z", "server_id": "mc", '
                b'"source": "log", "raw": null, "seq": 1, '
                b'"player": {"name": "Steve", "uuid": null}, "online_count": 1, '
                b'"address": null, "first_seen": false}\n\n'
            )
            return response

        app = web.Application()
        app.router.add_get("/events", handler)
        async for url in _serve(app):
            async with DaemonClient(url) as client:
                events = [event async for event in client.stream_events()]
            assert len(events) == 1
            assert isinstance(events[0], PlayerJoined)

    async def test_comments_and_keepalives_are_ignored(self) -> None:
        async def handler(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse()
            response.content_type = "text/event-stream"
            await response.prepare(request)
            await response.write(b": connected\n\n: keepalive 2026-07-25T22:58:20Z\n\n")
            return response

        app = web.Application()
        app.router.add_get("/events", handler)
        async for url in _serve(app):
            async with DaemonClient(url) as client:
                assert [event async for event in client.stream_events()] == []

    async def test_one_malformed_frame_does_not_end_the_stream(self) -> None:
        # A tail somebody is watching a player session through must survive one bad event.
        async def handler(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse()
            response.content_type = "text/event-stream"
            await response.prepare(request)
            await response.write(b'data: {"type": "NotAnEvent"}\n\n')
            await response.write(b"data: {not json at all}\n\n")
            await response.write(
                b'data: {"type": "ConsoleLog", "ts": "2026-07-25T22:58:20Z", '
                b'"server_id": "mc", "source": "log", "raw": null, "seq": 2, '
                b'"message": "made it", "level": null, "thread": null, '
                b'"origin": "raw", "stream": "stdout"}\n\n'
            )
            return response

        app = web.Application()
        app.router.add_get("/events", handler)
        async for url in _serve(app):
            async with DaemonClient(url) as client:
                events = [event async for event in client.stream_events()]
            assert len(events) == 1
            assert isinstance(events[0], ConsoleLog)

    async def test_a_type_filter_is_sent_as_a_comma_list(self) -> None:
        seen: list[str] = []

        async def handler(request: web.Request) -> web.StreamResponse:
            seen.append(request.query.get("type", ""))
            response = web.StreamResponse()
            response.content_type = "text/event-stream"
            await response.prepare(request)
            return response

        app = web.Application()
        app.router.add_get("/events", handler)
        async for url in _serve(app):
            async with DaemonClient(url) as client:
                _ = [event async for event in client.stream_events(types=["PlayerEvent", "Chat"])]
        assert seen == ["PlayerEvent,Chat"]


class TestFailures:
    async def test_nothing_listening_raises_daemon_unreachable_naming_the_url(self) -> None:
        client = DaemonClient("http://127.0.0.1:1")
        with pytest.raises(DaemonUnreachableError) as caught:
            await client.status()
        await client.aclose()
        assert "http://127.0.0.1:1" in str(caught.value)
        assert caught.value.exit_code == 69
        assert "docker exec mcmanager" in str(caught.value)

    async def test_a_stream_against_nothing_also_raises_unreachable(self) -> None:
        client = DaemonClient("http://127.0.0.1:1")
        with pytest.raises(DaemonUnreachableError):
            _ = [event async for event in client.stream_events()]
        await client.aclose()

    async def test_a_non_json_body_is_reported_as_such(self) -> None:
        async def handler(request: web.Request) -> web.Response:
            del request
            return web.Response(text="<html>nope</html>", content_type="text/html")

        app = web.Application()
        app.router.add_get("/status", handler)
        async for url in _serve(app):
            async with DaemonClient(url) as client:
                with pytest.raises(DaemonRequestError, match="not JSON"):
                    await client.status()

    async def test_an_error_body_message_is_surfaced(self) -> None:
        async def handler(request: web.Request) -> web.Response:
            del request
            return web.json_response(
                {"error": {"code": "teapot", "message": "I am a teapot"}}, status=418
            )

        app = web.Application()
        app.router.add_get("/status", handler)
        async for url in _serve(app):
            async with DaemonClient(url) as client:
                with pytest.raises(DaemonRequestError) as caught:
                    await client.status()
                assert caught.value.code == "teapot"
                assert "I am a teapot" in str(caught.value)

    async def test_closing_twice_is_harmless(self) -> None:
        client = DaemonClient("http://127.0.0.1:1")
        await client.aclose()
        await client.aclose()


class TestLogRecord:
    def test_a_full_payload_decodes(self) -> None:
        record = LogRecord.from_dict(
            {
                "ts": "2026-07-25T22:58:20Z",
                "seq": 7,
                "type": "ConsoleLog",
                "message": "Done",
                "level": "INFO",
                "thread": "Server thread",
                "origin": "server",
                "stream": "stdout",
            }
        )
        assert record.ts == NOW
        assert record.seq == 7
        assert record.level == "INFO"

    @pytest.mark.parametrize("payload", [{}, {"ts": None}, {"ts": "not a date"}, {"seq": True}])
    def test_a_broken_payload_degrades_rather_than_raising(self, payload: dict[str, Any]) -> None:
        # A console tail must not die because one line arrived malformed.
        record = LogRecord.from_dict(payload)
        assert record.ts is None
        assert record.seq == 0
