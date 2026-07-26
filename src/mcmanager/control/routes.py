"""Route handlers for the control surface.

``GET /healthz`` ``/readyz`` ``/status`` ``/players`` ``/sessions`` ``/events`` ``/logs``, and
``POST /control/{start,stop,restart}`` guarded by a bearer token from ``web.token``.

Every JSON body is produced by :mod:`mcmanager.control.views` or, for events,
:mod:`mcmanager.core.serde` - the same functions the CLI's ``--json`` path uses, so the two cannot
drift. The control endpoints call :class:`~mcmanager.services.controller.ServerController`, the
same object the CLI and the slash commands call; there is no second path to starting a server.

**``/healthz`` and ``/readyz`` are different questions and the difference is load-bearing.**
Liveness asks "should this process be restarted"; readiness asks "is it working right now".
``/healthz`` therefore consults neither Docker nor Discord: a dead Docker socket must not restart
the daemon, because restarting fixes nothing, and a crash loop hides the actual fault behind a
container that never lives long enough to read. ``/readyz`` returns a **per-subsystem** body, so a
red check names the subsystem instead of leaving somebody to guess between four of them.

Everything the handlers need arrives in a :class:`ControlContext`, which is a bag of providers
rather than a reference to the application. That is what lets the whole route table be tested
against an ``aiohttp`` test server with four lambdas and no daemon.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, cast, final

import structlog
from aiohttp import web

from mcmanager.clock import Clock
from mcmanager.control.sse import (
    SSE_CONTENT_TYPE,
    EventFilter,
    SseChannel,
    parse_since,
    resolve_types,
)
from mcmanager.control.views import (
    ControlResultView,
    LivenessView,
    PlayerView,
    ReadinessView,
    SessionView,
    StatusView,
)
from mcmanager.core.types import ControlAction, Source

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping
    from datetime import datetime

    from mcmanager.control.sse import StreamMode
    from mcmanager.services.controller import ControlOutcome

__all__ = [
    "CONTEXT",
    "ControlContext",
    "Controller",
    "error_middleware",
    "outcome_to_dict",
    "register_routes",
]

_log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger("mcmanager.control.routes")

_DEFAULT_SESSION_LIMIT: Final = 20
_MAX_SESSION_LIMIT: Final = 500
_MAX_BODY_BYTES: Final = 64 * 1024
"""A control POST carries at most an actor and a reason. Anything larger is not a control call."""

_ACTIONS: Final[Mapping[str, ControlAction]] = {
    "start": ControlAction.START,
    "stop": ControlAction.STOP,
    "restart": ControlAction.RESTART,
}


# ------------------------------------------------------------------------------------ context


class Controller(Protocol):
    """The mutating surface, narrowed to the three calls this module makes.

    A Protocol rather than an import of :class:`~mcmanager.services.controller.ServerController`
    for the same reason ``containers/manager.py`` narrows its sink: the routes publish nothing and
    subscribe to nothing, and depending on the concrete class would make every route test build a
    lifecycle service. ``ServerController`` satisfies this structurally.
    """

    async def start(self, *, actor: str, via: Source, reason: str | None) -> ControlOutcome: ...

    async def stop(self, *, actor: str, via: Source, reason: str | None) -> ControlOutcome: ...

    async def restart(self, *, actor: str, via: Source, reason: str | None) -> ControlOutcome: ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ControlContext:
    """Everything the handlers are allowed to reach.

    Providers rather than objects: the daemon owns the aggregation and this module owns the HTTP.
    The seam is deliberate - it is what stops ``/status`` growing business logic, and it is why the
    route tests need no daemon at all.

    Attributes:
        status: Builds the full status view. Called per request; must be cheap and must not do
            I/O, because a ``/status`` that blocks on a Docker inspect makes the endpoint as
            unreliable as the thing it describes.
        sessions: Takes a limit, returns the newest records first.
        readiness: The per-subsystem report. See
            :func:`~mcmanager.control.views.evaluate_readiness` for the policy.
        liveness: Loop and supervisor health only.
        controller: ``None`` disables ``POST /control/*`` entirely, which is what a read-only
            deployment wants.
        token: The bearer token for ``POST /control/*``. ``None`` means every mutating request is
            refused with 503 and a message saying which config key to set. Reads stay open: on the
            homelab 8787 is ``expose``d and never published, so the only way to reach it is
            ``docker exec``, and demanding a token to run ``mcmanager status`` inside the
            container would buy nothing.
    """

    server_id: str
    clock: Clock
    channel: SseChannel
    status: Callable[[], StatusView]
    players: Callable[[], Sequence[PlayerView]]
    sessions: Callable[[int], Sequence[SessionView]]
    readiness: Callable[[], ReadinessView]
    liveness: Callable[[], LivenessView]
    controller: Controller | None = None
    token: str | None = None


CONTEXT: Final = web.AppKey("mcmanager_control_context", ControlContext)
"""Typed application key. ``app[CONTEXT]`` is the only global state a handler touches."""


def _context(request: web.Request) -> ControlContext:
    return request.app[CONTEXT]


# -------------------------------------------------------------------------------- responses


def _json(payload: Mapping[str, Any] | Sequence[Any], *, status: int = 200) -> web.Response:
    """A JSON response with stable key order, so a golden test can compare bytes."""
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return web.Response(
        status=status,
        body=body.encode("utf-8"),
        content_type="application/json",
        charset="utf-8",
    )


def _error(
    status: int,
    code: str,
    message: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> web.Response:
    """The one error shape this API emits: ``{"error": {"code": ..., "message": ...}}``.

    A single shape means the CLI has one decoder and one way to print a failure, which is what
    keeps ``mcmanager stop`` from having a different vocabulary of failure than ``/stop``.
    """
    response = _json({"error": {"code": code, "message": message}}, status=status)
    for key, value in (headers or {}).items():
        response.headers[key] = value
    return response


@web.middleware
async def error_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """Turn anything that escapes a handler into the standard error shape.

    An unhandled exception in a route must not take the daemon down and must not answer with
    aiohttp's HTML error page, which a CLI cannot parse. The traceback goes to the log, where it
    belongs; the client gets a code it can branch on.
    """
    try:
        return await handler(request)
    except web.HTTPException as exc:
        if exc.status < 400:  # redirects and the like pass through untouched
            raise
        return _error(exc.status, _code_for(exc.status), exc.reason or "request failed")
    except Exception:
        _log.exception("control.handler_failed", path=request.path, method=request.method)
        return _error(500, "internal_error", "the handler raised; see the daemon log")


def _code_for(status: int) -> str:
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        503: "unavailable",
    }.get(status, "error")


# -------------------------------------------------------------------------------------- auth


def _authorize(request: web.Request) -> web.Response | None:
    """Check the bearer token. Returns a response to send, or ``None`` when authorised.

    The token is compared with :func:`hmac.compare_digest`, not ``==``: this is a secret compared
    against attacker-supplied input, and the constant-time version costs nothing.

    A token in the query string is deliberately **not** accepted. Query strings end up in access
    logs, shell history and proxy logs, which is how a credential outlives its rotation.
    """
    context = _context(request)
    if context.token is None:
        return _error(
            503,
            "control_disabled",
            "web.token is unset, so mutating endpoints are refused. Set it in secrets_dir/"
            'web_token, MCM_WEB__TOKEN, or web.token = "file:/run/secrets/web_token".',
        )
    header = request.headers.get("Authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        return _error(
            401,
            "unauthorized",
            "expected an 'Authorization: Bearer <token>' header",
            headers={"WWW-Authenticate": 'Bearer realm="mcmanager"'},
        )
    # Compare as bytes: hmac.compare_digest raises TypeError on str inputs containing non-ASCII,
    # which would turn a correct-but-non-ASCII token into a 500 instead of a 200, and any
    # attacker-supplied non-ASCII header into a 500 plus a logged traceback.
    if not hmac.compare_digest(presented.strip().encode("utf-8"), context.token.encode("utf-8")):
        _log.warning("control.bad_token", path=request.path, peer=_peer(request))
        return _error(403, "forbidden", "the bearer token was not accepted")
    return None


def _peer(request: web.Request) -> str:
    transport = request.transport
    if transport is None:  # pragma: no cover - only during teardown
        return "?"
    peer: object = transport.get_extra_info("peername")
    return str(peer)


# ---------------------------------------------------------------------------------- handlers


async def handle_index(request: web.Request) -> web.Response:
    """``GET /`` - the route table.

    Present because the first thing anybody does with an unfamiliar control port is curl the root,
    and answering 404 there wastes their next five minutes.
    """
    context = _context(request)
    return _json(
        {
            "service": "mcmanager",
            "server_id": context.server_id,
            "endpoints": {
                "GET /healthz": "liveness: loop and supervisor only, never Docker or Discord",
                "GET /readyz": "readiness, per subsystem",
                "GET /status": "server state, players, session, idle countdown",
                "GET /players": "the online roster",
                "GET /sessions": "archived session records (?limit=)",
                "GET /events": "SSE event stream (?type=&since=&seq=)",
                "GET /logs": "SSE console stream (?since=)",
                "POST /control/start|stop|restart": "bearer token from web.token",
            },
        }
    )


async def handle_healthz(request: web.Request) -> web.Response:
    """``GET /healthz`` - liveness. 200 unless the loop or the supervisor is broken."""
    view = _context(request).liveness()
    return _json(view.to_dict(), status=200 if view.alive and view.supervisor_ok else 503)


async def handle_readyz(request: web.Request) -> web.Response:
    """``GET /readyz`` - readiness, with a per-subsystem body in both the 200 and the 503 case."""
    view = _context(request).readiness()
    return _json(view.to_dict(), status=200 if view.ready else 503)


async def handle_status(request: web.Request) -> web.Response:
    """``GET /status`` - everything ``mcmanager status`` prints."""
    return _json(_context(request).status().to_dict())


async def handle_players(request: web.Request) -> web.Response:
    """``GET /players`` - the online roster with session durations."""
    players = tuple(_context(request).players())
    return _json({"online": len(players), "players": [player.to_dict() for player in players]})


async def handle_sessions(request: web.Request) -> web.Response:
    """``GET /sessions?limit=N`` - archived session records, newest first."""
    context = _context(request)
    raw = request.query.get("limit")
    limit = _DEFAULT_SESSION_LIMIT
    if raw is not None:
        try:
            limit = int(raw)
        except ValueError:
            return _error(400, "bad_request", f"limit must be an integer, got {raw!r}")
        if limit < 1 or limit > _MAX_SESSION_LIMIT:
            return _error(400, "bad_request", f"limit must be between 1 and {_MAX_SESSION_LIMIT}")
    sessions = tuple(context.sessions(limit))
    return _json({"count": len(sessions), "sessions": [item.to_dict() for item in sessions]})


async def handle_events(request: web.Request) -> web.StreamResponse:
    """``GET /events`` - SSE over the whole bus, filtered."""
    return await _stream(request, mode="events")


async def handle_logs(request: web.Request) -> web.StreamResponse:
    """``GET /logs`` - SSE over ``ConsoleLog`` plus every other event's ``raw`` field."""
    return await _stream(request, mode="logs")


async def handle_control(request: web.Request) -> web.Response:
    """``POST /control/{action}`` - the same controller call the CLI and Discord make."""
    context = _context(request)
    name = request.match_info.get("action", "")
    action = _ACTIONS.get(name)
    if action is None:
        known = ", ".join(sorted(_ACTIONS))
        return _error(
            404, "not_found", f"unknown control action {name!r}; expected one of: {known}"
        )

    denied = _authorize(request)
    if denied is not None:
        return denied

    if context.controller is None:
        return _error(
            503,
            "control_disabled",
            "this daemon was built without a controller, so nothing can be started or stopped",
        )

    body = await _read_body(request)
    if isinstance(body, web.Response):
        return body

    actor = _text(body, "actor") or "web"
    reason = _text(body, "reason")
    controller = context.controller
    if action is ControlAction.START:
        outcome = await controller.start(actor=actor, via=Source.WEB, reason=reason)
    elif action is ControlAction.STOP:
        outcome = await controller.stop(actor=actor, via=Source.WEB, reason=reason)
    else:
        outcome = await controller.restart(actor=actor, via=Source.WEB, reason=reason)

    payload = outcome_to_dict(outcome)
    # 409 for a refusal, 502 for a runtime failure: both are honest outcomes rather than server
    # faults, and a CLI that has to parse prose to tell them apart is a CLI that gets it wrong.
    if not outcome.accepted:
        return _json(payload, status=409)
    if outcome.error is not None:
        return _json(payload, status=502)
    return _json(payload)


async def _read_body(request: web.Request) -> Mapping[str, Any] | web.Response:
    """Decode an optional JSON body. An empty body is fine; a malformed one is a 400."""
    if request.content_length is not None and request.content_length > _MAX_BODY_BYTES:
        return _error(413, "payload_too_large", "a control request carries an actor and a reason")
    raw = await request.read()
    if not raw.strip():
        return {}
    try:
        decoded: object = json.loads(raw)
    except ValueError as exc:
        return _error(400, "bad_request", f"body is not valid JSON: {exc}")
    if not isinstance(decoded, dict):
        return _error(400, "bad_request", "body must be a JSON object")
    return cast("Mapping[str, Any]", decoded)


def _text(payload: Mapping[str, Any], key: str) -> str | None:
    value: object = payload.get(key)
    if value is None:
        return None
    return str(value).strip() or None


def outcome_to_dict(outcome: ControlOutcome) -> dict[str, Any]:
    """Serialise a :class:`~mcmanager.services.controller.ControlOutcome`.

    Kept here rather than on the dataclass because the wire format belongs to this API, and the
    controller has no business knowing it has an HTTP client.
    """
    return ControlResultView(
        action=outcome.action.value,
        actor=outcome.actor,
        accepted=outcome.accepted,
        ok=outcome.ok,
        rejection=outcome.rejection,
        dry_run=outcome.dry_run,
        error=outcome.error,
        state_before=outcome.state_before.value,
        message=str(outcome),
    ).to_dict()


# ---------------------------------------------------------------------------------- streaming


def _parse_filter(request: web.Request, *, now: datetime) -> EventFilter | web.Response:
    """Build an :class:`~mcmanager.control.sse.EventFilter` from the query string.

    ``?type=`` may repeat and may also be comma separated, because both spellings are what people
    try. An unknown type is a 400 naming every accepted value: a filter that silently matches
    nothing looks exactly like a broken daemon.
    """
    raw_types: list[str] = []
    for value in request.query.getall("type", []):
        raw_types.extend(part for part in value.split(",") if part.strip())
    try:
        types = resolve_types(raw_types) if raw_types else None
    except ValueError as exc:
        return _error(400, "bad_request", str(exc))

    since = None
    raw_since = request.query.get("since")
    if raw_since is not None:
        try:
            since = parse_since(raw_since, now=now)
        except ValueError as exc:
            return _error(400, "bad_request", str(exc))

    since_seq = None
    raw_seq = request.query.get("seq")
    if raw_seq is not None:
        try:
            since_seq = int(raw_seq)
        except ValueError:
            return _error(400, "bad_request", f"seq must be an integer, got {raw_seq!r}")

    return EventFilter(types=types, since=since, since_seq=since_seq)


async def _stream(request: web.Request, *, mode: StreamMode) -> web.StreamResponse:
    """Serve one SSE stream until the client goes away.

    The client's queue is bounded and lossy by construction (see
    :mod:`mcmanager.control.sse`), so a reader that stops reading costs one bounded buffer and a
    counter, never a stalled bus.
    """
    context = _context(request)
    built = _parse_filter(request, now=context.clock.now())
    if isinstance(built, web.Response):
        return built

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": f"{SSE_CONTENT_TYPE}; charset=utf-8",
            "Cache-Control": "no-cache, no-store",
            # nginx buffers proxied responses by default, which turns a live tail into a stream
            # that arrives in 4KB lumps minutes late. This header is how you switch that off.
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    client = context.channel.open(event_filter=built, mode=mode)
    try:
        await response.write(b": connected\n\n")
        async for frame in client:
            await response.write(frame.encode("utf-8"))
    except (ConnectionResetError, ConnectionAbortedError):
        # The reader hung up. Entirely normal for a live tail; not worth a stack trace.
        _log.debug("control.sse_disconnected", path=request.path, mode=mode)
    finally:
        context.channel.close_client(client)
    return response


# ------------------------------------------------------------------------------ registration


def register_routes(app: web.Application, context: ControlContext) -> None:
    """Install the context and every route onto ``app``.

    Separate from :func:`~mcmanager.control.server.build_app` so a test can mount the same routes
    on its own application, and so the route table is one readable block rather than a method.
    """
    app[CONTEXT] = context
    app.router.add_get("/", handle_index)
    app.router.add_get("/healthz", handle_healthz)
    app.router.add_get("/readyz", handle_readyz)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/players", handle_players)
    app.router.add_get("/sessions", handle_sessions)
    app.router.add_get("/events", handle_events)
    app.router.add_get("/logs", handle_logs)
    app.router.add_post("/control/{action}", handle_control)
