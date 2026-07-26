"""HTTP and SSE client for the control surface.

aiohttp, never ``requests`` (banned outright by the ruff config). The one job beyond decoding JSON
is turning a failure to connect into :class:`~mcmanager.errors.DaemonUnreachableError` **carrying
the URL that was tried**, so the command can exit 69 with a message somebody can act on rather
than a stack trace ending in ``ClientConnectorError``.

Why the CLI refuses to degrade when it cannot reach the daemon: a ``status`` that quietly prints
stale standalone numbers while the daemon is wedged is worse than an error, because it looks like
an answer. Commands that need the daemon say so and stop. Commands that can work standalone -
``inspect``, ``replay``, ``sessions``, ``logs`` without ``--follow`` - never come here at all.

**SSE is parsed by hand, from raw chunks.** ``StreamReader.readline`` has a line-length cap, and
the one event guaranteed to exceed it is exactly the one worth seeing: a 4KB chat message, or a
console line from a Paper crash dump. The parser below buffers on ``\\n`` with no limit other than
the caller's patience, which is the correct trade for a debugging tool.

Tested against a real ``aiohttp`` test server rather than a mock, because the interesting failures
are in partial frames and disconnects, and a mock cannot produce those.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Self, cast, final

import aiohttp

from mcmanager.control.views import (
    ControlResultView,
    LivenessView,
    PlayerView,
    ReadinessView,
    SessionView,
    StatusView,
)
from mcmanager.core.serde import SerdeError, event_from_dict
from mcmanager.errors import EXIT_UNAVAILABLE, DaemonUnreachableError, McManagerError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable, Mapping

    from mcmanager.core.events import Event

__all__ = [
    "DEFAULT_URL",
    "TOKEN_ENV",
    "URL_ENV",
    "DaemonClient",
    "DaemonRequestError",
    "LogRecord",
    "resolve_url",
]

DEFAULT_URL: Final = "http://127.0.0.1:8787"
URL_ENV: Final = "MCMANAGER_URL"
TOKEN_ENV: Final = "MCMANAGER_TOKEN"  # noqa: S105 - the name of a variable, not a token

_EXIT_NOPERM: Final = 77
"""sysexits.h ``EX_NOPERM``. A rejected bearer token is a human problem, not a retryable one.

Duplicated from :mod:`mcmanager.config`, which also had to define it because
:mod:`mcmanager.errors` has no home for it yet. Both copies should collapse into ``errors.py``.
"""

_DEFAULT_TIMEOUT: Final = 10.0
_UNREACHABLE_HINT: Final = (
    "Is the daemon running? On the homelab the control port is exposed, never published, so the "
    "access path is:\n"
    "  ssh minty@192.168.1.7 'docker exec mcmanager mcmanager <command>'\n"
    "Locally, `mcmanager run` serves it on 127.0.0.1:8787. Point elsewhere with --url or "
    "MCMANAGER_URL."
)


class DaemonRequestError(McManagerError):
    """The daemon answered, and the answer was a refusal.

    Distinct from :class:`~mcmanager.errors.DaemonUnreachableError`, which means nothing answered
    at all. The exit code follows suit: **77** for an authentication failure, because retrying
    against a rejected credential is pointless and a supervisor should give up rather than loop;
    **69** for everything else.

    Attributes:
        status: The HTTP status.
        code: The machine-readable code from the API's error body, when it had one.
        url: What was requested.
    """

    def __init__(self, message: str, *, status: int, code: str = "error", url: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.url = url
        self.exit_code = _EXIT_NOPERM if status in (401, 403) else EXIT_UNAVAILABLE


def resolve_url(
    *,
    flag: str | None = None,
    env: Mapping[str, str] | None = None,
    configured: str | None = None,
) -> str:
    """Resolve the control endpoint: ``--url`` -> ``MCMANAGER_URL`` -> ``web.url`` -> default.

    Config comes *third* on purpose. A flag and an environment variable are both deliberate acts
    aimed at one invocation, while ``web.url`` describes where this daemon serves; when somebody
    types ``--url``, they mean it.
    """
    if flag:
        return flag.rstrip("/")
    environ: Mapping[str, str] = env if env is not None else {}
    from_env = environ.get(URL_ENV)
    if from_env:
        return from_env.rstrip("/")
    if configured:
        return configured.rstrip("/")
    return DEFAULT_URL


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class LogRecord:
    """One line from the ``/logs`` SSE stream.

    ``message`` is the parsed console text for a ``ConsoleLog`` and the event's ``raw`` for
    everything else, because recognised lines deliberately do not also emit a ``ConsoleLog`` and a
    console view with holes where the joins were is worse than no console view.
    """

    ts: datetime | None
    seq: int = 0
    type: str = "ConsoleLog"
    message: str = ""
    level: str | None = None
    thread: str | None = None
    origin: str | None = None
    stream: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        raw_ts: object = payload.get("ts")
        parsed: datetime | None = None
        if isinstance(raw_ts, str):
            try:
                candidate = raw_ts[:-1] + "+00:00" if raw_ts.endswith("Z") else raw_ts
                parsed = datetime.fromisoformat(candidate).astimezone(UTC)
            except ValueError:
                parsed = None
        raw_seq: object = payload.get("seq")
        return cls(
            ts=parsed,
            seq=raw_seq if isinstance(raw_seq, int) and not isinstance(raw_seq, bool) else 0,
            type=_str_or(payload.get("type"), "ConsoleLog"),
            message=_str_or(payload.get("message"), ""),
            level=_opt_str(payload.get("level")),
            thread=_opt_str(payload.get("thread")),
            origin=_opt_str(payload.get("origin")),
            stream=_opt_str(payload.get("stream")),
        )


def _str_or(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


@final
class DaemonClient:
    """Talks to one control server.

    Use as an async context manager. Every method either returns a decoded view or raises one of
    the two errors above; nothing here prints, and nothing decides an exit code except by choosing
    which exception to raise.
    """

    __slots__ = ("_base", "_owns_session", "_session", "_timeout", "_token")

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        token: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Build a client. No connection is made until a method is called.

        Args:
            base_url: Where the daemon serves. See :func:`resolve_url`.
            token: Bearer token for ``POST /control/*``. Reads need none.
            timeout: Per-request deadline, in seconds. **Not** applied to the SSE streams, which
                are by definition open until somebody stops them.
            session: An existing session, for tests. When given, this client will not close it.
        """
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._session = session
        self._owns_session = session is None

    # -- lifecycle -----------------------------------------------------------------------------

    async def __aenter__(self) -> DaemonClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the session, if this client made it. Safe to call twice."""
        session = self._session
        if session is not None and self._owns_session and not session.closed:
            await session.close()
        if self._owns_session:
            self._session = None

    @property
    def base_url(self) -> str:
        return self._base

    def _ensure_session(self) -> aiohttp.ClientSession:
        session = self._session
        if session is None or session.closed:
            session = aiohttp.ClientSession()
            self._session = session
            self._owns_session = True
        return session

    def _url(self, path: str) -> str:
        return f"{self._base}{path}"

    def _headers(self, *, auth: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if auth and self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    # -- requests ------------------------------------------------------------------------------

    async def get_json(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        allow_status: Iterable[int] = (200,),
    ) -> dict[str, Any]:
        """``GET`` and decode a JSON object.

        ``allow_status`` exists for ``/readyz``, whose 503 is a **valid answer with a body** rather
        than a failure: the whole point of that endpoint is to say which subsystem is red.
        """
        return await self._request("GET", path, params=params, allow_status=allow_status)

    async def post_json(
        self,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        allow_status: Iterable[int] = (200,),
    ) -> dict[str, Any]:
        """``POST`` a JSON object with the bearer token attached."""
        return await self._request(
            "POST", path, payload=payload, allow_status=allow_status, auth=True
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        payload: Mapping[str, Any] | None = None,
        allow_status: Iterable[int] = (200,),
        auth: bool = False,
    ) -> dict[str, Any]:
        session = self._ensure_session()
        url = self._url(path)
        allowed = set(allow_status)
        try:
            async with session.request(
                method,
                url,
                params=dict(params or {}),
                json=dict(payload) if payload is not None else None,
                headers=self._headers(auth=auth),
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as response:
                body = await response.read()
                if response.status not in allowed:
                    raise _request_error(response.status, body, url)
                return _decode_object(body, url)
        except (TimeoutError, aiohttp.ClientError, OSError) as exc:
            raise _unreachable(url, exc) from exc

    # -- reads ---------------------------------------------------------------------------------

    async def ensure_reachable(self) -> None:
        """Fail now, with a good message, rather than during a stream.

        The streaming commands call this first. Without it a daemon that is simply not there
        surfaces as an SSE connection that never produces a frame, which the caller's idle timeout
        then reports as "no events" - an answer, and the wrong one. One cheap ``/healthz`` turns
        that into the exit 69 the plan asks for.
        """
        await self.get_json("/healthz", allow_status=(200, 503))

    async def liveness(self) -> LivenessView:
        """``GET /healthz``. A 503 is a legitimate answer and is decoded, not raised."""
        return LivenessView.from_dict(await self.get_json("/healthz", allow_status=(200, 503)))

    async def readiness(self) -> ReadinessView:
        """``GET /readyz``, per subsystem. A 503 carries the report that says which one is red."""
        return ReadinessView.from_dict(await self.get_json("/readyz", allow_status=(200, 503)))

    async def status(self) -> StatusView:
        """``GET /status``."""
        return StatusView.from_dict(await self.get_json("/status"))

    async def players(self) -> tuple[PlayerView, ...]:
        """``GET /players``."""
        payload = await self.get_json("/players")
        raw: object = payload.get("players", [])
        if not isinstance(raw, list):
            msg = "'players' must be an array"
            raise SerdeError(msg)
        return tuple(
            PlayerView.from_dict(cast("Mapping[str, Any]", item))
            for item in cast("list[object]", raw)
            if isinstance(item, dict)
        )

    async def sessions(self, *, limit: int = 20) -> tuple[SessionView, ...]:
        """``GET /sessions?limit=``."""
        payload = await self.get_json("/sessions", params={"limit": str(limit)})
        raw: object = payload.get("sessions", [])
        if not isinstance(raw, list):
            msg = "'sessions' must be an array"
            raise SerdeError(msg)
        return tuple(
            SessionView.from_dict(cast("Mapping[str, Any]", item))
            for item in cast("list[object]", raw)
            if isinstance(item, dict)
        )

    # -- control -------------------------------------------------------------------------------

    async def control(
        self,
        action: str,
        *,
        actor: str,
        reason: str | None = None,
    ) -> ControlResultView:
        """``POST /control/{action}``.

        A refusal (409) and a runtime failure (502) both come back as a decoded result rather than
        an exception, because both are answers: the daemon considered the request and said no, or
        tried and failed. Only a transport failure or an auth rejection raises.
        """
        payload: dict[str, Any] = {"actor": actor}
        if reason is not None:
            payload["reason"] = reason
        body = await self.post_json(
            f"/control/{action}",
            payload=payload,
            allow_status=(200, 409, 502),
        )
        return ControlResultView.from_dict(body)

    # -- streams -------------------------------------------------------------------------------

    async def stream_events(
        self,
        *,
        types: Iterable[str] = (),
        since: str | None = None,
        seq: int | None = None,
    ) -> AsyncGenerator[Event, None]:
        """``GET /events`` as decoded :class:`~mcmanager.core.events.Event` objects.

        Runs until the daemon closes the stream or the consumer stops iterating. A frame that
        fails to decode is skipped rather than fatal: one malformed event must not end a tail that
        somebody is watching a player session through.
        """
        params: dict[str, str] = {}
        chosen = [name for name in types if name]
        if chosen:
            params["type"] = ",".join(chosen)
        if since is not None:
            params["since"] = since
        if seq is not None:
            params["seq"] = str(seq)
        async for payload in self._stream("/events", params):
            try:
                yield event_from_dict(payload)
            except SerdeError:
                continue

    async def stream_logs(
        self,
        *,
        since: str | None = None,
    ) -> AsyncGenerator[LogRecord, None]:
        """``GET /logs`` as :class:`LogRecord` objects."""
        params: dict[str, str] = {}
        if since is not None:
            params["since"] = since
        async for payload in self._stream("/logs", params):
            yield LogRecord.from_dict(payload)

    async def _stream(
        self,
        path: str,
        params: Mapping[str, str],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Open an SSE stream and yield each ``data:`` payload, decoded.

        No total timeout: this is a live tail. ``sock_read`` is also unset, because a quiet server
        legitimately sends nothing between keepalives, and the keepalive comment is what proves the
        connection is alive.
        """
        session = self._ensure_session()
        url = self._url(path)
        try:
            async with session.get(
                url,
                params=dict(params),
                headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=self._timeout),
            ) as response:
                if response.status != 200:
                    raise _request_error(response.status, await response.read(), url)
                buffer = ""
                async for chunk in response.content.iter_any():
                    buffer += chunk.decode("utf-8", errors="replace")
                    frames = buffer.split("\n\n")
                    buffer = frames.pop()
                    for frame in frames:
                        decoded = _decode_frame(frame)
                        if decoded is not None:
                            yield decoded
        except (TimeoutError, aiohttp.ClientError, OSError) as exc:
            raise _unreachable(url, exc) from exc


# ----------------------------------------------------------------------------------- decoding


def _decode_frame(frame: str) -> dict[str, Any] | None:
    """Pull the ``data:`` payload out of one SSE frame.

    Comments (``: keepalive``) and frames with no data return ``None``. Multi-line ``data:`` is
    joined per the SSE spec even though this server never produces it, because the cost is one
    line and the failure mode of not doing it is silent truncation.
    """
    data_lines: list[str] = []
    for line in frame.splitlines():
        if line.startswith(":") or not line.strip():
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data_lines.append(value[1:] if value.startswith(" ") else value)
    if not data_lines:
        return None
    try:
        decoded: object = json.loads("\n".join(data_lines))
    except ValueError:
        return None
    if not isinstance(decoded, dict):
        return None
    return cast("dict[str, Any]", decoded)


def _decode_object(body: bytes, url: str) -> dict[str, Any]:
    try:
        decoded: object = json.loads(body)
    except ValueError as exc:
        msg = f"{url} answered with something that is not JSON: {body[:200]!r}"
        raise DaemonRequestError(msg, status=502, code="bad_response", url=url) from exc
    if not isinstance(decoded, dict):
        msg = f"{url} answered with a {type(decoded).__name__}, expected an object"
        raise DaemonRequestError(msg, status=502, code="bad_response", url=url)
    return cast("dict[str, Any]", decoded)


def _request_error(status: int, body: bytes, url: str) -> DaemonRequestError:
    """Build the error for a non-2xx, preferring the API's own message."""
    code = "error"
    message = f"{url} answered {status}"
    try:
        decoded: object = json.loads(body)
    except ValueError:
        decoded = None
    if isinstance(decoded, dict):
        error: object = cast("Mapping[str, Any]", decoded).get("error")
        if isinstance(error, dict):
            fields = cast("Mapping[str, Any]", error)
            code = _str_or(fields.get("code"), code)
            detail = _opt_str(fields.get("message"))
            if detail:
                message = f"{message}: {detail}"
    return DaemonRequestError(message, status=status, code=code, url=url)


def _unreachable(url: str, exc: BaseException) -> DaemonUnreachableError:
    """The exit-69 error, carrying the URL and the hint that solves it most of the time."""
    return DaemonUnreachableError(
        f"could not reach the mcmanager daemon at {url}: {exc}\n{_UNREACHABLE_HINT}",
        url=url,
    )
