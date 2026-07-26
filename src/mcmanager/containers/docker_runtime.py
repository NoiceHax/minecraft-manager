"""``ContainerRuntime`` backed by the official Docker SDK.

One of exactly two modules permitted to ``import docker``. docker-py ships no type stubs, so this
file carries file-level suppressions and turns ``Any``-shaped inspect dictionaries into the DTOs
in ``dto.py``.

Four corrections to the obvious implementation, each from plan section 4:

1. **``container.logs(stream=True)`` cannot reliably be cancelled**, so calling ``.close()`` on it
   is a no-op and the pump thread leaks. The fix - opening the stream through the same private
   path docker-py's own ``events()`` uses and wrapping it in ``CancellableStream`` - lives in
   ``streams.open_log_stream``, feature-detected at import with a public-generator fallback.
2. **``asyncio.to_thread`` is wrong for long-lived streams**: ``asyncio.run()`` joins the default
   executor at shutdown and a thread parked in a blocking docker read never returns. The two pumps
   are ``threading.Thread(daemon=True)`` (see ``streams.StreamPump``); ``to_thread`` is used here
   only for short request/response calls, which is what it is for.
3. **``requests.Session`` is not thread-safe** and docker-py's ``APIClient`` is one, so this class
   holds *three* ``DockerClient`` instances - request/response, log pump, event watcher - and
   never lets two threads share one.
4. **The thread-to-loop handoff must not raise**; that lives in ``streams.StreamHandoff``.

Also true here and worth not rediscovering: ``Config.Tty`` is **false** on this container, so the
multiplexed path applies; ``Actor.Attributes["exitCode"]`` is a **string** (parsed by
``RuntimeEvent.exit_code``, not here); ``CancellableStream.close()`` raises ``DockerException``
over ``ssh://``, which ``streams.close_stream_quietly`` downgrades.

This module and ``streams.py`` are exempt from the coverage target and are covered by
``@pytest.mark.live`` instead - mocking docker-py would only prove that our assumptions about
docker-py agree with themselves.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import docker
from docker.errors import APIError, DockerException, NotFound, NullResource

from mcmanager.containers.base import ContainerRuntime
from mcmanager.containers.dto import (
    ContainerSnapshot,
    ContainerState,
    ExecResult,
    HealthTiming,
    LogLine,
    MountInfo,
    NetworkAttachment,
    RuntimeEvent,
)
from mcmanager.containers.errors import (
    ContainerNotFoundError,
    ContainerOperationError,
    ExecFailedError,
    RuntimeUnavailableError,
)
from mcmanager.containers.streams import (
    LOG_STREAM_PATH,
    ClosableStream,
    LineSplitter,
    StreamHandoff,
    StreamPump,
    open_event_stream,
    open_log_stream,
    parse_docker_datetime,
    split_docker_timestamp,
)
from mcmanager.core.types import Stream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping, Sequence

    from mcmanager.clock import Clock

__all__ = ["DockerRuntime", "describe_process_identity"]

log: Final = logging.getLogger(__name__)

type _ClientRole = Literal["api", "logs", "events"]

_DEFAULT_ENDPOINT: Final = (
    "npipe:////./pipe/docker_engine" if os.name == "nt" else "unix:///var/run/docker.sock"
)

_NANOSECONDS_PER_MICROSECOND: Final = 1000


def describe_process_identity() -> str:
    """``uid=1000 gid=1000 groups=1000,983``, or a note that the platform has no such concept.

    Printed alongside the socket path when ``ping()`` fails at startup. On this host that error is
    "not a member of gid 983" about 95% of the time, and printing ``id`` turns a twenty-minute
    investigation into a five-second one.
    """
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    getgroups = getattr(os, "getgroups", None)
    if getuid is None or getgid is None:
        return "uid/gid not available on this platform"
    groups = ",".join(str(g) for g in getgroups()) if getgroups is not None else ""
    return f"uid={getuid()} gid={getgid()} groups={groups}"


# ------------------------------------------------------------------------------- attrs -> DTO


def _mapping(value: object) -> Mapping[str, Any]:
    """Coerce an inspect sub-object to a mapping. Docker omits keys; it also sends ``null``.

    Every ``Any`` in this module is introduced here and in the small number of ``x: Any = ...``
    locals below. That is the whole untyped surface: docker-py is described by typeshed's
    third-party stubs, which are accurate about the public API and silent about everything else,
    so the honest move is to name the boundary rather than turn off checking for the file.
    """
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return {}


def _sequence(value: object) -> Sequence[Any]:
    if isinstance(value, list):
        return cast("list[Any]", value)
    return ()


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _str_map(value: object) -> dict[str, str]:
    return {str(k): str(v) for k, v in _mapping(value).items() if v is not None}


def _duration_from_nanoseconds(value: object) -> timedelta | None:
    """Docker reports healthcheck durations in nanoseconds. Zero means "unset", not "instant"."""
    nanoseconds = _int_or_none(value)
    if not nanoseconds:
        return None
    return timedelta(microseconds=nanoseconds / _NANOSECONDS_PER_MICROSECOND)


def _health_timing(config: Mapping[str, Any]) -> HealthTiming:
    healthcheck = _mapping(config.get("Healthcheck"))
    if not healthcheck:
        return HealthTiming()
    return HealthTiming(
        interval=_duration_from_nanoseconds(healthcheck.get("Interval")),
        timeout=_duration_from_nanoseconds(healthcheck.get("Timeout")),
        start_period=_duration_from_nanoseconds(healthcheck.get("StartPeriod")),
        retries=_int_or_none(healthcheck.get("Retries")),
    )


def _networks(network_settings: Mapping[str, Any]) -> tuple[NetworkAttachment, ...]:
    attachments: list[NetworkAttachment] = []
    for name, raw in _mapping(network_settings.get("Networks")).items():
        entry = _mapping(raw)
        aliases = tuple(str(a) for a in _sequence(entry.get("Aliases")))
        attachments.append(
            NetworkAttachment(
                name=str(name),
                ip_address=_str_or_none(entry.get("IPAddress")),
                aliases=aliases,
            )
        )
    return tuple(attachments)


def _mounts(attrs: Mapping[str, Any]) -> tuple[MountInfo, ...]:
    mounts: list[MountInfo] = []
    for raw in _sequence(attrs.get("Mounts")):
        entry = _mapping(raw)
        mounts.append(
            MountInfo(
                source=str(entry.get("Source") or ""),
                destination=str(entry.get("Destination") or ""),
                mode=str(entry.get("Mode") or ""),
                rw=bool(entry.get("RW", True)),
                kind=str(entry.get("Type") or "bind"),
            )
        )
    return tuple(mounts)


def _container_state(status: str | None, *, running: bool) -> ContainerState:
    """Map ``State.Status`` onto our enum without ever guessing.

    An unrecognised status becomes ``UNKNOWN`` rather than being coerced to the nearest thing,
    because a new Docker status string quietly reported as ``RUNNING`` is the kind of bug that
    only shows up as "the daemon thinks the server is up and it is not".
    """
    if status:
        try:
            return ContainerState(status)
        except ValueError:
            log.warning("unrecognised container status from Docker: %r", status)
            return ContainerState.UNKNOWN
    return ContainerState.RUNNING if running else ContainerState.UNKNOWN


def snapshot_from_attrs(
    name: str,
    attrs: Mapping[str, Any],
    *,
    observed_at: datetime,
) -> ContainerSnapshot:
    """Normalise one full ``docker inspect`` payload.

    This is the ``Any``-to-DTO boundary. Above it, nothing has to know that ``State.Health.Status``
    is a string, that healthcheck intervals are nanoseconds, or that ``StartedAt`` on a container
    that never ran is ``0001-01-01T00:00:00Z``.
    """
    state = _mapping(attrs.get("State"))
    config = _mapping(attrs.get("Config"))
    host_config = _mapping(attrs.get("HostConfig"))
    health = _mapping(state.get("Health"))
    log_config = _mapping(host_config.get("LogConfig"))

    status = _str_or_none(state.get("Status"))
    running = bool(state.get("Running"))

    return ContainerSnapshot(
        name=name,
        id=_str_or_none(attrs.get("Id")),
        exists=True,
        state=_container_state(status, running=running),
        status_text=status,
        running=running,
        _health_raw=_str_or_none(health.get("Status")),
        health_failing_streak=_int_or_none(health.get("FailingStreak")),
        health_timing=_health_timing(config),
        tty=bool(config.get("Tty")),
        exit_code=_int_or_none(state.get("ExitCode")),
        oom_killed=bool(state.get("OOMKilled")),
        restart_count=_int_or_none(attrs.get("RestartCount")) or 0,
        image=_str_or_none(config.get("Image")),
        created_at=parse_docker_datetime(_str_or_none(attrs.get("Created"))),
        started_at=parse_docker_datetime(_str_or_none(state.get("StartedAt"))),
        finished_at=parse_docker_datetime(_str_or_none(state.get("FinishedAt"))),
        labels=_str_map(config.get("Labels")),
        networks=_networks(_mapping(attrs.get("NetworkSettings"))),
        mounts=_mounts(attrs),
        log_driver=_str_or_none(log_config.get("Type")),
        log_options=_str_map(log_config.get("Config")),
        restart_policy=_str_or_none(_mapping(host_config.get("RestartPolicy")).get("Name")),
        stop_signal=_str_or_none(config.get("StopSignal")),
        observed_at=observed_at,
    )


def runtime_event_from_payload(
    payload: Mapping[str, Any],
    *,
    fallback_ts: datetime,
) -> RuntimeEvent:
    """Normalise one decoded entry from ``client.events()``.

    ``Actor.Attributes`` is passed through verbatim, values and all. Every one of them is a
    string, including ``exitCode`` - which is why :attr:`RuntimeEvent.exit_code` parses it rather
    than any caller comparing ``attributes["exitCode"] == 0`` and getting ``False`` forever.
    """
    actor = _mapping(payload.get("Actor"))
    attributes = _str_map(actor.get("Attributes"))
    action = _str_or_none(payload.get("Action")) or _str_or_none(payload.get("status")) or ""

    ts = fallback_ts
    nano = _int_or_none(payload.get("timeNano"))
    seconds = _int_or_none(payload.get("time"))
    if nano:
        ts = datetime.fromtimestamp(nano / 1_000_000_000, tz=UTC)
    elif seconds:
        ts = datetime.fromtimestamp(seconds, tz=UTC)

    return RuntimeEvent(
        action=action,
        container_id=_str_or_none(actor.get("ID")) or _str_or_none(payload.get("id")),
        container_name=attributes.get("name"),
        ts=ts,
        attributes=attributes,
    )


# ------------------------------------------------------------------------------- the runtime


class DockerRuntime(ContainerRuntime):
    """The real thing: ``ContainerRuntime`` over the official Docker SDK.

    Construction does no I/O. The three clients are built lazily and independently, so a daemon
    that never follows logs never opens the log-pump client, and a failure to connect surfaces at
    the first real call - where it can be turned into ``RuntimeUnavailable`` - rather than inside
    ``__init__`` where the only option is to crash the process.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        host: str | None = None,
        timeout: int = 30,
    ) -> None:
        """Create a runtime.

        Args:
            clock: Injected time. Every ``observed_at`` and ``received_at`` comes from here.
            host: The Docker endpoint. An empty or absent value means "let docker-py read
                ``DOCKER_HOST`` and the CLI context", which is what the deployed daemon uses and
                what makes ``DOCKER_HOST=ssh://minty@192.168.1.7`` work for the standalone CLI.
            timeout: Request timeout in seconds for short calls. Stream calls override it to
                ``None``; a log stream must not time out because nobody played for six hours.
        """
        self._clock = clock
        self._host = host or None
        self._timeout = timeout
        self._clients: dict[_ClientRole, docker.DockerClient] = {}
        self._lock = asyncio.Lock()
        self._pumps: set[StreamPump[Any, Any]] = set()
        self._closed = False

    # -- identity ----------------------------------------------------------------------------

    @property
    def endpoint(self) -> str:
        """The endpoint we are talking to, for error messages. Never a secret."""
        if self._host:
            return self._host
        return os.environ.get("DOCKER_HOST") or _DEFAULT_ENDPOINT

    @property
    def log_stream_path(self) -> str:
        """``"private"`` or ``"fallback"`` - which log-stream implementation is active.

        Surfaced by ``mcmanager inspect`` so that the feature detection in ``streams.py`` is an
        observable fact rather than something you discover when a stream refuses to close.
        """
        return LOG_STREAM_PATH

    # -- clients -----------------------------------------------------------------------------

    def _build_client(self) -> docker.DockerClient:
        """Blocking. Runs in a worker thread.

        ``version="auto"`` negotiates against the daemon rather than pinning docker-py's compiled
        default (1.45 in 7.2.0). The engine here is 29.6.2 / API 1.55, and pinning low would
        silently give up newer fields for no benefit.

        ``use_ssh_client=True`` is forced for ``ssh://`` endpoints. That shells out to the system
        ``ssh``, which is the one place a subprocess exists anywhere near this project - it is
        inside docker-py, on the Windows dev path only, and production over the unix socket has no
        subprocess anywhere. The alternative transport needs ``paramiko``, which is not a
        dependency.
        """
        kwargs: dict[str, Any] = {"version": "auto", "timeout": self._timeout}
        host = self._host
        if host is not None:
            if host.startswith("ssh://"):
                kwargs["use_ssh_client"] = True
            return docker.DockerClient(base_url=host, **kwargs)
        env_host = os.environ.get("DOCKER_HOST", "")
        if env_host.startswith("ssh://"):
            kwargs["use_ssh_client"] = True
        return docker.from_env(**kwargs)

    async def _client(self, role: _ClientRole) -> docker.DockerClient:
        """Get (building if needed) the client for ``role``.

        **Correction 3.** ``APIClient`` subclasses ``requests.Session``, which is not thread-safe:
        two threads sharing one connection pool interleave reads and produce responses that belong
        to the other caller's request. The request/response path, the log pump and the event
        watcher each get their own.
        """
        existing = self._clients.get(role)
        if existing is not None:
            return existing
        async with self._lock:
            existing = self._clients.get(role)
            if existing is not None:
                return existing
            if self._closed:
                msg = "runtime is closed"
                raise RuntimeUnavailableError(msg, endpoint=self.endpoint)
            try:
                client = await asyncio.to_thread(self._build_client)
            except (DockerException, OSError) as exc:
                msg = f"cannot reach the Docker daemon at {self.endpoint}: {exc}"
                raise RuntimeUnavailableError(msg, endpoint=self.endpoint) from exc
            self._clients[role] = client
            log.debug("built docker client role=%s endpoint=%s", role, self.endpoint)
            return client

    # -- error translation -------------------------------------------------------------------

    async def _call[T](self, operation: str, name: str, fn: Callable[[], T]) -> T:
        """Run a short blocking docker call in a worker thread and translate its failures.

        Nothing above ``containers/`` ever sees a ``docker.errors.*`` exception. The distinction
        that matters is the last clause: a transport failure is ``RuntimeUnavailableError``, which
        is what makes lifecycle go BLIND instead of inventing a ``ServerStopped``.
        """
        try:
            return await asyncio.to_thread(fn)
        except (NotFound, NullResource) as exc:
            raise ContainerNotFoundError(name) from exc
        except APIError as exc:
            raise ContainerOperationError(operation, name, str(exc)) from exc
        except (DockerException, OSError) as exc:
            msg = f"{operation} failed: Docker at {self.endpoint} is unreachable: {exc}"
            raise RuntimeUnavailableError(msg, endpoint=self.endpoint) from exc

    # -- ContainerRuntime --------------------------------------------------------------------

    async def ping(self) -> bool:
        """Is the daemon reachable? Never raises."""
        try:
            client = await self._client("api")
            api: Any = client

            def _do() -> bool:
                return bool(api.ping())

            return await asyncio.to_thread(_do)
        except (RuntimeUnavailableError, DockerException, OSError) as exc:
            log.debug("ping failed against %s: %s", self.endpoint, exc)
            return False

    async def inspect(self, name: str) -> ContainerSnapshot:
        """Resolve ``name`` and normalise its inspect payload.

        By name, every single time. A cached container id survives a ``compose down/up`` and then
        describes a container that no longer exists, which looks exactly like "the server stopped
        logging" and is impossible to debug from the outside.
        """
        observed_at = self._clock.now()
        client = await self._client("api")

        def _do() -> ContainerSnapshot | None:
            try:
                container: Any = client.containers.get(name)
            except (NotFound, NullResource):
                return None
            return snapshot_from_attrs(name, _mapping(container.attrs), observed_at=observed_at)

        snapshot = await self._call("inspect", name, _do)
        if snapshot is None:
            return ContainerSnapshot.missing(name, observed_at=observed_at)
        return snapshot

    async def start(self, name: str) -> None:
        """Start the container. Docker answers 304 for an already-running one, so this is
        idempotent without us having to check first (and checking first would be a race)."""
        client = await self._client("api")

        def _do() -> None:
            client.containers.get(name).start()

        await self._call("start", name, _do)

    async def stop(self, name: str, *, timeout: int) -> None:  # noqa: ASYNC109
        """Stop the container, giving it ``timeout`` seconds before SIGKILL.

        The timeout goes *to Docker*, which is what decides whether the JVM finishes saving the
        world - hence the ASYNC109 suppression. Cancelling our side would not stop the container
        faster, it would only leave us not watching.
        """
        client = await self._client("api")

        def _do() -> None:
            client.containers.get(name).stop(timeout=timeout)

        await self._call("stop", name, _do)

    async def logs_tail(
        self,
        name: str,
        *,
        lines: int = 100,
        since: datetime | None = None,
    ) -> list[LogLine]:
        """Fetch recent output without following. Works on a stopped container."""
        received_at = self._clock.now()
        client = await self._client("api")

        def _do() -> bytes:
            container: Any = client.containers.get(name)
            raw: Any = container.logs(
                tail=lines if lines >= 0 else "all",
                since=since,
                timestamps=True,
                stream=False,
            )
            return raw if isinstance(raw, bytes) else bytes(raw)

        payload = await self._call("logs_tail", name, _do)
        splitter = LineSplitter()
        texts = [*splitter.feed(payload), *splitter.flush()]
        return [self._to_log_line(text, received_at=received_at) for text in texts]

    def follow_logs(
        self,
        name: str,
        *,
        since: datetime | None = None,
        tail: int = 0,
    ) -> AsyncIterator[LogLine]:
        """Stream output until the iterator is closed or the container goes away.

        Deliberately not ``async def``: the call itself does nothing, so a caller can write
        ``async for line in runtime.follow_logs(...)`` and the attach happens on first iteration.
        """
        return self._iter_log_lines(name, since=since, tail=tail)

    async def _iter_log_lines(
        self,
        name: str,
        *,
        since: datetime | None,
        tail: int,
    ) -> AsyncIterator[LogLine]:
        client = await self._client("logs")
        snapshot = await self.inspect(name)
        if snapshot.absent:
            raise ContainerNotFoundError(name)
        container_id = snapshot.id or name

        handoff: StreamHandoff[str] = StreamHandoff()
        splitter = LineSplitter()

        def _open() -> ClosableStream:
            return open_log_stream(
                client,
                container_id,
                tty=snapshot.tty,
                follow=True,
                tail=tail,
                since=since,
            )

        stream = await self._call("follow_logs", name, _open)
        pump: StreamPump[bytes, str] = StreamPump(
            name=f"mcm-logs-{name}",
            stream=stream,
            handoff=handoff,
            transform=splitter.feed,
            on_eof=splitter.flush,
        )
        self._pumps.add(pump)
        pump.start()
        try:
            async for text in handoff:
                yield self._to_log_line(text, received_at=self._clock.now())
        finally:
            pump.stop()
            self._pumps.discard(pump)

    def watch_events(
        self,
        name: str | None = None,
        *,
        since: datetime | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        """Stream container events, optionally filtered to one container."""
        return self._iter_events(name, since=since)

    async def _iter_events(
        self,
        name: str | None,
        *,
        since: datetime | None,
    ) -> AsyncIterator[RuntimeEvent]:
        client = await self._client("events")
        handoff: StreamHandoff[Mapping[str, Any]] = StreamHandoff()

        def _open() -> ClosableStream:
            return open_event_stream(client, container=name, since=since)

        def _transform(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
            # The conversion to RuntimeEvent needs the clock, and the clock belongs to the loop,
            # so the pump thread only ferries the decoded dict across.
            return (payload,)

        stream = await self._call("watch_events", name or "*", _open)
        pump: StreamPump[Mapping[str, Any], Mapping[str, Any]] = StreamPump(
            name=f"mcm-events-{name or 'all'}",
            stream=stream,
            handoff=handoff,
            transform=_transform,
        )
        self._pumps.add(pump)
        pump.start()
        try:
            async for payload in handoff:
                yield runtime_event_from_payload(payload, fallback_ts=self._clock.now())
        finally:
            pump.stop()
            self._pumps.discard(pump)

    async def exec(
        self,
        name: str,
        cmd: Sequence[str],
        *,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> ExecResult:
        """Run a command inside the container and collect its output.

        This is the RCON channel: ``rcon-cli`` reads its own credentials from the container's
        environment, so mcmanager never holds the RCON password.

        ``timeout`` is enforced on our side because docker-py's exec API has no server-side
        deadline. The worker thread is not killed when it expires - there is no way to kill one -
        but exec calls here are ``rcon-cli list``, which either answers in milliseconds or is
        symptomatic of a server that has bigger problems.
        """
        argv = tuple(cmd)
        client = await self._client("api")

        def _do() -> ExecResult:
            container: Any = client.containers.get(name)
            result: Any = container.exec_run(list(argv), demux=True)
            exit_code: Any = result[0]
            output: Any = result[1]
            # demux=True gives (stdout, stderr); without it, one blob on stdout. Both shapes are
            # handled because a demux-unaware engine can ignore the flag.
            stdout_raw: Any = output
            stderr_raw: Any = None
            if isinstance(output, tuple):
                stdout_raw, stderr_raw = cast("tuple[Any, Any]", output)
            code = _int_or_none(exit_code)
            return ExecResult(
                exit_code=code if code is not None else -1,
                stdout=_decode(stdout_raw),
                stderr=_decode(stderr_raw),
            )

        if timeout is None:
            return await self._call("exec", name, _do)
        try:
            async with asyncio.timeout(timeout):
                return await self._call("exec", name, _do)
        except TimeoutError as exc:
            raise ExecFailedError(argv, f"timed out after {timeout}s") from exc

    async def aclose(self) -> None:
        """Cancel every stream, close every client. Safe to call twice; never raises."""
        self._closed = True
        for pump in list(self._pumps):
            pump.stop()
        self._pumps.clear()
        clients = list(self._clients.values())
        self._clients.clear()
        for client in clients:
            try:
                await asyncio.to_thread(client.close)
            except (DockerException, OSError) as exc:
                log.debug("closing docker client failed: %s", exc)

    # -- helpers -----------------------------------------------------------------------------

    def _to_log_line(self, text: str, *, received_at: datetime) -> LogLine:
        """Split Docker's timestamp prefix off and build the DTO.

        ``LogLine.text`` is the line *without* the prefix, and ``LogLine.ts`` is the prefix parsed
        to tz-aware UTC. The parser is handed those two separately, which is what keeps it from
        ever trying to reconstruct a date from Paper's time-only ``[13:24:37]``.

        Stream attribution is ``STDOUT`` for every line: docker-py's multiplexed helper discards
        the frame header, and the stream id lives in that header. See the note in the return value
        of this milestone - preserving it means reading frames ourselves.
        """
        ts, message = split_docker_timestamp(text)
        return LogLine(text=message, ts=ts, received_at=received_at, stream=Stream.STDOUT)


def _decode(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes | bytearray):
        return bytes(raw).decode("utf-8", errors="replace")
    return str(raw)
