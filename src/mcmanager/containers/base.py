"""The container runtime interface.

This ABC is exactly as wide as the daemon needs and no wider. Deliberately absent:

- **``restart()``** - the daemon does stop-then-start itself, so it owns the transition and emits
  the right events in the right order. A runtime-level restart would produce a ``die`` and a
  ``start`` that lifecycle would have to guess about.
- **``remove()`` / ``create()``** - mcmanager manages a container's *lifecycle*, not its
  existence. Compose owns that, and a manager that can delete the thing it manages is a bad idea.
- **``stats()``** - memory and CPU belong to beszel, which already watches this host.

Two implementations sit behind ``containers.factory.build_runtime``: ``DockerRuntime`` and a
scriptable ``FakeRuntime``. Every service test runs against the fake, and the whole daemon runs
offline with ``runtime = "fake"``.

The ``timeout`` parameters on :meth:`ContainerRuntime.stop` and :meth:`ContainerRuntime.exec`
carry ``noqa: ASYNC109``. Ruff's advice - "use ``asyncio.timeout`` instead" - is right for a
client-side deadline and wrong here: these values are handed *to Docker*, which uses them to decide
how long to wait before SIGKILL. Cancelling our side of the call would not stop the container any
faster; it would just leave us not watching.

The testing rule that follows from this file: **fake this interface, never the ``docker``
library.** Mocking ``docker.DockerClient`` encodes your assumptions about docker-py into the test
suite, so a wrong assumption produces a passing test. Faking our own ABC means only one file can
be wrong about docker-py, and that file is covered by the live tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from datetime import datetime

    from mcmanager.containers.dto import ContainerSnapshot, ExecResult, LogLine, RuntimeEvent

__all__ = ["ContainerRuntime"]


class ContainerRuntime(ABC):
    """Everything the daemon may ask of a container platform.

    Implementations must:

    - never raise for "the container does not exist" from :meth:`inspect` - return an absent
      snapshot, because that is a legitimate state;
    - raise :class:`~mcmanager.containers.errors.RuntimeUnavailableError` when the platform itself
      is unreachable, because that is what tells lifecycle to go BLIND instead of inventing a
      ``ServerStopped``;
    - return tz-aware UTC datetimes everywhere;
    - be safe to :meth:`aclose` twice.
    """

    @abstractmethod
    async def ping(self) -> bool:
        """Is the platform reachable? Never raises; returns False instead.

        A False here at startup is fatal and exits 69 with a message naming the socket **and the
        process's uid/gid/groups** - that error is the docker-group-membership problem 95% of the
        time, and printing ``id`` saves twenty minutes.
        """

    @abstractmethod
    async def inspect(self, name: str) -> ContainerSnapshot:
        """Resolve ``name`` and return its current state.

        Resolution is by name every single time. A cached container id survives a
        ``compose down/up`` and then silently describes a container that no longer exists.

        Returns an absent snapshot (``exists=False``) if there is no such container. Raises only
        when the platform itself is unreachable.
        """

    @abstractmethod
    async def start(self, name: str) -> None:
        """Start the container. Idempotent: starting a running container is not an error."""

    @abstractmethod
    async def stop(self, name: str, *, timeout: int) -> None:  # noqa: ASYNC109
        """Stop the container, allowing ``timeout`` seconds before SIGKILL.

        ``timeout`` is required rather than defaulted: this is the number that decides whether a
        Minecraft world finishes saving, it comes from ``server.lifecycle.stop_timeout_seconds``,
        and it should never be inherited by accident. Idempotent on an already-stopped container.
        """

    @abstractmethod
    async def logs_tail(
        self,
        name: str,
        *,
        lines: int = 100,
        since: datetime | None = None,
    ) -> list[LogLine]:
        """Fetch recent output and return. Never blocks waiting for more.

        Works on a stopped container, which is what makes ``mcmanager logs`` useful after a crash.
        """

    @abstractmethod
    def follow_logs(
        self,
        name: str,
        *,
        since: datetime | None = None,
        tail: int = 0,
    ) -> AsyncIterator[LogLine]:
        """Stream output until the iterator is closed or the container goes away.

        Not ``async def``: the call itself does not block, so callers write
        ``async for line in runtime.follow_logs(...)``.

        Contract notes that cost real time to learn:

        - ``since`` is **second-granularity** and replays the whole of that second, so the caller
          must de-duplicate. The reconnect loop keeps a 256-entry ``(ts, hash)`` ring for this.
        - Lines arrive in arbitrarily split chunks. The splitter buffers across chunk boundaries.
        - A clean EOF means the container stopped. It corroborates the ``die`` event; it is not an
          error.
        - Timestamps are always requested from the platform; never parsed out of the line text.
        """

    @abstractmethod
    def watch_events(
        self,
        name: str | None = None,
        *,
        since: datetime | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        """Stream platform events, optionally filtered to one container.

        ``health_status`` events are **edge-triggered**: one is emitted when the status changes and
        never again. Treating their absence as "still healthy" is wrong, which is why a 60-second
        reconcile inspect exists as the safety net.
        """

    @abstractmethod
    async def exec(
        self,
        name: str,
        cmd: Sequence[str],
        *,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> ExecResult:
        """Run a command inside the container and collect its output.

        This is the default RCON channel (``rcon-cli``), which is why mcmanager needs no RCON
        password. It is also, honestly, root-equivalent - which is why the docker-socket-proxy
        idea is deferred rather than adopted: it would have to allow ``EXEC=1`` anyway.
        """

    @abstractmethod
    async def aclose(self) -> None:
        """Release every client, thread and stream. Safe to call twice, and never raises."""
