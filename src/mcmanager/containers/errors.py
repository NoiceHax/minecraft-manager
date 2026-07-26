"""Exceptions raised at the container boundary.

The distinction that carries real weight is :class:`RuntimeUnavailableError` versus everything
else. "The Docker daemon is unreachable" and "the server stopped" look identical from inside a
failed ``inspect()``, and conflating them makes the daemon publish a phantom ``ServerStopped``
every time the socket hiccups. So the runtime layer is required to raise this specific type, and
lifecycle turns it into ``RuntimeUnavailable`` and enters ``BLIND``, retaining last-known state.

Nothing above ``containers/`` ever sees a ``docker.errors.*`` exception; translating them is part
of ``docker_runtime.py``'s job.
"""

from __future__ import annotations

from mcmanager.errors import EXIT_UNAVAILABLE, McManagerError

__all__ = [
    "ContainerNotFoundError",
    "ContainerOperationError",
    "ContainerRuntimeError",
    "ExecFailedError",
    "LogStreamError",
    "RuntimeUnavailableError",
]


class ContainerRuntimeError(McManagerError):
    """Base for every failure originating in the container layer."""


class RuntimeUnavailableError(ContainerRuntimeError):
    """The container platform itself could not be reached.

    Not "the container is missing" and not "the container is stopped" - those are states, and
    states are returned in a snapshot, not raised.

    Attributes:
        endpoint: The socket or URL that failed. Printed on startup failure alongside the
            process's uid/gid/groups, because on this host the answer is almost always
            "not in gid 983".
    """

    exit_code = EXIT_UNAVAILABLE

    def __init__(self, message: str, *, endpoint: str | None = None) -> None:
        super().__init__(message)
        self.endpoint = endpoint


class ContainerNotFoundError(ContainerRuntimeError):
    """A container was required by name and does not exist.

    :meth:`~mcmanager.containers.base.ContainerRuntime.inspect` never raises this - absence is a
    legitimate state and comes back as a snapshot. Operations that cannot proceed without the
    container (start, stop, exec) do raise it.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"no such container: {name!r}")
        self.name = name


class ContainerOperationError(ContainerRuntimeError):
    """A start/stop/inspect was rejected by the platform for a reason of its own.

    Attributes:
        operation: Which call failed.
        name: The container it was about.
    """

    def __init__(self, operation: str, name: str, message: str) -> None:
        super().__init__(f"{operation} failed for {name!r}: {message}")
        self.operation = operation
        self.name = name


class ExecFailedError(ContainerRuntimeError):
    """A command run inside the container could not be executed at all.

    A command that ran and returned non-zero is **not** this: that is a successful exec with a
    non-zero :class:`~mcmanager.containers.dto.ExecResult`. This is for "the exec could not be
    created", "the container is not running", "it timed out".

    Attributes:
        cmd: The argv that was attempted.
    """

    def __init__(self, cmd: tuple[str, ...], message: str) -> None:
        super().__init__(f"exec {' '.join(cmd)!r} failed: {message}")
        self.cmd = cmd


class LogStreamError(ContainerRuntimeError):
    """The log stream or event stream broke in a way the reconnect loop should back off from.

    Clean EOF is not this: a clean EOF means the container stopped, and it corroborates the ``die``
    event rather than signalling a fault.
    """
