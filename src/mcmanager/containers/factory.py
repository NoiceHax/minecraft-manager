"""Builds the configured ``ContainerRuntime``.

The one place that decides between real and fake, so that nothing else has to import either
implementation and the ``import docker`` confinement holds.

``DockerRuntime`` is imported lazily inside the branch that needs it. That is not a micro
optimisation: it is what lets ``mcmanager replay`` and the whole ``runtime = "fake"`` dev path run
without docker-py ever being imported, which in turn is what proves the confinement is real rather
than a naming convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcmanager.containers.fake import FakeRuntime
from mcmanager.errors import ConfigError

if TYPE_CHECKING:
    from mcmanager.clock import Clock
    from mcmanager.containers.base import ContainerRuntime
    from mcmanager.core.types import RuntimeKind

__all__ = ["build_runtime"]


def build_runtime(
    kind: RuntimeKind,
    *,
    clock: Clock,
    docker_host: str | None = None,
) -> ContainerRuntime:
    """Construct the runtime named by ``kind``.

    ``docker_host`` of ``None`` means "let docker-py read ``DOCKER_HOST`` / the default socket",
    which is what the deployed daemon uses and what makes
    ``DOCKER_HOST=ssh://minty@192.168.1.7`` work for the standalone CLI from Windows.

    ``kind`` is typed as a ``Literal``, but it arrives from a TOML file, so the final branch is
    reachable in practice and raises :class:`~mcmanager.errors.ConfigError` (exit 78) rather than
    quietly defaulting. Defaulting to the fake would mean a typo in production produces a daemon
    that cheerfully reports a server it is not managing.
    """
    if kind == "fake":
        return FakeRuntime(clock=clock)
    if kind == "real":
        from mcmanager.containers.docker_runtime import DockerRuntime

        return DockerRuntime(clock=clock, host=docker_host)
    msg = f"unknown runtime {kind!r}: expected 'real' or 'fake'"
    raise ConfigError(msg)
