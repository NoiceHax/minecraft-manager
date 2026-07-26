"""``mcmanager inspect`` - raw truth from Docker, no daemon required.

Prints container state, the **derived** health (which reads ``unknown`` on a stopped container even
while Docker's raw string still says ``unhealthy`` - the observed gotcha this design encodes once),
exit code, OOM flag, ``Tty``, the healthcheck timings the lifecycle machine reads its deadlines
from, network attachments and aliases, mounts and log-driver caps. ``--json`` emits the same facts
as an object.

This is the command that matters most when the daemon itself is the broken thing: set
``DOCKER_HOST=ssh://minty@192.168.1.7`` and it reads the real box from a Windows laptop, with no
bus, no supervisor, no Discord import and no daemon anywhere.

The ``--json`` shape is defined here rather than in ``core/serde.py`` because serde's subject is
the event vocabulary, and widening it to container DTOs would make "the wire format for an event"
mean two things. What matters for consistency is that ``health`` in this output is the **derived**
value and ``health_reported_raw`` is what Docker literally said - the same split the DTO enforces.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from mcmanager.cli import render
from mcmanager.containers.errors import RuntimeUnavailableError
from mcmanager.containers.factory import build_runtime
from mcmanager.errors import EXIT_OK, EXIT_UNAVAILABLE

if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from mcmanager.cli.main import CliContext
    from mcmanager.containers.base import ContainerRuntime
    from mcmanager.containers.dto import ContainerSnapshot

__all__ = ["run", "snapshot_to_dict"]


async def run(ctx: CliContext, *, container: str | None = None) -> int:
    """Inspect one container and print it.

    Returns 69 when the *platform* is unreachable, and **0** when the container simply does not
    exist. That asymmetry is deliberate and matches the rest of the system: an absent container is
    a legitimate state that the daemon warns about and keeps running through, while an unreachable
    Docker socket is a hard failure with a message naming the socket and this process's uid, gid
    and groups - because that error is the docker-group problem most of the time.
    """
    from mcmanager.config import runtime_unreachable_message

    name = container or ctx.settings.server.container
    runtime = build_runtime(
        ctx.settings.runtime,
        clock=ctx.clock,
        docker_host=ctx.settings.docker.host,
    )
    try:
        snapshot = await runtime.inspect(name)
        extra = _diagnostics(runtime)
    except RuntimeUnavailableError as exc:
        render.emit_error(runtime_unreachable_message(endpoint=_endpoint(runtime), error=str(exc)))
        return EXIT_UNAVAILABLE
    finally:
        await runtime.aclose()

    if ctx.json_output:
        payload = snapshot_to_dict(snapshot)
        payload["runtime"] = dict(extra)
        render.emit(json.dumps(payload, indent=2, sort_keys=True))
    else:
        render.emit(render.render_inspect(snapshot, palette=ctx.palette, extra=extra))
    return EXIT_OK


def snapshot_to_dict(snapshot: ContainerSnapshot) -> dict[str, Any]:
    """The ``--json`` form of one inspect.

    ``health`` is the derived value; ``health_reported_raw`` is Docker's own string, present so a
    bug report can show both. A consumer that wants to know whether the server is answering reads
    ``health``; a consumer investigating why Docker is confused reads the raw one.
    """
    timing = snapshot.health_timing
    return {
        "name": snapshot.name,
        "id": snapshot.id,
        "exists": snapshot.exists,
        "state": snapshot.state.value,
        "status_text": snapshot.status_text,
        "running": snapshot.running,
        "health": snapshot.health.value,
        "health_reported_raw": snapshot.health_reported_raw,
        "health_failing_streak": snapshot.health_failing_streak,
        "healthcheck": {
            "interval_seconds": _seconds(timing.interval),
            "timeout_seconds": _seconds(timing.timeout),
            "start_period_seconds": _seconds(timing.start_period),
            "retries": timing.retries,
            "guard_window_seconds": _seconds(snapshot.guard_window),
        },
        "tty": snapshot.tty,
        "exit_code": snapshot.exit_code,
        "oom_killed": snapshot.oom_killed,
        "restart_count": snapshot.restart_count,
        "restart_policy": snapshot.restart_policy,
        "stop_signal": snapshot.stop_signal,
        "image": snapshot.image,
        "created_at": _iso(snapshot.created_at),
        "started_at": _iso(snapshot.started_at),
        "finished_at": _iso(snapshot.finished_at),
        "labels": dict(snapshot.labels),
        "networks": [
            {
                "name": network.name,
                "ip_address": network.ip_address,
                "aliases": list(network.aliases),
            }
            for network in snapshot.networks
        ],
        "mounts": [
            {
                "source": mount.source,
                "destination": mount.destination,
                "mode": mount.mode,
                "rw": mount.rw,
                "kind": mount.kind,
            }
            for mount in snapshot.mounts
        ],
        "log_driver": snapshot.log_driver,
        "log_options": dict(snapshot.log_options),
        "observed_at": _iso(snapshot.observed_at),
    }


def _diagnostics(runtime: ContainerRuntime) -> dict[str, str]:
    """Runtime-specific facts worth printing, read defensively.

    ``log_stream_path`` says which of the two log-stream implementations is live (the private
    ``_multiplexed_response_stream_helper`` path or the public generator fallback). Nobody wants
    to learn that from a stack trace at 2am, and the ABC deliberately does not declare it, so it
    is read only if the implementation offers it.
    """
    found: dict[str, str] = {}
    for attribute, label in (("endpoint", "docker host"), ("log_stream_path", "log stream")):
        value: object = getattr(runtime, attribute, None)
        if isinstance(value, str) and value:
            found[label] = value
    found["runtime class"] = type(runtime).__name__
    return found


def _endpoint(runtime: ContainerRuntime) -> str | None:
    value: object = getattr(runtime, "endpoint", None)
    return value if isinstance(value, str) else None


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _seconds(value: timedelta | None) -> float | None:
    return None if value is None else value.total_seconds()
