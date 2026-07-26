"""The command channel. **Phase 9 - body lands with the control work.**

Two modes, and the default is the unusual one on purpose:

- ``exec`` (default): ``docker exec minecraft rcon-cli <cmd>`` through
  ``ContainerRuntime.exec``. The itzg image configures ``rcon-cli`` from the container's own
  environment, so **mcmanager never needs the RCON password at all** - nothing to store, rotate or
  leak. It also means rotating ``RCON_PASSWORD`` on the host has zero impact on this daemon.
- ``network``: ``aio-mc-rcon`` to ``minecraft:25575``, available for latency, behind the optional
  ``rcon`` extra, and requiring the password to exist somewhere. Not the default.

RCON is the only command channel available: the container runs with ``OpenStdin: false``, so there
is no console to attach to.
"""

from __future__ import annotations

# Will export: RconClient
__all__: list[str] = []
