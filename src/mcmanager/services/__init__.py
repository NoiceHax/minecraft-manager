"""The business logic: lifecycle, log pipeline, players, polling, control, idle, sessions.

Imports neither ``docker`` nor ``discord``. Dependencies on the container platform and on the game
arrive as constructor-injected interfaces (``ContainerRuntime``, ``GameAdapter``, ``Clock``), which
is what lets every module here be tested with no I/O at all.
"""

from __future__ import annotations
