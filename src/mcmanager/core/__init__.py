"""Game-agnostic, platform-agnostic core: the event vocabulary, the bus, task supervision, serde.

Nothing in this package imports ``docker``, ``discord``, ``mcstatus`` or anything under
``games/``. That is the property that makes the whole design testable without I/O.
"""

from __future__ import annotations
