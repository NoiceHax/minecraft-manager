"""The aiohttp control surface: ``/healthz`` ``/readyz`` ``/status`` ``/players`` ``/sessions``
``/events`` ``/logs`` ``/control/*``.

One API with three clients - the CLI, Discord, and any future web UI - rather than three parallel
implementations. Named ``control`` rather than ``web`` because it is primarily a control plane and
only incidentally HTTP.
"""

from __future__ import annotations
