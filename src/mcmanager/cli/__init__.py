"""The ``mcmanager`` command line.

A first-class client, not an afterthought: it makes every subsystem observable with Discord
offline, and it is the forcing function for "Discord contains no business logic" - if
``mcmanager status`` and ``/status`` both render from the same controller and aggregator, no logic
can hide in a slash-command handler.

``render.py`` holds the only ``print`` in the project.
"""

from __future__ import annotations
