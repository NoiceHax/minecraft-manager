"""Minecraft Manager: an event-driven, Docker-native manager for game servers.

Import graph, top to bottom. Nothing ever points upward:

``clock`` / ``errors``  ->  ``core.types``  ->  ``core.events``  ->  ``core.bus`` ->
``containers`` / ``games`` -> ``services`` -> ``control`` / ``cli`` / ``discordbot``

``services`` talks to ``containers`` through the injected ``ContainerRuntime`` ABC and to
``games`` through the injected ``GameAdapter`` Protocol, never by importing an implementation.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
