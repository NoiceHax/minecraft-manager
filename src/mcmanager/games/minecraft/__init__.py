"""The Minecraft adapter: ANSI stripping, line grammar, patterns, death table, parser, SLP probe.

Pipeline order inside this package, and the order these modules may import each other in:
``ansi -> lines -> patterns -> deaths -> parser -> adapter``. ``probe`` and ``rcon`` hang off the
side and are the only modules here that touch the network.
"""

from __future__ import annotations
