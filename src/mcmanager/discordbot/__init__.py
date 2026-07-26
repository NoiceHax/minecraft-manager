"""The Discord client.

Named ``discordbot`` so it cannot shadow the installed ``discord`` package. Written strictly as a
second client of the same ``ServerController`` the CLI already exercises: no business logic lives
in here, and nothing outside this package imports ``discord``.
"""

from __future__ import annotations
