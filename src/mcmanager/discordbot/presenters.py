"""Event to Discord message. **File and signatures only; body in M6.**

The one place with a per-game table, and it takes that table from a registry rather than
hardcoding Minecraft, which is the honest boundary of the "game-agnostic" claim.

Non-negotiable, and each has a named test:

- **``@everyone``, ``@here`` and role mentions are neutralised.** Chat is attacker-controlled text.
- **Backticks and code fences are escaped**, and long messages are truncated with a marker.
- **``PlayerJoined.address`` is never rendered.** It is on the event because idle logic and abuse
  investigation want it; relaying a player's IP to a chat channel is not acceptable.
- A ``match`` over the event union ends in ``assert_never`` so a new event cannot silently render
  as its base class.

Pure functions, string in and string out. No gateway, no clock, no config - which is what makes the
adversarial tests (``@everyone``, backticks, 4KB of text, null bytes, lone surrogates) a plain
table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcmanager.core.events import AnyEvent, Event

__all__ = ["MAX_MESSAGE_LENGTH", "escape_markdown", "neutralise_mentions", "render_event"]

MAX_MESSAGE_LENGTH = 1900
"""Discord's limit is 2000; the margin leaves room for the truncation marker and a code fence."""


def neutralise_mentions(text: str) -> str:
    """Make ``@everyone``, ``@here``, ``<@id>`` and ``<@&role>`` inert.

    Applied to **every** string that reaches Discord, not only to chat, because a player name, an
    advancement title and a death message are all attacker-influenced too.
    """
    raise NotImplementedError


def escape_markdown(text: str) -> str:
    """Escape backticks, code fences and the rest of Discord's markdown."""
    raise NotImplementedError


def render_event(event: AnyEvent) -> str | None:
    """Render one event, or ``None`` when this event is not announced at all.

    A ``match`` over the union ending in ``assert_never``, so adding a nineteenth event is a type
    error here rather than a silent fall-through to a base-class rendering.
    """
    raise NotImplementedError


def render_raw(event: Event) -> str | None:
    """Render ``Event.raw`` for the console relay, escaped and truncated."""
    raise NotImplementedError
