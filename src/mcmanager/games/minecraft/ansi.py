"""Sanitising raw log text.

Step one of parsing, applied unconditionally and before anything else, because ANSI is confirmed
in *both* sinks: ``^[[93mHypixelite joined the game^[[0m`` in the log4j2 archives and
``^[[39m[mc-image-helper]`` in the docker stream. The colour is embedded in log4j2's ``%msg`` and
``SysOut``/``File`` share a ``PatternLayout``, so both carry it.

Not stripping this is one of the three defects in the script this project replaces
(``~/homelab/scripts/minecraft-discord-bridge.py``): the escape bytes end up inside captured
player names, so a Discord message reads ``\\x1b[93mHypixelite joined the game``.

What is removed, and why each one is here:

* **CSI sequences** - ``\\x1b[93m``, ``\\x1b[0m``, ``\\x1b[3m``, ``\\x1b[0;39m`` and truecolor
  ``\\x1b[38;2;R;G;Bm``. All real, all present in the sampled archives.
* **Two-byte escapes** - ``\\x1bM``, ``\\x1b7``. Not observed, but they are the other half of the
  escape grammar and leaving one in would put a stray byte inside a captured name.
* **Minecraft section codes** - ``\\u00a7a``, ``\\u00a7l``, and Bungee's hex form
  ``\\u00a7x\\u00a7R\\u00a7R\\u00a7G\\u00a7G\\u00a7B\\u00a7B`` (which is seven of the simple form in
  a row, so one pattern covers it). Plugins inject these into chat and death messages.
* **C0 control characters** other than tab - notably NUL. A player cannot type one, but a
  malformed frame or a plugin can emit one, and a ``\\x00`` inside a name would survive every
  anchored regex in :mod:`~mcmanager.games.minecraft.patterns` and land in Discord.
* **Trailing ``\\r`` and ``\\n``** - the docker stream is ``\\n``-terminated, but a plugin writing
  CRLF would otherwise leave a ``\\r`` glued to the last captured group.

Tab is deliberately preserved: it is the field separator of the ``mc-server-runner`` wrapper
grammar and the leading marker of a stack-trace continuation line, both of which
:mod:`~mcmanager.games.minecraft.lines` classifies on.
"""

from __future__ import annotations

import re

__all__ = ["ANSI_RE", "CONTROL_RE", "SECTION_RE", "sanitize_line", "strip_ansi"]

ANSI_RE = re.compile(
    r"\x1b(?:"
    r"\[[0-?]*[ -/]*[@-~]"  # CSI:  ESC [ params intermediates final
    r"|\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC:  ESC ] ... BEL or ST
    r"|[ -/]*[0-~]"  # any other escape: ESC intermediates final
    r")"
)
"""Every ANSI escape sequence.

The CSI branch is the standard ``\\x1b[`` plus parameter bytes ``[0-?]``, intermediate bytes
``[ -/]`` and one final byte ``[@-~]``, which covers ``\\x1b[0m`` and ``\\x1b[38;2;255;0;0m`` alike
without enumerating colour codes. Enumerating them is how a truecolor sequence survives stripping
and half of it ends up in a player name.

Two branches beyond the plan's ``\\x1b(?:[@-Z\\\\-_]|\\[[0-?]*[ -/]*[@-~])``, both strict
supersets of it, both there because a *partially* stripped escape is worse than an unstripped one:

* **OSC** (``ESC ] ... BEL``) sets the terminal title and can carry arbitrary text. Under the
  narrower pattern only the two-byte ``ESC ]`` is removed and the payload survives as visible
  garbage.
* **The generic escape form** ``ESC intermediates final`` rather than only the C1 range
  ``[@-Z\\-_]``. ``ESC 7`` (save cursor) has final byte ``0x37``, outside that range, so the
  narrower pattern leaves a bare ``\\x1b`` in the output. Ordering matters here: CSI is tried
  first, because ``[`` is inside ``[0-~]`` and the generic branch would otherwise consume
  ``ESC [`` on its own and leave the parameters behind.
"""

SECTION_RE = re.compile("§.", re.DOTALL)
"""Minecraft's own colour codes: a section sign and whatever follows it.

Deliberately ``§.`` rather than ``§[0-9a-fk-or]``. A section sign is not a character
anybody types on purpose, the set of valid codes has grown twice (hex colours, then
``§x``-prefixed RGB), and a code we failed to enumerate leaves a visible ``§`` in a
Discord message. Dropping one extra character on a malformed sequence is the cheaper mistake.
"""

CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
"""C0 control characters and DEL, excluding tab (``\\x09``) and the newlines handled separately.

Tab survives because it is load-bearing: it separates the wrapper grammar's fields and marks a
stack-trace continuation.
"""


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences and Minecraft section codes. Nothing else."""
    return SECTION_RE.sub("", ANSI_RE.sub("", text))


def sanitize_line(raw: str) -> str:
    """Turn one raw log line into the text the grammar patterns are allowed to see.

    Total and pure: any ``str`` in, a ``str`` out. Strings holding lone surrogates, or bytes
    round-tripped through ``surrogateescape``, are ordinary ``str`` values as far as :mod:`re` is
    concerned and pass through unharmed rather than raising - which is what keeps
    :func:`mcmanager.games.minecraft.parser.parse` total.
    """
    return CONTROL_RE.sub("", strip_ansi(raw.rstrip("\r\n")))
