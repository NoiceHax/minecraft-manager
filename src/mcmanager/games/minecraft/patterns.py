"""Compiled payload patterns.

**Every pattern is ``^``-anchored to the message payload**, after the prefix has been stripped by
:func:`mcmanager.games.minecraft.lines.classify`. That single property fixes the first defect in
``~/homelab/scripts/minecraft-discord-bridge.py``: ``re.compile(r"(.+) joined the game")``
``.search(line)`` against ``[22:41:23] [Server thread/INFO]: Steve joined the game`` captures
``[22:41:23] [Server thread/INFO]: Steve`` as the player's name, because ``.+`` is greedy, the
pattern is unanchored, and ``search`` is happy to start at column zero.

Fixed, tested match order (see :data:`MATCH_ORDER`)::

    READY -> VERSION -> STOPPING -> CHAT -> EMOTE -> SAY -> RCON_ECHO -> JOIN -> LEAVE
          -> LOGIN -> LOST_CONN -> KICKED -> UUID -> ADVANCEMENT -> DEATHS -> ConsoleLog

Chat before join is what makes ``<Bob> Alice joined the game`` produce exactly one chat event and
zero join events - the old script's third defect, which let any player forge a join notification
just by typing it.

``VERSION`` is inserted after ``READY``; the plan's order omitted it because it is a server
lifecycle line rather than a player one, and it cannot be confused with any pattern after it.

Real lines that must be handled, sampled verbatim from the live archives::

    Starting minecraft server version 26.2
    Done (32.521s)! For help, type "help"
    UUID of player Hypixelite is 403d4fb2-f466-3716-b9cb-a3e769bb40c9   (offline v3, still present)
    Hypixelite[/115.99.245.156:49237] logged in with entity id 74 at (...)  (a player IP)
    bharath_720 lost connection: Server closed              (a shutdown, not a voluntary leave)
    bharath_720 was kicked due to keepalive timeout!
    [Not Secure] <bharath_720> yo                           (chat, on an Async Chat Thread)
    [Not Secure] [Rcon] Hello from HCP
    [Rcon: Stopping the server]
    Hypixelite has made the advancement [Stone Age]

Some recognised lines have no event to become. ``LOGIN``, ``LOST_CONN``, ``KICKED`` and ``UUID``
carry facts that belong to a *different* event: the address that the next ``PlayerJoined`` should
report, the reason the next ``PlayerLeft`` should classify, the UUID the next ``PlayerRef`` should
carry. Inventing four new event types for them would push stitching into every consumer, so the
parser emits a :class:`~mcmanager.core.events.ConsoleLog` for these and exports the matchers
below for ``services/log_pipeline.py``, which owns the stitching because it is the only component
allowed to hold state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mcmanager.core.types import LeaveReason

__all__ = [
    "ADVANCEMENT_RE",
    "CHAT_RE",
    "COMMAND_RE",
    "EMOTE_RE",
    "JOIN_RE",
    "KICKED_RE",
    "LEAVE_RE",
    "LOGIN_RE",
    "LOST_CONN_RE",
    "MATCH_ORDER",
    "PLAYER_NAME",
    "RCON_BRACKET_RE",
    "RCON_ECHO_RE",
    "RCON_PSEUDO_NAME",
    "READY_RE",
    "SAY_RE",
    "STOPPING_RE",
    "UUID_RE",
    "VERSION_RE",
    "CommandMatch",
    "KickMatch",
    "LoginMatch",
    "LostConnectionMatch",
    "UuidMatch",
    "classify_leave_reason",
    "match_command",
    "match_kick",
    "match_login",
    "match_lost_connection",
    "match_uuid",
]

PLAYER_NAME = r"[A-Za-z0-9_]{1,16}"
"""What a Minecraft account name can be, and nothing else.

The vanilla server enforces ``^[A-Za-z0-9_]{3,16}$`` on registration; the lower bound is relaxed
to 1 here only so a test can use a one-character name. **No dot, no space, no punctuation.**

This is the load-bearing half of the fix for the old script's greedy-capture bug. An anchored
``.+`` would still match ``[22:41:23] [Server thread/INFO]: Steve`` if the prefix were ever left
in place; a character class that cannot contain ``[``, ``]``, ``:`` or a space cannot. The two
defences are deliberately redundant, because the failure mode is silent and ends up in Discord.

The cost is honest and worth stating: a Bedrock player bridged in through Geyser has a name with a
space or a ``.`` prefix, and their join line will not match here. It falls through to
``ConsoleLog``, which is visible in the unrecognised-line ratio, rather than matching something
wrong. No Geyser on this deployment.
"""

RCON_PSEUDO_NAME = "Rcon"
"""The speaker attributed to ``[Rcon] ...`` output. Not a player; quite possibly us."""

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
"""A canonical UUID. Offline v3 on this server - see :data:`UUID_RE`."""

# ------------------------------------------------------------------------------ server lifecycle

READY_RE = re.compile(r'^Done \((?P<seconds>[0-9][0-9,.]*)s\)! For help, type "help"')
"""``Done (32.521s)! For help, type "help"``.

The digit group tolerates a thousands separator because log4j2 renders durations with the JVM's
default locale: ``Finished initialising converters for DataConverter in 1,044.8ms`` appears in the
sampled archives, so ``Done (1,032.5s)!`` is reachable on a slow start.
"""

VERSION_RE = re.compile(r"^Starting minecraft server version (?P<version>\S+)$")
"""``Starting minecraft server version 26.2``. The earliest signal that this run has begun."""

STOPPING_RE = re.compile(r"^(?:\[Rcon: Stopping the server\]|Stopping the server|Stopping server)$")
"""A shutdown has begun.

``[Rcon: Stopping the server]`` is what the idle script and mcmanager both produce; the bare forms
are what a console ``stop`` or a plugin produces. Load-bearing beyond the obvious: a
``lost connection: Server closed`` arriving *after* this is a shutdown casualty, and counting it
as a voluntary quit corrupts every session summary.
"""

# ---------------------------------------------------------------------------------- chat family

CHAT_RE = re.compile(rf"^<(?P<name>{PLAYER_NAME})> (?P<message>.*)$", re.DOTALL)
"""``<Steve> hello``. First of the player patterns, and that ordering is the whole point.

``<Steve> Alex joined the game`` matches here and stops. Under the old script's
``if "joined the game" in line`` it fired a join notification for ``Alex``, which is a forgeable
event: any player could announce anyone's arrival.
"""

EMOTE_RE = re.compile(rf"^\* (?P<name>{PLAYER_NAME}) (?P<message>.*)$", re.DOTALL)
"""``* Steve waves``, from ``/me``."""

SAY_RE = re.compile(
    rf"^\[(?!{RCON_PSEUDO_NAME}[\]:])(?P<name>{PLAYER_NAME})\] (?P<message>.*)$",
    re.DOTALL,
)
"""``[Steve] hello``, from ``/say``.

The negative lookahead is what lets ``SAY`` sit before ``RCON_ECHO`` in the fixed match order
without swallowing ``[Rcon] Hello from HCP`` and attributing it to a player named ``Rcon``.
"""

RCON_ECHO_RE = re.compile(rf"^\[{RCON_PSEUDO_NAME}\] (?P<message>.*)$", re.DOTALL)
"""``[Rcon] Hello from HCP`` - a broadcast issued over the command channel.

Sampled real line: ``[Not Secure] [Rcon] Hello from HCP``, emitted on ``Server thread``.
"""

RCON_BRACKET_RE = re.compile(rf"^\[{RCON_PSEUDO_NAME}: (?P<result>.*)\]$", re.DOTALL)
"""``[Rcon: Saved the game]``, ``[Rcon: Made Hypixelite a server operator]``.

A command *result* echoed back, not a broadcast: nobody in the game saw it, so it is not chat.
Recognised so that it never reaches the death table, and so ``[Rcon: Stopping the server]`` is
the only one of this family with special meaning (handled earlier by :data:`STOPPING_RE`).
"""

# ------------------------------------------------------------------------------ player lifecycle

JOIN_RE = re.compile(rf"^(?P<name>{PLAYER_NAME}) joined the game$")
"""``Steve joined the game``. Fully anchored at both ends: the old script anchored at neither."""

LEAVE_RE = re.compile(rf"^(?P<name>{PLAYER_NAME}) left the game$")
"""``Steve left the game``. The reason arrives on a *different* line - see :data:`LOST_CONN_RE`."""

LOGIN_RE = re.compile(
    rf"^(?P<name>{PLAYER_NAME})\[/(?P<address>[^\]]+)\] logged in with entity id "
    r"(?P<entity_id>\d+) at \((?P<position>.*)\)$"
)
"""``Hypixelite[/115.99.245.156:49237] logged in with entity id 74 at ([minecraft:overworld]...)``.

**Contains a player's IP address.** It is captured because the log pipeline stitches it onto the
following :class:`~mcmanager.core.events.PlayerJoined` (idle logic and abuse investigation want
it, and it cannot be recovered later), and because ``Event.raw`` would carry it regardless.
Presenters must drop it; it is never relayed to Discord.
"""

LOST_CONN_RE = re.compile(
    rf"^(?P<name>{PLAYER_NAME})"
    rf"(?: \((?:(?P<uuid>{_UUID})|/(?P<address>[^)]+))\))?"
    r" lost connection: (?P<reason>.*)$",
    re.DOTALL,
)
"""Three real spellings of the same fact, all present in ``tests/fixtures/logs/archives/``::

    Hypixelite lost connection: Disconnected
    bharath_720 lost connection: Server closed
    Hypixelite (403d4fb2-f466-3716-b9cb-a3e769bb40c9) lost connection: Disconnected
    KittyScan (/176.65.148.158:58184) lost connection: Disconnected

Emitted immediately *before* the corresponding ``left the game`` line, which is why the reason is
stashed by the pipeline rather than attached here: the parser is stateless by contract.

The second and third spellings are **not** speculative padding. Neither was in the plan's sampled
lines; both were found by the unrecognised-line canary in
``tests/games/test_parser.py`` running over the real archives, which is the entire argument for
committing real fixtures instead of lines somebody typed from memory. The third comes from a
port scanner that authenticated and hung up without ever joining, so ``lost connection`` does not
imply a preceding ``joined the game`` and the pipeline must not assume one.

The parenthesised address is another **player IP**. Never relay it.
"""

COMMAND_RE = re.compile(
    rf"^(?P<name>{PLAYER_NAME}) issued server command: (?P<command>.*)$", re.DOTALL
)
"""``Hypixelite issued server command: /gamemode creative``.

The single most common player-attributed line in the archives after chat. Recognised rather than
left to fall through, for three reasons: it is audit-relevant (somebody gave themselves creative
mode), it keeps the canary honest, and a command like ``/say Notch fell from a high place`` would
otherwise reach the death table on the *echo* line as well as producing a real chat line.
"""

KICKED_RE = re.compile(rf"^(?P<name>{PLAYER_NAME}) was kicked due to (?P<reason>.*)$", re.DOTALL)
"""``bharath_720 was kicked due to keepalive timeout!``.

Matched before the death table on purpose. No vanilla death template begins ``was kicked``, but
the table is a linear scan of ~100 patterns and one future addition colliding here would silently
turn a disconnect into a death.
"""

UUID_RE = re.compile(rf"^UUID of player (?P<name>{PLAYER_NAME}) is (?P<uuid>{_UUID})$")
"""``UUID of player Hypixelite is 403d4fb2-f466-3716-b9cb-a3e769bb40c9``.

``online-mode=false`` on this server, so this is an **offline v3 UUID**: derived from
``OfflinePlayer:<name>``, stable per name, and *not* a Mojang UUID. Never send it to a Mojang API.
Session statistics key on the name.
"""

# --------------------------------------------------------------------------------- advancements

ADVANCEMENT_RE = re.compile(
    rf"^(?P<name>{PLAYER_NAME}) has "
    r"(?P<kind>made the advancement|reached the goal|completed the challenge) "
    r"\[(?P<title>.+)\]$"
)
"""``Hypixelite has made the advancement [Stone Age]``.

The title group is greedy so that a title containing a ``]`` survives; the pattern is anchored at
both ends, so greed cannot escape past the final bracket.
"""

MATCH_ORDER: tuple[str, ...] = (
    "READY",
    "VERSION",
    "STOPPING",
    "CHAT",
    "EMOTE",
    "SAY",
    "RCON_ECHO",
    "JOIN",
    "LEAVE",
    "LOGIN",
    "LOST_CONN",
    "KICKED",
    "COMMAND",
    "UUID",
    "ADVANCEMENT",
    "DEATHS",
)
"""The single source of truth for the order :func:`mcmanager.games.minecraft.parser.parse` tries.

Exported so a test can assert the order the parser actually implements matches the documented one,
rather than the two drifting silently.
"""


# --------------------------------------------------------------- enrichment matchers (stateless)
#
# The parser stays pure and stateless; these are what services/log_pipeline.py calls to stitch
# adjacent lines together. They live here because the regexes do, and because a second copy of
# LOGIN_RE in the pipeline is exactly how the two would drift.


@dataclass(frozen=True, slots=True, kw_only=True)
class LoginMatch:
    """``name[/address] logged in ...``, split up.

    Attributes:
        name: The player.
        address: ``ip:port`` as logged. Never relay this.
        entity_id: The entity the player was assigned. Diagnostic only.
        position: The spawn position, as written. Diagnostic only.
    """

    name: str
    address: str
    entity_id: int
    position: str

    @property
    def ip(self) -> str:
        """The address with the ephemeral port removed. Still never relay it.

        IPv4 addresses log as ``115.99.245.156:49237``; a bare IPv6 address contains colons of its
        own, so only a *trailing* ``:digits`` is removed, and only when what precedes it is not
        itself unparsable.
        """
        head, sep, tail = self.address.rpartition(":")
        if sep and tail.isdigit():
            return head
        return self.address


@dataclass(frozen=True, slots=True, kw_only=True)
class LostConnectionMatch:
    """``name lost connection: reason``, with the reason already classified.

    Attributes:
        name: The player.
        uuid: Present only in the log spelling that embeds it. Offline v3; see :data:`UUID_RE`.
        address: Present only in the spelling that embeds an address instead. ``ip:port``.
            **Never relay this.**
        raw_reason: The reason exactly as logged, e.g. ``"Server closed"``.
        reason: :attr:`~mcmanager.core.types.LeaveReason.SERVER_CLOSED` and friends.
    """

    name: str
    uuid: str | None
    address: str | None
    raw_reason: str
    reason: LeaveReason


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandMatch:
    """``name issued server command: /gamemode creative``.

    Attributes:
        name: The player who ran it.
        command: The command including its leading slash. Player-controlled text; escape it.
    """

    name: str
    command: str


@dataclass(frozen=True, slots=True, kw_only=True)
class KickMatch:
    """``name was kicked due to reason``."""

    name: str
    raw_reason: str
    reason: LeaveReason = LeaveReason.KICKED


@dataclass(frozen=True, slots=True, kw_only=True)
class UuidMatch:
    """``UUID of player name is uuid``. The UUID is offline v3; see :data:`UUID_RE`."""

    name: str
    uuid: str


_SERVER_CLOSED_REASONS = ("server closed", "server is restarting", "server shutdown")
_TIMEOUT_REASONS = ("timed out", "timeout", "keepalive")
_KICK_REASONS = ("kicked", "banned", "you are banned")


def classify_leave_reason(raw_reason: str) -> LeaveReason:
    """Map a disconnect reason string onto a :class:`~mcmanager.core.types.LeaveReason`.

    Substring matching is safe *here* and only here: this input is a single already-extracted
    reason field, not a whole log line, so there is no prefix for a substring to accidentally hit
    and no way for a player to inject into it.

    ``Server closed`` is the one that has to be right. It means the player was dropped by a
    shutdown, and counting that as a voluntary quit is what corrupts a session summary. Anything
    unmapped returns :attr:`~mcmanager.core.types.LeaveReason.UNKNOWN` - never a guess.
    """
    lowered = raw_reason.strip().lower()
    if any(marker in lowered for marker in _SERVER_CLOSED_REASONS):
        return LeaveReason.SERVER_CLOSED
    if any(marker in lowered for marker in _TIMEOUT_REASONS):
        return LeaveReason.TIMED_OUT
    if any(marker in lowered for marker in _KICK_REASONS):
        return LeaveReason.KICKED
    if lowered.startswith("disconnected"):
        return LeaveReason.QUIT
    return LeaveReason.UNKNOWN


def match_login(message: str) -> LoginMatch | None:
    """Extract the address the next ``PlayerJoined`` should carry, or ``None``."""
    found = LOGIN_RE.match(message)
    if found is None:
        return None
    return LoginMatch(
        name=found["name"],
        address=found["address"],
        entity_id=int(found["entity_id"]),
        position=found["position"],
    )


def match_lost_connection(message: str) -> LostConnectionMatch | None:
    """Extract the reason the next ``PlayerLeft`` should carry, or ``None``."""
    found = LOST_CONN_RE.match(message)
    if found is None:
        return None
    raw_reason = found["reason"]
    return LostConnectionMatch(
        name=found["name"],
        uuid=found["uuid"],
        address=found["address"],
        raw_reason=raw_reason,
        reason=classify_leave_reason(raw_reason),
    )


def match_command(message: str) -> CommandMatch | None:
    """Extract a player-issued server command, or ``None``."""
    found = COMMAND_RE.match(message)
    if found is None:
        return None
    return CommandMatch(name=found["name"], command=found["command"])


def match_kick(message: str) -> KickMatch | None:
    """Extract a kick, or ``None``."""
    found = KICKED_RE.match(message)
    if found is None:
        return None
    return KickMatch(name=found["name"], raw_reason=found["reason"])


def match_uuid(message: str) -> UuidMatch | None:
    """Extract an offline v3 UUID announcement, or ``None``."""
    found = UUID_RE.match(message)
    if found is None:
        return None
    return UuidMatch(name=found["name"], uuid=found["uuid"])
