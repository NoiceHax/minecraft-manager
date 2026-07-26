"""Event to Discord message.

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

**Why sanitisation happens here and not at the gateway.** ``DiscordGateway.send`` takes an
already-rendered, already-escaped string and says so. If neutralising lived there instead, only the
messages that happened to travel that path would be safe, and the first caller that formatted its
own string would be an injection. Doing it here means the unsafe value never becomes a message at
all.

**Why this matters more than it looks.** The server runs ``online-mode=false``, so a player picks
their own name with no authentication. ``@everyone`` minus the ``@`` is a legal Minecraft username,
and a death message embeds the name verbatim. Every fragment that originates outside this process
goes through :func:`_clean`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, assert_never

from mcmanager.core.events import (
    ChatMessage,
    CommandIssued,
    ConsoleLog,
    IdleCancelled,
    IdleStarted,
    IdleStopTriggered,
    IdleWarning,
    PlayerAdvancement,
    PlayerDeath,
    PlayerJoined,
    PlayerLeft,
    RuntimeRestored,
    RuntimeUnavailable,
    ServerCrashed,
    ServerReady,
    ServerStarting,
    ServerStopped,
    ServerStopping,
)
from mcmanager.core.types import AdvancementKind, ChatKind, LeaveReason

if TYPE_CHECKING:
    from mcmanager.core.events import (
        AnyEvent,
        Event,
        IdleEventU,
        PlayerEventU,
        ServerEventU,
    )

__all__ = [
    "MAX_MESSAGE_LENGTH",
    "TRUNCATION_MARKER",
    "escape_markdown",
    "neutralise_mentions",
    "render_event",
    "render_raw",
    "sanitize",
]

MAX_MESSAGE_LENGTH = 1900
"""Discord's limit is 2000; the margin leaves room for the truncation marker and a code fence."""

TRUNCATION_MARKER: Final = " [...truncated]"

_ZWSP: Final = "​"
"""Zero-width space. Inserted after the ``@`` so the text still reads as ``@everyone`` to a human
while Discord's parser no longer sees a mention."""

_MENTION_KEYWORD_RE: Final = re.compile(r"@(everyone|here)")
_MENTION_TAG_RE: Final = re.compile(r"<(@[!&]?|#)(\d+)>")

_MARKDOWN_CHARS: Final = "\\`*_~|>[]()"
_MARKDOWN_RE: Final = re.compile("([" + re.escape(_MARKDOWN_CHARS) + "])")

_CONTROL_RE: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
"""Everything unprintable except tab and newline. Paper writes neither into a chat body, but a
client can, and a null byte in a JSON payload is a 400 for the whole batched message."""

# ------------------------------------------------------------------------------- sanitisation


def sanitize(text: str) -> str:
    """Make an arbitrary string safe to put in a JSON payload at all.

    Strips control characters and replaces lone surrogates. Surrogates are reachable because the
    log stream is decoded with ``errors="replace"`` upstream, but a plugin can still emit one, and
    a lone surrogate raises ``UnicodeEncodeError`` at serialisation time - which would take out the
    whole batched message rather than the single bad line.
    """
    cleaned = _CONTROL_RE.sub("", text)
    return cleaned.encode("utf-8", "replace").decode("utf-8", "replace")


def neutralise_mentions(text: str) -> str:
    """Make ``@everyone``, ``@here``, ``<@id>``, ``<@&role>`` and ``<#channel>`` inert.

    Applied to **every** string that reaches Discord, not only to chat, because a player name, an
    advancement title and a death message are all attacker-influenced too.
    """
    out = _MENTION_KEYWORD_RE.sub(lambda m: "@" + _ZWSP + m.group(1), text)
    return _MENTION_TAG_RE.sub(lambda m: "<" + _ZWSP + m.group(1) + m.group(2) + ">", out)


def escape_markdown(text: str) -> str:
    """Escape backticks, code fences and the rest of Discord's markdown."""
    return _MARKDOWN_RE.sub(r"\\\1", text)


def _clean(text: str) -> str:
    """The full inbound pipeline for any attacker-influenced fragment."""
    return neutralise_mentions(escape_markdown(sanitize(text)))


def _truncate(text: str, limit: int = MAX_MESSAGE_LENGTH) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER


# ------------------------------------------------------------------------------------ helpers


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "an unknown time"
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _players(count: int) -> str:
    return "1 player" if count == 1 else f"{count} players"


_LEAVE_SUFFIX: Final[dict[LeaveReason, str]] = {
    LeaveReason.TIMED_OUT: " (timed out)",
    LeaveReason.KICKED: " (kicked)",
    LeaveReason.SERVER_CLOSED: " (server closing)",
}

_ADVANCEMENT_VERB: Final[dict[AdvancementKind, str]] = {
    AdvancementKind.ADVANCEMENT: "earned the advancement",
    AdvancementKind.GOAL: "reached the goal",
    AdvancementKind.CHALLENGE: "completed the challenge",
}

# ------------------------------------------------------------------------------------- render


def _render_server(event: ServerEventU) -> str:
    match event:
        case ServerStarting():
            who = f" (requested by {_clean(event.requested_by)})" if event.requested_by else ""
            return f":hourglass: **Server starting**{who}"
        case ServerReady():
            version = f" `{_clean(event.version)}`" if event.version else ""
            took = f" in {event.startup_seconds:.1f}s" if event.startup_seconds else ""
            return f":white_check_mark: **Server ready**{version}{took}"
        case ServerStopping():
            who = f" (requested by {_clean(event.requested_by)})" if event.requested_by else ""
            return f":octagonal_sign: **Server stopping**{who}"
        case ServerStopped():
            if event.forced:
                return (
                    ":warning: **Server stopped, but had to be force-killed** after "
                    f"{_duration(event.uptime_seconds)} up. It did not finish saving in time."
                )
            return f":black_circle: **Server stopped** after {_duration(event.uptime_seconds)} up"
        case ServerCrashed():
            why = " (out of memory)" if event.oom_killed else ""
            code = f", exit {event.exit_code}" if event.exit_code is not None else ""
            body = f":boom: **Server crashed**{why}{code}"
            if event.tail:
                tail = "\n".join(sanitize(line) for line in event.tail[-8:])
                body += f"\n```\n{_truncate(tail, 1200)}\n```"
            return body
        case _ as unreachable:
            assert_never(unreachable)


def _render_player(event: PlayerEventU) -> str:
    match event:
        case PlayerJoined():
            # event.address is deliberately never rendered. See the module docstring.
            first = " *(first time!)*" if event.first_seen else ""
            return (
                f":green_circle: **{_clean(event.player.name)}** joined{first} "
                f"({_players(event.online_count)} online)"
            )
        case PlayerLeft():
            suffix = _LEAVE_SUFFIX.get(event.reason, "")
            played = (
                f" after {_duration(event.session_seconds)}"
                if event.session_seconds is not None
                else ""
            )
            return (
                f":red_circle: **{_clean(event.player.name)}** left{suffix}{played} "
                f"({_players(event.online_count)} online)"
            )
        case PlayerDeath():
            return _truncate(f":skull: {_clean(event.message)}")
        case PlayerAdvancement():
            verb = _ADVANCEMENT_VERB.get(event.kind, "earned the advancement")
            return f":trophy: **{_clean(event.player.name)}** {verb} **{_clean(event.title)}**"
        case _ as unreachable:
            assert_never(unreachable)


def _render_idle(event: IdleEventU) -> str | None:
    match event:
        case IdleStarted():
            prefix = "*(dry run)* " if event.dry_run else ""
            return (
                f":new_moon: {prefix}Nobody is online. Stopping in "
                f"{_duration(event.timeout_seconds)} unless somebody joins."
            )
        case IdleWarning():
            prefix = "*(dry run)* " if event.dry_run else ""
            return (
                f":warning: {prefix}Idle shutdown in {_duration(event.remaining_seconds)}. "
                "Join now to cancel it."
            )
        case IdleCancelled():
            # Only the shutdown case is worth announcing. "Somebody joined" already produced a
            # PlayerJoined message a moment earlier, so announcing the cancellation too is noise.
            if event.reason == "daemon_shutdown":
                return ":information_source: Manager restarting; idle countdown cancelled."
            return None
        case IdleStopTriggered():
            if event.dry_run:
                return (
                    ":new_moon: *(dry run)* Would have stopped the server after "
                    f"{_duration(event.idle_seconds)} idle."
                )
            return f":new_moon: Stopping the server after {_duration(event.idle_seconds)} idle."
        case _ as unreachable:
            assert_never(unreachable)


def render_event(event: AnyEvent) -> str | None:
    """Render one event, or ``None`` when this event is not announced at all.

    A ``match`` over the union ending in ``assert_never``, so adding a nineteenth event is a type
    error here rather than a silent fall-through to a base-class rendering.

    Returning ``None`` is a decision, not a gap. ``ConsoleLog`` belongs to the console relay rather
    than the events channel; an accepted ``CommandIssued`` is redundant because the resulting
    ``Server*`` event says the same thing better; and an RCON chat echo is our own console output
    coming back, which would loop.
    """
    match event:
        case (
            ServerStarting() | ServerReady() | ServerStopping() | ServerStopped() | ServerCrashed()
        ):
            return _render_server(event)

        case PlayerJoined() | PlayerLeft() | PlayerDeath() | PlayerAdvancement():
            return _render_player(event)

        case IdleStarted() | IdleWarning() | IdleCancelled() | IdleStopTriggered():
            return _render_idle(event)

        case ChatMessage():
            name = _clean(event.player.name)
            body = _clean(event.message)
            match event.kind:
                case ChatKind.EMOTE:
                    return _truncate(f":speech_balloon: \\* {name} {body}")
                case ChatKind.SAY:
                    return _truncate(f":loudspeaker: **\\[Server]** {body}")
                case ChatKind.RCON:
                    return None  # our own console echo; relaying it would loop
                case ChatKind.CHAT:
                    return _truncate(f":speech_balloon: **{name}**: {body}")
            return _truncate(f":speech_balloon: **{name}**: {body}")

        case RuntimeUnavailable():
            return (
                ":satellite: **Lost contact with Docker.** The server's state is unknown until it "
                "comes back, and nothing will be started or stopped in the meantime."
            )

        case RuntimeRestored():
            return f":satellite: Docker is back after {_duration(event.downtime_seconds)}."

        case ConsoleLog():
            return None  # the console relay's job, not the events channel's

        case CommandIssued():
            if event.accepted:
                return None  # the resulting Server* event is the interesting one
            reason = _clean(event.rejection) if event.rejection else "not permitted"
            return (
                f":no_entry: `/{event.action.value}` from **{_clean(event.actor)}** "
                f"was refused: {reason}"
            )

        case _ as unreachable:
            assert_never(unreachable)


def render_raw(event: Event) -> str | None:
    """Render ``Event.raw`` for the console relay, escaped and truncated.

    ``Event.raw`` rather than ``ConsoleLog.message`` because recognised lines deliberately do not
    also emit a ``ConsoleLog``: a relay reading only ``ConsoleLog`` would show gaps exactly where
    the joins, the chat and the deaths were.

    Markdown is **not** escaped here, because the relay wraps its batch in a code fence where
    markdown does not apply. Mentions still are: a code fence does not stop a ping.
    """
    if event.raw is None:
        return None
    line = sanitize(event.raw).rstrip()
    if not line:
        return None
    return _truncate(neutralise_mentions(line).replace("```", "`​``"))
