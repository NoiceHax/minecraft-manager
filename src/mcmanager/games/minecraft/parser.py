"""Raw line to event.

The purity contract, restated because everything else depends on it:

- ``parse(raw, *, ts, server_id, stream) -> Event`` is **total**: it always returns an ``Event``,
  never ``None``, never raises. Unrecognised input becomes a ``ConsoleLog``.
- Module state is compiled regexes. No clock, no config, no bus, no I/O, no logging. ``ts`` is
  injected.
- **Stateless.** Anything needing memory - stitching a login IP onto the following join, a
  disconnect reason onto the following leave, knowing the online roster - is
  ``services/log_pipeline.py``'s job. See :mod:`~mcmanager.games.minecraft.patterns` for the
  matchers it calls.

Because of that contract, ``mcmanager replay`` can run a ``.log.gz`` archive through the parser
with no daemon, no Docker and no network, which is both the development loop for the parser and
the test of the contract itself.

:func:`scan` exports the unrecognised-line ratio: the canary for a Paper upgrade breaking the
patterns. It fails loudly in CI instead of letting events silently disappear in production, which
is the actual failure mode - a parser that stops matching does not crash, it goes quiet.

Two decisions worth stating outright, because both deviate from the letter of the plan and both
are driven by real sampled lines:

**1. The thread guard is per-pattern, not global.** The plan says only ``Server thread`` /
``INFO`` lines are candidates for player *and chat* events. On this server that would discard
**100% of chat**: Paper emits every chat line from ``Async Chat Thread - #N``. Verbatim from the
archive::

    [14:48:06] [Async Chat Thread - #0/INFO]: [Not Secure] <bharath_720> yo

So the guard is: player-lifecycle patterns require ``Server thread``; chat patterns accept
``Server thread`` (``/say`` and ``[Rcon]`` are emitted there) or ``Async Chat Thread - #N``; the
UUID matcher accepts ``User Authenticator #N``. Every other thread - ``ServerMain``,
``RCON Listener``, ``Worker-Main-1`` - is console output and nothing else. The guard is still
strict; it is just correct about which threads exist.

**2. ``Event.raw`` carries the *sanitised* line, not the byte-for-byte original.** Every event
carries ``raw`` so the Discord console relay can render recognised lines too. Putting escape bytes
back into that field would re-create, one layer down, the exact defect this project exists to fix.
``mcmanager logs --raw`` bypasses the parser entirely and still shows the true original.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mcmanager.core.events import (
    ChatMessage,
    ConsoleLog,
    Event,
    PlayerAdvancement,
    PlayerDeath,
    PlayerJoined,
    PlayerLeft,
    ServerReady,
    ServerStarting,
    ServerStopping,
)
from mcmanager.core.types import (
    AdvancementKind,
    ChatKind,
    LineOrigin,
    PlayerRef,
    ReadySignal,
    Source,
    Stream,
)
from mcmanager.games.minecraft.ansi import sanitize_line
from mcmanager.games.minecraft.deaths import match_death
from mcmanager.games.minecraft.lines import ParsedLine, classify
from mcmanager.games.minecraft.patterns import (
    ADVANCEMENT_RE,
    CHAT_RE,
    COMMAND_RE,
    EMOTE_RE,
    JOIN_RE,
    KICKED_RE,
    LEAVE_RE,
    LOGIN_RE,
    LOST_CONN_RE,
    PLAYER_NAME,
    RCON_BRACKET_RE,
    RCON_ECHO_RE,
    RCON_PSEUDO_NAME,
    READY_RE,
    SAY_RE,
    STOPPING_RE,
    UUID_RE,
    VERSION_RE,
    match_uuid,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mcmanager.core.types import ServerId

__all__ = ["ParseStats", "parse", "scan"]

_ADVANCEMENT_KINDS = {
    "made the advancement": AdvancementKind.ADVANCEMENT,
    "reached the goal": AdvancementKind.GOAL,
    "completed the challenge": AdvancementKind.CHALLENGE,
}

_METRIC_TS = datetime(1970, 1, 1, tzinfo=UTC)
"""The timestamp :func:`scan` hands to :func:`parse`.

:func:`scan` measures classification, not chronology, and the parser must not read a clock. A
fixed epoch keeps the function pure and its output reproducible.
"""


def parse(raw: str, *, ts: datetime, server_id: ServerId, stream: Stream) -> Event:
    """Parse one raw log line into exactly one event. Total; never raises.

    Args:
        raw: One line of container output, newline already removed, ANSI still present.
        ts: The event timestamp, injected. Comes from Docker's RFC3339Nano log prefix, never from
            the ``[13:24:37]`` in the line: that is time-only, in ``Asia/Kolkata``, and simply
            wrong for a backfilled line.
        server_id: Which managed server this line came from.
        stream: stdout or stderr.

    Returns:
        Exactly one event. A line that matches no pattern becomes a
        :class:`~mcmanager.core.events.ConsoleLog`, which is what keeps the log pipeline alive
        through a Paper format change instead of taking it down.
    """
    try:
        return _parse(raw, ts=ts, server_id=server_id, stream=stream)
    except Exception as exc:
        # A parser that can throw takes the log pipeline down with it, and the input is attacker
        # -influenced (chat is a player-controlled string that reaches these regexes). The
        # fallback keeps the line and records the failure in the event itself rather than
        # swallowing it - this module is not allowed to log.
        return ConsoleLog(
            ts=ts,
            server_id=server_id,
            source=Source.LOG,
            raw=sanitize_line(raw),
            message=f"[mcmanager: parser error {type(exc).__name__}] {sanitize_line(raw)}",
            origin=LineOrigin.RAW,
            stream=stream,
        )


def _parse(raw: str, *, ts: datetime, server_id: ServerId, stream: Stream) -> Event:
    sanitized = sanitize_line(raw)
    line = classify(sanitized)

    if line.origin is LineOrigin.SERVER:
        event = _match_server_line(line, ts=ts, server_id=server_id, sanitized=sanitized)
        if event is not None:
            return event

    return _console(line, ts=ts, server_id=server_id, sanitized=sanitized, stream=stream)


def _match_server_line(
    line: ParsedLine,
    *,
    ts: datetime,
    server_id: ServerId,
    sanitized: str,
) -> Event | None:
    """Try the fixed match order against a ``SERVER``-origin payload.

    Returns ``None`` when nothing matched, or when the line matched something that is deliberately
    represented as a ``ConsoleLog`` (``LOGIN``, ``LOST_CONN``, ``KICKED``, ``UUID``, the
    ``[Rcon: ...]`` echo). Those are recognised - they just have no event of their own, and the
    facts they carry belong to a neighbouring event that only the stateful pipeline can build.
    """
    message = line.message

    if line.is_server_thread_info:
        # READY
        ready = READY_RE.match(message)
        if ready is not None:
            return ServerReady(
                ts=ts,
                server_id=server_id,
                source=Source.LOG,
                raw=sanitized,
                startup_seconds=float(ready["seconds"].replace(",", "")),
                detected_by=ReadySignal.LOG_DONE,
            )

        # VERSION
        version = VERSION_RE.match(message)
        if version is not None:
            return ServerStarting(
                ts=ts,
                server_id=server_id,
                source=Source.LOG,
                raw=sanitized,
                version=version["version"],
            )

        # STOPPING
        if STOPPING_RE.match(message) is not None:
            return ServerStopping(
                ts=ts,
                server_id=server_id,
                source=Source.LOG,
                raw=sanitized,
                reason=message,
            )

    if line.is_server_thread_info or line.is_chat_thread_info:
        chat = _match_chat(line, ts=ts, server_id=server_id, sanitized=sanitized)
        if chat is not None:
            return chat

    if line.is_server_thread_info:
        # JOIN - after chat, so "<Bob> Alice joined the game" can never forge one.
        join = JOIN_RE.match(message)
        if join is not None:
            return PlayerJoined(
                ts=ts,
                server_id=server_id,
                source=Source.LOG,
                raw=sanitized,
                player=PlayerRef(name=join["name"]),
            )

        # LEAVE. reason stays UNKNOWN here: it arrives on the preceding "lost connection" line,
        # and correlating two lines is state, which belongs to the pipeline.
        leave = LEAVE_RE.match(message)
        if leave is not None:
            return PlayerLeft(
                ts=ts,
                server_id=server_id,
                source=Source.LOG,
                raw=sanitized,
                player=PlayerRef(name=leave["name"]),
            )

        # LOGIN / LOST_CONN / KICKED / COMMAND - recognised, but they enrich a neighbouring event
        # or an audit trail rather than becoming an event of their own. See patterns.match_login /
        # match_lost_connection / match_kick / match_command.
        if (
            LOGIN_RE.match(message) is not None
            or LOST_CONN_RE.match(message) is not None
            or KICKED_RE.match(message) is not None
            or COMMAND_RE.match(message) is not None
        ):
            return None

        # ADVANCEMENT
        advancement = ADVANCEMENT_RE.match(message)
        if advancement is not None:
            return PlayerAdvancement(
                ts=ts,
                server_id=server_id,
                source=Source.LOG,
                raw=sanitized,
                player=PlayerRef(name=advancement["name"]),
                title=advancement["title"],
                kind=_ADVANCEMENT_KINDS[advancement["kind"]],
            )

        # DEATHS - the linear scan, reached only by lines nothing else claimed.
        death = match_death(message)
        if death is not None:
            return PlayerDeath(
                ts=ts,
                server_id=server_id,
                source=Source.LOG,
                raw=sanitized,
                player=PlayerRef(name=death.player),
                message=message,
                killer=death.killer,
                item=death.item,
                template=death.template,
            )

    if line.is_authenticator_info and UUID_RE.match(message) is not None:
        # Recognised; the name/uuid pair is stitched onto the next PlayerJoined by the pipeline.
        return None

    return None


def _match_chat(
    line: ParsedLine,
    *,
    ts: datetime,
    server_id: ServerId,
    sanitized: str,
) -> ChatMessage | None:
    """CHAT -> EMOTE -> SAY -> RCON_ECHO, in that fixed order."""
    message = line.message

    chat = CHAT_RE.match(message)
    if chat is not None:
        return ChatMessage(
            ts=ts,
            server_id=server_id,
            source=Source.LOG,
            raw=sanitized,
            player=PlayerRef(name=chat["name"]),
            message=chat["message"],
            kind=ChatKind.CHAT,
        )

    emote = EMOTE_RE.match(message)
    if emote is not None:
        return ChatMessage(
            ts=ts,
            server_id=server_id,
            source=Source.LOG,
            raw=sanitized,
            player=PlayerRef(name=emote["name"]),
            message=emote["message"],
            kind=ChatKind.EMOTE,
        )

    say = SAY_RE.match(message)
    if say is not None:
        return ChatMessage(
            ts=ts,
            server_id=server_id,
            source=Source.LOG,
            raw=sanitized,
            player=PlayerRef(name=say["name"]),
            message=say["message"],
            kind=ChatKind.SAY,
        )

    rcon = RCON_ECHO_RE.match(message)
    if rcon is not None:
        return ChatMessage(
            ts=ts,
            server_id=server_id,
            source=Source.LOG,
            raw=sanitized,
            player=PlayerRef(name=RCON_PSEUDO_NAME),
            message=rcon["message"],
            kind=ChatKind.RCON,
        )

    return None


def _console(
    line: ParsedLine,
    *,
    ts: datetime,
    server_id: ServerId,
    sanitized: str,
    stream: Stream,
) -> ConsoleLog:
    return ConsoleLog(
        ts=ts,
        server_id=server_id,
        source=Source.LOG,
        raw=sanitized,
        message=line.message,
        level=line.level,
        thread=line.thread,
        origin=line.origin,
        stream=stream,
    )


# ------------------------------------------------------------------------------- the canary


def _is_recognised_console(line: ParsedLine) -> bool:
    """Did a ``ConsoleLog`` line nonetheless match a known pattern?

    These are the four enrichment lines plus the ``[Rcon: ...]`` command echo. They are
    represented as ``ConsoleLog`` but they are *not* evidence that the grammar has drifted, so the
    canary must not count them.
    """
    message = line.message
    if line.is_authenticator_info:
        return UUID_RE.match(message) is not None
    if not line.is_server_thread_info:
        return False
    return (
        LOGIN_RE.match(message) is not None
        or LOST_CONN_RE.match(message) is not None
        or KICKED_RE.match(message) is not None
        or COMMAND_RE.match(message) is not None
        or RCON_BRACKET_RE.match(message) is not None
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ParseStats:
    """What :func:`scan` measured over a batch of lines.

    **Why the ratio is not simply "unrecognised / all server lines".** That was the obvious
    design, it was implemented, and measuring it against the 3,419 real archive lines produced
    **0.895**. It is not broken - roughly nine in ten ``Server thread/INFO`` lines genuinely are
    console chatter (``Preparing spawn area: 2%``, ``Saving chunks for level ...``,
    ``Halted I/O scheduler for world 'minecraft:the_end'``) and always will be. A canary that
    idles at 0.895 cannot detect nineteen join lines going quiet, which is exactly the failure it
    exists to catch.

    So the population is narrowed to lines that provably concern a **known player**, where "known"
    means the same batch elsewhere produced a chat message, a join, a leave or a UUID
    announcement for that name. Console chatter never begins with a player's name, and a name can
    only enter the set by way of a pattern that already matched, so the denominator cannot be
    poisoned by a parser that is already broken. If Paper renames ``joined the game``, those lines
    stop matching but the players are still known from chat, they land in
    :attr:`unrecognised_player_lines`, and the ratio jumps off zero.

    This two-pass shape is legitimate here and nowhere else: :func:`scan` is a batch analysis tool,
    not the parser, and it is allowed to hold state across lines.

    Attributes:
        total: Every line handed in.
        server_lines: ``Server thread`` / ``INFO`` lines.
        console_lines: Of those, how many produced a bare ``ConsoleLog``. Reported for context, not
            used as a gate.
        known_players: Names the batch proved real. The denominator's basis.
        player_lines: ``Server thread`` / ``INFO`` lines whose payload starts with a known player's
            name. The canary's population.
        unrecognised_player_lines: Player lines that produced a bare ``ConsoleLog`` and matched no
            enrichment pattern. The canary's numerator.
        samples: Up to ``sample_limit`` unrecognised player payloads, so a failing assertion says
            *what* stopped matching rather than only that something did.
        by_event: Count per event type name, for a plausibility check on a replay. Independently
            useful as a canary: ``PlayerJoined`` falling to zero over an archive that contains
            chat is unambiguous.
    """

    total: int
    server_lines: int
    console_lines: int
    known_players: frozenset[str]
    player_lines: int
    unrecognised_player_lines: int
    samples: tuple[str, ...]
    by_event: dict[str, int]

    @property
    def unrecognised_ratio(self) -> float:
        """Unrecognised player lines over all player lines. ``0.0`` when there were none.

        This is ``parser.unrecognized_server_lines`` from the plan, with the denominator corrected
        as described above. A rise here is the visible symptom of a Paper upgrade changing a
        message, which otherwise manifests as events silently ceasing to exist.
        """
        if self.player_lines == 0:
            return 0.0
        return self.unrecognised_player_lines / self.player_lines

    @property
    def console_ratio(self) -> float:
        """Bare ``ConsoleLog`` over all server lines. Context only; ~0.9 is normal."""
        if self.server_lines == 0:
            return 0.0
        return self.console_lines / self.server_lines

    def __repr__(self) -> str:
        return (
            f"ParseStats(total={self.total}, server_lines={self.server_lines}, "
            f"players={len(self.known_players)}, player_lines={self.player_lines}, "
            f"unrecognised={self.unrecognised_player_lines}, "
            f"ratio={self.unrecognised_ratio:.4f})"
        )


def _named_player(event: Event) -> str | None:
    """The player a successfully parsed event proves exists, if any."""
    if isinstance(event, PlayerJoined | PlayerLeft | PlayerDeath | PlayerAdvancement):
        return event.player.name
    if isinstance(event, ChatMessage) and event.kind is not ChatKind.RCON:
        return event.player.name
    return None


def scan(
    lines: Iterable[str],
    *,
    server_id: ServerId = "scan",
    sample_limit: int = 20,
) -> ParseStats:
    """Run lines through :func:`parse` and report the unrecognised-player-line ratio.

    Pure: no clock, no I/O, no logging. The caller supplies the lines (``mcmanager replay`` reads
    the ``.log.gz``; the test suite reads the committed fixtures), and every event is timestamped
    with a fixed epoch because only the classification is being measured.

    The input is materialised into a list: this is a batch tool run over an archive or in a test,
    never on the hot path, and two passes are what make the denominator meaningful.
    """
    batch = list(lines)
    parsed: list[tuple[ParsedLine, Event]] = []
    by_event: dict[str, int] = {}
    known: set[str] = set()

    for raw in batch:
        event = parse(raw, ts=_METRIC_TS, server_id=server_id, stream=Stream.STDOUT)
        line = classify(sanitize_line(raw))
        parsed.append((line, event))
        by_event[event.name] = by_event.get(event.name, 0) + 1

        name = _named_player(event)
        if name is not None:
            known.add(name)
        uuid_match = match_uuid(line.message) if line.is_authenticator_info else None
        if uuid_match is not None:
            known.add(uuid_match.name)

    server_lines = 0
    console_lines = 0
    player_lines = 0
    unrecognised = 0
    samples: list[str] = []

    for line, event in parsed:
        if not line.is_server_thread_info:
            continue
        server_lines += 1
        bare_console = isinstance(event, ConsoleLog) and not _is_recognised_console(line)
        if isinstance(event, ConsoleLog):
            console_lines += 1
        if _leading_name(line.message) not in known:
            continue
        player_lines += 1
        if bare_console:
            unrecognised += 1
            if len(samples) < sample_limit:
                samples.append(line.message)

    return ParseStats(
        total=len(batch),
        server_lines=server_lines,
        console_lines=console_lines,
        known_players=frozenset(known),
        player_lines=player_lines,
        unrecognised_player_lines=unrecognised,
        samples=tuple(samples),
        by_event=by_event,
    )


_LEADING_NAME_RE = re.compile(rf"^(?P<name>{PLAYER_NAME})(?=[ \[])")


def _leading_name(message: str) -> str | None:
    """The first token of a payload, if it could be a player name.

    ``[`` is accepted as a terminator as well as a space, because the login line is
    ``Hypixelite[/115.99.245.156:49237] logged in ...``.
    """
    found = _LEADING_NAME_RE.match(message)
    return found["name"] if found is not None else None
