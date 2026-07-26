"""Line grammar classification.

Step two: decide which of the interleaved grammars a sanitised line belongs to, producing a
:class:`ParsedLine`. Three real shapes plus a catch-all, all on one stream::

    [13:24:37] [Server thread/INFO]: Hypixelite joined the game        SERVER
    [22:41:13] [Server thread/INFO] [WorldEdit]: Loaded 1 world        SERVER, with logger
    2026-07-25T22:58:20.809+0530\\tINFO\\tmc-server-runner\\tDone        WRAPPER
    \\tat net.minecraft.world.level.levelgen.WorldDimensions.<init>()   CONTINUATION
    [init] Starting the Minecraft server...                            RAW

The server pattern includes an **optional second bracket group** -
``[13:24:37] [Server thread/INFO] [PluginName]: msg`` - which is the single most common reason a
naive regex silently stops matching after a Paper update or a plugin install.

``clock_hint`` (the ``[13:24:37]``) is kept for display only and is **never** authoritative: it
carries no date, it is in ``Asia/Kolkata`` on this deployment, and it is meaningless for
backfilled lines. Event timestamps come from Docker's RFC3339Nano prefix instead.

Thread names are not a fixed vocabulary, and the guards below are built on the ones actually
observed across 3,419 real archive lines plus the docker stream:

* ``Server thread`` (2,505 lines) - joins, leaves, deaths, advancements, readiness, shutdown.
* ``ServerMain`` (297) - boot chatter, emitted before the world exists.
* ``Async Chat Thread - #N`` (~300) - **all player chat**. See :data:`ASYNC_CHAT_THREAD_RE`.
* ``User Authenticator #N`` (22) - ``UUID of player X is ...``.
* ``RCON Listener #N`` / ``RCON Client /0:0:0:0:0:0:0:1 #N`` (~230) - command-channel noise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mcmanager.core.types import LineOrigin

__all__ = [
    "ASYNC_CHAT_THREAD_RE",
    "CONTINUATION_RE",
    "NOT_SECURE_PREFIX",
    "SERVER_RE",
    "SERVER_THREAD",
    "USER_AUTHENTICATOR_THREAD_RE",
    "WRAPPER_RE",
    "ParsedLine",
    "classify",
]

SERVER_RE = re.compile(
    r"""^
    \[(?P<clock>\d{1,2}:\d{2}:\d{2})\]         # log4j2 %d{HH:mm:ss}: time only, no date
    \s
    \[(?P<thread>[^\]]+)/(?P<level>[A-Z]+)\]   # greedy thread: "RCON Client /0:0:0:0:0:0:0:1 #2"
    (?:\s\[(?P<logger>[^\]]*)\])?              # the optional plugin/logger group, see module docs
    :\ ?
    (?P<message>.*)
    $""",
    re.VERBOSE | re.DOTALL,
)
"""The log4j2 ``PatternLayout`` this server is configured with.

``thread`` is greedy on purpose: a real thread name is ``RCON Client /0:0:0:0:0:0:0:1 #2`` and
contains slashes, so only the *last* slash inside the bracket separates it from the level.
"""

WRAPPER_RE = re.compile(
    r"""^
    (?P<ts>\d{4}-\d{2}-\d{2}T[0-9:.]+(?:Z|[+-]\d{2}:?\d{2}))
    \t(?P<level>[A-Za-z]+)
    \t(?P<logger>[^\t]*)
    \t(?P<message>.*)
    $""",
    re.VERBOSE | re.DOTALL,
)
"""``mc-server-runner``, the itzg image's entrypoint supervisor.

Tab separated, full date with offset, and present **only in the docker stream** - log4j2 never
sees these, so a fixture set built from ``/data/logs/*.log.gz`` alone would miss this grammar
entirely. Sampled verbatim::

    2026-07-25T22:58:18.720+0530\\tINFO\\tmc-server-runner\\tgracefully stopping server...
    2026-07-25T22:58:20.809+0530\\tINFO\\tmc-server-runner\\tDone

Its ``Done`` is **not** the server's ``Done (32.521s)!``: this one means the JVM has exited. They
are told apart by origin, never by substring, which is exactly the class of mistake the old bridge
script made with ``if "joined the game" in line``.
"""

CONTINUATION_RE = re.compile(r"^[\t ]+(?:at\s|\.{3}\s|Caused by:|Suppressed:)")
"""A stack frame or wrapped message belonging to the line above.

Anchored to the four shapes the JVM actually produces rather than "starts with whitespace",
because an indented line is otherwise indistinguishable from a plugin writing pretty output, and
misclassifying that as a continuation hides it from the unrecognised-line canary.
"""

SERVER_THREAD = "Server thread"
"""The main game loop. Every player lifecycle fact is emitted from here."""

ASYNC_CHAT_THREAD_RE = re.compile(r"^Async Chat Thread - #\d+$")
"""Paper dispatches chat off the main thread, on a pool with a per-message index.

This matters more than it looks. The obvious guard - "only ``Server thread`` lines are candidates
for player and chat events" - would silently discard **100% of chat** on this server, because
every sampled chat line reads
``[14:48:06] [Async Chat Thread - #0/INFO]: [Not Secure] <bharath_720> yo``. The guard is
therefore per-pattern rather than global: chat is accepted from this thread and from
``Server thread`` (which carries ``/say`` and ``[Rcon]`` output), and from nowhere else.
"""

USER_AUTHENTICATOR_THREAD_RE = re.compile(r"^User Authenticator #\d+$")
"""Where ``UUID of player X is ...`` is logged, again not on the main thread."""

NOT_SECURE_PREFIX = "[Not Secure] "
"""Paper's unsigned-chat marker, prepended to the payload of every chat line in offline mode.

``online-mode=false`` here, so it is on *every* chat message and must be removed before the
``^<name> message`` anchor can match. It is stripped in :func:`classify` rather than in each
pattern so that exactly one place knows about it.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedLine:
    """One sanitised log line, split into grammar and payload.

    Attributes:
        origin: Which of the interleaved grammars this line belongs to.
        message: The payload, with the prefix and any ``[Not Secure] `` marker removed. For
            :attr:`~mcmanager.core.types.LineOrigin.RAW` and ``CONTINUATION`` lines this is the
            whole sanitised line, because there is no prefix to remove.
        level: ``INFO`` / ``WARN`` / ``ERROR`` where the grammar provides one.
        thread: e.g. ``"Server thread"``, ``"Async Chat Thread - #0"``.
        logger: The optional second bracket group, i.e. the plugin name. Always ``None`` on this
            server, which runs zero plugins - captured anyway, because the day a plugin is
            installed is the day a parser that ignored that group stops matching.
        clock_hint: The ``[13:24:37]`` as written. **Display only, never authoritative**: no date,
            local timezone, and wrong for any backfilled line.
        was_not_secure: The payload carried Paper's unsigned-chat marker. Kept because it is the
            honest answer to "is this chat line attributable", and silently discarding the fact
            would be worse than carrying a flag nobody reads yet.
    """

    origin: LineOrigin
    message: str
    level: str | None = None
    thread: str | None = None
    logger: str | None = None
    clock_hint: str | None = None
    was_not_secure: bool = False

    @property
    def is_server_thread_info(self) -> bool:
        """The guard for player-lifecycle patterns: main game loop, informational level."""
        return (
            self.origin is LineOrigin.SERVER
            and self.thread == SERVER_THREAD
            and self.level == "INFO"
        )

    @property
    def is_chat_thread_info(self) -> bool:
        """The guard for chat patterns: Paper's async chat pool, informational level."""
        return (
            self.origin is LineOrigin.SERVER
            and self.level == "INFO"
            and self.thread is not None
            and ASYNC_CHAT_THREAD_RE.match(self.thread) is not None
        )

    @property
    def is_authenticator_info(self) -> bool:
        """The guard for ``UUID of player X is ...``."""
        return (
            self.origin is LineOrigin.SERVER
            and self.level == "INFO"
            and self.thread is not None
            and USER_AUTHENTICATOR_THREAD_RE.match(self.thread) is not None
        )


def classify(sanitized: str) -> ParsedLine:
    """Split one already-sanitised line into grammar and payload.

    Total and pure: never raises, never returns ``None``. Anything matching no known grammar comes
    back as :attr:`~mcmanager.core.types.LineOrigin.RAW` with the whole line as the message, which
    is what feeds the unrecognised-line ratio.

    The order is fixed and matters: ``CONTINUATION`` is tested first because an indented stack
    frame can contain anything at all, including text that looks like another grammar.
    """
    if CONTINUATION_RE.match(sanitized):
        return ParsedLine(origin=LineOrigin.CONTINUATION, message=sanitized)

    server = SERVER_RE.match(sanitized)
    if server is not None:
        payload = server["message"]
        was_not_secure = payload.startswith(NOT_SECURE_PREFIX)
        if was_not_secure:
            payload = payload[len(NOT_SECURE_PREFIX) :]
        return ParsedLine(
            origin=LineOrigin.SERVER,
            message=payload,
            level=server["level"],
            thread=server["thread"],
            logger=server["logger"],
            clock_hint=server["clock"],
            was_not_secure=was_not_secure,
        )

    wrapper = WRAPPER_RE.match(sanitized)
    if wrapper is not None:
        return ParsedLine(
            origin=LineOrigin.WRAPPER,
            message=wrapper["message"],
            level=wrapper["level"].upper(),
            thread=wrapper["logger"] or None,
            logger=wrapper["logger"] or None,
            clock_hint=wrapper["ts"],
        )

    return ParsedLine(origin=LineOrigin.RAW, message=sanitized)
