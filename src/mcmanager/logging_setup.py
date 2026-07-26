"""structlog configuration, routed **through** stdlib logging.

The important structural decision: structlog is not a parallel logging system here. Everything -
our own ``get_logger("mcmanager.bus")`` calls, docker-py's ``logging.getLogger("docker")``,
discord.py's gateway chatter, aiohttp's access log - goes through one stdlib handler wearing a
:class:`structlog.stdlib.ProcessorFormatter`. So a third-party warning about a dropped websocket
frame comes out with the same shape, the same timestamp format and the same bound context as our
own events, and ``docker logs -f mcmanager | jq 'select(.level=="warning")'`` sees all of it.

``json`` for the deployed daemon, ``console`` for humans, selected by ``logs.format``.

**Chat bodies never appear in the ops log at INFO.** Chat is user-generated content, which makes
it two problems at once: a log-injection surface (a player can type a newline and a fake JSON
object) and a privacy surface (a Discord relay is a deliberate republication; the ops log is not).
So the INFO record carries the *shape* - kind, player, message length - and the body is gated
behind DEBUG and control-character-escaped on the way out. :func:`log_chat` is the sanctioned way
to do that, and :data:`CHAT_BODY_KEY` is enforced by a processor so an ad-hoc log call cannot get
it wrong either.

Ruff's ``G`` and ``LOG`` rules are on project-wide: no f-strings in log calls, no ``logging.warn``,
structured key-value pairs rather than interpolated prose.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from typing import TYPE_CHECKING, Any, Final, Literal, TextIO

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from contextlib import AbstractContextManager
    from contextvars import Token
    from pathlib import Path
    from types import TracebackType

    from structlog.typing import EventDict, Processor, WrappedLogger

    from mcmanager.clock import Clock
    from mcmanager.config import LogsConfig

__all__ = [
    "CHAT_BODY_KEY",
    "NOISY_LOGGERS",
    "bind_context",
    "bound_context",
    "chat_log_fields",
    "clear_context",
    "configure_from_settings",
    "configure_logging",
    "get_logger",
    "iter_configured_handlers",
    "log_chat",
    "sanitise_for_log",
]

CHAT_BODY_KEY: Final = "chat_body"
"""The one key allowed to carry raw user-generated text, and only at DEBUG.

A processor removes it from any record above DEBUG and replaces it with ``message_length``. That
makes "don't log chat at INFO" a property of the pipeline rather than a rule people remember.
"""

NOISY_LOGGERS: Final[tuple[str, ...]] = (
    "asyncio",
    "aiohttp",
    "aiohttp.access",
    "aiohttp.client",
    "aiohttp.server",
    "charset_normalizer",
    "discord",
    "discord.client",
    "discord.gateway",
    "discord.http",
    "discord.state",
    "docker",
    "docker.auth",
    "docker.utils.config",
    "urllib3",
    "urllib3.connectionpool",
    "websocket",
)
"""Third-party loggers pinned to WARNING.

discord.py logs every gateway heartbeat at DEBUG and every reconnect at INFO; docker-py logs a
line per API call. At ``logs.level = "DEBUG"`` - which is the setting you reach for when something
is wrong - that is thousands of lines an hour of somebody else's bookkeeping burying the one line
that matters. Their warnings and errors still come through.
"""

_CONTROL_TRANSLATION: Final = {
    ord("\n"): "\\n",
    ord("\r"): "\\r",
    ord("\t"): "\\t",
    ord("\x1b"): "\\x1b",
    0: "\\x00",
}


def sanitise_for_log(text: str, *, limit: int = 512) -> str:
    """Make user-generated text safe to put in a log line.

    Escapes the characters that let a player forge a log record - newline, carriage return, tab and
    ESC (which is also how the old bridge script's ANSI problem got into player names) - and caps
    the length, because "4KB of text" is an expected chat input and a log line is not the place to
    find that out.
    """
    escaped = text.translate(_CONTROL_TRANSLATION)
    if len(escaped) > limit:
        return f"{escaped[:limit]}...[{len(escaped) - limit} more]"
    return escaped


# ------------------------------------------------------------------------------- processors


class _ClockTimestamper:
    """Timestamps from the injected :class:`~mcmanager.clock.Clock`.

    Used when a clock is supplied, so a test can assert on log timestamps and a replay can be
    stamped with the replayed time rather than now. Without one, structlog's own ``TimeStamper``
    is used - the single sanctioned exception to the "only clock.py knows the time" rule, because
    it lives inside a third-party library and stamping a log line is not program logic.
    """

    __slots__ = ("_clock", "_key")

    def __init__(self, clock: Clock, *, key: str = "timestamp") -> None:
        self._clock = clock
        self._key = key

    def __call__(
        self,
        logger: WrappedLogger,
        method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        del logger, method_name
        stamp = self._clock.now().isoformat(timespec="milliseconds")
        event_dict[self._key] = stamp.replace("+00:00", "Z")
        return event_dict


def _chat_body_guard(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Drop :data:`CHAT_BODY_KEY` above DEBUG; escape it at DEBUG.

    This is the mechanical half of the chat-privacy rule. :func:`log_chat` is the polite half.
    """
    del logger
    body = event_dict.get(CHAT_BODY_KEY)
    if body is None:
        return event_dict
    text = body if isinstance(body, str) else str(body)
    event_dict.setdefault("message_length", len(text))
    if method_name == "debug":
        event_dict[CHAT_BODY_KEY] = sanitise_for_log(text)
    else:
        del event_dict[CHAT_BODY_KEY]
        event_dict["message_redacted"] = True
    return event_dict


def _shared_processors(clock: Clock | None) -> list[Processor]:
    """The chain applied to our records and to foreign stdlib records alike.

    ``merge_contextvars`` first, so ``session_id`` and ``container`` bound once at startup appear
    on every line including docker-py's.
    """
    timestamper: Processor = (
        _ClockTimestamper(clock)
        if clock is not None
        else structlog.processors.TimeStamper(fmt="iso", utc=True)
    )
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        _chat_body_guard,
    ]


def _want_colour(stream: object) -> bool:
    """Colour only on a real terminal, and never when ``NO_COLOR`` is set.

    This project exists partly because ANSI escapes corrupted player names. Emitting them into a
    piped log would be a poor joke.
    """
    if os.environ.get("NO_COLOR"):
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty()) if callable(isatty) else False


def _build_renderer(fmt: Literal["json", "console"], stream: object) -> list[Processor]:
    if fmt == "json":
        # The event name stays under "event" rather than being renamed to "message": half the
        # records here are foreign, chat records already carry message_length, and one key meaning
        # two things is how a jq filter starts lying.
        return [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(sort_keys=True),
        ]
    return [structlog.dev.ConsoleRenderer(colors=_want_colour(stream))]


# ---------------------------------------------------------------------------- configuration


def configure_logging(
    *,
    level: str = "INFO",
    fmt: Literal["json", "console"] = "json",
    stream: TextIO | None = None,
    log_file: Path | None = None,
    clock: Clock | None = None,
    force: bool = True,
) -> None:
    """Install the one handler everything logs through.

    Args:
        level: Root level for our own loggers. Third-party loggers named in :data:`NOISY_LOGGERS`
            are pinned to WARNING regardless.
        fmt: ``json`` for the deployed daemon (``docker logs | jq``), ``console`` for humans.
        stream: Where to write. Defaults to ``sys.stdout``, because a container's log driver reads
            stdout and splitting our own output across two streams by severity makes the ordering
            of an interleaved read undefined.
        log_file: Optional rotating file copy, in the same format. The directory must already
            exist; this function never creates one, because on the daemon that directory is a
            mount and creating it locally would mask a missing volume.
        clock: Injected clock for timestamps. ``None`` uses structlog's own timestamper.
        force: Replace any handlers already on the root logger. False is for a host application
            that owns logging and merely wants our processors.
    """
    target = stream if stream is not None else sys.stdout
    shared = _shared_processors(clock)

    structlog.configure(
        processors=[
            *shared,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # Foreign records - docker-py, discord.py, aiohttp - go through the same chain, so they
        # arrive with a level, a timestamp and our bound context rather than as bare strings.
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.UnicodeDecoder(),
            *_build_renderer(fmt, target),
        ],
    )

    handler = logging.StreamHandler(target)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    if force:
        for existing in list(root.handlers):
            root.removeHandler(existing)
            existing.close()
    root.addHandler(handler)
    root.setLevel(level.upper())

    if log_file is not None and log_file.parent.is_dir():
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def configure_from_settings(
    logs: LogsConfig,
    *,
    stream: TextIO | None = None,
    clock: Clock | None = None,
) -> None:
    """:func:`configure_logging` driven by ``[logs]``.

    The daemon's own log directory is used only if it already exists: a missing directory means a
    missing volume, and quietly creating it inside the container would hide that until the next
    restart threw the logs away.
    """
    directory = logs.daemon_log_dir
    configure_logging(
        level=logs.level,
        fmt=logs.format,
        stream=stream,
        log_file=directory / "mcmanager.log" if directory.is_dir() else None,
        clock=clock,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """A logger. Name it after the subsystem: ``mcmanager.bus``, ``mcmanager.lifecycle``.

    ``mcmanager.deaths.unmatched`` is a named example from the parser design: the tier-2 death
    fallback logs there so the vanilla death table can be grown from real data.
    """
    return structlog.stdlib.get_logger(name or "mcmanager")


# --------------------------------------------------------------------------------- context


def bind_context(
    *,
    session_id: str | None = None,
    container: str | None = None,
    player: str | None = None,
    event_kind: str | None = None,
    **extra: object,
) -> None:
    """Bind ambient fields onto every subsequent log line in this task and its children.

    The four named ones are the questions actually asked of these logs: *which session*, *which
    container*, *which player*, *what kind of event*. They are keyword-only and explicit so they
    are spelled the same way everywhere; ``**extra`` covers everything else.

    Uses context variables, so a value bound inside one handler does not leak into a concurrently
    running one.
    """
    values: dict[str, object] = {
        "session_id": session_id,
        "container": container,
        "player": player,
        "event_kind": event_kind,
        **extra,
    }
    structlog.contextvars.bind_contextvars(
        **{key: value for key, value in values.items() if value is not None}
    )


def bound_context(**values: object) -> AbstractContextManager[None]:
    """Scoped :func:`bind_context`: fields are unbound again on exit."""
    return _BoundContext(values)


class _BoundContext:
    """Implements :func:`bound_context` without a decorator, so pyright sees a real type."""

    __slots__ = ("_reset", "_values")

    def __init__(self, values: dict[str, object]) -> None:
        self._values = {key: value for key, value in values.items() if value is not None}
        self._reset: Mapping[str, Token[Any]] | None = None

    def __enter__(self) -> None:
        self._reset = structlog.contextvars.bind_contextvars(**self._values)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self._reset is not None:
            structlog.contextvars.reset_contextvars(**self._reset)
            self._reset = None


def clear_context() -> None:
    """Drop every bound field. Called between sessions, and by tests."""
    structlog.contextvars.clear_contextvars()


# ------------------------------------------------------------------------------------ chat


def chat_log_fields(*, kind: str, player: str, message: str) -> dict[str, object]:
    """The INFO-safe shape of a chat message: no body, just enough to reason about it.

    ``message_length`` is what answers the operational questions - is somebody pasting 4KB of
    text, did the relay truncate - without putting a player's words in the ops log.
    """
    return {
        "chat_kind": kind,
        "player": player,
        "message_length": len(message),
    }


def log_chat(
    logger: structlog.stdlib.BoundLogger,
    *,
    kind: str,
    player: str,
    message: str,
    event: str = "chat",
) -> None:
    """Log a chat message at two levels: shape at INFO, body at DEBUG.

    The body is passed under :data:`CHAT_BODY_KEY`, so even if this function is bypassed the
    processor will still strip it from anything above DEBUG.
    """
    fields = chat_log_fields(kind=kind, player=player, message=message)
    body_event = f"{event}.body"
    logger.info(event, **fields)
    logger.debug(body_event, **fields, **{CHAT_BODY_KEY: message})


def iter_configured_handlers() -> Iterator[logging.Handler]:
    """The handlers :func:`configure_logging` installed. For tests and for ``/readyz``."""
    yield from logging.getLogger().handlers
