"""Logging: one stream, one shape, and no chat bodies at INFO.

The chat tests are the ones that matter. Chat is user-generated content, which makes the ops log
both a log-injection surface (a player types a newline and a plausible JSON object) and a privacy
surface (a Discord relay is a deliberate republication; the ops log is not). "Don't log chat at
INFO" is a rule people forget, so it is enforced by a processor and asserted here.
"""

from __future__ import annotations

import json
import logging
from io import StringIO
from typing import TYPE_CHECKING, Any

import pytest
import structlog

from mcmanager.logging_setup import (
    CHAT_BODY_KEY,
    NOISY_LOGGERS,
    bind_context,
    bound_context,
    chat_log_fields,
    clear_context,
    configure_logging,
    get_logger,
    log_chat,
    sanitise_for_log,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mcmanager.clock import ManualClock


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    """Put stdlib logging and structlog back exactly as they were.

    ``configure_logging(force=True)`` removes every root handler, pytest's included, so without
    this a single test in this file would silence the rest of the suite's caplog.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_levels = {name: logging.getLogger(name).level for name in NOISY_LOGGERS}
    clear_context()
    try:
        yield
    finally:
        clear_context()
        structlog.reset_defaults()
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)


def records(stream: StringIO) -> list[dict[str, Any]]:
    """Every JSON line written so far."""
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


# ------------------------------------------------------------------------------ json shape


def test_json_records_carry_level_timestamp_and_logger() -> None:
    stream = StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)

    get_logger("mcmanager.bus").info("published", event_name="PlayerJoined", seq=7)

    (record,) = records(stream)
    assert record["event"] == "published"
    assert record["level"] == "info"
    assert record["logger"] == "mcmanager.bus"
    assert record["event_name"] == "PlayerJoined"
    assert record["seq"] == 7
    assert record["timestamp"]


def test_third_party_loggers_land_in_the_same_stream_with_the_same_shape() -> None:
    """The whole reason structlog is routed *through* stdlib logging.

    docker-py and discord.py log to their own stdlib loggers. If those came out as bare strings
    while ours came out as JSON, ``docker logs -f mcmanager | jq`` would choke on half the stream.
    """
    stream = StringIO()
    configure_logging(level="DEBUG", fmt="json", stream=stream)

    logging.getLogger("docker.api.client").warning("Connection refused: %s", "socket")

    (record,) = records(stream)
    assert record["event"] == "Connection refused: socket"
    assert record["level"] == "warning"
    assert record["logger"] == "docker.api.client"
    assert record["timestamp"]


def test_noisy_loggers_are_pinned_to_warning() -> None:
    stream = StringIO()
    configure_logging(level="DEBUG", fmt="json", stream=stream)

    for name in NOISY_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING

    logging.getLogger("discord.gateway").debug("keeping alive")
    logging.getLogger("discord.gateway").warning("shard disconnected")

    events = [record["event"] for record in records(stream)]
    assert events == ["shard disconnected"]


def test_exceptions_are_rendered_structurally_in_json() -> None:
    stream = StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)

    try:
        raise RuntimeError("socket went away")
    except RuntimeError:
        get_logger("mcmanager.manager").exception("inspect failed")

    (record,) = records(stream)
    assert record["event"] == "inspect failed"
    assert "socket went away" in json.dumps(record["exception"])


def test_console_format_emits_no_ansi_into_a_pipe() -> None:
    """This project exists partly because ANSI escapes corrupted player names."""
    stream = StringIO()
    configure_logging(level="INFO", fmt="console", stream=stream)

    get_logger("mcmanager").info("hello", player="Steve")

    output = stream.getvalue()
    assert "hello" in output
    assert "\x1b[" not in output


def test_level_filters_below_the_threshold() -> None:
    stream = StringIO()
    configure_logging(level="WARNING", fmt="json", stream=stream)

    logger = get_logger("mcmanager")
    logger.info("quiet")
    logger.warning("loud")

    assert [record["event"] for record in records(stream)] == ["loud"]


def test_configure_logging_is_idempotent() -> None:
    """Calling it twice must not produce every line twice, which is what a leaked handler does."""
    stream = StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)
    configure_logging(level="INFO", fmt="json", stream=stream)

    get_logger("mcmanager").info("once")

    assert len(records(stream)) == 1


# ---------------------------------------------------------------------------------- clock


def test_timestamps_come_from_the_injected_clock(clock: ManualClock) -> None:
    stream = StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream, clock=clock)

    get_logger("mcmanager").info("first")

    (record,) = records(stream)
    assert record["timestamp"] == "2026-01-01T00:00:00.000Z"


# -------------------------------------------------------------------------------- context


def test_bound_context_appears_on_every_line_including_foreign_ones() -> None:
    stream = StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)

    bind_context(session_id="s-1", container="minecraft")
    get_logger("mcmanager").info("ours")
    logging.getLogger("aiohttp").warning("theirs")

    for record in records(stream):
        assert record["session_id"] == "s-1"
        assert record["container"] == "minecraft"


def test_none_values_are_not_bound() -> None:
    stream = StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)

    bind_context(session_id="s-1", player=None)
    get_logger("mcmanager").info("x")

    (record,) = records(stream)
    assert "player" not in record


def test_bound_context_is_scoped() -> None:
    stream = StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)

    with bound_context(event_kind="PlayerJoined"):
        get_logger("mcmanager").info("inside")
    get_logger("mcmanager").info("outside")

    inside, outside = records(stream)
    assert inside["event_kind"] == "PlayerJoined"
    assert "event_kind" not in outside


def test_clear_context_drops_everything() -> None:
    stream = StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)

    bind_context(container="minecraft")
    clear_context()
    get_logger("mcmanager").info("x")

    (record,) = records(stream)
    assert "container" not in record


# ----------------------------------------------------------------------------------- chat


CHAT = "hello @everyone `rm -rf`"


def test_chat_body_never_appears_at_info() -> None:
    stream = StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)

    log_chat(get_logger("mcmanager.chat"), kind="chat", player="Steve", message=CHAT)

    (record,) = records(stream)
    assert record["event"] == "chat"
    assert record["player"] == "Steve"
    assert record["chat_kind"] == "chat"
    assert record["message_length"] == len(CHAT)
    assert CHAT not in json.dumps(record)


def test_chat_body_appears_at_debug() -> None:
    stream = StringIO()
    configure_logging(level="DEBUG", fmt="json", stream=stream)

    log_chat(get_logger("mcmanager.chat"), kind="chat", player="Steve", message=CHAT)

    shape, body = records(stream)
    assert CHAT not in json.dumps(shape)
    assert body[CHAT_BODY_KEY] == CHAT
    assert body["event"] == "chat.body"


def test_an_ad_hoc_info_call_carrying_a_body_is_still_stripped() -> None:
    """The processor, not the helper, is what makes this a property of the pipeline."""
    stream = StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)

    get_logger("mcmanager").info("careless", **{CHAT_BODY_KEY: CHAT})

    (record,) = records(stream)
    assert CHAT_BODY_KEY not in record
    assert record["message_redacted"] is True
    assert record["message_length"] == len(CHAT)


def test_a_forged_log_line_in_chat_is_escaped_at_debug() -> None:
    """A player can type a newline. They must not be able to type a second log record."""
    forged = 'ok\n{"level":"critical","event":"server on fire"}'
    stream = StringIO()
    configure_logging(level="DEBUG", fmt="json", stream=stream)

    log_chat(get_logger("mcmanager.chat"), kind="chat", player="Mallory", message=forged)

    written = records(stream)
    assert len(written) == 2
    assert all(record["logger"] == "mcmanager.chat" for record in written)
    assert "\\n" in written[1][CHAT_BODY_KEY]


def test_chat_log_fields_has_no_body() -> None:
    fields = chat_log_fields(kind="say", player="Steve", message=CHAT)
    assert fields == {"chat_kind": "say", "player": "Steve", "message_length": len(CHAT)}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain", "plain"),
        ("two\nlines", "two\\nlines"),
        ("tab\there", "tab\\there"),
        ("\x1b[93mcoloured\x1b[0m", "\\x1b[93mcoloured\\x1b[0m"),
        ("null\x00byte", "null\\x00byte"),
    ],
)
def test_sanitise_escapes_control_characters(raw: str, expected: str) -> None:
    assert sanitise_for_log(raw) == expected


def test_sanitise_truncates_a_wall_of_text() -> None:
    """4KB of chat is an expected input, not an incident."""
    result = sanitise_for_log("x" * 5000, limit=100)
    assert len(result) < 200
    assert result.startswith("x" * 100)
    assert "4900 more" in result
