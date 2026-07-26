"""Step one of the parser: sanitisation.

Every fixture line here is built with **explicit escape bytes** (``\\x1b``), never with a pretty
``^[`` that only looks like one. A test that asserts on a string containing the two characters
``^`` and ``[`` proves nothing about a file containing byte 0x1b, and this is the exact defect the
module exists to fix.
"""

from __future__ import annotations

import pytest

from mcmanager.games.minecraft.ansi import sanitize_line, strip_ansi

ESC = "\x1b"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The real archive line, byte for byte.
        (
            f"[13:24:37] [Server thread/INFO]: {ESC}[93mHypixelite joined the game{ESC}[0m",
            "[13:24:37] [Server thread/INFO]: Hypixelite joined the game",
        ),
        # Real docker-stream lines: a bare colour and a compound one.
        (f"{ESC}[39m[mc-image-helper] 22:41:13.778 INFO", "[mc-image-helper] 22:41:13.778 INFO"),
        (f"{ESC}[0;39m[init] Generating log4j2.xml", "[init] Generating log4j2.xml"),
        # Real advancement line: colour wraps the bracketed title only.
        (
            f"Hypixelite has made the advancement {ESC}[92m[Stone Age]{ESC}[0m",
            "Hypixelite has made the advancement [Stone Age]",
        ),
        # Real rcon echo: italic + grey.
        (f"{ESC}[3m{ESC}[37m[Rcon: Stopping the server]{ESC}[0m", "[Rcon: Stopping the server]"),
        # Truecolor. Enumerating colour codes instead of parsing the grammar is how half of one of
        # these survives and lands inside a captured name.
        (f"{ESC}[38;2;255;0;0mSteve{ESC}[0m", "Steve"),
        (f"{ESC}[48;5;226mSteve{ESC}[m", "Steve"),
        # Cursor movement and erase: not colour, still an escape sequence.
        (f"{ESC}[2K{ESC}[1;31mSteve", "Steve"),
        # The two-byte, non-CSI form.
        (f"{ESC}MSteve{ESC}7", "Steve"),
        # Minecraft section codes, including Bungee's hex spelling.
        ("§aSteve§r", "Steve"),
        ("§x§f§f§0§0§0§0Steve", "Steve"),
        # Nothing to strip is not an error.
        ("plain text", "plain text"),
        ("", ""),
    ],
)
def test_strip_ansi(raw: str, expected: str) -> None:
    assert strip_ansi(raw) == expected


def test_strip_ansi_leaves_no_escape_bytes_in_the_real_archive_line() -> None:
    """The regression, stated as the property that actually matters."""
    raw = f"[13:24:37] [Server thread/INFO]: {ESC}[93mHypixelite joined the game{ESC}[0m"
    assert ESC not in strip_ansi(raw)
    assert "[93m" not in strip_ansi(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Steve joined the game\r", "Steve joined the game"),
        ("Steve joined the game\r\n", "Steve joined the game"),
        ("Steve joined the game\n", "Steve joined the game"),
        # A CR in the middle is a control character, not a line terminator.
        ("Steve\rjoined", "Stevejoined"),
    ],
)
def test_sanitize_line_strips_line_terminators(raw: str, expected: str) -> None:
    assert sanitize_line(raw) == expected


def test_sanitize_line_removes_nul_but_keeps_tab() -> None:
    """Tab is load-bearing: it separates the wrapper grammar's fields.

    NUL is not: a name containing one would survive every anchored pattern and reach Discord.
    """
    assert sanitize_line("a\x00b") == "ab"
    assert sanitize_line("2026-07-25T22:58:20.809+0530\tINFO\tmc-server-runner\tDone") == (
        "2026-07-25T22:58:20.809+0530\tINFO\tmc-server-runner\tDone"
    )


@pytest.mark.parametrize(
    "hostile",
    [
        "\x00\x01\x02\x03",
        "\ud800",  # a lone high surrogate
        "Steve \udcff\udcfe",  # what surrogateescape produces from invalid UTF-8
        b"\xff\xfe\x80".decode("utf-8", "surrogateescape"),
        "\x7f\x1b",  # DEL, and a truncated escape with nothing after it
        f"{ESC}",
        f"{ESC}[",
        f"{ESC}[38;2;",  # a truncated truecolor sequence
        "§",  # a section sign with nothing following it
        "\U0001f4a3" * 100,
    ],
)
def test_sanitize_line_never_raises_on_hostile_input(hostile: str) -> None:
    """Totality starts here. Every caller above depends on this not throwing."""
    assert isinstance(sanitize_line(hostile), str)


def test_sanitize_line_is_idempotent() -> None:
    raw = f"{ESC}[93mSteve§a joined\x00{ESC}[0m\r"
    once = sanitize_line(raw)
    assert sanitize_line(once) == once
