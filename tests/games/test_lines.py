"""Step two of the parser: telling the three interleaved grammars apart."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mcmanager.core.types import LineOrigin
from mcmanager.games.minecraft.ansi import sanitize_line
from mcmanager.games.minecraft.lines import classify

if TYPE_CHECKING:
    from collections.abc import Callable

ESC = "\x1b"


def test_server_line_is_split_into_thread_level_and_payload() -> None:
    line = classify("[13:24:37] [Server thread/INFO]: Hypixelite joined the game")
    assert line.origin is LineOrigin.SERVER
    assert line.thread == "Server thread"
    assert line.level == "INFO"
    assert line.message == "Hypixelite joined the game"
    assert line.clock_hint == "13:24:37"
    assert line.logger is None
    assert line.is_server_thread_info


def test_server_line_with_a_plugin_bracket_group_still_matches() -> None:
    """The optional second bracket group.

    The plan calls this the single most common reason a naive regex silently stops matching after
    a Paper update or a plugin install: the prefix grows a group and every anchored pattern below
    it stops seeing a payload it recognises.
    """
    line = classify("[13:24:37] [Server thread/INFO] [WorldEdit]: Loaded 1 world")
    assert line.origin is LineOrigin.SERVER
    assert line.thread == "Server thread"
    assert line.logger == "WorldEdit"
    assert line.message == "Loaded 1 world"
    assert line.is_server_thread_info


def test_thread_name_containing_slashes_is_split_at_the_last_one() -> None:
    """Real thread name: ``RCON Client /0:0:0:0:0:0:0:1 #2``."""
    line = classify(
        "[22:41:35] [RCON Client /0:0:0:0:0:0:0:1 #2/INFO]: "
        "Thread RCON Client /0:0:0:0:0:0:0:1 shutting down"
    )
    assert line.thread == "RCON Client /0:0:0:0:0:0:0:1 #2"
    assert line.level == "INFO"
    assert not line.is_server_thread_info


@pytest.mark.parametrize(
    ("raw", "level"),
    [
        ("[13:24:37] [Server thread/WARN]: bharath_720 moved too quickly! 1.0,0.0,1.0", "WARN"),
        ("[13:24:37] [Worker-Main-1/ERROR]: boom", "ERROR"),
        ("[13:24:37] [ServerMain/INFO]: Loaded 1688 advancements", "INFO"),
    ],
)
def test_levels_are_captured(raw: str, level: str) -> None:
    assert classify(raw).level == level


def test_non_info_server_thread_line_is_not_a_candidate() -> None:
    """The level half of the guard.

    ``bharath_720 moved too quickly!`` is a real WARN line that starts with a real player name.
    Without the level check it is one plausible future death template away from being parsed as a
    death.
    """
    line = classify("[13:24:37] [Server thread/WARN]: bharath_720 moved too quickly! 1.0,0.0,1.0")
    assert not line.is_server_thread_info


def test_chat_thread_is_recognised_and_not_secure_prefix_is_stripped() -> None:
    """The finding that forced a deviation from the plan.

    The plan's guard was "only ``Server thread`` / ``INFO`` lines are candidates for player and
    chat events". Every chat line on this server is emitted from ``Async Chat Thread - #N``, so
    that guard would have discarded 100% of chat. Verbatim archive line below.
    """
    line = classify("[14:48:06] [Async Chat Thread - #0/INFO]: [Not Secure] <bharath_720> yo")
    assert line.origin is LineOrigin.SERVER
    assert line.thread == "Async Chat Thread - #0"
    assert line.is_chat_thread_info
    assert not line.is_server_thread_info
    assert line.was_not_secure
    assert line.message == "<bharath_720> yo"


def test_high_numbered_chat_thread_is_recognised() -> None:
    line = classify("[16:00:50] [Async Chat Thread - #31/INFO]: [Not Secure] <bharath_720> lol")
    assert line.is_chat_thread_info


def test_user_authenticator_thread_is_recognised() -> None:
    line = classify(
        "[00:49:05] [User Authenticator #0/INFO]: "
        "UUID of player Hypixelite is 403d4fb2-f466-3716-b9cb-a3e769bb40c9"
    )
    assert line.is_authenticator_info
    assert not line.is_server_thread_info


def test_wrapper_grammar() -> None:
    """``mc-server-runner``: tab separated, full date, docker stream only."""
    line = classify("2026-07-25T22:58:20.809+0530\tINFO\tmc-server-runner\tDone")
    assert line.origin is LineOrigin.WRAPPER
    assert line.level == "INFO"
    assert line.logger == "mc-server-runner"
    assert line.message == "Done"
    assert not line.is_server_thread_info


def test_wrapper_done_is_not_confused_with_the_server_done() -> None:
    """Two different facts spelled the same way.

    ``mc-server-runner``'s ``Done`` means the JVM exited. Paper's ``Done (32.521s)!`` means the
    server came up. They are told apart by grammar, never by substring - which is the same class
    of mistake as the old bridge script's ``if "joined the game" in line``.
    """
    wrapper = classify("2026-07-25T22:58:20.809+0530\tINFO\tmc-server-runner\tDone")
    server = classify('[22:41:47] [Server thread/INFO]: Done (14.117s)! For help, type "help"')
    assert wrapper.origin is LineOrigin.WRAPPER
    assert server.origin is LineOrigin.SERVER
    assert wrapper.message != server.message


@pytest.mark.parametrize(
    "raw",
    [
        "\tat net.minecraft.server.MinecraftServer.runServer(MinecraftServer.java:1385)",
        "\tat java.base/java.lang.Thread.run(Unknown Source)",
        "\t... 12 more",
        "\tCaused by: java.io.IOException",
        "    at com.mojang.serialization.DataResult$Error.mapOrElse(DataResult.java:309)",
    ],
)
def test_continuation_lines(raw: str) -> None:
    assert classify(raw).origin is LineOrigin.CONTINUATION


@pytest.mark.parametrize(
    "raw",
    [
        "[init] Starting the Minecraft server...",
        "[mc-image-helper] 22:41:13.778 INFO  : Created/updated 1 property",
        "Starting org.bukkit.craftbukkit.Main",
        "WARNING: A terminally deprecated method in sun.misc.Unsafe has been called",
        "",
        "   ",
    ],
)
def test_raw_lines(raw: str) -> None:
    line = classify(raw)
    assert line.origin is LineOrigin.RAW
    assert line.message == raw
    assert not line.is_server_thread_info


def test_classify_never_raises_on_hostile_input() -> None:
    for hostile in ("\ud800", "[" * 5000, "[13:24:37] [" + "a" * 10_000 + "/INFO]: x", "\x00"):
        assert classify(hostile) is not None


def test_every_grammar_appears_in_the_committed_fixtures(
    load_fixture: Callable[[str], list[str]],
) -> None:
    """The reason the docker stream is committed as well as the ``.gz`` archives.

    Only the docker stream carries WRAPPER and the ``[init]`` RAW lines. A fixture set built from
    ``/data/logs/*.log.gz`` alone would exercise one grammar out of three and look complete.
    """
    stream_origins = {
        classify(sanitize_line(raw)).origin for raw in load_fixture("docker_stream.log")
    }
    archive_origins = {
        classify(sanitize_line(raw)).origin for raw in load_fixture("run_normal_session.log")
    }
    assert LineOrigin.WRAPPER in stream_origins
    assert LineOrigin.RAW in stream_origins
    assert LineOrigin.SERVER in stream_origins
    assert LineOrigin.WRAPPER not in archive_origins
