"""The SLP probe, the adapter that wraps it, and the registry that builds it.

No network: ``mcstatus``' ``JavaServer`` is replaced wholesale. The autouse guard in
``tests/conftest.py`` would catch a real connection anyway, but a test that depends on a firewall
to stay offline is a test that fails for the wrong reason.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Self

import pytest

from mcmanager.clock import ManualClock
from mcmanager.core.events import ServerReady, ServerStopped, ServerStopping
from mcmanager.core.types import ReadySignal, Source, Stream
from mcmanager.errors import ConfigError
from mcmanager.games.base import GameAdapter, ProbeResult
from mcmanager.games.minecraft import probe as probe_module
from mcmanager.games.minecraft.adapter import MinecraftAdapter
from mcmanager.games.registry import get_adapter, known_games

if TYPE_CHECKING:
    from collections.abc import Sequence

TS = datetime(2026, 7, 25, 17, 11, 33, tzinfo=UTC)


# --------------------------------------------------------------------------- mcstatus doubles


class _Players:
    def __init__(self, online: int, maximum: int, sample: Sequence[object] | None) -> None:
        self.online = online
        self.max = maximum
        self.sample = sample


class _Player:
    def __init__(self, name: str) -> None:
        self.name = name
        self.id = "403d4fb2-f466-3716-b9cb-a3e769bb40c9"


class _Version:
    def __init__(self, name: str) -> None:
        self.name = name


class _Motd:
    def __init__(self, plain: str) -> None:
        self._plain = plain

    def to_plain(self) -> str:
        return self._plain


class _Status:
    def __init__(
        self,
        *,
        online: int = 2,
        maximum: int = 5,
        sample: Sequence[object] | None = None,
        version: str = "26.2",
        motd: str = "A Minecraft Server",
        latency: float = 12.5,
    ) -> None:
        self.players = _Players(online, maximum, sample)
        self.version = _Version(version)
        self.motd = _Motd(motd)
        self.latency = latency


class _FakeJavaServer:
    """Stands in for ``mcstatus.JavaServer``. Records what it was asked for."""

    status: _Status | None = None
    raises: BaseException | None = None
    lookup_raises: BaseException | None = None
    seen_address: str | None = None
    seen_timeout: float | None = None

    @classmethod
    def async_lookup(cls, address: str, timeout: float = 3) -> Any:
        cls.seen_address = address
        cls.seen_timeout = timeout
        failure = cls.lookup_raises
        if failure is not None:
            raise failure

        async def _build() -> Self:
            return cls()

        return _build()

    async def async_status(self) -> Any:
        failure = type(self).raises
        if failure is not None:
            raise failure
        return type(self).status


@pytest.fixture(autouse=True)
def fake_java_server(monkeypatch: pytest.MonkeyPatch) -> type[_FakeJavaServer]:
    _FakeJavaServer.status = _Status()
    _FakeJavaServer.raises = None
    _FakeJavaServer.lookup_raises = None
    _FakeJavaServer.seen_address = None
    _FakeJavaServer.seen_timeout = None
    monkeypatch.setattr(probe_module, "JavaServer", _FakeJavaServer)
    return _FakeJavaServer


# ------------------------------------------------------------------------------------ probe


async def test_a_successful_probe_is_fully_populated(clock: ManualClock) -> None:
    _FakeJavaServer.status = _Status(
        online=2,
        maximum=5,
        sample=[_Player("Hypixelite"), _Player("bharath_720")],
    )
    result = await probe_module.query_status("minecraft", 25565, timeout=3.0, clock=clock)

    assert result.reachable
    assert result.players_online == 2
    assert result.players_max == 5
    assert result.sample == ("Hypixelite", "bharath_720")
    assert result.version == "26.2"
    assert result.motd == "A Minecraft Server"
    assert result.latency_ms == pytest.approx(12.5)
    assert result.error is None
    assert result.probed_at == clock.now()
    assert result.probed_at.tzinfo is not None


async def test_the_host_and_port_are_passed_through(clock: ManualClock) -> None:
    """``minecraft``, not ``localhost``: inside our container localhost is our own loopback."""
    await probe_module.query_status("minecraft", 25565, timeout=1.5, clock=clock)
    assert _FakeJavaServer.seen_address == "minecraft:25565"
    assert _FakeJavaServer.seen_timeout == pytest.approx(1.5)


async def test_sample_is_complete_when_the_sample_accounts_for_everyone(
    clock: ManualClock,
) -> None:
    _FakeJavaServer.status = _Status(online=2, sample=[_Player("a"), _Player("b")])
    result = await probe_module.query_status("minecraft", 25565, timeout=3.0, clock=clock)
    assert result.sample_is_complete


def test_sample_is_complete_is_players_online_equals_len_sample() -> None:
    """The exact rule, asserted directly rather than through a probe.

    ``players.sample`` is protocol-capped at 12 on this server and Paper may randomise or omit it.
    Reconciling a roster against a truncated sample produces a storm of fake ``PlayerLeft``
    events, which is why every consumer must gate on this.
    """
    truncated = ProbeResult(
        reachable=True,
        players_online=20,
        sample=tuple(str(i) for i in range(12)),
        probed_at=TS,
    )
    assert not truncated.sample_is_complete

    omitted = ProbeResult(reachable=True, players_online=3, sample=(), probed_at=TS)
    assert not omitted.sample_is_complete

    empty = ProbeResult(reachable=True, players_online=0, sample=(), probed_at=TS)
    assert empty.sample_is_complete


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionRefusedError(111, "Connection refused"),
        TimeoutError(),
        OSError("dns"),
        ValueError("malformed status payload"),
        RuntimeError(),
    ],
)
async def test_any_failure_means_unknown_and_never_zero_players(
    failure: BaseException, clock: ManualClock
) -> None:
    """The failure mode that loses somebody's build session.

    A probe that cannot reach the server knows **nothing** about how many people are playing.
    Reporting zero, combined with idle shutdown, is precisely how a populated server gets stopped.
    """
    _FakeJavaServer.raises = failure
    result = await probe_module.query_status("minecraft", 25565, timeout=3.0, clock=clock)

    assert not result.reachable
    assert result.players_online is None
    assert result.players_online != 0
    assert not result.is_empty
    assert not result.sample_is_complete
    assert result.error is not None
    assert type(failure).__name__ in result.error
    assert "Traceback" not in result.error


async def test_a_lookup_failure_is_also_unknown(clock: ManualClock) -> None:
    """DNS is the first thing that breaks when the compose network changes."""
    _FakeJavaServer.lookup_raises = OSError("Name or service not known")
    result = await probe_module.query_status("minecraft", 25565, timeout=3.0, clock=clock)
    assert not result.reachable
    assert result.players_online is None
    assert result.probed_at == clock.now()


async def test_is_empty_requires_an_answer(clock: ManualClock) -> None:
    _FakeJavaServer.status = _Status(online=0, sample=[])
    answered = await probe_module.query_status("minecraft", 25565, timeout=3.0, clock=clock)
    assert answered.is_empty

    _FakeJavaServer.raises = TimeoutError()
    unreachable = await probe_module.query_status("minecraft", 25565, timeout=3.0, clock=clock)
    assert not unreachable.is_empty


async def test_a_missing_sample_is_an_empty_tuple_not_none(clock: ManualClock) -> None:
    """Paper anti-scrape settings omit the sample entirely; mcstatus reports ``None``."""
    _FakeJavaServer.status = _Status(online=4, sample=None)
    result = await probe_module.query_status("minecraft", 25565, timeout=3.0, clock=clock)
    assert result.sample == ()
    assert result.players_online == 4
    assert not result.sample_is_complete


async def test_the_clock_is_injected_not_read_from_the_wall(clock: ManualClock) -> None:
    """``probed_at`` comes from the injected clock, which is what makes the poller testable."""
    before = await probe_module.query_status("minecraft", 25565, timeout=3.0, clock=clock)
    await clock.advance(45)
    after = await probe_module.query_status("minecraft", 25565, timeout=3.0, clock=clock)
    assert (after.probed_at - before.probed_at).total_seconds() == pytest.approx(45)


# ---------------------------------------------------------------------------------- adapter


def test_the_adapter_satisfies_the_protocol() -> None:
    assert isinstance(MinecraftAdapter(), GameAdapter)


def test_adapter_constants() -> None:
    adapter = MinecraftAdapter()
    assert adapter.game_id == "minecraft"
    assert adapter.default_port == 25565
    assert adapter.default_stop_timeout == 90


def test_adapter_parse_line_delegates_to_the_pure_parser() -> None:
    adapter = MinecraftAdapter()
    event = adapter.parse_line(
        "[13:24:37] [Server thread/INFO]: \x1b[93mHypixelite joined the game\x1b[0m",
        ts=TS,
        server_id="minecraft",
        stream=Stream.STDOUT,
    )
    assert event.ts == TS
    assert event.source is Source.LOG


def test_ready_signal() -> None:
    adapter = MinecraftAdapter()
    from_log = ServerReady(
        ts=TS, server_id="minecraft", source=Source.LOG, detected_by=ReadySignal.LOG_DONE
    )
    from_health = ServerReady(
        ts=TS, server_id="minecraft", source=Source.RUNTIME, detected_by=ReadySignal.HEALTHCHECK
    )
    other = ServerStopping(ts=TS, server_id="minecraft", source=Source.LOG)

    assert adapter.ready_signal(from_log) is ReadySignal.LOG_DONE
    assert adapter.ready_signal(from_health) is ReadySignal.HEALTHCHECK
    assert adapter.ready_signal(other) is None


def test_stop_signal() -> None:
    adapter = MinecraftAdapter()
    stopping = ServerStopping(ts=TS, server_id="minecraft", source=Source.LOG)
    stopped = ServerStopped(ts=TS, server_id="minecraft", source=Source.RUNTIME, clean=True)
    assert adapter.stop_signal(stopping)
    assert not adapter.stop_signal(stopped)


async def test_adapter_probe_uses_its_injected_clock() -> None:
    clock = ManualClock()
    adapter = MinecraftAdapter(clock=clock)
    result = await adapter.probe("minecraft", 25565, timeout=3.0)
    assert result.probed_at == clock.now()


# --------------------------------------------------------------------------------- registry


def test_registry_resolves_minecraft() -> None:
    adapter = get_adapter("minecraft")
    assert adapter.game_id == "minecraft"
    assert isinstance(adapter, GameAdapter)


def test_registry_is_forgiving_about_case_and_whitespace() -> None:
    assert get_adapter("  Minecraft ").game_id == "minecraft"


def test_registry_passes_the_clock_through() -> None:
    clock = ManualClock()
    adapter = get_adapter("minecraft", clock=clock)
    assert isinstance(adapter, MinecraftAdapter)


def test_registry_rejects_an_unknown_game_and_says_what_it_knows() -> None:
    """The message lists the known ids: the difference between a five-second fix and a grep."""
    with pytest.raises(ConfigError) as caught:
        get_adapter("valheim")
    assert "valheim" in str(caught.value)
    assert "minecraft" in str(caught.value)


def test_known_games() -> None:
    assert known_games() == ("minecraft",)
