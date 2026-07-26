"""Resolves ``server.game`` to a :class:`~mcmanager.games.base.GameAdapter`.

Adding a game is one package under ``games/`` plus one line in :data:`_FACTORIES`. Imports of
concrete adapters are done lazily inside :func:`get_adapter` so that ``mcmanager replay`` pulls in
only the parser it needs, and so that a broken optional dependency in one game's probe cannot stop
the daemon booting for a different game.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcmanager.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mcmanager.clock import Clock
    from mcmanager.games.base import GameAdapter

__all__ = ["get_adapter", "known_games"]


def _minecraft(clock: Clock | None) -> GameAdapter:
    from mcmanager.games.minecraft.adapter import MinecraftAdapter

    return MinecraftAdapter(clock=clock)


_FACTORIES: Mapping[str, Callable[[Clock | None], GameAdapter]] = {
    "minecraft": _minecraft,
}
"""Game id to factory. The single edit point for a new game."""


def get_adapter(game_id: str, *, clock: Clock | None = None) -> GameAdapter:
    """Return the adapter registered under ``game_id``.

    Args:
        game_id: The value of ``server.game``. Matched case-insensitively after stripping: a TOML
            file is hand-edited, and ``game = "Minecraft"`` refusing to boot the daemon is a poor
            trade for a strictness nobody asked for.
        clock: Passed through to the adapter, which needs one only to timestamp probe results.
            ``None`` lets the adapter default to :class:`~mcmanager.clock.SystemClock`, which is
            what the standalone CLI commands want. This keyword is an addition to the signature
            the scaffold declared; the positional call ``get_adapter("minecraft")`` is unchanged.

    Raises:
        ConfigError: if no such game is registered. The message lists the known ids, which is the
            difference between a five-second fix and a grep.
    """
    key = game_id.strip().lower()
    factory = _FACTORIES.get(key)
    if factory is None:
        known = ", ".join(known_games())
        msg = f"unknown game {game_id!r}; server.game must be one of: {known}"
        raise ConfigError(msg)
    return factory(clock)


def known_games() -> tuple[str, ...]:
    """Every registered game id, sorted, for error messages and ``check-config``."""
    return tuple(sorted(_FACTORIES))
