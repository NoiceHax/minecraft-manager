"""Who may run the mutating commands. **File and signatures only; body in M6.**

A single ``admin_role_id`` check plus a rate limit, applied before the command body runs and
audited through ``CommandIssued(accepted=False, rejection=...)`` when it rejects - a refused stop
should be as visible as an accepted one.

Takes role ids as plain integers rather than a discord.py member object, so this file needs no
gateway import and its tests are a table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection

    from mcmanager.clock import Clock

__all__ = ["PermissionVerdict", "RateLimiter", "check_admin"]


class PermissionVerdict:
    """Allowed, or refused with a reason. Refusals are audited, not silently dropped."""

    def __init__(self, *, allowed: bool, reason: str | None = None) -> None:
        """Hold the decision and, when refused, why."""
        raise NotImplementedError

    def __bool__(self) -> bool:
        """True when allowed."""
        raise NotImplementedError


def check_admin(role_ids: Collection[int], *, admin_role_id: int) -> PermissionVerdict:
    """Does this member hold the admin role?

    ``admin_role_id == 0`` means unconfigured, which refuses everything. An unconfigured gate that
    defaulted to "allow" would hand ``/stop`` to the whole guild.
    """
    raise NotImplementedError


class RateLimiter:
    """Per-user command rate limiting. Body in M6."""

    def __init__(self, *, clock: Clock, per_minute: int = 5) -> None:
        """Hold the injected clock and the budget."""
        raise NotImplementedError

    def allow(self, user_id: int) -> PermissionVerdict:
        """Consume one token for ``user_id``, or refuse with the time remaining."""
        raise NotImplementedError
