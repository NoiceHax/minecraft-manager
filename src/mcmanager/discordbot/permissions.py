"""Who may run the mutating commands.

A single ``admin_role_id`` check plus a rate limit, applied before the command body runs and
audited through ``CommandIssued(accepted=False, rejection=...)`` when it rejects - a refused stop
should be as visible as an accepted one.

Takes role ids as plain integers rather than a discord.py member object, so this file needs no
gateway import and its tests are a table.

**Two defaults that are deliberately hostile.** An unconfigured ``admin_role_id`` of ``0`` refuses
everything rather than allowing everything: a gate that fails open is worse than no gate, because
it looks like protection. And the rate limiter is keyed per user rather than globally, so one
person hammering ``/restart`` cannot lock everyone else out of ``/stop``.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Final, final

if TYPE_CHECKING:
    from collections.abc import Collection

    from mcmanager.clock import Clock

__all__ = [
    "NOT_CONFIGURED",
    "NOT_PERMITTED",
    "PermissionVerdict",
    "RateLimiter",
    "check_admin",
]

NOT_CONFIGURED: Final = (
    "the admin role is not configured, so nobody may run this; "
    "set discord.admin_role_id in mcmanager.toml"
)
NOT_PERMITTED: Final = "you do not hold the admin role"


@final
class PermissionVerdict:
    """Allowed, or refused with a reason. Refusals are audited, not silently dropped."""

    __slots__ = ("_allowed", "_reason")

    def __init__(self, *, allowed: bool, reason: str | None = None) -> None:
        """Hold the decision and, when refused, why."""
        if not allowed and reason is None:
            raise ValueError("a refusal must carry a reason; it is audited and shown to the user")
        self._allowed = allowed
        self._reason = reason

    @property
    def allowed(self) -> bool:
        """True when the command may proceed."""
        return self._allowed

    @property
    def reason(self) -> str | None:
        """Why it was refused, or ``None`` when allowed."""
        return self._reason

    def __bool__(self) -> bool:
        """True when allowed."""
        return self._allowed

    def __repr__(self) -> str:
        if self._allowed:
            return "PermissionVerdict(allowed=True)"
        return f"PermissionVerdict(allowed=False, reason={self._reason!r})"


_ALLOWED: Final = PermissionVerdict(allowed=True)


def check_admin(role_ids: Collection[int], *, admin_role_id: int) -> PermissionVerdict:
    """Does this member hold the admin role?

    ``admin_role_id == 0`` means unconfigured, which refuses everything. An unconfigured gate that
    defaulted to "allow" would hand ``/stop`` to the whole guild.

    Note that the ``@everyone`` role id equals the guild id and is present on every member, so
    configuring it would be equivalent to no gate at all. That is a deployment mistake this
    function cannot detect, and it is why the setup notes say to create a dedicated role.
    """
    if admin_role_id <= 0:
        return PermissionVerdict(allowed=False, reason=NOT_CONFIGURED)
    if admin_role_id in role_ids:
        return _ALLOWED
    return PermissionVerdict(allowed=False, reason=NOT_PERMITTED)


@final
class RateLimiter:
    """Per-user command rate limiting.

    A sliding window of recent command timestamps per user, on the injected clock's monotonic
    reading rather than the wall clock, so a step in system time cannot hand somebody an unlimited
    budget or lock them out for an hour.
    """

    def __init__(self, *, clock: Clock, per_minute: int = 5) -> None:
        """Hold the injected clock and the budget."""
        if per_minute < 1:
            raise ValueError("per_minute must be at least 1")
        self._clock = clock
        self._per_minute = per_minute
        self._window = 60.0
        self._hits: dict[int, deque[float]] = {}

    @property
    def per_minute(self) -> int:
        """The budget each user gets per rolling minute."""
        return self._per_minute

    def allow(self, user_id: int) -> PermissionVerdict:
        """Consume one token for ``user_id``, or refuse with the time remaining."""
        now = self._clock.monotonic()
        hits = self._hits.setdefault(user_id, deque())
        cutoff = now - self._window
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self._per_minute:
            retry_in = hits[0] + self._window - now
            return PermissionVerdict(
                allowed=False,
                reason=f"rate limited, try again in {max(retry_in, 0.0):.0f}s",
            )
        hits.append(now)
        return _ALLOWED

    def reset(self, user_id: int | None = None) -> None:
        """Forget one user's history, or everyone's. Only used by tests and ``aclose``."""
        if user_id is None:
            self._hits.clear()
        else:
            self._hits.pop(user_id, None)
