"""Tests for the admin gate and the per-user rate limit.

The gate's defaults are deliberately hostile, and both of those defaults have a named test here:
an unconfigured role refuses everybody rather than allowing everybody, and the rate limit is keyed
per user so one person cannot lock everyone else out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mcmanager.discordbot.permissions import (
    NOT_CONFIGURED,
    NOT_PERMITTED,
    PermissionVerdict,
    RateLimiter,
    check_admin,
)

if TYPE_CHECKING:
    from mcmanager.clock import ManualClock

ADMIN = 1530965589303103590
OTHER = 999


class TestAdminGate:
    def test_holding_the_role_allows(self) -> None:
        assert check_admin([OTHER, ADMIN], admin_role_id=ADMIN)

    def test_not_holding_the_role_refuses_with_a_reason(self) -> None:
        verdict = check_admin([OTHER], admin_role_id=ADMIN)
        assert not verdict
        assert verdict.reason == NOT_PERMITTED

    def test_an_unconfigured_role_refuses_everybody(self) -> None:
        """A gate that fails open is worse than no gate, because it looks like protection."""
        assert not check_admin([OTHER, ADMIN], admin_role_id=0)
        assert check_admin([], admin_role_id=0).reason == NOT_CONFIGURED

    def test_a_negative_role_id_also_refuses(self) -> None:
        assert not check_admin([ADMIN], admin_role_id=-1)

    def test_no_roles_at_all_refuses(self) -> None:
        assert not check_admin([], admin_role_id=ADMIN)


class TestVerdict:
    def test_a_refusal_must_carry_a_reason(self) -> None:
        """Refusals are audited and shown to the user; an empty one is a bug, not a state."""
        with pytest.raises(ValueError, match="reason"):
            PermissionVerdict(allowed=False)

    def test_truthiness_matches_allowed(self) -> None:
        assert bool(PermissionVerdict(allowed=True))
        assert not bool(PermissionVerdict(allowed=False, reason="no"))


class TestRateLimiter:
    def test_the_budget_is_spent_then_refused(self, clock: ManualClock) -> None:
        limiter = RateLimiter(clock=clock, per_minute=3)
        assert all(limiter.allow(1) for _ in range(3))
        refused = limiter.allow(1)
        assert not refused
        assert "rate limited" in (refused.reason or "")

    async def test_the_window_slides(self, clock: ManualClock) -> None:
        limiter = RateLimiter(clock=clock, per_minute=2)
        limiter.allow(1)
        limiter.allow(1)
        assert not limiter.allow(1)

        await clock.advance(61)
        assert limiter.allow(1), "the window should have expired"

    def test_one_user_cannot_exhaust_another_users_budget(self, clock: ManualClock) -> None:
        """Keyed per user, so somebody hammering /restart cannot lock everyone out of /stop."""
        limiter = RateLimiter(clock=clock, per_minute=2)
        limiter.allow(1)
        limiter.allow(1)
        assert not limiter.allow(1)
        assert limiter.allow(2), "a different user has their own budget"

    def test_a_zero_budget_is_rejected_at_construction(self, clock: ManualClock) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            RateLimiter(clock=clock, per_minute=0)
