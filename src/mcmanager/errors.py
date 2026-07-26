"""The project-wide exception hierarchy, and the exit codes that go with it.

Every exception mcmanager raises on purpose descends from :class:`McManagerError`, so ``except
McManagerError`` at the top of ``main()`` is a complete catch for expected failure, and anything
else reaching that point is a bug worth a traceback.

Exit codes follow ``sysexits.h`` because the daemon runs under compose and its exit code is the
only thing a healthcheck or a runbook sees:

- **78** (``EX_CONFIG``) - the config is wrong. Print *every* validation error, not the first.
- **69** (``EX_UNAVAILABLE``) - something we depend on is not there: the Docker socket, or the
  daemon the CLI wanted to talk to. Never degrade silently into a half-working mode.
- **70** (``EX_SOFTWARE``) - an internal invariant broke.
"""

from __future__ import annotations

__all__ = [
    "EXIT_CONFIG",
    "EXIT_INTERNAL",
    "EXIT_OK",
    "EXIT_UNAVAILABLE",
    "ConfigError",
    "DaemonUnreachableError",
    "McManagerError",
    "NotSupportedError",
    "StateError",
]

EXIT_OK = 0
EXIT_CONFIG = 78
"""sysexits.h EX_CONFIG. Emitted after printing all of pydantic's ``e.errors()``."""
EXIT_UNAVAILABLE = 69
"""sysexits.h EX_UNAVAILABLE. The Docker socket, or a daemon the CLI required."""
EXIT_INTERNAL = 70
"""sysexits.h EX_SOFTWARE."""


class McManagerError(Exception):
    """Base of every deliberate failure in mcmanager.

    Attributes:
        exit_code: What ``main()`` should exit with if this escapes.
    """

    exit_code: int = EXIT_INTERNAL


class ConfigError(McManagerError):
    """The configuration is invalid, incomplete, or internally contradictory.

    Raised for cross-field failures as well as schema ones: ``discord.enabled`` without a token,
    ``runtime = "fake"`` together with a live Discord (refusing to post fake events to a real
    channel), or an idle poll interval that cannot possibly hit its own deadline.

    Fail fast, loudly, once. Never start in a degraded configuration.
    """

    exit_code = EXIT_CONFIG


class DaemonUnreachableError(McManagerError):
    """A CLI command that requires the running daemon could not reach it.

    Carries the URL that was tried so the message can say what to fix. Commands that require the
    daemon exit 69 rather than quietly falling back to standalone data: a ``status`` that shows
    stale standalone numbers while the daemon is wedged is worse than an error.

    Attributes:
        url: The endpoint that was tried.
    """

    exit_code = EXIT_UNAVAILABLE

    def __init__(self, message: str, *, url: str | None = None) -> None:
        super().__init__(message)
        self.url = url


class StateError(McManagerError):
    """An operation was asked for in a state that cannot service it.

    E.g. stopping a container that is already absent. Usually surfaced to the user as a rejected
    :class:`~mcmanager.core.events.CommandIssued`, not as a traceback.
    """


class NotSupportedError(McManagerError):
    """The configured game adapter or runtime does not implement this capability.

    E.g. asking for a command channel when ``rcon.mode = "disabled"``.
    """
