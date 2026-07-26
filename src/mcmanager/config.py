"""Configuration model and loading.

One of the two places pydantic belongs (the other is :mod:`mcmanager.core.serde`). TOML through
``pydantic-settings``' :class:`~pydantic_settings.TomlConfigSettingsSource`, with precedence:

    defaults  <  TOML file  <  ``secrets_dir`` files  <  ``MCM_*`` env  <  CLI overrides

Decisions worth not relitigating:

- **``extra = "forbid"`` on every nested model.** A typo'd ``stop_timeout`` silently doing nothing
  is exactly the 2am bug to design out.
- **``frozen = True`` plus a single ``@lru_cache`` :func:`get_settings`**, with
  ``get_settings.cache_clear()`` in a test fixture.
- **No secret is ever a ``str``.** ``discord.token``, ``web.token`` and ``rcon.password`` are
  ``SecretStr | None``, so ``repr()`` and ``model_dump()`` render ``**********`` and an accidental
  ``log.info("config", cfg=settings)`` cannot leak. There is a unit test that walks the model and
  fails if a plain-``str`` secret field is ever added.
- **Discord snowflakes are ``int``**, not ``""``. An empty string defers every misconfiguration to
  a runtime ``int("")`` three hours later.
- **Indirection references.** Any string value anywhere in the config may be written as
  ``"env:NAME"`` or ``"file:/path"`` and is resolved before validation. That is how a secret gets
  into the model without ever being written down: ``token = "file:/run/secrets/discord_token"``.
  A missing target is a hard validation error, never a silent empty string.
- **A literal secret in the TOML is accepted but warned about**, loudly, once, at startup. Refusing
  it outright would make a local experiment impossible; staying quiet about it is how a token ends
  up in git.

Startup contract: **fail fast, loudly, once - never degrade silently.**

- A config error prints *every* ``ValidationError`` entry, one line each, and exits **78**
  (``EX_CONFIG``).
- A Docker ping failure exits **69** (``EX_UNAVAILABLE``) with a message naming the socket **and
  the process's uid/gid/groups**, because that error is the docker-group (gid 983) problem 95% of
  the time and printing ``id`` saves twenty minutes.
- A Discord login failure exits **77** (``EX_NOPERM``): the token is wrong or the bot was removed
  from the guild, and retrying forever against a rejected credential is how you get rate limited.
- An absent container is a warning, not a failure. A failed status probe is a warning.

Nothing in this module prints. It builds messages and raises; ``cli/render.py`` is the only module
allowed to write to a stream.
"""

from __future__ import annotations

import os
import tomllib
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    ValidationError,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from mcmanager.core.types import ReadySignal, RuntimeKind
from mcmanager.errors import EXIT_UNAVAILABLE, ConfigError, McManagerError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from pydantic.fields import FieldInfo

__all__ = [
    "CONFIG_FILE_ENV",
    "DEFAULT_SECRETS_DIR",
    "EXIT_NOPERM",
    "SCHEMA_VERSION",
    "SECRETS_DIR_ENV",
    "SECRET_FILE_NAMES",
    "DiscordAuthError",
    "DiscordConfig",
    "DockerConfig",
    "EventsConfig",
    "FlatFileSecretsSource",
    "IdleConfig",
    "LifecycleConfig",
    "LogsConfig",
    "ProbeConfig",
    "RconConfig",
    "RuntimeUnreachableError",
    "ServerConfig",
    "Settings",
    "StateConfig",
    "WebConfig",
    "discord_auth_message",
    "format_validation_errors",
    "get_settings",
    "load_settings",
    "process_identity",
    "redact",
    "runtime_unreachable_message",
    "startup_banner",
]

SCHEMA_VERSION: Final = 1
"""Bumped when a backwards-incompatible schema change lands. The loader refuses a version it does
not understand rather than guessing at what the old keys meant."""

CONFIG_FILE_ENV: Final = "MCM_CONFIG_FILE"
SECRETS_DIR_ENV: Final = "MCM_SECRETS_DIR"
ENV_PREFIX: Final = "MCM_"
DEFAULT_SECRETS_DIR: Final = Path("/run/secrets")
DEFAULT_CONFIG_FILE: Final = Path("config/mcmanager.toml")

EXIT_NOPERM: Final = 77
"""sysexits.h ``EX_NOPERM``. Discord refused the credential: bad token, or the bot is no longer in
the guild. Distinct from 69 on purpose - 69 means "try again later", 77 means "a human must fix a
secret", and a supervisor that restarts on 69 should give up on 77.

This constant belongs next to its siblings in :mod:`mcmanager.errors`; it lives here only because
that module is owned by another change in flight. Move it, and re-export from here.
"""


class RuntimeUnreachableError(McManagerError):
    """The container runtime did not answer at startup.

    Carries the endpoint and the process identity so the message can say the thing that is almost
    always true: the daemon is not in the ``docker`` group.

    Belongs in :mod:`mcmanager.errors` alongside :class:`~mcmanager.errors.DaemonUnreachableError`.
    """

    exit_code = EXIT_UNAVAILABLE


class DiscordAuthError(McManagerError):
    """Discord rejected the bot token, or the bot is not in the configured guild.

    Belongs in :mod:`mcmanager.errors`.
    """

    exit_code = EXIT_NOPERM


# ------------------------------------------------------------------- indirection references
#
# "env:NAME" and "file:/path" are resolved for every string value in the merged config, before
# validation. Two forms rather than a general template language on purpose: these are the only two
# indirections a container actually needs, and anything more expressive becomes a place where
# config grows behaviour.

ENV_REF_PREFIX: Final = "env:"
FILE_REF_PREFIX: Final = "file:"


def _resolve_ref(value: str) -> str:
    """Resolve one ``env:``/``file:`` reference. A plain string is returned unchanged."""
    if value.startswith(ENV_REF_PREFIX):
        name = value[len(ENV_REF_PREFIX) :].strip()
        if not name:
            msg = f"{value!r} is an env reference with no variable name"
            raise ValueError(msg)
        resolved = os.environ.get(name)
        if resolved is None:
            msg = (
                f"{value!r} refers to environment variable {name!r}, which is not set. "
                "Set it, or write the value literally."
            )
            raise ValueError(msg)
        return resolved
    if value.startswith(FILE_REF_PREFIX):
        raw = value[len(FILE_REF_PREFIX) :].strip()
        if not raw:
            msg = f"{value!r} is a file reference with no path"
            raise ValueError(msg)
        path = Path(raw).expanduser()
        if not path.is_file():
            msg = f"{value!r} refers to {path}, which does not exist or is not a file"
            raise ValueError(msg)
        # Trailing newlines are what `echo secret > file` leaves behind, and a token with a
        # newline on the end fails authentication in a way that reads as "wrong token".
        return path.read_text(encoding="utf-8").strip()
    return value


def _resolve_refs(value: object) -> object:
    """Recursively resolve indirection references in a decoded config tree."""
    if isinstance(value, str):
        return _resolve_ref(value)
    if isinstance(value, dict):
        items = cast("dict[str, object]", value)
        return {key: _resolve_refs(item) for key, item in items.items()}
    if isinstance(value, list):
        items_list = cast("list[object]", value)
        return [_resolve_refs(item) for item in items_list]
    return value


def _is_ref(value: object) -> bool:
    return isinstance(value, str) and (
        value.startswith(ENV_REF_PREFIX) or value.startswith(FILE_REF_PREFIX)
    )


# ------------------------------------------------------------------------ small field types


def _upper(value: object) -> object:
    return value.upper() if isinstance(value, str) else value


def _lower(value: object) -> object:
    return value.lower() if isinstance(value, str) else value


type LogLevel = Annotated[
    Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], BeforeValidator(_upper)
]
type LogFormat = Annotated[Literal["json", "console"], BeforeValidator(_lower)]
type RconMode = Annotated[Literal["exec", "network", "disabled"], BeforeValidator(_lower)]
type DiscordMode = Annotated[Literal["live", "dryrun", "disabled"], BeforeValidator(_lower)]


class _Section(BaseModel):
    """Base of every nested config table.

    ``extra="forbid"`` is the whole point: ``stop_timeout = 90`` next to a field actually called
    ``stop_timeout_seconds`` is a startup error, not a silently ignored line.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        str_strip_whitespace=True,
    )


# ---------------------------------------------------------------------------------- sections


class LifecycleConfig(_Section):
    """``[server.lifecycle]`` - how the daemon starts and stops the server."""

    stop_timeout_seconds: int = Field(default=90, ge=1, le=3600)
    """Seconds given to the JVM to save and exit before Docker sends SIGKILL. This was a bare
    ``-t 90`` in the old bash script with no explanation attached."""

    start_deadline_extra_seconds: int = Field(default=60, ge=0, le=3600)
    """Extra seconds past the container's own healthcheck guard window before a server stuck in
    STARTING is declared DEGRADED. The guard window itself is read from the container
    (``StartPeriod + Interval * (Retries + 1)``), never hardcoded."""

    ready_signals: tuple[ReadySignal, ...] = (
        ReadySignal.LOG_DONE,
        ReadySignal.HEALTHCHECK,
        ReadySignal.PROBE,
    )
    """Which signals may promote STARTING -> READY, first one wins. Set to ``["health"]`` for the
    strict, spec-literal behaviour."""

    treat_unexpected_exit_as_crash: bool = True

    @model_validator(mode="after")
    def _check(self) -> LifecycleConfig:
        if not self.ready_signals:
            msg = "server.lifecycle.ready_signals must list at least one signal"
            raise ValueError(msg)
        if len(set(self.ready_signals)) != len(self.ready_signals):
            msg = "server.lifecycle.ready_signals contains duplicates"
            raise ValueError(msg)
        return self


class ServerConfig(_Section):
    """``[server]`` - which server this daemon manages."""

    id: str = Field(default="minecraft", min_length=1)
    """Stable identity stamped onto every event as ``Event.server_id``."""

    game: str = Field(default="minecraft", min_length=1)
    """Which ``games/`` adapter parses this server's logs."""

    container: str = Field(default="minecraft", min_length=1)
    """Docker container name. Re-resolved by name on every reconnect, never cached by id."""

    host: str = Field(default="minecraft", min_length=1)
    """Hostname for SLP status probes. **Not ``localhost``**: from inside our container that is our
    own loopback where nothing listens, and idle shutdown treating a failed probe as "zero players"
    would then kill a populated server."""

    port: int = Field(default=25565, ge=1, le=65535)

    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)

    @model_validator(mode="after")
    def _check(self) -> ServerConfig:
        if self.host == "localhost" or self.host.startswith("127."):
            msg = (
                "server.host is loopback. Inside a container that is the daemon's own loopback, "
                "where no Minecraft server listens; every probe would fail. Use the container's "
                'DNS name on the shared docker network (e.g. "minecraft").'
            )
            raise ValueError(msg)
        return self


class DockerConfig(_Section):
    """``[docker]`` - how we talk to the container platform."""

    host: str | None = None
    """``None`` means "let docker-py read ``DOCKER_HOST`` or the default socket", which is what the
    deployed daemon uses and what makes ``DOCKER_HOST=ssh://minty@192.168.1.7`` work for the
    standalone CLI from Windows."""

    ping_timeout_seconds: float = Field(default=5.0, gt=0)
    reconnect_initial_seconds: float = Field(default=1.0, gt=0)
    reconnect_max_seconds: float = Field(default=30.0, gt=0)
    reconcile_interval_seconds: float = Field(default=60.0, gt=0)
    dedupe_ring_size: int = Field(default=256, ge=1)
    """Docker's ``since`` is second-granularity and replays the whole of that second, so lines are
    de-duplicated through a ring of ``(timestamp, hash)`` pairs."""

    @model_validator(mode="after")
    def _check(self) -> DockerConfig:
        if self.reconnect_max_seconds < self.reconnect_initial_seconds:
            msg = "docker.reconnect_max_seconds must be >= docker.reconnect_initial_seconds"
            raise ValueError(msg)
        return self


class EventsConfig(_Section):
    """``[events]`` - bus queue and rate-limit tuning."""

    queue_max: int = Field(default=5000, ge=1)
    handler_timeout_seconds: float = Field(default=5.0, gt=0)
    max_consecutive_failures: int = Field(default=5, ge=1)
    rate_limit_per_second: int = Field(default=200, ge=1)
    rate_limit_burst: int = Field(default=1000, ge=1)
    crash_tail_lines: int = Field(default=40, ge=0)
    drain_timeout_seconds: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def _check(self) -> EventsConfig:
        if self.rate_limit_burst < self.rate_limit_per_second:
            msg = "events.rate_limit_burst must be >= events.rate_limit_per_second"
            raise ValueError(msg)
        return self


class ProbeConfig(_Section):
    """``[probe]`` - the Server List Ping poller."""

    enabled: bool = True
    interval_seconds: float = Field(default=45.0, gt=0)
    timeout_seconds: float = Field(default=5.0, gt=0)
    trust_partial_sample: bool = False
    """``players.sample`` is capped at 12 by the protocol and can be randomised by plugins. Leave
    this false: roster reconciliation against a truncated sample is a storm of fake leave events."""

    @model_validator(mode="after")
    def _check(self) -> ProbeConfig:
        if self.timeout_seconds >= self.interval_seconds:
            msg = "probe.timeout_seconds must be < probe.interval_seconds, or probes overlap"
            raise ValueError(msg)
        return self


class RconConfig(_Section):
    """``[rcon]`` - the command channel.

    ``exec`` is the default because the itzg image configures ``rcon-cli`` from the container's own
    environment, so mcmanager never needs the RCON password at all: nothing to store, rotate or
    leak.
    """

    mode: RconMode = "exec"
    host: str = Field(default="minecraft", min_length=1)
    port: int = Field(default=25575, ge=1, le=65535)
    timeout_seconds: float = Field(default=10.0, gt=0)
    password: SecretStr | None = None
    """Only needed for ``mode = "network"``. Never written in the TOML: ``secrets_dir/
    rcon_password``, ``MCM_RCON__PASSWORD``, or ``password = "file:/run/secrets/rcon_password"``."""

    @model_validator(mode="after")
    def _check(self) -> RconConfig:
        if self.mode == "network" and self.password is None:
            msg = (
                'rcon.mode = "network" needs rcon.password. Provide it as a file in secrets_dir '
                "(rcon_password), as MCM_RCON__PASSWORD, or leave the default exec mode, which "
                "needs no password at all."
            )
            raise ValueError(msg)
        return self


class IdleConfig(_Section):
    """``[idle]`` - automatic shutdown of an empty server."""

    enabled: bool = False
    """Off by default. Turn it on only after the dry-run soak in the cutover plan."""

    dry_run: bool = True
    """Compute and log "would stop", never actually stop. The cutover setting."""

    timeout_minutes: float = Field(default=15, gt=0)
    poll_interval_seconds: float = Field(default=60, gt=0)
    warn_minutes: float = Field(default=2, ge=0)
    min_uptime_minutes: float = Field(default=20, ge=0)
    """Never idle-stop a server that just booted: "start -> nobody joined in 15 min -> instant
    shutdown" reads as broken to everyone watching."""

    treat_probe_failure_as_empty: bool = False
    """A failed status probe means UNKNOWN, not "zero players". Leave this false. Setting it true
    is how you lose somebody's build session."""

    @property
    def timeout_seconds(self) -> float:
        return self.timeout_minutes * 60.0

    @model_validator(mode="after")
    def _check(self) -> IdleConfig:
        limit = self.timeout_seconds / 3.0
        if self.poll_interval_seconds > limit:
            msg = (
                f"idle.poll_interval_seconds ({self.poll_interval_seconds:g}) must be <= a third "
                f"of idle.timeout_minutes ({self.timeout_minutes:g} min = {limit:g}s), or the "
                "countdown cannot observe its own deadline with any precision"
            )
            raise ValueError(msg)
        if self.warn_minutes >= self.timeout_minutes:
            msg = (
                f"idle.warn_minutes ({self.warn_minutes:g}) must be < idle.timeout_minutes "
                f"({self.timeout_minutes:g}), or the warning fires before the countdown starts"
            )
            raise ValueError(msg)
        return self


class LogsConfig(_Section):
    """``[logs]`` - three different directories for three different things.

    Where the *server's* logs are, where *our* archives go, and where the *daemon's* own logs go
    are three separate concerns with three separate mounts, and collapsing them into one
    ``directory`` key is how a daemon ends up writing into a read-only bind.
    """

    server_log_dir: Path = Path("/mnt/mc-logs")
    """The server's own log4j2 output, mounted read-only. log4j2 already gzips per-run archives
    here; the daemon must never create or delete files in this directory."""

    archive_dir: Path = Path("/var/lib/mcmanager/archives")
    """Where our gap-filling archives go. log4j2 rolls ``latest.log`` at the next *start*, so a
    session never followed by another start would otherwise have no archive at all."""

    daemon_log_dir: Path = Path("/var/lib/mcmanager/logs")
    level: LogLevel = "INFO"
    format: LogFormat = "json"
    archive_on_stop: bool = True
    retain_archives: int = Field(default=50, ge=0)


class StateConfig(_Section):
    """``[state]`` - session records, the resume marker and the idle deadline."""

    dir: Path = Path("/var/lib/mcmanager/state")
    checkpoint_interval_seconds: float = Field(default=60.0, gt=0)


class DiscordConfig(_Section):
    """``[discord]`` - the bot.

    Snowflakes are ``int`` and default to ``0`` meaning "unset". ``0`` is not a valid snowflake, so
    the cross-field validator can demand ``> 0`` without inventing a sentinel type.
    """

    enabled: bool = False
    mode: DiscordMode = "dryrun"
    """``live`` posts for real, ``dryrun`` logs what it would send and sends nothing (the 48h
    shadow-run setting), ``disabled`` never opens the gateway."""

    guild_id: int = Field(default=0, ge=0)
    channel_id: int = Field(default=0, ge=0)
    console_channel_id: int = Field(default=0, ge=0)
    """Optional raw console relay. ``0`` disables it."""

    admin_role_id: int = Field(default=0, ge=0)
    token: SecretStr | None = None
    register_commands_globally: bool = False
    """Guild-scoped registration is instant; global takes about an hour."""

    @property
    def live(self) -> bool:
        """True only when the gateway will actually post. The one predicate worth sharing."""
        return self.enabled and self.mode == "live"

    @model_validator(mode="after")
    def _check(self) -> DiscordConfig:
        if not self.enabled:
            return self
        problems: list[str] = []
        if self.mode == "disabled":
            problems.append('discord.enabled is true but discord.mode is "disabled"')
        if self.token is None:
            problems.append(
                "discord.enabled is true but no token was found. Provide "
                "secrets_dir/discord_token, MCM_DISCORD__TOKEN, or "
                'token = "file:/run/secrets/discord_token"'
            )
        for name in ("guild_id", "channel_id", "admin_role_id"):
            if getattr(self, name) <= 0:
                problems.append(
                    f"discord.enabled is true but discord.{name} is not set (must be > 0)"
                )
        if problems:
            raise ValueError("\n".join(problems))
        return self


class WebConfig(_Section):
    """``[web]`` - the aiohttp control surface the CLI, Discord and any web UI all speak to."""

    enabled: bool = True
    host: str = Field(default="127.0.0.1", min_length=1)
    """Bind loopback only. On the homelab 8787 is ``expose``d and never published; the documented
    access path is ``docker exec mcmanager mcmanager status``, so there is no LAN auth surface."""

    port: int = Field(default=8787, ge=1, le=65535)
    url: str = "http://127.0.0.1:8787"
    """What the CLI dials when neither ``--url`` nor ``MCMANAGER_URL`` is set."""

    sse_queue_max: int = Field(default=500, ge=1)
    token: SecretStr | None = None
    """Required for ``POST /control/*``. Absent means the mutating endpoints refuse everything."""

    @model_validator(mode="after")
    def _check(self) -> WebConfig:
        if not self.url.startswith(("http://", "https://")):
            msg = f"web.url must start with http:// or https://, got {self.url!r}"
            raise ValueError(msg)
        return self


# ----------------------------------------------------------------------------------- settings

SECRET_FILE_NAMES: Final[Mapping[str, tuple[str, str]]] = {
    "discord_token": ("discord", "token"),
    "web_token": ("web", "token"),
    "rcon_password": ("rcon", "password"),
}
"""File name in ``secrets_dir`` -> ``(section, field)``.

Deliberately flat rather than pydantic-settings' nested ``discord__token`` convention: these names
are what ``deploy/docker-compose.yml`` declares under ``secrets:`` and what the annotated TOML
documents, and a Docker secret's file name is fixed by the compose file, not by us. The nested
spelling is accepted too, so neither convention surprises anybody.
"""


class FlatFileSecretsSource(PydanticBaseSettingsSource):
    """Reads one secret per file from ``secrets_dir``, using the flat names above.

    The built-in :class:`~pydantic_settings.SecretsSettingsSource` maps file names to *top level*
    fields, so ``discord.token`` would have to live in a file called ``discord__token``. Compose
    names its secrets ``discord_token``. Rather than bend the deployment to the library, this
    source does the mapping.
    """

    def __init__(self, settings_cls: type[BaseSettings], secrets_dir: Path | None) -> None:
        super().__init__(settings_cls)
        self._secrets_dir = secrets_dir

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        """Unused: this source builds its whole mapping in :meth:`__call__`."""
        del field
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        directory = self._secrets_dir
        if directory is None or not directory.is_dir():
            return {}
        found: dict[str, Any] = {}
        for file_name, (section, field) in SECRET_FILE_NAMES.items():
            for candidate in (file_name, f"{section}__{field}"):
                path = directory / candidate
                if not path.is_file():
                    continue
                value = path.read_text(encoding="utf-8").strip()
                if not value:
                    continue
                found.setdefault(section, {})[field] = value
                break
        return found

    def __repr__(self) -> str:
        return f"{type(self).__name__}(secrets_dir={self._secrets_dir})"


@dataclass(frozen=True, slots=True)
class _LoadContext:
    """Where :class:`Settings` should look for its file-backed sources.

    Carried in a :class:`~contextvars.ContextVar` rather than passed as an argument because
    ``settings_customise_sources`` is a classmethod that pydantic-settings calls for us and does
    not forward constructor keywords to. :func:`load_settings` is the only thing that sets it.
    """

    toml_file: Path | None = None
    secrets_dir: Path | None = None


_LOAD_CONTEXT: ContextVar[_LoadContext] = ContextVar("mcmanager_load_context")


def _current_context() -> _LoadContext:
    return _LOAD_CONTEXT.get(_LoadContext())


class Settings(BaseSettings):
    """The whole configuration, resolved.

    Frozen, so nothing can reconfigure the daemon at runtime by accident, and hashable, so
    :func:`get_settings`'s cache is well defined.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
        nested_model_default_partial_update=True,
    )

    schema_version: int = SCHEMA_VERSION
    runtime: RuntimeKind = "real"
    """``real`` talks to the Docker daemon. ``fake`` drives a scriptable in-memory runtime.
    Deliberately not the default: a homelab daemon silently running against a fake in production is
    worse than a startup failure."""

    secrets_dir: Path = DEFAULT_SECRETS_DIR

    server: ServerConfig = Field(default_factory=ServerConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    rcon: RconConfig = Field(default_factory=RconConfig)
    idle: IdleConfig = Field(default_factory=IdleConfig)
    logs: LogsConfig = Field(default_factory=LogsConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    _warnings: list[str] = PrivateAttr(default_factory=list)
    _config_file: Path | None = PrivateAttr(default=None)

    # -- sources ----------------------------------------------------------------------------

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Highest priority first: CLI overrides, env, secret files, TOML, defaults.

        The built-in ``file_secret_settings`` is replaced by :class:`FlatFileSecretsSource`, and
        ``dotenv_settings`` is dropped: ``.env`` is a convenience for the *process environment*
        (compose reads it via ``env_file``), not a second config format for the daemon to parse.
        """
        del dotenv_settings, file_secret_settings
        context = _current_context()
        return (
            init_settings,
            env_settings,
            FlatFileSecretsSource(settings_cls, context.secrets_dir),
            TomlConfigSettingsSource(settings_cls, toml_file=context.toml_file),
        )

    # -- validation -------------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _resolve_indirection(cls, data: object) -> object:
        """Turn every ``env:``/``file:`` string in the merged config into its target."""
        if isinstance(data, dict):
            return _resolve_refs(cast("dict[str, object]", data))
        return data

    @model_validator(mode="after")
    def _cross_field_checks(self) -> Settings:
        problems: list[str] = []

        if self.schema_version != SCHEMA_VERSION:
            problems.append(
                f"schema_version {self.schema_version} is not understood by this build "
                f"(expected {SCHEMA_VERSION}). Refusing to guess at what the old keys meant."
            )

        # Refusing to post fake events into a real channel is worth a hard failure: a shadow run
        # that turns out to have been driving a fake runtime tells you nothing.
        if self.runtime == "fake" and self.discord.live:
            problems.append(
                'runtime = "fake" with discord.mode = "live": the daemon would post events from a '
                'scriptable in-memory runtime into a real channel. Use discord.mode = "dryrun" '
                'or runtime = "real".'
            )

        if self.logs.archive_on_stop:
            problems.extend(self._check_server_log_dir())

        if problems:
            raise ValueError("\n".join(problems))

        self._add_soft_warnings()
        return self

    def _check_server_log_dir(self) -> list[str]:
        """``logs.archive_on_stop`` promises to gzip ``latest.log``; check we can read it.

        A directory that is *missing* is a warning, not an error: the example config names
        ``/mnt/mc-logs``, which exists only on the homelab, and ``mcmanager check-config
        config/mcmanager.example.toml`` has to pass on a developer laptop. A directory that
        *exists but cannot be read* is a hard error, because that is the real failure - a bind
        mount with the wrong uid, which would otherwise surface as silently missing archives.
        """
        directory = self.logs.server_log_dir
        if not directory.exists():
            return []
        if not directory.is_dir():
            return [f"logs.server_log_dir ({directory}) exists but is not a directory"]
        if not os.access(directory, os.R_OK | os.X_OK):
            return [
                f"logs.archive_on_stop is true but logs.server_log_dir ({directory}) is not "
                f"readable by this process ({process_identity()}). The bind mount's owner and the "
                "container's uid have to agree."
            ]
        return []

    def _add_soft_warnings(self) -> None:
        if self.logs.archive_on_stop and not self.logs.server_log_dir.exists():
            self.add_warning(
                f"logs.server_log_dir ({self.logs.server_log_dir}) does not exist; "
                "session archiving is disabled until the bind mount appears"
            )
        if self.web.enabled and self.web.token is None:
            self.add_warning(
                "web.enabled is true but web.token is unset: POST /control/* will refuse every "
                "request. Reads (/status, /events, /logs) still work."
            )
        if self.runtime == "fake":
            self.add_warning(
                'runtime = "fake": no Docker socket is touched and nothing real is started or '
                "stopped. This must never be the production setting."
            )
        if self.idle.enabled and self.idle.dry_run:
            self.add_warning(
                "idle.enabled is true but idle.dry_run is set: shutdowns are computed and logged, "
                "never performed."
            )
        if self.idle.treat_probe_failure_as_empty:
            self.add_warning(
                "idle.treat_probe_failure_as_empty is true: an unreachable server counts as "
                "empty, so one bad probe can stop a populated server. That is how you lose "
                "somebody's build session."
            )

    # -- warnings ---------------------------------------------------------------------------

    @property
    def warnings(self) -> tuple[str, ...]:
        """Non-fatal problems found while loading. Logged once at startup; never raised."""
        return tuple(self._warnings)

    def add_warning(self, message: str) -> None:
        """Record a non-fatal problem. Idempotent, so re-validation cannot duplicate a line."""
        if message not in self._warnings:
            self._warnings.append(message)

    @property
    def config_file(self) -> Path | None:
        """The TOML that was loaded, if any. For the startup banner and ``check-config``."""
        return self._config_file

    def record_source(self, path: Path | None) -> None:
        """Remember which file this was loaded from. Called once, by :func:`load_settings`.

        Not a model field: it is provenance, not configuration, and making it a field would mean a
        TOML could claim to have come from somewhere else.
        """
        self._config_file = path


# ------------------------------------------------------------------------------------ loading


def _resolve_config_file(explicit: Path | str | None) -> Path | None:
    if explicit is not None:
        return Path(explicit).expanduser()
    from_env = os.environ.get(CONFIG_FILE_ENV)
    if from_env:
        return Path(from_env).expanduser()
    if DEFAULT_CONFIG_FILE.is_file():
        return DEFAULT_CONFIG_FILE
    return None


def _peek_toml(path: Path | None) -> dict[str, Any]:
    """Decode the TOML once, before validation, so we can find ``secrets_dir`` in it.

    Chicken and egg: the secrets source has to be built before validation, but the directory it
    reads is itself a config value. Decoding the file twice is cheaper than every alternative and
    keeps the precedence rule honest.
    """
    if path is None or not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        msg = f"could not read config file {path}: {exc}"
        raise ConfigError(msg) from exc


def _resolve_secrets_dir(explicit: Path | str | None, toml_data: Mapping[str, Any]) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    from_env = os.environ.get(SECRETS_DIR_ENV)
    if from_env:
        return Path(from_env).expanduser()
    from_toml = toml_data.get("secrets_dir")
    if isinstance(from_toml, str) and from_toml:
        # This resolution happens before validation, so a bad reference here would otherwise
        # escape as a bare ValueError instead of the exit-78 report everything else gets.
        try:
            return Path(_resolve_ref(from_toml)).expanduser()
        except ValueError as exc:
            msg = f"secrets_dir: {exc}"
            raise ConfigError(msg) from exc
    return DEFAULT_SECRETS_DIR


def _literal_secret_warnings(toml_data: Mapping[str, Any]) -> Iterator[str]:
    """Warn - never fail - when a secret is written literally in the TOML.

    Failing would make a five-minute local experiment impossible. Staying quiet is how a token
    ends up in git, which is the mistake this whole project exists to stop repeating.
    """
    for file_name, (section, field) in SECRET_FILE_NAMES.items():
        table = toml_data.get(section)
        if not isinstance(table, dict):
            continue
        value = cast("dict[str, object]", table).get(field)
        if value is None or _is_ref(value):
            continue
        yield (
            f"{section}.{field} is written literally in the config file. Move it to "
            f"secrets_dir/{file_name}, to MCM_{section.upper()}__{field.upper()}, or to "
            f'{field} = "file:/run/secrets/{file_name}" - and rotate it if this file is in git.'
        )


def load_settings(
    *,
    config_file: Path | str | None = None,
    secrets_dir: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Build :class:`Settings` from every layer, in precedence order.

    Args:
        config_file: TOML to load. Defaults to ``$MCM_CONFIG_FILE``, then
            ``config/mcmanager.toml`` if it exists, then nothing (pure defaults).
        secrets_dir: Directory of one-secret-per-file. Defaults to ``$MCM_SECRETS_DIR``, then the
            ``secrets_dir`` key in the TOML, then ``/run/secrets``.
        overrides: The CLI layer. Highest priority of all, and nested like the TOML:
            ``{"logs": {"level": "DEBUG"}}``.

    Raises:
        ConfigError: with every validation failure listed, one per line. Exit code 78.
    """
    resolved_config = _resolve_config_file(config_file)
    toml_data = _peek_toml(resolved_config)
    resolved_secrets = _resolve_secrets_dir(secrets_dir, toml_data)

    token = _LOAD_CONTEXT.set(_LoadContext(toml_file=resolved_config, secrets_dir=resolved_secrets))
    try:
        settings = Settings(**dict(overrides or {}))
    except ValidationError as exc:
        raise ConfigError(format_validation_errors(exc, config_file=resolved_config)) from exc
    finally:
        _LOAD_CONTEXT.reset(token)

    settings.record_source(resolved_config)
    for warning in _literal_secret_warnings(toml_data):
        settings.add_warning(warning)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings, loaded once.

    ``get_settings.cache_clear()`` is what a test fixture calls between cases; nothing else should
    ever call it.
    """
    return load_settings()


# ------------------------------------------------------------------------ failure reporting


def format_validation_errors(
    exc: ValidationError,
    *,
    config_file: Path | None = None,
) -> str:
    """Render *every* validation failure, one per line.

    Printing only the first error means three restarts to fix three typos, which is precisely the
    experience a fail-fast daemon is supposed to avoid. Cross-field validators raise with embedded
    newlines so that several related problems still come out as several lines.
    """
    where = f" in {config_file}" if config_file is not None else ""
    header = f"{exc.error_count()} configuration error(s){where}:"
    lines: list[str] = [header]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        for index, part in enumerate(str(error["msg"]).split("\n")):
            prefix = f"  {location}: " if index == 0 else "    "
            lines.append(f"{prefix}{part}")
    lines.append(
        "Fix the file, or run `mcmanager check-config --config <file>` to re-validate without "
        "starting anything."
    )
    return "\n".join(lines)


def process_identity() -> str:
    """``uid=1000 gid=1000 groups=1000,983``, or a note that this platform has no such thing.

    Printed by :func:`runtime_unreachable_message` because a Docker socket permission error is the
    group-membership problem 95% of the time, and this line is what makes that obvious instead of
    a twenty-minute detour.
    """
    getuid = cast("Callable[[], int] | None", getattr(os, "getuid", None))
    getgid = cast("Callable[[], int] | None", getattr(os, "getgid", None))
    getgroups = cast("Callable[[], Sequence[int]] | None", getattr(os, "getgroups", None))
    if getuid is None or getgid is None:
        return "uid/gid unavailable on this platform (not POSIX)"
    groups = ",".join(str(gid) for gid in getgroups()) if getgroups is not None else "?"
    return f"uid={getuid()} gid={getgid()} groups={groups}"


def runtime_unreachable_message(*, endpoint: str | None, error: str) -> str:
    """The exit-69 message. Names the socket **and** the process identity."""
    target = endpoint or os.environ.get("DOCKER_HOST") or "unix:///var/run/docker.sock"
    return (
        f"Docker is unreachable at {target}: {error}\n"
        f"This process runs as {process_identity()}.\n"
        "If that is a permission error, the socket is owned by root:docker (gid 983 on the "
        "homelab) and this process is not in that group. In compose that is\n"
        '  user: "1000:1000"\n'
        '  group_add: ["${DOCKER_GID:-983}"]\n'
        "The gid does not need to exist inside the image; the kernel only compares the number."
    )


def discord_auth_message(*, error: str, guild_id: int | None = None) -> str:
    """The exit-77 message. A rejected credential is a human problem, not a retryable one."""
    guild = f" for guild {guild_id}" if guild_id else ""
    return (
        f"Discord refused the bot credential{guild}: {error}\n"
        "The token is wrong, was rotated, or the bot was removed from the guild. Retrying will "
        "not fix it and will get the app rate limited.\n"
        "Check secrets_dir/discord_token (or MCM_DISCORD__TOKEN), then re-invite the bot with the "
        "applications.commands scope."
    )


# ------------------------------------------------------------------------------- redaction


_SECRET_PLACEHOLDER: Final = "<set>"  # noqa: S105 - the point is that it is not one
_UNSET_PLACEHOLDER: Final = "<unset>"


def _redact_value(value: object) -> object:
    if isinstance(value, SecretStr):
        return _SECRET_PLACEHOLDER
    if isinstance(value, Enum):
        return cast("object", value.value)
    if isinstance(value, Path):
        # POSIX form even on Windows: the banner is compared against the deployed config more
        # often than it is read on the dev box, and backslashes make that needlessly hard.
        return value.as_posix()
    if isinstance(value, dict):
        items = cast("dict[str, object]", value)
        return {key: _redact_value(item) for key, item in items.items()}
    if isinstance(value, list | tuple):
        items_seq = cast("Sequence[object]", value)
        return [_redact_value(item) for item in items_seq]
    return value


def redact(settings: Settings) -> dict[str, Any]:
    """A JSON-safe view of the config with every secret replaced by a placeholder.

    Redaction works by **type**, not by field name: anything that is a :class:`SecretStr` becomes
    ``<set>`` and a ``None`` in a secret field becomes ``<unset>``. Since no secret in the model is
    ever a plain ``str`` - there is a unit test that enforces exactly that - nothing can leak here
    by being named something unexpected.
    """
    dumped = settings.model_dump(mode="python")
    redacted = cast("dict[str, Any]", _redact_value(dumped))
    for section, field in SECRET_FILE_NAMES.values():
        table = redacted.get(section)
        if isinstance(table, dict) and cast("dict[str, object]", table).get(field) is None:
            cast("dict[str, object]", table)[field] = _UNSET_PLACEHOLDER
    return redacted


def _banner_lines(prefix: str, value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in sorted(cast("dict[str, object]", value).items()):
            yield from _banner_lines(f"{prefix}.{key}" if prefix else str(key), item)
        return
    yield f"  {prefix} = {value!r}"


def startup_banner(settings: Settings) -> str:
    """The block logged once at startup, and what ``mcmanager check-config`` prints.

    Contains the fully resolved configuration - every default made explicit - because "what did it
    actually load" is the first question of every config bug, and answering it from three files and
    an env var is not an answer. Secrets appear only as ``<set>``/``<unset>``.
    """
    lines: list[str] = ["mcmanager configuration (resolved):"]
    source = settings.config_file
    lines.append(f"  <config_file> = {str(source) if source is not None else '<none: defaults>'}")
    lines.extend(_banner_lines("", redact(settings)))
    if settings.warnings:
        lines.append("warnings:")
        lines.extend(f"  ! {warning}" for warning in settings.warnings)
    return "\n".join(lines)
