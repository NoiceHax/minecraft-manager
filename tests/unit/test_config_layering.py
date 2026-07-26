"""Config: layering, cross-field validation, indirection, and redaction.

The layering tests are the reason this file exists. "Which of my five config sources won" is not
something anyone should have to determine by experiment at 2am, so every boundary between the
layers - defaults, TOML, secrets files, env, CLI - is asserted here explicitly, including the
partial-override case where two sources contribute different keys of the same table.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr, ValidationError

from mcmanager.config import (
    DEFAULT_SECRETS_DIR,
    SCHEMA_VERSION,
    SECRET_FILE_NAMES,
    DiscordConfig,
    RconConfig,
    Settings,
    WebConfig,
    format_validation_errors,
    get_settings,
    load_settings,
    process_identity,
    redact,
    runtime_unreachable_message,
    startup_banner,
)
from mcmanager.core.types import ReadySignal
from mcmanager.errors import EXIT_CONFIG, ConfigError

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_TOML = REPO_ROOT / "config" / "mcmanager.example.toml"
DEV_TOML = REPO_ROOT / "config" / "mcmanager.dev.toml"

# Obvious non-secrets. S105 is suppressed rather than the strings obfuscated: a test that
# proves redaction works has to have something to redact.
TOKEN = "not-a-real-token-0123456789"  # noqa: S105
OTHER_TOKEN = "also-not-real-9876543210"  # noqa: S105


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No MCM_* variable from the developer's shell may reach a test.

    Without this, a single ``MCM_LOGS__LEVEL=DEBUG`` exported months ago makes half of this file
    fail on one machine and pass on every other one.
    """
    for name in list(os.environ):
        if name.startswith("MCM_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def write_toml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


# ------------------------------------------------------------------ the file is the spec


def test_example_toml_is_exactly_the_model_defaults() -> None:
    """The committed reference config must equal the defaults, key for key.

    This is the guard against the models and ``config/mcmanager.example.toml`` drifting apart. If
    it fails, one of the two changed without the other, and the TOML is the spec.
    """
    from_file = redact(load_settings(config_file=EXAMPLE_TOML))
    from_defaults = redact(load_settings())
    assert from_file == from_defaults


def test_example_toml_loads_and_has_no_secrets_in_it() -> None:
    settings = load_settings(config_file=EXAMPLE_TOML)
    assert settings.discord.token is None
    assert settings.web.token is None
    assert settings.rcon.password is None
    assert settings.config_file == EXAMPLE_TOML


def test_dev_toml_is_fully_offline() -> None:
    settings = load_settings(config_file=DEV_TOML)
    assert settings.runtime == "fake"
    assert settings.probe.enabled is False
    assert settings.discord.enabled is False
    assert settings.logs.format == "console"


# ---------------------------------------------------------------------------- precedence


def test_defaults_apply_with_no_file_at_all() -> None:
    settings = load_settings()
    assert settings.schema_version == SCHEMA_VERSION
    assert settings.runtime == "real"
    assert settings.secrets_dir == DEFAULT_SECRETS_DIR
    assert settings.server.lifecycle.stop_timeout_seconds == 90


def test_toml_beats_defaults(tmp_path: Path) -> None:
    path = write_toml(
        tmp_path / "c.toml",
        "[server.lifecycle]\nstop_timeout_seconds = 120\n",
    )
    settings = load_settings(config_file=path)
    assert settings.server.lifecycle.stop_timeout_seconds == 120
    # An unrelated key in the same table keeps its default: sources deep-merge, they do not
    # replace whole tables.
    assert settings.server.lifecycle.start_deadline_extra_seconds == 60


def test_secrets_dir_beats_toml(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "web_token").write_text(TOKEN, encoding="utf-8")
    path = write_toml(tmp_path / "c.toml", '[web]\ntoken = "from-the-toml"\n')

    settings = load_settings(config_file=path, secrets_dir=secrets)

    assert settings.web.token is not None
    assert settings.web.token.get_secret_value() == TOKEN


def test_env_beats_secrets_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "web_token").write_text(TOKEN, encoding="utf-8")
    monkeypatch.setenv("MCM_WEB__TOKEN", OTHER_TOKEN)

    settings = load_settings(secrets_dir=secrets)

    assert settings.web.token is not None
    assert settings.web.token.get_secret_value() == OTHER_TOKEN


def test_env_beats_toml_with_nested_delimiter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_toml(tmp_path / "c.toml", '[logs]\nlevel = "INFO"\nformat = "json"\n')
    monkeypatch.setenv("MCM_LOGS__LEVEL", "DEBUG")

    settings = load_settings(config_file=path)

    assert settings.logs.level == "DEBUG"
    assert settings.logs.format == "json"


def test_cli_overrides_beat_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_toml(tmp_path / "c.toml", '[logs]\nlevel = "INFO"\n')
    monkeypatch.setenv("MCM_LOGS__LEVEL", "WARNING")

    settings = load_settings(config_file=path, overrides={"logs": {"level": "ERROR"}})

    assert settings.logs.level == "ERROR"


def test_config_file_env_var_is_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_toml(tmp_path / "c.toml", 'runtime = "fake"\n')
    monkeypatch.setenv("MCM_CONFIG_FILE", str(path))

    assert load_settings().runtime == "fake"


def test_secrets_dir_comes_from_the_toml_when_not_given(tmp_path: Path) -> None:
    """The chicken-and-egg case: the directory of secrets is itself a config value."""
    secrets = tmp_path / "s"
    secrets.mkdir()
    (secrets / "rcon_password").write_text("hunter2", encoding="utf-8")
    path = write_toml(tmp_path / "c.toml", f'secrets_dir = "{secrets.as_posix()}"\n')

    settings = load_settings(config_file=path)

    assert settings.rcon.password is not None
    assert settings.rcon.password.get_secret_value() == "hunter2"


def test_secrets_dir_env_beats_the_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    losing = tmp_path / "losing"
    winning = tmp_path / "winning"
    losing.mkdir()
    winning.mkdir()
    (losing / "web_token").write_text("wrong", encoding="utf-8")
    (winning / "web_token").write_text(TOKEN, encoding="utf-8")
    path = write_toml(tmp_path / "c.toml", f'secrets_dir = "{losing.as_posix()}"\n')
    monkeypatch.setenv("MCM_SECRETS_DIR", str(winning))

    settings = load_settings(config_file=path)

    assert settings.web.token is not None
    assert settings.web.token.get_secret_value() == TOKEN


def test_missing_secrets_dir_is_not_an_error(tmp_path: Path) -> None:
    settings = load_settings(secrets_dir=tmp_path / "does-not-exist")
    assert settings.web.token is None


def test_empty_secret_file_is_ignored(tmp_path: Path) -> None:
    secrets = tmp_path / "s"
    secrets.mkdir()
    (secrets / "web_token").write_text("   \n", encoding="utf-8")
    assert load_settings(secrets_dir=secrets).web.token is None


def test_secret_file_trailing_newline_is_stripped(tmp_path: Path) -> None:
    """``echo token > file`` is how every one of these files gets written."""
    secrets = tmp_path / "s"
    secrets.mkdir()
    (secrets / "discord_token").write_text(f"{TOKEN}\n", encoding="utf-8")
    settings = load_settings(secrets_dir=secrets)
    assert settings.discord.token is not None
    assert settings.discord.token.get_secret_value() == TOKEN


def test_nested_secret_file_name_also_works(tmp_path: Path) -> None:
    """``discord__token`` is pydantic-settings' own convention; accept it too."""
    secrets = tmp_path / "s"
    secrets.mkdir()
    (secrets / "discord__token").write_text(TOKEN, encoding="utf-8")
    settings = load_settings(secrets_dir=secrets)
    assert settings.discord.token is not None
    assert settings.discord.token.get_secret_value() == TOKEN


# --------------------------------------------------------------------------- extra = forbid


def test_a_typo_in_a_nested_table_is_a_startup_error(tmp_path: Path) -> None:
    """The 2am bug this whole design point exists to prevent."""
    path = write_toml(tmp_path / "c.toml", "[server.lifecycle]\nstop_timeout = 120\n")

    with pytest.raises(ConfigError) as excinfo:
        load_settings(config_file=path)

    assert "stop_timeout" in str(excinfo.value)
    assert excinfo.value.exit_code == EXIT_CONFIG


def test_a_typo_at_the_top_level_is_a_startup_error(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "c.toml", 'runtim = "fake"\n')
    with pytest.raises(ConfigError, match="runtim"):
        load_settings(config_file=path)


def test_an_unknown_section_is_a_startup_error(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "c.toml", "[minecraft]\nport = 1\n")
    with pytest.raises(ConfigError, match="minecraft"):
        load_settings(config_file=path)


@pytest.mark.parametrize(
    "model",
    [Settings, DiscordConfig],
)
def test_models_are_frozen(model: type[Any]) -> None:
    assert model.model_config.get("frozen") is True
    assert model.model_config.get("extra") == "forbid"


def test_every_nested_section_forbids_extras() -> None:
    """Walk the whole model rather than trusting that every author remembered the base class."""
    seen: set[type[Any]] = set()

    def walk(model: type[Any]) -> None:
        if model in seen:
            return
        seen.add(model)
        assert model.model_config.get("extra") == "forbid", f"{model.__name__} allows extras"
        for field in model.model_fields.values():
            annotation = field.annotation
            if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
                walk(annotation)

    walk(Settings)
    assert len(seen) > 8


# ------------------------------------------------------------------------ no plain-str secrets


def test_no_secret_field_is_a_plain_str() -> None:
    """Redaction is by type, so a ``str`` named ``token`` would leak into every log line.

    This walks the model instead of listing the three known fields, so the guarantee holds for a
    secret somebody adds next year.
    """
    suspicious = ("token", "password", "secret", "apikey", "api_key", "credential")
    # A plain `str` is the failure mode. `secrets_dir: Path` matches the name filter and is not a
    # secret at all, so the type - not the name - decides.
    stringy = (str, str | None)
    offenders: list[str] = []
    seen: set[type[Any]] = set()

    def walk(model: type[Any], prefix: str) -> None:
        if model in seen:
            return
        seen.add(model)
        for name, field in model.model_fields.items():
            annotation = field.annotation
            looks_secret = any(word in name.lower() for word in suspicious)
            if looks_secret and any(annotation == candidate for candidate in stringy):
                offenders.append(f"{prefix}{name}: {annotation}")
            if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
                walk(annotation, f"{prefix}{name}.")

    walk(Settings, "")
    assert not offenders, f"secret-looking fields that are not SecretStr: {offenders}"

    # And the three that exist really are SecretStr, so the walk above is not vacuous.
    assert DiscordConfig.model_fields["token"].annotation == SecretStr | None
    assert WebConfig.model_fields["token"].annotation == SecretStr | None
    assert RconConfig.model_fields["password"].annotation == SecretStr | None


# ------------------------------------------------------------------- cross-field validators


def test_discord_enabled_requires_token_and_all_three_ids(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "c.toml", '[discord]\nenabled = true\nmode = "live"\n')

    with pytest.raises(ConfigError) as excinfo:
        load_settings(config_file=path)

    message = str(excinfo.value)
    assert "no token" in message
    for field in ("guild_id", "channel_id", "admin_role_id"):
        assert field in message


def test_discord_enabled_is_satisfied_by_all_four(tmp_path: Path) -> None:
    path = write_toml(
        tmp_path / "c.toml",
        "[discord]\n"
        "enabled = true\n"
        'mode = "dryrun"\n'
        "guild_id = 1\n"
        "channel_id = 2\n"
        "admin_role_id = 3\n"
        f'token = "{TOKEN}"\n',
    )
    settings = load_settings(config_file=path)
    assert settings.discord.enabled is True
    assert settings.discord.live is False


def test_fake_runtime_refuses_a_live_discord(tmp_path: Path) -> None:
    """Posting events from a scriptable in-memory runtime into a real channel is not a drill."""
    path = write_toml(
        tmp_path / "c.toml",
        'runtime = "fake"\n'
        "[discord]\n"
        "enabled = true\n"
        'mode = "live"\n'
        "guild_id = 1\n"
        "channel_id = 2\n"
        "admin_role_id = 3\n"
        f'token = "{TOKEN}"\n',
    )
    with pytest.raises(ConfigError, match="scriptable in-memory runtime"):
        load_settings(config_file=path)


def test_fake_runtime_with_a_dryrun_discord_is_fine(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "c.toml", 'runtime = "fake"\n[discord]\nmode = "dryrun"\n')
    assert load_settings(config_file=path).runtime == "fake"


def test_idle_poll_interval_must_be_at_most_a_third_of_the_timeout(tmp_path: Path) -> None:
    path = write_toml(
        tmp_path / "c.toml",
        "[idle]\ntimeout_minutes = 15\npoll_interval_seconds = 600\nwarn_minutes = 2\n",
    )
    with pytest.raises(ConfigError, match="a third"):
        load_settings(config_file=path)


def test_idle_poll_interval_exactly_a_third_is_allowed(tmp_path: Path) -> None:
    path = write_toml(
        tmp_path / "c.toml",
        "[idle]\ntimeout_minutes = 15\npoll_interval_seconds = 300\nwarn_minutes = 2\n",
    )
    assert load_settings(config_file=path).idle.poll_interval_seconds == 300


def test_idle_warning_must_land_inside_the_countdown(tmp_path: Path) -> None:
    path = write_toml(
        tmp_path / "c.toml",
        "[idle]\ntimeout_minutes = 15\npoll_interval_seconds = 60\nwarn_minutes = 20\n",
    )
    with pytest.raises(ConfigError, match="warn_minutes"):
        load_settings(config_file=path)


def test_archive_on_stop_requires_a_readable_server_log_dir(tmp_path: Path) -> None:
    """A *present but unreadable* mount is the real failure, and it is fatal."""
    logs_dir = tmp_path / "mc-logs"
    logs_dir.mkdir()
    path = write_toml(
        tmp_path / "c.toml",
        f'[logs]\narchive_on_stop = true\nserver_log_dir = "{logs_dir.as_posix()}"\n',
    )

    # Readable: fine.
    assert load_settings(config_file=path).logs.archive_on_stop is True

    # Not a directory at all: fatal, because the operator clearly meant something by it.
    not_a_dir = tmp_path / "file-not-dir"
    not_a_dir.write_text("", encoding="utf-8")
    bad = write_toml(
        tmp_path / "bad.toml",
        f'[logs]\narchive_on_stop = true\nserver_log_dir = "{not_a_dir.as_posix()}"\n',
    )
    with pytest.raises(ConfigError, match="not a directory"):
        load_settings(config_file=bad)


def test_a_missing_server_log_dir_is_a_warning_not_an_error(tmp_path: Path) -> None:
    """``mcmanager check-config config/mcmanager.example.toml`` has to pass on a laptop."""
    missing = tmp_path / "nope"
    path = write_toml(
        tmp_path / "c.toml",
        f'[logs]\narchive_on_stop = true\nserver_log_dir = "{missing.as_posix()}"\n',
    )
    settings = load_settings(config_file=path)
    assert any("does not exist" in warning for warning in settings.warnings)


def test_loopback_server_host_is_rejected(tmp_path: Path) -> None:
    """From inside a container, ``localhost`` is our own loopback where nothing listens."""
    path = write_toml(tmp_path / "c.toml", '[server]\nhost = "localhost"\n')
    with pytest.raises(ConfigError, match="loopback"):
        load_settings(config_file=path)


def test_network_rcon_requires_a_password(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "c.toml", '[rcon]\nmode = "network"\n')
    with pytest.raises(ConfigError, match=r"rcon\.password"):
        load_settings(config_file=path)


def test_exec_rcon_needs_no_password(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "c.toml", '[rcon]\nmode = "exec"\n')
    assert load_settings(config_file=path).rcon.password is None


def test_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "c.toml", "schema_version = 99\n")
    with pytest.raises(ConfigError, match="schema_version"):
        load_settings(config_file=path)


def test_ready_signals_are_parsed_and_deduplicated(tmp_path: Path) -> None:
    path = write_toml(
        tmp_path / "c.toml",
        '[server.lifecycle]\nready_signals = ["health"]\n',
    )
    assert load_settings(config_file=path).server.lifecycle.ready_signals == (
        ReadySignal.HEALTHCHECK,
    )

    dup = write_toml(
        tmp_path / "d.toml",
        '[server.lifecycle]\nready_signals = ["log", "log"]\n',
    )
    with pytest.raises(ConfigError, match="duplicates"):
        load_settings(config_file=dup)


def test_empty_ready_signals_is_refused(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "c.toml", "[server.lifecycle]\nready_signals = []\n")
    with pytest.raises(ConfigError, match="at least one"):
        load_settings(config_file=path)


# ---------------------------------------------------------------------------- indirection


def test_env_reference_is_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_EXTERNAL_TOKEN", TOKEN)
    path = write_toml(tmp_path / "c.toml", '[web]\ntoken = "env:SOME_EXTERNAL_TOKEN"\n')

    settings = load_settings(config_file=path)

    assert settings.web.token is not None
    assert settings.web.token.get_secret_value() == TOKEN


def test_missing_env_reference_fails_loudly(tmp_path: Path) -> None:
    """Never a silent empty string: that defers the failure to a 401 hours later."""
    path = write_toml(tmp_path / "c.toml", '[web]\ntoken = "env:DEFINITELY_NOT_SET_ANYWHERE"\n')
    with pytest.raises(ConfigError, match="DEFINITELY_NOT_SET_ANYWHERE"):
        load_settings(config_file=path)


def test_file_reference_is_resolved_and_stripped(tmp_path: Path) -> None:
    secret_file = tmp_path / "token.txt"
    secret_file.write_text(f"  {TOKEN}\n", encoding="utf-8")
    path = write_toml(
        tmp_path / "c.toml",
        f'[discord]\ntoken = "file:{secret_file.as_posix()}"\n',
    )

    settings = load_settings(config_file=path)

    assert settings.discord.token is not None
    assert settings.discord.token.get_secret_value() == TOKEN


def test_missing_file_reference_fails_loudly(tmp_path: Path) -> None:
    path = write_toml(
        tmp_path / "c.toml",
        f'[discord]\ntoken = "file:{(tmp_path / "nope").as_posix()}"\n',
    )
    with pytest.raises(ConfigError, match="does not exist"):
        load_settings(config_file=path)


def test_indirection_works_for_non_secret_values_too(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MC_CONTAINER_NAME", "minecraft-staging")
    path = write_toml(tmp_path / "c.toml", '[server]\ncontainer = "env:MC_CONTAINER_NAME"\n')
    assert load_settings(config_file=path).server.container == "minecraft-staging"


def test_indirection_applies_to_env_supplied_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_file = tmp_path / "t"
    secret_file.write_text(TOKEN, encoding="utf-8")
    monkeypatch.setenv("MCM_WEB__TOKEN", f"file:{secret_file.as_posix()}")

    settings = load_settings()

    assert settings.web.token is not None
    assert settings.web.token.get_secret_value() == TOKEN


# ------------------------------------------------------------------------ literal secrets


def test_a_literal_secret_in_the_toml_warns_but_loads(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "c.toml", f'[web]\ntoken = "{TOKEN}"\n')

    settings = load_settings(config_file=path)

    assert settings.web.token is not None
    assert any("written literally" in warning for warning in settings.warnings)
    assert all(TOKEN not in warning for warning in settings.warnings)


def test_a_referenced_secret_in_the_toml_does_not_warn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOME_EXTERNAL_TOKEN", TOKEN)
    path = write_toml(tmp_path / "c.toml", '[web]\ntoken = "env:SOME_EXTERNAL_TOKEN"\n')
    settings = load_settings(config_file=path)
    assert all("written literally" not in warning for warning in settings.warnings)


def test_a_secret_from_secrets_dir_does_not_warn(tmp_path: Path) -> None:
    secrets = tmp_path / "s"
    secrets.mkdir()
    (secrets / "web_token").write_text(TOKEN, encoding="utf-8")
    settings = load_settings(secrets_dir=secrets)
    assert all("written literally" not in warning for warning in settings.warnings)


# ---------------------------------------------------------------------------- redaction


def _all_secret_values(settings: Settings) -> list[str]:
    values: list[str] = []
    for section, field in SECRET_FILE_NAMES.values():
        secret = getattr(getattr(settings, section), field)
        if isinstance(secret, SecretStr):
            values.append(secret.get_secret_value())
    return values


def _fully_loaded(tmp_path: Path) -> Settings:
    secrets = tmp_path / "s"
    secrets.mkdir()
    (secrets / "discord_token").write_text(TOKEN, encoding="utf-8")
    (secrets / "web_token").write_text(OTHER_TOKEN, encoding="utf-8")
    (secrets / "rcon_password").write_text("hunter2-not-real", encoding="utf-8")
    path = write_toml(
        tmp_path / "c.toml",
        "[discord]\n"
        "enabled = true\n"
        'mode = "dryrun"\n'
        "guild_id = 111\n"
        "channel_id = 222\n"
        "admin_role_id = 333\n"
        "[rcon]\n"
        'mode = "network"\n',
    )
    return load_settings(config_file=path, secrets_dir=secrets)


def test_no_secret_value_appears_in_the_startup_banner(tmp_path: Path) -> None:
    """The assertion this whole redaction design exists to make true."""
    settings = _fully_loaded(tmp_path)
    secrets = _all_secret_values(settings)
    assert len(secrets) == 3

    banner = startup_banner(settings)

    for value in secrets:
        assert value not in banner
    assert banner.count("<set>") == 3


def test_no_secret_value_appears_in_redact_output(tmp_path: Path) -> None:
    settings = _fully_loaded(tmp_path)
    rendered = json.dumps(redact(settings))
    for value in _all_secret_values(settings):
        assert value not in rendered


def test_no_secret_value_appears_in_repr_or_model_dump(tmp_path: Path) -> None:
    """Belt and braces: even an accidental ``log.info("config", cfg=settings)`` is safe."""
    settings = _fully_loaded(tmp_path)
    blobs = [repr(settings), str(settings), json.dumps(settings.model_dump(mode="json"))]
    for value in _all_secret_values(settings):
        for blob in blobs:
            assert value not in blob


def test_unset_secrets_render_as_unset() -> None:
    rendered = redact(load_settings())
    assert rendered["discord"]["token"] == "<unset>"  # noqa: S105
    assert rendered["web"]["token"] == "<unset>"  # noqa: S105
    assert rendered["rcon"]["password"] == "<unset>"  # noqa: S105


def test_redact_output_is_json_serialisable() -> None:
    """The banner is also what ``mcmanager check-config --json`` will emit."""
    json.dumps(redact(load_settings(config_file=EXAMPLE_TOML)))


def test_banner_names_the_config_file_and_lists_warnings(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "c.toml", 'runtime = "fake"\n')
    banner = startup_banner(load_settings(config_file=path))
    assert str(path) in banner or path.as_posix() in banner
    assert 'runtime = "fake"' in banner or "'fake'" in banner
    assert "warnings:" in banner


# --------------------------------------------------------------------- failure reporting


def test_every_validation_error_is_reported_not_just_the_first(tmp_path: Path) -> None:
    """Three typos should take one restart to find, not three."""
    path = write_toml(
        tmp_path / "c.toml",
        "[server]\nport = 70000\n[events]\nqueue_max = 0\n[probe]\ninterval_seconds = -1\n",
    )

    with pytest.raises(ConfigError) as excinfo:
        load_settings(config_file=path)

    message = str(excinfo.value)
    assert "server.port" in message
    assert "events.queue_max" in message
    assert "probe.interval_seconds" in message


def test_format_validation_errors_is_one_line_per_problem() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Settings.model_validate({"web": {"port": -1}, "events": {"queue_max": 0}})

    report = format_validation_errors(excinfo.value)
    lines = report.splitlines()

    assert lines[0].startswith("2 configuration error(s)")
    assert any(line.strip().startswith("web.port:") for line in lines)
    assert any(line.strip().startswith("events.queue_max:") for line in lines)


def test_config_error_carries_exit_code_78(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "c.toml", "[web]\nport = -1\n")
    with pytest.raises(ConfigError) as excinfo:
        load_settings(config_file=path)
    assert excinfo.value.exit_code == EXIT_CONFIG == 78


def test_unreadable_config_file_is_a_config_error(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "c.toml", "this is not = = toml\n")
    with pytest.raises(ConfigError, match="could not read config file"):
        load_settings(config_file=path)


def test_runtime_unreachable_message_names_the_socket_and_the_identity() -> None:
    """The gid-983 problem, 95% of the time. Printing ``id`` saves twenty minutes."""
    message = runtime_unreachable_message(
        endpoint="unix:///var/run/docker.sock",
        error="Permission denied",
    )
    assert "unix:///var/run/docker.sock" in message
    assert "Permission denied" in message
    assert process_identity() in message
    assert "group_add" in message


def test_process_identity_does_not_explode_on_windows() -> None:
    identity = process_identity()
    assert identity
    assert "uid=" in identity or "not POSIX" in identity


def test_discord_auth_message_says_retrying_will_not_help() -> None:
    from mcmanager.config import EXIT_NOPERM, discord_auth_message

    assert EXIT_NOPERM == 77
    message = discord_auth_message(error="401 Unauthorized", guild_id=123)
    assert "401 Unauthorized" in message
    assert "123" in message
    assert "Retrying will not fix it" in message


# ------------------------------------------------------------------------------- caching


def test_get_settings_is_cached_and_clearable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_file = write_toml(tmp_path / "a.toml", 'runtime = "fake"\n')
    monkeypatch.setenv("MCM_CONFIG_FILE", str(first_file))

    first = get_settings()
    assert first is get_settings()
    assert first.runtime == "fake"

    second_file = write_toml(tmp_path / "b.toml", 'runtime = "real"\n')
    monkeypatch.setenv("MCM_CONFIG_FILE", str(second_file))
    assert get_settings() is first, "still cached until cache_clear()"

    get_settings.cache_clear()
    assert get_settings().runtime == "real"


def test_settings_are_immutable() -> None:
    settings = load_settings()
    with pytest.raises(ValidationError):
        settings.runtime = "fake"


def test_settings_warnings_are_deduplicated() -> None:
    settings = load_settings()
    before = len(settings.warnings)
    settings.add_warning(settings.warnings[0] if settings.warnings else "x")
    settings.add_warning("x")
    assert len(settings.warnings) <= before + 1
