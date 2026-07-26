"""Argument parsing, dispatch, exit codes, and the import-isolation guarantee.

The load-bearing test in this file is :class:`TestReplayIsStandalone`. ``mcmanager replay`` is the
proof of the parser's purity contract, so its module-level import closure is asserted to contain no
``docker``, no ``aiohttp``, no ``discord`` and no ``pydantic``. That check is static - an AST walk
over the transitive ``mcmanager`` imports - rather than a ``sys.modules`` inspection, because by
the time this test runs the rest of the suite has imported half the world into the interpreter.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mcmanager.cli.main import build_parser, main
from mcmanager.errors import EXIT_CONFIG, EXIT_OK, EXIT_UNAVAILABLE

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mcmanager.config import Settings

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "logs"

_MISSING_REF_TOML = '[web]\ntoken = "file:/definitely/not/here"\n'
_REDACTED_UNSET = "<unset>"
"""What ``redact()`` prints for a secret that was never set.

Compared against by name rather than inline, so ruff's hardcoded-password rule does not have to
make a judgement about a string literal sitting next to the word "token".
"""


class TestParser:
    def test_no_command_prints_help_and_exits_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([]) == EXIT_OK
        assert "usage: mcmanager" in capsys.readouterr().out

    def test_an_unknown_command_is_an_argparse_error(self) -> None:
        with pytest.raises(SystemExit) as caught:
            main(["frobnicate"])
        assert caught.value.code == 2

    @pytest.mark.parametrize(
        "argv",
        [
            ["--json", "status"],
            ["status", "--json"],
        ],
    )
    def test_a_global_flag_is_accepted_on_either_side_of_the_subcommand(
        self, argv: list[str]
    ) -> None:
        """The argparse.SUPPRESS trick, tested.

        Without it a subparser's own default would overwrite the root parser's value, and
        ``mcmanager --json status`` would silently print a table.
        """
        args = build_parser().parse_args(argv)
        assert args.json is True

    def test_the_defaults_survive_when_a_flag_is_given_nowhere(self) -> None:
        args = build_parser().parse_args(["status"])
        assert args.json is False
        assert args.no_color is False
        assert args.url is None
        assert args.config is None

    def test_every_documented_command_exists(self) -> None:
        parser = build_parser()
        for command in (
            "run",
            "check-config",
            "inspect",
            "status",
            "players",
            "events",
            "logs",
            "replay",
            "sessions",
            "start",
            "stop",
            "restart",
        ):
            assert parser.parse_args([command, *(["x.log"] if command == "replay" else [])])

    def test_type_filters_accumulate(self) -> None:
        args = build_parser().parse_args(
            ["events", "--type", "PlayerEvent", "--type", "ChatMessage"]
        )
        assert args.type == ["PlayerEvent", "ChatMessage"]


class TestCheckConfig:
    def test_a_valid_config_prints_the_resolved_banner_and_exits_zero(
        self, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["check-config", "--config", str(config_file)]) == EXIT_OK
        out = capsys.readouterr().out
        assert "mcmanager configuration (resolved):" in out
        assert "server.container = 'minecraft'" in out

    def test_json_output_is_the_redacted_config(
        self, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        assert main(["check-config", "--config", str(config_file), "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["runtime"] == "fake"
        assert payload["web"]["token"] == _REDACTED_UNSET

    def test_a_broken_config_exits_78_listing_every_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Three typos, one run. Printing only the first means three restarts to fix three keys.

        Note the errors are all *field-level*: pydantic runs field validation before the
        cross-field model validators, so a config that fails both reports the field errors and
        stops. That is still one error per typo, which is what matters here.
        """
        broken = tmp_path / "broken.toml"
        broken.write_text(
            "\n".join(
                [
                    "[server]",
                    "port = 70000",
                    'id = ""',
                    "[probe]",
                    "interval_seconds = -1",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        assert main(["check-config", "--config", str(broken)]) == EXIT_CONFIG
        err = capsys.readouterr().err
        assert "3 configuration error(s)" in err
        assert "server.port" in err
        assert "server.id" in err
        assert "probe.interval_seconds" in err

    def test_an_unknown_key_is_refused_rather_than_ignored(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # `extra = "forbid"`: a typo'd stop_timeout silently doing nothing is the 2am bug.
        broken = tmp_path / "typo.toml"
        broken.write_text("[server.lifecycle]\nstop_timeout = 90\n", encoding="utf-8")
        assert main(["check-config", "--config", str(broken)]) == EXIT_CONFIG
        assert "stop_timeout" in capsys.readouterr().err

    def test_a_missing_secret_reference_is_a_config_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        broken = tmp_path / "missing-ref.toml"
        broken.write_text(_MISSING_REF_TOML, encoding="utf-8")
        assert main(["check-config", "--config", str(broken)]) == EXIT_CONFIG
        assert "does not exist" in capsys.readouterr().err


class TestDaemonRequiredCommands:
    """The commands that refuse to degrade, and the exit code that says so."""

    def test_events_exits_69_naming_the_url_it_tried(
        self, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["events", "--config", str(config_file), "--url", "http://127.0.0.1:1"])
        assert code == EXIT_UNAVAILABLE
        err = capsys.readouterr().err
        assert "http://127.0.0.1:1" in err
        assert "docker exec mcmanager" in err

    def test_stop_without_a_token_refuses_before_connecting(
        self, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["stop", "--config", str(config_file), "--url", "http://127.0.0.1:1"])
        assert code == EXIT_UNAVAILABLE
        assert "web.token" in capsys.readouterr().err

    def test_start_with_a_token_but_no_daemon_exits_69(
        self, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "start",
                "--config",
                str(config_file),
                "--url",
                "http://127.0.0.1:1",
                "--token",
                "anything",
            ]
        )
        assert code == EXIT_UNAVAILABLE
        assert "could not reach the mcmanager daemon" in capsys.readouterr().err

    def test_logs_with_follow_needs_the_daemon(
        self, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            ["logs", "--follow", "--config", str(config_file), "--url", "http://127.0.0.1:1"]
        )
        assert code == EXIT_UNAVAILABLE
        assert "could not reach" in capsys.readouterr().err

    def test_status_falls_back_instead_of_failing(
        self, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The asymmetry the plan is explicit about: reads degrade loudly, events do not degrade.
        code = main(["status", "--config", str(config_file), "--url", "http://127.0.0.1:1"])
        assert code == EXIT_OK
        captured = capsys.readouterr()
        assert "daemon: offline" in captured.out
        assert "falling back to a standalone view" in captured.err


class TestRunDaemon:
    """The CLI-to-composition-root seam.

    ``mcmanager run`` is the one command with no daemon to talk to and no output of its own: all
    it does is hand the resolved :class:`Settings` to :func:`mcmanager.app.run` and return its
    exit code. Both halves of that contract are asserted here, because nothing else in the suite
    covers the handoff and an argument-order slip would only show up in production.
    """

    def test_run_hands_the_resolved_settings_to_app_run(
        self, config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mcmanager import app

        seen: list[Settings] = []

        def fake_run(settings: Settings, *, clock: object = None) -> int:
            seen.append(settings)
            return EXIT_OK

        monkeypatch.setattr(app, "run", fake_run)
        assert main(["run", "--config", str(config_file)]) == EXIT_OK
        assert len(seen) == 1
        # Not a fresh default-constructed Settings: the one parsed from --config.
        assert seen[0].server.container

    def test_the_daemons_exit_code_is_the_processs_exit_code(
        self, config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-zero return from the daemon must not be flattened to success by the CLI."""
        from mcmanager import app

        def fake_run(settings: Settings, *, clock: object = None) -> int:
            return EXIT_UNAVAILABLE

        monkeypatch.setattr(app, "run", fake_run)
        assert main(["run", "--config", str(config_file)]) == EXIT_UNAVAILABLE


class TestReplay:
    def test_replay_prints_events_and_a_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["replay", str(FIXTURES / "run_normal_session.log")])
        assert code == EXIT_OK
        out = capsys.readouterr().out
        assert "PlayerJoined" in out
        assert "replay summary" in out
        assert "ratio 0.0000" in out

    def test_replay_needs_no_config_at_all(self, capsys: pytest.CaptureFixture[str]) -> None:
        # No --config, no MCM_CONFIG_FILE, no state dir. If this ever needs settings, the purity
        # contract has been broken.
        assert main(["replay", str(FIXTURES / "run_normal_session.log"), "--quiet"]) == EXIT_OK
        assert "replay summary" in capsys.readouterr().out

    def test_replay_reads_a_gzip_archive(self, capsys: pytest.CaptureFixture[str]) -> None:
        import gzip

        archives = sorted((FIXTURES / "archives").glob("*.log"))
        assert archives, "the real log fixtures should be committed"
        packed = Path(str(archives[0]) + ".gz")
        packed.write_bytes(gzip.compress(archives[0].read_bytes()))
        try:
            assert main(["replay", str(packed), "--quiet"]) == EXIT_OK
        finally:
            packed.unlink()
        assert "replay summary" in capsys.readouterr().out

    def test_json_output_is_one_serde_object_per_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        from mcmanager.core.serde import event_from_dict

        assert main(["replay", str(FIXTURES / "run_normal_session.log"), "--json"]) == EXIT_OK
        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line.strip()]
        assert len(lines) > 50
        for line in lines:
            event_from_dict(json.loads(line))
        # The summary goes to stderr on the JSON paths so a redirect stays a clean fixture.
        assert json.loads(captured.err)["unrecognised_ratio"] == 0.0

    def test_json_array_output_is_one_document(self, capsys: pytest.CaptureFixture[str]) -> None:
        import json

        code = main(["replay", str(FIXTURES / "run_normal_session.log"), "--json-array"])
        assert code == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list)
        assert payload[0]["type"]

    def test_strict_mode_fails_when_the_ratio_is_over_the_threshold(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The canary, wired to an exit code so CI can hold the line.

        A player is 'known' because chat proved the name real; the following line then names that
        player and matches nothing, which is exactly what a Paper format change looks like.
        """
        drifted = tmp_path / "drifted.log"
        drifted.write_text(
            "\n".join(
                [
                    "[13:24:37] [Async Chat Thread - #0/INFO]: <Steve> hello",
                    "[13:24:38] [Server thread/INFO]: Steve has teleported to the shadow realm",
                ]
            ),
            encoding="utf-8",
        )
        code = main(["replay", str(drifted), "--quiet", "--strict", "--threshold", "0.0"])
        assert code != EXIT_OK
        assert "log format has probably drifted" in capsys.readouterr().err

    def test_a_missing_file_is_reported_rather_than_traced(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["replay", str(tmp_path / "nope.log")])
        assert code == EXIT_CONFIG
        assert "could not read" in capsys.readouterr().err


# ------------------------------------------------------------------- the isolation guarantee


FORBIDDEN = ("docker", "aiohttp", "discord", "pydantic", "pydantic_settings", "mcstatus")
"""What ``mcmanager replay`` must not pull in at import time.

``pydantic`` is on the list alongside the obvious three: replay takes no config, and an import of
``mcmanager.config`` would mean it had grown one.
"""


def _module_path(name: str) -> Path | None:
    """Where ``name`` lives, or ``None`` if it is not an importable Python module.

    ``from x import y`` is ambiguous - ``y`` may be a submodule or just a name - so the walker
    tries both and lets this return ``None`` for the names.
    """
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin)


def _module_level_imports(path: Path) -> set[str]:
    """Top-level import targets, skipping function bodies and ``if TYPE_CHECKING`` blocks.

    Function-scoped imports are deliberately not counted: importing ``mcmanager.config`` inside a
    branch that only ``check-config`` reaches is the whole mechanism being tested.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()

    def visit(body: Iterable[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    found.add(node.module)
                    # `from mcmanager.cli import render` imports a *module*, not a name, and the
                    # walker has to follow it or the closure stops at the package.
                    found.update(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.If):
                if _is_type_checking(node.test):
                    continue
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Try):
                visit(node.body)
                visit(node.orelse)
                visit(node.finalbody)
                for handler in node.handlers:
                    visit(handler.body)

    visit(tree.body)
    return found


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _closure(root: str) -> set[str]:
    """Every module ``root`` imports at module scope, transitively through ``mcmanager``."""
    seen: set[str] = set()
    pending = [root]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = _module_path(name)
        if path is None:
            continue
        for imported in _module_level_imports(path):
            if imported not in seen:
                seen.add(imported)
                if imported.startswith("mcmanager"):
                    pending.append(imported)
    return seen


class TestReplayIsStandalone:
    def test_the_replay_command_imports_nothing_heavy(self) -> None:
        closure = _closure("mcmanager.cli.cmd_replay")
        offenders = {name for name in closure for bad in FORBIDDEN if name.split(".")[0] == bad}
        assert offenders == set(), (
            f"mcmanager replay must run with only the parser: {sorted(offenders)} reached it. "
            "The parser's purity contract, or cmd_replay's import discipline, has slipped."
        )

    def test_the_replay_command_does_reach_the_renderer(self) -> None:
        # A sanity check on the walker itself: if it found nothing at all the test above is vacuous.
        assert "mcmanager.cli.render" in _closure("mcmanager.cli.cmd_replay")

    def test_the_walker_would_catch_a_forbidden_import(self) -> None:
        # The negative control: cmd_control legitimately reaches aiohttp, and the walker sees it.
        closure = _closure("mcmanager.cli.client")
        assert "aiohttp" in closure

    def test_the_entry_point_itself_stays_light(self) -> None:
        """``main.py`` may import pydantic (via config) but must not import docker or discord.

        Every command module is imported inside its own dispatch branch, so a broken optional
        dependency in one command cannot stop the others running.
        """
        closure = _closure("mcmanager.cli.main")
        assert not {name for name in closure if name.split(".")[0] in ("docker", "discord")}
        assert "mcmanager.cli.cmd_inspect" not in closure
        assert "mcmanager.cli.cmd_status" not in closure
