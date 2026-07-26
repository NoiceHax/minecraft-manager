# Minecraft Manager

One event-driven daemon that owns a Dockerised game server's lifecycle: log streaming, player
tracking, Discord integration, idle auto-shutdown and session history. It replaces two ad-hoc
homelab scripts (`minecraft-discord-bridge.py` and `minecraft-idle-stop.sh`) with something
testable, restart-safe and observable.

## Design rules

These are enforced mechanically, not by convention:

- **Python 3.12+, async-first, full type hints.** `ruff check` and `pyright --strict` are clean.
- **Official Python Docker SDK only.** No `subprocess`, no `docker` CLI, no shell-outs, no
  `requests`. `subprocess` and `requests` are in ruff's `banned-api` list.
- **Nothing calls the clock directly.** `asyncio.sleep`, `time.sleep`, `time.monotonic` and
  `datetime.now` are banned outside `src/mcmanager/clock.py`; every component takes an injected
  `Clock`. Tests use `ManualClock` and run a 15-minute idle timeout in about a millisecond.
- **All datetimes are tz-aware UTC.**
- **`import docker` is confined** to `containers/docker_runtime.py` and `containers/streams.py`.
  Everything else sees only `ContainerRuntime` and the DTOs.
- **No `print()`** outside `cli/render.py`.
- **Events are frozen slotted dataclasses**, never pydantic. Pydantic lives at the two real
  boundaries: config load and JSON serialisation.
- **Business logic imports neither Discord nor docker-py.**

## Layout

```
src/mcmanager/
  clock.py        Clock protocol, SystemClock, ManualClock
  core/           events, bus, supervisor, serde, shared types
  containers/     ContainerRuntime ABC, DTOs, Docker + Fake implementations
  games/          GameAdapter seam; games/minecraft/ holds every Minecraft-specific line
  services/       lifecycle FSM, log pipeline, poller, players, controller, idle, session
  persistence/    state store, session log
  control/        aiohttp surface: /healthz /readyz /status /players /events /logs /control/*
  cli/            mcmanager run|inspect|status|events|logs|replay|sessions|start|stop|restart
  discordbot/     a client of ServerController, containing no business logic
```

## Development

```
uv sync --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -m "not live"
```

Fully offline daemon (fake container runtime, Discord disabled):

```
uv run mcmanager run --config config/mcmanager.dev.toml
```

Standalone commands need no daemon and work against the homelab directly:

```
$env:DOCKER_HOST = "ssh://minty@192.168.1.7"
uv run mcmanager inspect
uv run mcmanager replay tests/fixtures/logs/<file>.log.gz
```

`@pytest.mark.live` tests touch the real homelab and are opt-in via `MCMANAGER_LIVE=1`. An autouse
fixture blocks outbound sockets for every other test.

## Security note, stated plainly

The daemon needs `/var/run/docker.sock`. On the homelab that socket is `root:983` and the container
joins gid 983 via `group_add`. **Membership in that group is root-equivalent on the host** - anyone
who reaches the socket can `docker run --privileged -v /:/host`. Running as uid 1000 is
blast-radius reduction, not an isolation boundary. A `docker-socket-proxy` sidecar is deferred with
reason: the RCON path needs `EXEC=1`, and exec is itself root-equivalent.

No secret is ever a plain `str` in the config model. Tokens and passwords are `SecretStr` fed from
`secrets_dir`, so `repr()` and `model_dump()` redact them.
