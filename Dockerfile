# syntax=docker/dockerfile:1
#
# Two uv sync layers so the dependency layer caches independently of the source: a 90-second
# rebuild on the homelab becomes about 6 seconds when only src/ changed.
#
# Rejected: alpine (musl means source builds for aiohttp and multidict), and distroless (no shell,
# and you *will* end up debugging socket permissions inside this container).

FROM ghcr.io/astral-sh/uv:0.11.29 AS uv

FROM python:3.12-slim-bookworm AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Layer 1: dependencies only. Invalidated by the lockfile, not by source edits.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Layer 2: the project itself.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm AS runtime

# uid/gid 1000 mirrors `minty` on the homelab, so bind-mounted files are readable without chmod
# games. Membership in the docker group is added at runtime via compose `group_add`, because the
# gid is host-specific (983 here) and baking it into a portable image is host coupling.
# The three subdirectories are created here, not just their parent: compose mounts a named volume
# at /var/lib/mcmanager, a named volume seeds itself from the image's contents at that path, and a
# StateStore pointed at a path that is not a directory treats it as a file - so without this,
# /var/lib/mcmanager/state becomes a JSON file where a directory was meant.
RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin app \
 && mkdir -p /var/lib/mcmanager/state /var/lib/mcmanager/archives /var/lib/mcmanager/logs \
 && chown -R app:app /var/lib/mcmanager

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MCM_CONFIG_FILE=/app/config/mcmanager.toml

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src ./src
COPY --chown=app:app config/mcmanager.example.toml ./config/mcmanager.example.toml

USER app

# mc-server-runner traps SIGTERM; so do we. Compose gives us 30s because shutdown does a Discord
# round trip, though it targets under 5.
STOPSIGNAL SIGTERM

EXPOSE 8787

# urllib rather than adding curl to the image for one healthcheck.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8787/healthz', timeout=4).status == 200 else 1)"]

ENTRYPOINT ["mcmanager"]
CMD ["run"]
