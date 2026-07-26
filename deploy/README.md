# Deploy runbook

## 0. Prerequisite, once: put `minecraft` on the `homelab` network

Do this first. It unblocks everything else and costs nothing while the server is stopped.

The daemon needs to reach the server on 25565 (SLP) and, optionally, 25575 (RCON). Two obvious
options do not work:

- **`host.docker.internal` via `host-gateway`** - container-to-host-IP traffic hits the host
  INPUT chain, and ufw's default input policy on this box is DROP. LAN clients work only because
  Docker's published-port FORWARD rules bypass ufw entirely, and that bypass does not apply here.
  Also leaves RCON unreachable.
- **Joining `minecraft_default` as `external: true`** - works until `docker compose down` on the
  minecraft project removes that network, after which mcmanager (`restart: unless-stopped`) cannot
  start. That is a bootstrap deadlock: the server is down, so the manager that would start it will
  not boot.

So: add the existing external `homelab` network to the minecraft compose. It already holds eight
containers, is owned by no single project, and is therefore never garbage-collected.
Container-to-container traffic on a bridge never touches ufw.

In `~/homelab/compose/minecraft/docker-compose.yml`:

```yaml
services:
  minecraft:
    # ... unchanged ...
    networks: [homelab]

networks:
  homelab:
    external: true
```

Then `docker compose up -d`. Rollback is deleting those four lines. Verify:

```
docker run --rm --network homelab busybox nslookup minecraft
```

## 1. Secrets

On the homelab, in `~/homelab/secrets/`:

```
printf '%s' '<bot token>'      > discord_token
printf '%s' "$(openssl rand -hex 32)" > web_token
chmod 600 discord_token web_token
```

These are bind-mounted as files at `/run/secrets/`. No secret is ever a plain string in the config
model, and a literal token in the TOML is accepted but warned about loudly at startup.

The daemon's config file lives next to the compose file and is mounted read-only:

```
cp deploy/../config/mcmanager.example.toml ~/homelab/compose/mcmanager/mcmanager.toml
# then edit: discord ids, idle settings, whatever this host needs
```

## 1a. The docker group id

The socket is `root:983` and the container runs as `1000:1000`, so mcmanager needs a supplementary
group. Discover the gid on the host rather than assuming it:

```
getent group docker
# docker:x:983:minty
```

Put it in this deploy directory's `.env`:

```
printf 'DOCKER_GID=%s\n' "$(getent group docker | cut -d: -f3)" > ~/homelab/compose/mcmanager/.env
```

`group_add: ["${DOCKER_GID:-983}"]` in the compose file adds that group at runtime. The gid does
**not** need to exist in the image's `/etc/group` - the kernel compares numbers, not names - which
is why the image stays portable and a host where docker is gid 999 is a config change rather than a
rebuild.

Rejected alternatives, so nobody re-litigates them: baking gid 983 into the image (host-specific
coupling inside a portable artifact), and running as root (buys nothing, since the socket group
already grants everything root has).

### Read this part: gid 983 is root-equivalent on the host

**Anyone who can reach `/var/run/docker.sock` can run `docker run --privileged -v /:/host` and own
the machine.** That is what membership in the docker group means, and it is true of mcmanager's
container exactly as it is true of the `minty` account.

Running as uid 1000 instead of root is **blast-radius reduction, not an isolation boundary**: it
means a parser bug cannot scribble on `/app` or on the image's own files. It does not mean the
container is contained. Do not read the `user:` line in the compose file as security that it is not
providing.

A `docker-socket-proxy` sidecar was considered and **deferred, with a reason**: the daemon uses
`docker exec minecraft rcon-cli` as its command channel, so the proxy would have to allow `EXEC=1` -
and exec alone is root-equivalent, because you can exec into a privileged container. The proxy would
therefore buy very little until the exec path is dropped in favour of network RCON. Revisit it if
and when `rcon.mode = "network"` becomes the default.

## 2. Deploy

```
rsync -a --delete ./ minty@192.168.1.7:~/homelab/apps/mcmanager/
ssh minty@192.168.1.7 '~/homelab/scripts/deploy.sh mcmanager'
```

## 3. Gate before anything else

```
ssh minty@192.168.1.7 'docker exec mcmanager mcmanager check-config'
```

Exit 0 or exit 78 with every validation error listed. This is also a CI job.

## 4. Prove the socket permissions

```
ssh minty@192.168.1.7 'docker exec mcmanager id'
# expect: uid=1000 gid=1000 groups=1000,983
ssh minty@192.168.1.7 'docker exec mcmanager python -c "import docker; print(docker.from_env().ping())"'
# expect: True
```

If `ping()` is False, the daemon exits 69 with a message naming the socket and this process's
uid/gid/groups - that error is the gid-983 problem 95% of the time. `DOCKER_GID` is parameterised
in this directory's `.env`; a host where docker is gid 999 is a config change, not a rebuild.

## 5. Operate

```
docker exec mcmanager mcmanager status
docker exec mcmanager mcmanager events --follow --type PlayerEvent
docker logs -f mcmanager | jq
```

8787 is `expose`d, never published. The access path is `docker exec`, so there is no LAN auth
surface and ufw is not involved at all.

## 6. Stopping the manager

`docker stop mcmanager` sends SIGTERM and the daemon runs this sequence, in this order:

```
Discord -> control surface -> idle -> status poller -> DockerManager
        -> session checkpoint -> state flush -> bus drain
        -> supervised tasks cancelled (reverse spawn order) -> runtime close
```

Three things about it are worth knowing on a bad night:

- **It never stops the Minecraft server.** The idle manager's teardown cancels its countdown and
  publishes `IdleCancelled(reason="daemon_shutdown")`; issuing a stop there would mean restarting
  the *manager* kills the *server*. There is a named test for this.
- **The bus drains before tasks are cancelled**, so the last events - including that
  `IdleCancelled` - are actually delivered rather than discarded with the dispatch loop.
- **The session record stays open.** A session spans the server's life, not the daemon's, so the
  next boot resumes it if `(container_id, started_at)` still matches and closes it as
  `daemon_missed_shutdown` if it does not.

The whole sequence is budgeted at 4.5 seconds and each step gets a slice; `stop_grace_period` is
30s so a step that overruns still exits cleanly rather than being SIGKILLed. A **second** SIGTERM
or SIGINT during shutdown is an immediate hard exit - use it only when the orderly path is visibly
stuck, because it skips the state flush.

## Known gaps at this milestone

- **Log backfill after a manager restart is not wired yet.** The resume marker (`last_log_ts`) is
  persisted and read, but `DockerManager` has no way to be seeded with a starting `since`, so a
  restarted daemon streams from now and the lines in between are only in `docker logs`. Boot logs
  `app.resume_marker` with `backfilled=false` so this is visible rather than assumed.
- **The idle stop itself is not implemented** (M4). The countdown arms, warns, cancels and
  publishes `IdleStopTriggered`, and stops nothing. `idle.stats.stops_issued` is structurally
  incapable of being non-zero today.
- **Session summaries and the `latest.log` gap-filling archive are not implemented** (M5). The
  server's own log4j2 archiving is untouched and unaffected; nothing writes to `/mnt/mc-logs`.
- **Discord connects to nothing** (M6). `discord.enabled = false` is the default.

## Cutover

Phased, and every phase rolls back to "stop the new daemon", because nothing is running today.

1. **Shadow, 48h** - `discord.mode = "dryrun"`, `idle.enabled = false`, sessions on. Gate: every
   join/leave/chat/death in `latest.log` appears exactly once; zero unhandled exceptions; clean
   player names; memory flat in beszel. Verified with `mcmanager events --follow` while someone
   plays, then `mcmanager replay` over that session's archive - the two streams must agree.
2. **Discord read-only** - a **new** channel so old and new output never interleave; guild-scoped
   commands; only `/status` `/players` `/logs`. Gate: `@everyone` in chat does not ping; the admin
   gate rejects a non-admin; the rate limit holds.
3. **Mutating commands** - `/start` `/stop` `/restart`. Gate: `/stop` uses the 90s timeout and
   `latest.log` shows the world saved cleanly.
4. **Idle dry-run, 3-5 days** - logs "would stop", stops nothing. Gate: every "would stop" matches
   a genuinely empty server; `min_uptime_minutes` prevents a stop right after boot; and critically,
   **mcstatus failure is treated as UNKNOWN, not zero players** - verify by stopping the container
   mid-poll. That failure mode is the one that loses somebody's build session.
5. **Idle live**, watch for a week.
6. **Decommission** - move both old scripts to `scripts/deprecated/`, **delete both Discord
   webhooks in Discord's UI** (removing the files is not revocation: anyone holding the URL can
   still post), rotate `RCON_PASSWORD` (free of mcmanager impact, thanks to the exec-based RCON
   path), and give `~/homelab` a `.gitignore` and a first commit. Delete `deprecated/` after two
   clean weeks.

**Rollback caveat, flagged because it is a real trap:** the path back to the old scripts is
unproven. The bridge needs `requests` and the host has no pip3/uv/pipx; the idle script needs `jq`
and `curl`. Verify `python3 -c 'import requests'` and `which jq curl` **before** treating rollback
as available.
