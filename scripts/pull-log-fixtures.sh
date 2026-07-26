#!/usr/bin/env bash
# Pull real log archives off the homelab into tests/fixtures/logs/.
#
# 164K in total for 36 files: trivially cheap to commit, and the highest-value test assets in the
# project. Golden files built from these plus an unmatched-line-ratio canary are what turn "Paper
# changed its log format" from silent data loss into a visible test failure.
#
# Note the second half: a `docker logs --timestamps` capture is taken as well, because only that
# contains the mc-server-runner and [init] wrapper lines. File-only fixtures would miss two of the
# three interleaved grammars entirely.
set -euo pipefail

HOST="${MCMANAGER_HOST:-minty@192.168.1.7}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/tests/fixtures/logs"

mkdir -p "$DEST"

scp "$HOST:~/homelab/data/minecraft/logs/*.log.gz" "$DEST/"
scp "$HOST:~/homelab/data/minecraft/logs/latest.log" "$DEST/latest.log" || true

# The only source of wrapper-grammar lines.
ssh "$HOST" 'docker logs --timestamps minecraft 2>&1' > "$DEST/docker-stream.timestamped.log"

echo "fixtures in $DEST:"
ls -la "$DEST"
