#!/usr/bin/env bash
# Push this working tree to the homelab and redeploy.
#
# Phase 10. Kept as a shell script deliberately: it is a developer convenience that runs on a
# workstation, not part of the daemon. The "no subprocess" rule is about mcmanager's own code
# talking to Docker through the SDK rather than shelling out.
set -euo pipefail

HOST="${MCMANAGER_HOST:-minty@192.168.1.7}"
DEST="${MCMANAGER_DEST:-~/homelab/apps/mcmanager}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -az --delete \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude '.ruff_cache' \
    --exclude 'var' \
    --exclude '.env' \
    "$ROOT/" "$HOST:$DEST/"

ssh "$HOST" '~/homelab/scripts/deploy.sh mcmanager'
ssh "$HOST" 'docker exec mcmanager mcmanager check-config'
