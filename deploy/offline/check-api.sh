#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if docker compose version >/dev/null 2>&1; then
  compose=(docker compose)
else
  compose=("$PWD/host-tools/docker-compose-linux-x86_64")
fi
"${compose[@]}" --env-file .env.offline -f compose.yaml -p issue-filter-offline exec -T issue-filter python - < check_api.py
