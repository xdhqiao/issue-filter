#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

command -v docker >/dev/null || { echo 'Docker is required. See README-OFFLINE.md or run sudo bash install-docker-linux.sh.' >&2; exit 1; }
awk '$2 ~ /^(images|host-tools)\// {print}' SHA256SUMS | sha256sum --check -
docker load --input images/issue-filter-offline-20261104-cpu.tar
if [[ "${1:-}" == "--load-only" ]]; then exit 0; fi
if docker compose version >/dev/null 2>&1; then
  compose=(docker compose)
else
  chmod u+x host-tools/docker-compose-linux-x86_64
  compose=("$PWD/host-tools/docker-compose-linux-x86_64")
fi
"${compose[@]}" --env-file .env.offline -f compose.yaml -p issue-filter-offline up --pull never -d
echo 'Started. Check readiness with: bash check-api.sh'
