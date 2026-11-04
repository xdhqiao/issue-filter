#!/usr/bin/env bash
# Optional host installation. Application dependencies are already in the image.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Only Linux x86_64 is supported by these host tools.' >&2; exit 1; }
[[ "$EUID" -eq 0 ]] || { echo 'Run with sudo/root.' >&2; exit 1; }
awk '$2 ~ /^(images|host-tools)\// {print}' SHA256SUMS | sha256sum --check -

if ! command -v docker >/dev/null; then
  command -v iptables >/dev/null || { echo 'Host prerequisite missing: iptables. Use your OS offline installation media first.' >&2; exit 1; }
  command -v systemctl >/dev/null || { echo 'This helper requires systemd. Install Docker manually on this host.' >&2; exit 1; }
  task_tmp_dir="$(mktemp -d)"
  trap 'rm -rf -- "$task_tmp_dir"' EXIT
  tar -xzf host-tools/docker-29.9.3.tgz -C "$task_tmp_dir"
  # Refuse to replace an independently installed containerd/runc/docker binary.
  for binary in "$task_tmp_dir"/docker/*; do
    if [[ -e "/usr/local/bin/$(basename "$binary")" ]]; then
      echo "Existing /usr/local/bin/$(basename "$binary"); refusing to overwrite it." >&2
      exit 1
    fi
  done
  install -m 0755 "$task_tmp_dir"/docker/* /usr/local/bin/
  if ! getent group docker >/dev/null; then groupadd --system docker; fi
  if [[ ! -e /etc/systemd/system/docker.service && ! -e /lib/systemd/system/docker.service && ! -e /usr/lib/systemd/system/docker.service ]]; then
    cat > /etc/systemd/system/docker.service <<'SERVICE'
[Unit]
Description=Docker Application Container Engine (offline static install)
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
ExecStart=/usr/local/bin/dockerd --host=unix:///var/run/docker.sock
Restart=on-failure
RestartSec=5
Delegate=yes
KillMode=process
LimitNOFILE=1048576
TasksMax=infinity
[Install]
WantedBy=multi-user.target
SERVICE
  fi
  systemctl daemon-reload
  systemctl enable --now docker.service
fi
if ! docker compose version >/dev/null 2>&1; then
  install -d /usr/local/lib/docker/cli-plugins
  install -m 0755 host-tools/docker-compose-linux-x86_64 /usr/local/lib/docker/cli-plugins/docker-compose
fi
docker version
docker compose version
echo 'Docker host tools are ready. Run: sudo bash load-and-start.sh'
