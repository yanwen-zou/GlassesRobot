#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[INFO] Starting headpose listener..."
python "$repo_root/src/egodata_record/egodata_record/headpos_listener.py" &
listener_pid=$!
sleep 1

cleanup() {
  if kill -0 "$listener_pid" 2>/dev/null; then
    echo "[INFO] Stopping headpose listener (pid=$listener_pid)..."
    kill "$listener_pid"
  fi
}
trap cleanup EXIT

echo "[INFO] Starting record.py..."
python "$repo_root/glasses_hardware/hardware/record.py" "$@"
