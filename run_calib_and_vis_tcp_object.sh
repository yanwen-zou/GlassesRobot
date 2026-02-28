#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_NAME="${1:-}"
if [[ -z "${TASK_NAME}" ]]; then
  echo "Usage: $0 <task-name> [extra calib args...]"
  echo "Example: $0 book"
  exit 1
fi
shift || true

TS="$(date +%Y%m%d_%H%M%S)"
OUT_NPZ="/tmp/task_tcp_object_${TASK_NAME}_${TS}.npz"
cleanup() {
  rm -f "${OUT_NPZ}" || true
}
trap cleanup EXIT

# Calibration must inherit current shell ROS env (rclpy), so use current python directly.
CALIB_PY_CMD=(python)
echo "[INFO] Using current python for calibration (expects active 'glasses' + sourced ROS in this shell)"

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "vis"; then
  VIS_PY_CMD=(conda run -n vis python)
  echo "[INFO] Using conda env: vis (for rerun)"
else
  VIS_PY_CMD=(python)
  echo "[WARN] conda env 'vis' not found; fallback to current python for rerun"
fi

echo "[INFO] Running calibration for task=${TASK_NAME}"
"${CALIB_PY_CMD[@]}" "${ROOT_DIR}/src/egodata_eval/calib_task_tcp_object_se3.py" \
  --task "${TASK_NAME}" \
  --out-npz "${OUT_NPZ}" \
  "$@"

echo "[INFO] Launch rerun visualization"
"${VIS_PY_CMD[@]}" "${ROOT_DIR}/vis_task_tcp_to_object_rerun.py" \
  --task "${TASK_NAME}" \
  --calib-npz "${OUT_NPZ}" \
  --spawn

echo "[OK] Done. Temporary npz cleaned: ${OUT_NPZ}"
