#!/usr/bin/env bash
# Run run_ball_pipeline.sh for all subdirectories under a data root that contain masks_balls.
set -euo pipefail

DATA_ROOT="${1:-}"
if [[ -z "${DATA_ROOT}" ]]; then
  echo "Usage: $0 <data-root>"
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Data root does not exist or is not a directory: ${DATA_ROOT}"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="${SCRIPT_DIR}/run_ball_pipeline.sh"
if [[ ! -x "${PIPELINE}" ]]; then
  echo "Pipeline script not found or not executable: ${PIPELINE}"
  exit 1
fi

shopt -s nullglob
for subdir in "${DATA_ROOT}"/*; do
  [[ -d "${subdir}" ]] || continue
  if [[ -d "${subdir}/masks_balls" ]]; then
    echo "[INFO] Running pipeline in ${subdir}"
    "${PIPELINE}" --data-dir "${subdir}"
  else
    echo "[INFO] Skipping ${subdir} (no masks_balls)"
  fi
done
shopt -u nullglob
