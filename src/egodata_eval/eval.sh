#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/humble/setup.bash
set -u

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

TASK="book"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --task=*)
      TASK="${1#*=}"
      shift 1
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
  esac
done

python -u glasses_hardware/hardware/grasp_test.py --task "$TASK"

python -u src/egodata_record/egodata_record/headpos_listener.py &
HEADPOS_PID=$!

cleanup() {
  if kill -0 "$HEADPOS_PID" >/dev/null 2>&1; then
    kill "$HEADPOS_PID"
  fi
}
trap cleanup EXIT

python -u src/egodata_eval/eval.py --task "$TASK" "${EXTRA_ARGS[@]}"
