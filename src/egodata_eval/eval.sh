#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/humble/setup.bash
set -u

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

TASK="book"
DATA_ROOT=""
REALSENSE_DEVICE_INDEX=0
REALSENSE_SERIAL=""
ARM_HARDWARE="ur5"
UR5_ROBOT_IP="192.168.2.102"
DH_GRIPPER_PORT="/dev/ttyUSB0"
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
    --data-dir)
      DATA_ROOT="$2"
      shift 2
      ;;
    --data-dir=*)
      DATA_ROOT="${1#*=}"
      shift 1
      ;;
    --realsense-device-index)
      REALSENSE_DEVICE_INDEX="$2"
      shift 2
      ;;
    --realsense-device-index=*)
      REALSENSE_DEVICE_INDEX="${1#*=}"
      shift 1
      ;;
    --realsense-serial)
      REALSENSE_SERIAL="$2"
      shift 2
      ;;
    --realsense-serial=*)
      REALSENSE_SERIAL="${1#*=}"
      shift 1
      ;;
    --arm-hardware)
      ARM_HARDWARE="$2"
      shift 2
      ;;
    --arm-hardware=*)
      ARM_HARDWARE="${1#*=}"
      shift 1
      ;;
    --ur5-robot-ip)
      UR5_ROBOT_IP="$2"
      shift 2
      ;;
    --ur5-robot-ip=*)
      UR5_ROBOT_IP="${1#*=}"
      shift 1
      ;;
    --dh-gripper-port)
      DH_GRIPPER_PORT="$2"
      shift 2
      ;;
    --dh-gripper-port=*)
      DH_GRIPPER_PORT="${1#*=}"
      shift 1
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
  esac
done

case "$TASK" in
  teapot|book|sword|cup|bread) ;;
  *)
    echo "[ERROR] Unsupported task: $TASK (supported: teapot, book, sword, cup, bread)" >&2
    exit 1
    ;;
esac

if [[ -z "$DATA_ROOT" ]]; then
  DATA_ROOT="src/egodata_eval/eval_output"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DATA_DIR="${DATA_ROOT}/${TIMESTAMP}"
mkdir -p "$DATA_DIR"

case "$ARM_HARDWARE" in
  flexiv)
    python -u glasses_hardware/hardware/grasp_test.py --task "$TASK" --arm-hardware flexiv
    ;;
  ur5)
    python -u glasses_hardware/hardware/grasp_test.py \
      --task "$TASK" \
      --arm-hardware ur5 \
      --ur5-robot-ip "$UR5_ROBOT_IP" \
      --dh-gripper-port "$DH_GRIPPER_PORT"
    ;;
  *)
    echo "[ERROR] Unsupported arm hardware: $ARM_HARDWARE" >&2
    exit 1
    ;;
esac

python -u src/egodata_record/egodata_record/headpos_listener.py &
HEADPOS_PID=$!

REALSENSE_CMD=(
  python -u scripts_data_processing/realsense_record_mp4.py
  --data-dir "$DATA_DIR"
  --device-index "$REALSENSE_DEVICE_INDEX"
)
if [[ -n "$REALSENSE_SERIAL" ]]; then
  REALSENSE_CMD+=(--serial "$REALSENSE_SERIAL")
fi
"${REALSENSE_CMD[@]}" &
REALSENSE_PID=$!

cleanup() {
  if kill -0 "$HEADPOS_PID" >/dev/null 2>&1; then
    kill "$HEADPOS_PID"
  fi
  if kill -0 "$REALSENSE_PID" >/dev/null 2>&1; then
    kill "$REALSENSE_PID"
  fi
}
trap cleanup EXIT

python -u src/egodata_eval/eval.py \
  --task "$TASK" \
  --out-dir "$DATA_DIR" \
  --arm-hardware "$ARM_HARDWARE" \
  --ur5-robot-ip "$UR5_ROBOT_IP" \
  --dh-gripper-port "$DH_GRIPPER_PORT" \
  "${EXTRA_ARGS[@]}"
