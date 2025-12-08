#!/usr/bin/env bash
set -euo pipefail

# Run vis_train.py under the "glasses" conda environment to generate temp data,
# then visualize it with vis_train_rerun.py under the "vis" environment.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
VIS_TRAIN_SCRIPT="${PROJECT_ROOT}/src/egodata_eval/visualize_scripts/vis_train.py"
VIS_RERUN_SCRIPT="${PROJECT_ROOT}/src/egodata_eval/visualize_scripts/vis_train_rerun.py"
DEFAULT_TEMP="${PROJECT_ROOT}/outputs/vis_train_temp.npy"
DEFAULT_T_ROBOT="${PROJECT_ROOT}/glasses_hardware/calib/T_robot_base.npy"
DEFAULT_HEAD_TO_ZED="${PROJECT_ROOT}/glasses_hardware/calib/T_tcp_zed.npy"

DATA_PATH=""
TEMP_FILE="$DEFAULT_TEMP"
T_ROBOT_BASE="$DEFAULT_T_ROBOT"
HEAD_TO_ZED="$DEFAULT_HEAD_TO_ZED"
AXIS_LEN="0.25"
MAX_SEQS=""
SPAWN_FLAG=0

usage() {
  cat <<EOF
Usage: $(basename "$0") --data-path DIR [options]

Options:
  --temp-file PATH         Path for the temp .npy file (default: $DEFAULT_TEMP)
  --T_robot_base PATH      Override T_robot_base file (default: $DEFAULT_T_ROBOT)
  --head-to-zed PATH       Override T_tcp_zed file (default: $DEFAULT_HEAD_TO_ZED)
  --axis-len VALUE         Axis length for visualization (default: 0.25)
  --max-seqs N             Limit number of sequences processed
  --spawn                  Spawn rerun viewer when visualizing
  -h, --help               Show this help message
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --data-path)
      DATA_PATH="${2:-}"
      shift 2
      ;;
    --temp-file)
      TEMP_FILE="${2:-}"
      shift 2
      ;;
    --T_robot_base)
      T_ROBOT_BASE="${2:-}"
      shift 2
      ;;
    --head-to-zed)
      HEAD_TO_ZED="${2:-}"
      shift 2
      ;;
    --axis-len)
      AXIS_LEN="${2:-}"
      shift 2
      ;;
    --max-seqs)
      MAX_SEQS="${2:-}"
      shift 2
      ;;
    --spawn)
      SPAWN_FLAG=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "❌ Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [ -z "$DATA_PATH" ]; then
  echo "❌ --data-path is required." >&2
  usage
  exit 1
fi

if [[ "$DATA_PATH" != /* ]]; then
  DATA_PATH="${PROJECT_ROOT}/${DATA_PATH}"
fi
if [[ "$TEMP_FILE" != /* ]]; then
  TEMP_FILE="${PROJECT_ROOT}/${TEMP_FILE}"
fi
if [[ "$T_ROBOT_BASE" != /* ]]; then
  T_ROBOT_BASE="${PROJECT_ROOT}/${T_ROBOT_BASE}"
fi
if [[ "$HEAD_TO_ZED" != /* ]]; then
  HEAD_TO_ZED="${PROJECT_ROOT}/${HEAD_TO_ZED}"
fi

if [ ! -d "$DATA_PATH" ]; then
  echo "❌ Data directory not found: $DATA_PATH" >&2
  exit 1
fi
mkdir -p "$(dirname "$TEMP_FILE")"

TRAIN_ARGS=(--data-path "$DATA_PATH" --output "$TEMP_FILE" --T_robot_base "$T_ROBOT_BASE" --head-to-zed "$HEAD_TO_ZED" --axis-len "$AXIS_LEN")
if [ -n "$MAX_SEQS" ]; then
  TRAIN_ARGS+=(--max-seqs "$MAX_SEQS")
fi

echo "🧮 Generating temp data via vis_train.py (glasses env)..."
conda run --no-capture-output -n glasses python "$VIS_TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"

RERUN_ARGS=(--temp-file "$TEMP_FILE")
if [ "$SPAWN_FLAG" -eq 1 ]; then
  RERUN_ARGS+=(--spawn)
fi

echo "🛰️  Visualizing with vis_train_rerun.py (vis env)..."
conda run --no-capture-output -n vis python "$VIS_RERUN_SCRIPT" "${RERUN_ARGS[@]}"

echo "✅ Done."
