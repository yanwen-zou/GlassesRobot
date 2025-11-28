#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --data-dir <path-to-sequence>" >&2
  exit 1
}

DATA_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
done

[[ -z "$DATA_DIR" ]] && usage

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(realpath "$DATA_DIR")"

echo "[run_ball_pipeline] Data dir: $DATA_DIR"

python "$SCRIPT_DIR/apply_ball_masks_to_images.py" --data-dir "$DATA_DIR"
python "$SCRIPT_DIR/calculate_ball_centers.py" --data-dir "$DATA_DIR"
python "$SCRIPT_DIR/compute_base_from_ball_centers.py" --ball-centers "$DATA_DIR/ball_centers.txt"

echo "[run_ball_pipeline] Done."
