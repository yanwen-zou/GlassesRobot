#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <data-dir> [extra-load-args] -- [extra-vis-args]"
  echo "Example: $0 /path/to/data -- --spawn"
  exit 1
fi

DATA_DIR="$1"
shift

LOAD_ARGS=()
VIS_ARGS=()

vis_flag_with_value() {
  case "$1" in
    --max-frames|--axis-len|--tmp-dir) return 0 ;;
    *) return 1 ;;
  esac
}

vis_flag_no_value() {
  case "$1" in
    --spawn) return 0 ;;
    *) return 1 ;;
  esac
}

if [ $# -gt 0 ]; then
  if [[ " $* " == *" -- "* ]]; then
    while [ $# -gt 0 ]; do
      if [ "$1" == "--" ]; then
        shift
        break
      fi
      LOAD_ARGS+=("$1")
      shift
    done
    while [ $# -gt 0 ]; do
      VIS_ARGS+=("$1")
      shift
    done
  else
    while [ $# -gt 0 ]; do
      if vis_flag_no_value "$1"; then
        VIS_ARGS+=("$1")
        shift
      elif vis_flag_with_value "$1"; then
        if [ $# -lt 2 ]; then
          echo "Missing value for $1"
          exit 1
        fi
        VIS_ARGS+=("$1" "$2")
        shift 2
      else
        LOAD_ARGS+=("$1")
        shift
      fi
    done
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

conda run -n glasses python "${SCRIPT_DIR}/load_dataset.py" --data-dir "${DATA_DIR}" "${LOAD_ARGS[@]}"
conda run -n vis python "${SCRIPT_DIR}/vis_dataset.py" --data-dir "${DATA_DIR}" "${VIS_ARGS[@]}"
