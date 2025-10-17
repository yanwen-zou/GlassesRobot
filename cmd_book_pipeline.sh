#!/usr/bin/env bash
set -euo pipefail

# Automates the post-recording steps from cmd_book.md:
# 1. Extract RGB frames (left eye) and convert PNG -> JPG for SAM.
# 2. Launch SAM-based mask generation.
# 3. Run FoundationStereo depth generation.
# 4. Save the 3x3 camera intrinsic matrix into each episode directory.
# 5. Run FoundationPose to estimate object poses per episode.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="$SCRIPT_DIR"
FOUNDATION_STEREO_DIR="${PROJECT_ROOT}/src/FoundationStereo"
DATA_ROOT="${PROJECT_ROOT}/data"
INTRINSICS_SRC="${FOUNDATION_STEREO_DIR}/assets/K_ZED.txt"

if [ ! -f "$INTRINSICS_SRC" ]; then
  echo "❌ Missing camera intrinsics: $INTRINSICS_SRC" >&2
  exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "❌ Data directory not found: $DATA_ROOT" >&2
  exit 1
fi

usage() {
  cat <<EOF
Usage: $(basename "$0") [episode_name ...]

Without arguments, all directories under data/ are processed.
Specify one or more episode names (matching subdirectories of data/) to
limit processing to those recordings.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

declare -a EPISODES=()
declare -a READY_EPISODES=()

if [ "$#" -gt 0 ]; then
  for episode in "$@"; do
    episode_name=$(basename "$episode")
    episode_dir="${DATA_ROOT}/${episode_name}"
    if [ -d "$episode_dir" ]; then
      EPISODES+=("$episode_name")
    else
      echo "⚠️  Skipping unknown episode: $episode_name" >&2
    fi
  done
else
  shopt -s nullglob
  for dir in "${DATA_ROOT}"/*/; do
    EPISODES+=("$(basename "${dir%/}")")
  done
  shopt -u nullglob
fi

if [ "${#EPISODES[@]}" -eq 0 ]; then
  echo "⚠️  No episode directories to process under $DATA_ROOT" >&2
  exit 0
fi

read -r -a K_VALUES <<<"$(head -n 1 "$INTRINSICS_SRC")"

if [ "${#K_VALUES[@]}" -ne 9 ]; then
  echo "❌ Expected 9 values for camera intrinsics, got ${#K_VALUES[@]}" >&2
  exit 1
fi

fill_head_pose_nans() {
  local head_dir="$1"
  if [ ! -d "$head_dir" ]; then
    return
  fi
  if ! find "$head_dir" -maxdepth 1 -type f -name '*.txt' -print -quit >/dev/null; then
    return
  fi
  conda run --no-capture-output -n foundation_stereo python - "$head_dir" <<'PY'
import os
import sys
import numpy as np

head_dir = sys.argv[1]
files = [f for f in os.listdir(head_dir) if f.lower().endswith('.txt')]
if not files:
    sys.exit(0)

def sort_key(name):
    stem = os.path.splitext(name)[0]
    try:
        return int(stem)
    except ValueError:
        return stem

files.sort(key=sort_key)
arrays = []
for fname in files:
    path = os.path.join(head_dir, fname)
    data = np.loadtxt(path, dtype=np.float32)
    arrays.append(np.atleast_1d(data))
shapes = {arr.shape for arr in arrays}
if len(shapes) != 1:
    print(f"[head_pos] ⚠️ inconsistent shapes {shapes} in {head_dir}", file=sys.stderr)
    sys.exit(1)
modified = False
for idx in range(len(arrays) - 2, -1, -1):
    cur = arrays[idx]
    nxt = arrays[idx + 1]
    mask = np.isnan(cur)
    if mask.any():
        arrays[idx] = np.where(mask, nxt, cur)
        modified = True
for idx in range(1, len(arrays)):
    cur = arrays[idx]
    prev = arrays[idx - 1]
    mask = np.isnan(cur)
    if mask.any():
        arrays[idx] = np.where(mask, prev, cur)
        modified = True
remaining = sum(np.isnan(arr).sum() for arr in arrays)
if remaining:
    for idx, arr in enumerate(arrays):
        mask = np.isnan(arr)
        if mask.any():
            arrays[idx] = np.where(mask, 0.0, arr)
    modified = True
if not modified:
    sys.exit(0)
for fname, arr in zip(files, arrays):
    np.savetxt(os.path.join(head_dir, fname), np.atleast_2d(arr), fmt='%.6f')
print(f"[head_pos] ✅ filled NaNs in {head_dir} ({len(files)} frames)")
PY
}

echo "🎯 Episodes to process: ${EPISODES[*]}"

echo "🧼 Cleaning head pose NaNs (using next-frame fill)..."
for episode in "${EPISODES[@]}"; do
  episode_dir="${DATA_ROOT}/${episode}"
  shopt -s nullglob
  head_dirs=("${episode_dir}/head_pos" "${episode_dir}"/head_pos_* "${episode_dir}"/glasses_pose)
  for head_dir in "${head_dirs[@]}"; do
    if [ -d "$head_dir" ]; then
      fill_head_pose_nans "$head_dir"
    fi
  done
  shopt -u nullglob
done

prepare_frames() {
  local episode="$1"
  local episode_dir="${DATA_ROOT}/${episode}"
  local rgb_dir="${episode_dir}/rgb"
  local jpg_dir="${episode_dir}/jpg"

  echo "=============================="
  echo "🎬 Preparing frames for episode: $episode"

  # Locate left/right image directories
  local left_dir=""
  local right_dir=""
  shopt -s nullglob
  local candidate
  for candidate in "${episode_dir}/zed_left" "${episode_dir}"/zed_left_* "${episode_dir}/left"; do
    if [ -d "$candidate" ]; then
      left_dir="$candidate"
      break
    fi
  done
  for candidate in "${episode_dir}/zed_right" "${episode_dir}"/zed_right_* "${episode_dir}/right"; do
    if [ -d "$candidate" ]; then
      right_dir="$candidate"
      break
    fi
  done
  shopt -u nullglob

  if [ -z "$left_dir" ]; then
    echo "⚠️  Missing left camera frame directory for $episode" >&2
    return
  fi
  if [ -z "$right_dir" ]; then
    echo "⚠️  Missing right camera frame directory for $episode" >&2
    return
  fi

  if ! find "$left_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
    echo "⚠️  No PNG frames in ${left_dir}; skipping episode." >&2
    return
  fi

  if [ -L "$rgb_dir" ]; then
    local target
    target=$(readlink -f "$rgb_dir")
    if [ "$target" != "$(realpath "$left_dir")" ]; then
      rm -f "$rgb_dir"
      ln -s "$(realpath "$left_dir")" "$rgb_dir"
      echo "🔗 Updated rgb symlink -> $(realpath "$left_dir")"
    else
      echo "✅ rgb symlink already points to $(realpath "$left_dir")"
    fi
  elif [ -d "$rgb_dir" ]; then
    echo "ℹ️ rgb directory already exists at ${rgb_dir}"
  else
    ln -s "$(realpath "$left_dir")" "$rgb_dir"
    echo "🔗 Created rgb symlink to $(realpath "$left_dir")"
  fi

  if [ ! -d "$jpg_dir" ] || ! find "$jpg_dir" -maxdepth 1 -name '*.jpg' -print -quit >/dev/null; then
    echo "🖼️ Converting PNG -> JPG for $episode..."
    conda run --no-capture-output -n foundation_stereo python -u \
      "${FOUNDATION_STEREO_DIR}/scripts/png2jpg.py" \
      --input_dir "$left_dir" \
      --output_dir "$jpg_dir"
  else
    echo "✅ JPG frames already present in ${jpg_dir}"
  fi

  READY_EPISODES+=("$episode")
}

for episode in "${EPISODES[@]}"; do
  prepare_frames "$episode"
done

if [ "${#READY_EPISODES[@]}" -eq 0 ]; then
  echo "⚠️  No episodes have RGB frames prepared. Aborting subsequent steps." >&2
  exit 1
fi

SAM_TEMP_ROOT=$(mktemp -d)
cleanup_sam() {
  rm -rf "$SAM_TEMP_ROOT"
}
trap cleanup_sam EXIT

for episode in "${READY_EPISODES[@]}"; do
  ln -s "${DATA_ROOT}/${episode}" "${SAM_TEMP_ROOT}/${episode}"
done

echo "=============================="
echo "🪄 Launching SAM for selected episodes..."
conda run --no-capture-output -n foundation_stereo python -u \
  "${FOUNDATION_STEREO_DIR}/scripts/batch_sam_segmentation.py" \
  --data_root "$SAM_TEMP_ROOT"

cleanup_sam
trap - EXIT

for episode in "${READY_EPISODES[@]}"; do
  echo "=============================="
  echo "🚀 Post-processing episode: $episode"

  episode_dir="${DATA_ROOT}/${episode}"
  depth_dir="${episode_dir}/depth"
  masks_dir="${episode_dir}/masks"

  echo "🔄 Generating depth with FoundationStereo for $episode..."
  pushd "$FOUNDATION_STEREO_DIR" >/dev/null
  ./scripts/zed2depth.sh "$episode"
  popd >/dev/null

  intrinsics_out="${episode_dir}/cam_K.txt"
  camera_intrinsics_out="${episode_dir}/camera_intrinsics.txt"
  {
    printf "%s %s %s\n" "${K_VALUES[0]}" "${K_VALUES[1]}" "${K_VALUES[2]}"
    printf "%s %s %s\n" "${K_VALUES[3]}" "${K_VALUES[4]}" "${K_VALUES[5]}"
    printf "%s %s %s\n" "${K_VALUES[6]}" "${K_VALUES[7]}" "${K_VALUES[8]}"
  } >"$intrinsics_out"
  cp "$intrinsics_out" "$camera_intrinsics_out"
  echo "📐 Wrote camera intrinsics to ${intrinsics_out} (and ${camera_intrinsics_out})"

  if [ ! -d "$masks_dir" ] || ! find "$masks_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
    echo "⚠️  Masks not found for $episode, skipping FoundationPose." >&2
    continue
  fi

  if [ ! -d "$depth_dir" ] || ! find "$depth_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
    echo "⚠️  Depth maps missing for $episode, skipping FoundationPose." >&2
    continue
  fi

  echo "🤖 Running FoundationPose for $episode..."

  conda run --no-capture-output -n foundationpose python -u \
   foundationpose/FoundationPose/run_1010_only.py \
    --demo_name "$episode" \
    --data_root "$DATA_ROOT"
done

echo "✅ Pipeline finished."
