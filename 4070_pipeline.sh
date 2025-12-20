#!/usr/bin/env bash
set -euo pipefail

# Automates the post-recording steps from cmd_book.md:
# 1. Extract RGB frames (left eye) and convert PNG -> JPG for SAM.
# 2. Launch SAM-based mask generation.
# 3. Run FoundationStereo depth generation.
# 4. Save the 3x3 camera intrinsic matrix into each episode directory.
# 5. Run FoundationPose to estimate object poses per episode, using book as mesh.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="$SCRIPT_DIR"
FOUNDATION_STEREO_DIR="${PROJECT_ROOT}/src/FoundationStereo"
BALL_PIPELINE="${PROJECT_ROOT}/scripts_calib_balls/run_ball_pipeline.sh"
DEFAULT_DATA_ROOT="${PROJECT_ROOT}/data"
DATA_ROOT="$DEFAULT_DATA_ROOT"
SCALE=0.75
INTRINSICS_SRC="${FOUNDATION_STEREO_DIR}/assets/K_ZED.txt"

if [ ! -f "$INTRINSICS_SRC" ]; then
  echo "❌ Missing camera intrinsics: $INTRINSICS_SRC" >&2
  exit 1
fi

usage() {
  cat <<EOF
Usage: $(basename "$0") [--data-root PATH] [episode_name ...][--scale VALUE] [--mesh-name NAME] [--mesh-root PATH]

Without episode arguments, all directories under the selected data root are processed.
Specify one or more episode names (matching subdirectories of the data root) to
limit processing to those recordings.
EOF
}

POSITIONAL_ARGS=()
MESH_NAME="book" # Set Mesh Name
MESH_ROOT="${PROJECT_ROOT}/data"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --data-root|--data_root)
      if [ "${2:-}" = "" ]; then
        echo "❌ Missing path argument for --data-root" >&2
        exit 1
      fi
      DATA_ROOT="$2"
      shift 2
      ;;
    --mesh-root|--mesh_root)
      if [ "${2:-}" = "" ]; then
        echo "❌ Missing path argument for --mesh-root" >&2
        exit 1
      fi
      MESH_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      POSITIONAL_ARGS+=("$@")
      break
      ;;
    -*)
      echo "❌ Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"

if [[ "$DATA_ROOT" != /* ]]; then
  DATA_ROOT="${PROJECT_ROOT}/${DATA_ROOT}"
fi

if ! DATA_ROOT=$(realpath "$DATA_ROOT"); then
  echo "❌ Failed to resolve data root path." >&2
  exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "❌ Data directory not found: $DATA_ROOT" >&2
  exit 1
fi

if [[ "$MESH_ROOT" != /* ]]; then
  MESH_ROOT="${PROJECT_ROOT}/${MESH_ROOT}"
fi

if ! MESH_ROOT=$(realpath "$MESH_ROOT"); then
  echo "❌ Failed to resolve mesh root path." >&2
  exit 1
fi

if [ ! -d "$MESH_ROOT" ]; then
  echo "❌ Mesh directory not found: $MESH_ROOT" >&2
  exit 1
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

write_intrinsics() {
  local episode_dir="$1"
  local intrinsics_out="${episode_dir}/cam_K.txt"
  local camera_intrinsics_out="${episode_dir}/camera_intrinsics.txt"
  {
    printf "%s %s %s\n" "${K_VALUES[0]}" "${K_VALUES[1]}" "${K_VALUES[2]}"
    printf "%s %s %s\n" "${K_VALUES[3]}" "${K_VALUES[4]}" "${K_VALUES[5]}"
    printf "%s %s %s\n" "${K_VALUES[6]}" "${K_VALUES[7]}" "${K_VALUES[8]}"
  } >"$intrinsics_out"
  cp "$intrinsics_out" "$camera_intrinsics_out"
  echo "📐 Wrote camera intrinsics to ${intrinsics_out} (and ${camera_intrinsics_out})"
}

fill_head_pose_nans() {
  local head_dir="$1"
  if [ ! -d "$head_dir" ]; then
    return
  fi
  if ! find "$head_dir" -maxdepth 1 -type f -name '*.txt' -print -quit >/dev/null; then
    return
  fi
  conda run --no-capture-output -n glasses python - "$head_dir" <<'PY'
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
    rm -f "$rgb_dir"
  fi

  mkdir -p "$rgb_dir"
  shopt -s nullglob
  local src_file
  local copied=0
  for src_file in "${left_dir}"/*; do
    if [ -f "$src_file" ]; then
      cp -f "$src_file" "${rgb_dir}/"
      copied=1
    fi
  done
  shopt -u nullglob

  if [ "$copied" -eq 1 ]; then
    echo "📁 Copied left camera frames into ${rgb_dir}"
  else
    echo "⚠️  No files copied into ${rgb_dir}; check ${left_dir}" >&2
    return
  fi

  if [ ! -d "$jpg_dir" ] || ! find "$jpg_dir" -maxdepth 1 -name '*.jpg' -print -quit >/dev/null; then
    echo "🖼️ Converting PNG -> JPG for $episode..."
    conda run --no-capture-output -n glasses python -u \
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

declare -a SAM_EPISODES=()
for episode in "${READY_EPISODES[@]}"; do
  episode_dir="${DATA_ROOT}/${episode}"
  masks_dir="${episode_dir}/masks"
  if [ -d "$masks_dir" ] && find "$masks_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
    echo "⏭️  Masks already exist for $episode; skipping SAM segmentation."
    continue
  fi
  SAM_EPISODES+=("$episode")
done

if [ "${#SAM_EPISODES[@]}" -gt 0 ]; then
  SAM_TEMP_ROOT=$(mktemp -d)
  cleanup_sam() {
    rm -rf "$SAM_TEMP_ROOT"
  }
  trap cleanup_sam EXIT

  for episode in "${SAM_EPISODES[@]}"; do
    ln -s "${DATA_ROOT}/${episode}" "${SAM_TEMP_ROOT}/${episode}"
  done

  echo "=============================="
  echo "🪄 Launching SAM for selected episodes..."
  conda run --no-capture-output -n glasses python -u \
    "${FOUNDATION_STEREO_DIR}/scripts/batch_sam_segmentation.py" \
    --data_root "$SAM_TEMP_ROOT"

  cleanup_sam
  trap - EXIT
else
  echo "⏭️  All episodes already have SAM masks; skipping SAM segmentation."
fi

# Ensure intrinsics are present before ball processing.
for episode in "${READY_EPISODES[@]}"; do
  episode_dir="${DATA_ROOT}/${episode}"
  write_intrinsics "$episode_dir"
done

# === Ball masks (interactive) and downstream ball pipeline ===
declare -a BALL_EPISODES=()
for episode in "${READY_EPISODES[@]}"; do
  episode_dir="${DATA_ROOT}/${episode}"
  mask_balls_dir="${episode_dir}/masks_balls"
  if [ -d "$mask_balls_dir" ] && find "$mask_balls_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
    echo "⏭️  Ball masks already exist for $episode; skipping ball SAM."
    continue
  fi
  BALL_EPISODES+=("$episode")
done

if [ "${#BALL_EPISODES[@]}" -gt 0 ]; then
  BALL_TEMP_ROOT=$(mktemp -d)
  cleanup_ball() {
    rm -rf "$BALL_TEMP_ROOT"
  }
  trap cleanup_ball EXIT

  for episode in "${BALL_EPISODES[@]}"; do
    ln -s "${DATA_ROOT}/${episode}" "${BALL_TEMP_ROOT}/${episode}"
  done

  echo "=============================="
  echo "🟢 Launching SAM for balls (3 objects) ..."
  conda run --no-capture-output -n glasses python -u \
    "${FOUNDATION_STEREO_DIR}/scripts/multi_object_sam_segmentation.py" \
    --data_root "$BALL_TEMP_ROOT" \
    --num_objects 3 \
    --output_dirname masks_balls

  cleanup_ball
  trap - EXIT

else
  echo "⏭️  All episodes already have ball masks; skipping ball sam."
fi

# Decide which episodes still need ball pipeline based on cam_to_base.txt presence.
declare -a BALL_PIPELINE_EPISODES=()
for episode in "${READY_EPISODES[@]}"; do
  episode_dir="${DATA_ROOT}/${episode}"
  if [ -f "${episode_dir}/cam_to_base.txt" ]; then
    echo "⏭️  cam_to_base.txt already exists for $episode; skipping ball pipeline."
    continue
  fi
  BALL_PIPELINE_EPISODES+=("$episode")
done

for episode in "${READY_EPISODES[@]}"; do
  echo "=============================="
  echo "🔄 Generating depth with FoundationStereo for $episode..."

  episode_dir="${DATA_ROOT}/${episode}"
  depth_dir="${episode_dir}/depth"
  if [ -d "$depth_dir" ] && find "$depth_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
    echo "⏭️  Depth already exists in ${depth_dir}; skipping generation."
  else
    pushd "$FOUNDATION_STEREO_DIR" >/dev/null
    ./scripts/zed2depth.sh --data-root "$DATA_ROOT" --scale "$SCALE" "$episode"
    popd >/dev/null
  fi
done

if [ "${#BALL_PIPELINE_EPISODES[@]}" -gt 0 ]; then
  echo "=============================="
  echo "🎯 Running ball post-processing pipeline (masks -> centers -> cam_to_base)..."
  for episode in "${BALL_PIPELINE_EPISODES[@]}"; do
    conda run --no-capture-output -n glasses bash "$BALL_PIPELINE" --data-dir "${DATA_ROOT}/${episode}"
  done

  # echo "=============================="
  # echo "📐 Computing aligned cam poses (base frame) from head_pos..."
  # for episode in "${BALL_PIPELINE_EPISODES[@]}"; do
  #   conda run --no-capture-output -n glasses python -u \
  #     "${PROJECT_ROOT}/scripts_calib_balls/compute_aligned_cam_pose.py" \
  #     --episode_dir "${DATA_ROOT}/${episode}"
  # done
else
  echo "⏭️  cam_to_base.txt already present for all episodes; skipping ball pipeline/alignment."
fi

for episode in "${READY_EPISODES[@]}"; do
  echo "=============================="
  echo "🚀 Post-processing episode: $episode"

  episode_dir="${DATA_ROOT}/${episode}"
  depth_dir="${episode_dir}/depth"
  masks_dir="${episode_dir}/masks"

  if [ ! -d "$masks_dir" ] || ! find "$masks_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
    echo "⚠️  Masks not found for $episode, skipping FoundationPose." >&2
    continue
  fi

  if [ ! -d "$depth_dir" ] || ! find "$depth_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
    echo "⚠️  Depth maps missing for $episode, skipping FoundationPose." >&2
    continue
  fi

  vis_video_path="${episode_dir}/foundationpose_vis.mp4"
  if [ -f "$vis_video_path" ]; then
    echo "⏭️  FoundationPose output already exists for $episode (foundationpose_vis.mp4). Skipping."
    continue
  fi

  echo "🤖 Running FoundationPose for $episode..."

  set +e
  fp_output=$(conda run --no-capture-output -n glasses python -u \
    foundationpose/FoundationPose/run_from_mesh.py \
    --demo-name "$episode" \
    --data-root "$DATA_ROOT" \
    --mesh-root "$MESH_ROOT" \
    --mesh-name "$MESH_NAME" 2>&1)
  fp_status=$?
  set -e

  if [ "$fp_status" -ne 0 ]; then
    if echo "$fp_output" | grep -qiE 'No such file or directory|FileNotFoundError' && echo "$fp_output" | grep -qiE '\.obj'; then
      echo "⚠️  FoundationPose mesh missing for $episode, skipping."
      continue
    fi
    echo "$fp_output" >&2
    exit "$fp_status"
  fi

  printf "%s\n" "$fp_output"
done

echo "✅ Pipeline finished."
