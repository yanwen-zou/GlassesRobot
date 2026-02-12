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
ZED_INTR_READER="${PROJECT_ROOT}/scripts_data_processing/zed_intr_reader.py"
HAND_MASK_SCRIPT="${PROJECT_ROOT}/scripts_data_processing/grounded_sam_hand_masks.py"

if [ ! -f "$ZED_INTR_READER" ]; then
  echo "❌ Missing ZED intrinsics reader: $ZED_INTR_READER" >&2
  exit 1
fi
if [ ! -f "$HAND_MASK_SCRIPT" ]; then
  echo "❌ Missing hand mask script: $HAND_MASK_SCRIPT" >&2
  exit 1
fi

usage() {
  cat <<EOF
Usage: $(basename "$0") [--data-root PATH] [episode_name ...][--scale VALUE] [--mesh-name NAME] [--mesh-root PATH] [--clear] [--run-fp] [--run-ball-calib]

Without episode arguments, all directories under the selected data root are processed.
Specify one or more episode names (matching subdirectories of the data root) to
limit processing to those recordings.
EOF
}

run_glasses() {\
  conda run --no-capture-output -n glasses "$@" 2> >(sed -e '/zstandard could not be imported/d' -e '/Install zstandard Python bindings/d' >&2)
}

POSITIONAL_ARGS=()
MESH_NAME="book" # Set Mesh Name
MESH_ROOT="${PROJECT_ROOT}/data"
CLEAR_EPISODE=0
RUN_FP=0
RUN_BALL_CALIB=0
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
    --mesh-name|--mesh_name)
      if [ "${2:-}" = "" ]; then
        echo "❌ Missing name argument for --mesh-name" >&2
        exit 1
      fi
      MESH_NAME="$2"
      shift 2
      ;;
    --clear)
      CLEAR_EPISODE=1
      shift
      ;;
    --run-fp|--run_fp)
      RUN_FP=1
      shift
      ;;
    --run-ball-calib|--run_ball_calib)
      RUN_BALL_CALIB=1
      shift
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

clear_episode_dirs() {
  local episode_dir="$1"
  local entry
  shopt -s nullglob
  for entry in "${episode_dir}"/*; do
    if [ ! -d "$entry" ]; then
      continue
    fi
    case "$(basename "$entry")" in
      head_pos|zed_left|zed_right)
        ;;
      *)
        rm -rf "$entry"
        ;;
    esac
  done
  shopt -u nullglob
}

if [ "$CLEAR_EPISODE" -eq 1 ]; then
  echo "🧹 Clearing episode directories (keep: head_pos, zed_left, zed_right)..."
  for episode in "${EPISODES[@]}"; do
    if [[ "$episode" != 2026* ]]; then
      echo "  ⏭️  Skip non-2026 episode: $episode"
      continue
    fi
    episode_dir="${DATA_ROOT}/${episode}"
    clear_episode_dirs "$episode_dir"
  done
fi

if ! run_glasses python "$ZED_INTR_READER" --resolution WVGA >/dev/null; then
  echo "❌ Failed to read ZED intrinsics from camera." >&2
  exit 1
fi
INTRINSICS_SRC="${FOUNDATION_STEREO_DIR}/assets/K_ZED.txt"
if [ ! -f "$INTRINSICS_SRC" ]; then
  echo "❌ Missing camera intrinsics: $INTRINSICS_SRC" >&2
  exit 1
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
  local head_path="$1"
  if [ ! -f "$head_path" ]; then
    return
  fi
  conda run --no-capture-output -n glasses python - "$head_path" 2>&1 <<'PY' | sed \
    -e '/zstandard could not be imported/d' \
    -e '/Install zstandard Python bindings/d' \
    -e '/CUDA environment configured/d' \
    -e '/^nvcc: NVIDIA (R) Cuda compiler driver/d' \
    -e '/^Copyright (c) 2005-2024 NVIDIA Corporation/d' \
    -e '/^Built on Thu_Mar_28_02:18:24_PDT_2024/d' \
    -e '/^Cuda compilation tools, release 12.4, V12.4.131/d' \
    -e '/^Build cuda_12.4.r12.4\/compiler.34097967_0/d'
import sys
import numpy as np

head_path = sys.argv[1]
rows = np.loadtxt(head_path, dtype=np.float32)
if rows.size == 0:
    sys.exit(0)
if rows.ndim == 1:
    rows = rows[None, :]
arrays = [row.copy() for row in rows]
shapes = {arr.shape for arr in arrays}
if len(shapes) != 1:
    print(f"[head_pos] ⚠️ inconsistent shapes {shapes} in {head_path}", file=sys.stderr)
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
np.savetxt(head_path, np.stack(arrays, axis=0), fmt='%.6f')
print(f"[head_pos] ✅ filled NaNs in {head_path} ({len(arrays)} rows)")
PY
}

echo "🎯 Episodes to process: ${EPISODES[*]}"

echo "🧼 Cleaning head pose NaNs (using next-frame fill)..."
for episode in "${EPISODES[@]}"; do
  episode_dir="${DATA_ROOT}/${episode}"
  shopt -s nullglob
  head_path="${episode_dir}/head_pos.txt"
  if [ -f "$head_path" ]; then
    fill_head_pose_nans "$head_path"
  fi
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

  local copied=0
  if [ -d "$rgb_dir" ]; then
    echo "⏭️  RGB directory already exists at ${rgb_dir}; skipping copy."
    copied=1
  else
    mkdir -p "$rgb_dir"
    shopt -s nullglob
    local src_file
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
  fi

  if [ -d "$jpg_dir" ]; then
    echo "⏭️  JPG directory already exists at ${jpg_dir}; skipping conversion."
  else
    echo "🖼️ Converting PNG -> JPG for $episode (overwrite)..."
    run_glasses python -u \
      "${FOUNDATION_STEREO_DIR}/scripts/png2jpg.py" \
      --input_dir "$left_dir" \
      --output_dir "$jpg_dir"
  fi

  READY_EPISODES+=("$episode")
}

generate_hand_masks() {
  local episode="$1"
  local episode_dir="${DATA_ROOT}/${episode}"
  local rgb_dir="${episode_dir}/rgb"
  local masks_dir="${episode_dir}/mask_hand"
  if [ ! -d "$rgb_dir" ]; then
    echo "⚠️  Missing rgb directory for $episode; skipping hand masks." >&2
    return
  fi
  if [ -d "$masks_dir" ] && find "$masks_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
    echo "⏭️  Hand masks already exist for $episode; skipping."
    return
  fi
  echo "🖐️  Generating hand masks for episode: $episode"
  run_glasses python "$HAND_MASK_SCRIPT" --data-root "$episode_dir"
}

for episode in "${EPISODES[@]}"; do
  prepare_frames "$episode"
done

if [ "${#READY_EPISODES[@]}" -eq 0 ]; then
  echo "⚠️  No episodes have RGB frames prepared. Aborting subsequent steps." >&2
  exit 1
fi

# === Robot arm masks (interactive click prompt) saved to mask_arm ===
declare -a ARM_EPISODES=()
for episode in "${READY_EPISODES[@]}"; do
  episode_dir="${DATA_ROOT}/${episode}"
  arm_dir="${episode_dir}/mask_arm"
  if [ -d "$arm_dir" ] && find "$arm_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
    echo "⏭️  mask_arm already exists for $episode; skipping robot arm mask."
    continue
  fi
  ARM_EPISODES+=("$episode")
done

if [ "${#ARM_EPISODES[@]}" -gt 0 ]; then
  ARM_TEMP_ROOT=$(mktemp -d)
  cleanup_arm() {
    rm -rf "$ARM_TEMP_ROOT"
  }
  trap cleanup_arm EXIT

  for episode in "${ARM_EPISODES[@]}"; do
    episode_dir="${DATA_ROOT}/${episode}"
    temp_episode="${ARM_TEMP_ROOT}/${episode}"
    mkdir -p "$temp_episode"
    if [ -d "${episode_dir}/jpg" ]; then
      ln -s "${episode_dir}/jpg" "${temp_episode}/jpg"
    elif [ -d "${episode_dir}/color" ]; then
      ln -s "${episode_dir}/color" "${temp_episode}/color"
    else
      echo "⚠️  Missing jpg/color for $episode; skipping robot arm mask." >&2
    fi
  done

  echo "=============================="
  echo "🦾 Launching SAM (click prompt) for robot arm masks..."
  run_glasses python -u \
    "${FOUNDATION_STEREO_DIR}/scripts/batch_sam_segmentation.py" \
    --data_root "$ARM_TEMP_ROOT"

  # Store robot arm masks to mask_arm (merge happens after grounded-SAM hand masks).
  for episode in "${ARM_EPISODES[@]}"; do
    episode_dir="${DATA_ROOT}/${episode}"
    temp_masks="${ARM_TEMP_ROOT}/${episode}/masks"
    arm_dir="${episode_dir}/mask_arm"
    if [ ! -d "$temp_masks" ] || ! find "$temp_masks" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
      echo "⚠️  No robot arm masks for $episode; skipping save."
      continue
    fi
    mkdir -p "$arm_dir"
    cp -f "$temp_masks"/*.png "$arm_dir"/
  done

  cleanup_arm
  trap - EXIT
else
  echo "⏭️  No episodes eligible for robot arm masks; skipping."
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
  run_glasses python -u \
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
  run_glasses python -u \
    "${FOUNDATION_STEREO_DIR}/scripts/multi_object_sam_segmentation.py" \
    --data_root "$BALL_TEMP_ROOT" \
    --num_objects 3 \
    --output_dirname masks_balls

  cleanup_ball
  trap - EXIT

else
  echo "⏭️  All episodes already have ball masks; skipping ball sam."
fi

declare -a HAND_EPISODES=()
for episode in "${READY_EPISODES[@]}"; do
  episode_dir="${DATA_ROOT}/${episode}"
  hand_dir="${episode_dir}/mask_hand"
  if [ -d "$hand_dir" ] && find "$hand_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
    echo "⏭️  Hand masks already exist for $episode; skipping."
    continue
  fi
  HAND_EPISODES+=("$episode")
done

if [ "${#HAND_EPISODES[@]}" -gt 0 ]; then
  HAND_TEMP_ROOT=$(mktemp -d)
  cleanup_hand() {
    rm -rf "$HAND_TEMP_ROOT"
  }
  trap cleanup_hand EXIT

  for episode in "${HAND_EPISODES[@]}"; do
    ln -s "${DATA_ROOT}/${episode}" "${HAND_TEMP_ROOT}/${episode}"
  done

  echo "=============================="
  echo "🖐️  Launching Grounded-SAM for hand masks (batch episodes)..."
  run_glasses python -u "$HAND_MASK_SCRIPT" --data-root "$HAND_TEMP_ROOT" --batch-size 4

  cleanup_hand
  trap - EXIT
else
  echo "⏭️  All episodes already have hand masks; skipping grounded-sam."
fi

# Merge robot arm masks into mask_hand (after grounded-SAM hand masks).
for episode in "${READY_EPISODES[@]}"; do
  episode_dir="${DATA_ROOT}/${episode}"
  arm_dir="${episode_dir}/mask_arm"
  hand_dir="${episode_dir}/mask_hand"
  if [ ! -d "$arm_dir" ] || ! find "$arm_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
    continue
  fi
  if [ ! -d "$hand_dir" ] || ! find "$hand_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
    echo "⚠️  mask_hand missing for $episode; skipping robot arm merge." >&2
    continue
  fi
  run_glasses python - "$arm_dir" "$hand_dir" <<'PY'
import sys
from pathlib import Path
import numpy as np
from PIL import Image

arm_dir = Path(sys.argv[1])
hand_dir = Path(sys.argv[2])
hand_dir.mkdir(parents=True, exist_ok=True)

def read_mask(path: Path):
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return arr

def write_mask(path: Path, arr: np.ndarray):
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)

for mask_path in sorted(arm_dir.glob("*.png")):
    out_path = hand_dir / mask_path.name
    arm = read_mask(mask_path)
    if out_path.exists():
        hand = read_mask(out_path)
        if hand.shape != arm.shape:
            # Prefer arm shape; resize hand if needed
            hand = np.array(Image.fromarray(hand).resize((arm.shape[1], arm.shape[0])))
        merged = np.maximum(hand, arm)
    else:
        merged = arm
    write_mask(out_path, merged)
PY
done

# Decide which episodes still need ball pipeline based on cam_to_base.txt presence,
# unless explicitly requested to re-run ball calibration.
declare -a BALL_PIPELINE_EPISODES=()
for episode in "${READY_EPISODES[@]}"; do
  episode_dir="${DATA_ROOT}/${episode}"
  if [ "$RUN_BALL_CALIB" -eq 0 ] && [ -f "${episode_dir}/cam_to_base.txt" ]; then
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
    run_glasses bash "$BALL_PIPELINE" --data-dir "${DATA_ROOT}/${episode}"
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
  if [ -f "$vis_video_path" ] && [ "$RUN_FP" -eq 0 ]; then
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
    --mesh-name "$MESH_NAME" 2>&1 | sed -e '/zstandard could not be imported/d' -e '/Install zstandard Python bindings/d')
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
