#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FS_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${FS_ROOT}/../.." && pwd)
DEFAULT_DATA_DIR="${PROJECT_ROOT}/data/train"
DATA_DIR="$DEFAULT_DATA_DIR"
SCALE=1

usage() {
    cat <<EOF
Usage: $(basename "$0") [--data-root PATH] [--scale FLOAT] <timestamp_dir>
EOF
}

POSITIONAL_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --data-root|--data_root)
            if [ "${2:-}" = "" ]; then
                echo "❌ Missing path argument for --data-root" >&2
                exit 1
            fi
            DATA_DIR="$2"
            shift 2
            ;;
        --scale)
            if [ "${2:-}" = "" ]; then
                echo "❌ Missing value for --scale" >&2
                exit 1
            fi
            SCALE="$2"
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

if [ "${#POSITIONAL_ARGS[@]}" -ne 1 ]; then
    usage
    exit 1
fi

video_name="${POSITIONAL_ARGS[0]}"

if [[ "$DATA_DIR" != /* ]]; then
    DATA_DIR="${PROJECT_ROOT}/${DATA_DIR}"
fi

if ! DATA_DIR=$(realpath "$DATA_DIR"); then
    echo "❌ Failed to resolve data root path." >&2
    exit 1
fi

if [ -z "$video_name" ]; then
    echo "Usage: $(basename "$0") <timestamp_dir>"
    exit 1
fi

episode_dir="${DATA_DIR}/${video_name}"
if [ ! -d "$episode_dir" ]; then
    echo "❌ 未找到时间戳目录: ${episode_dir}"
    exit 1
fi

pushd "$FS_ROOT" >/dev/null

timestamp=$(basename "$episode_dir")
left_file="${episode_dir}/zed_left_${timestamp}.mp4"
right_file="${episode_dir}/zed_right_${timestamp}.mp4"
left_dir=""
right_dir=""

for candidate in "${episode_dir}/zed_left" "${episode_dir}"/zed_left_* "${episode_dir}/left"; do
    if [ -z "$left_dir" ] && [ -d "$candidate" ]; then
        left_dir="$candidate"
    fi
done
for candidate in "${episode_dir}/zed_right" "${episode_dir}"/zed_right_* "${episode_dir}/right"; do
    if [ -z "$right_dir" ] && [ -d "$candidate" ]; then
        right_dir="$candidate"
    fi
done

# 判断是否已处理：depth 目录下存在 png 文件则认为已处理
depth_dir="${episode_dir}/depth"
processed=$(find "$depth_dir" -maxdepth 1 -name '*.png' | head -n 1)
if [ -n "$processed" ]; then
    echo "⏩ 跳过 ${timestamp}: 已经处理过 (depth 下存在 png 文件)"
    popd >/dev/null
    exit 0
fi

input_left=""
input_right=""
if [ -f "$left_file" ] && [ -f "$right_file" ]; then
    input_left="$left_file"
    input_right="$right_file"
elif [ -n "$left_dir" ] && [ -n "$right_dir" ]; then
    if ! find "$left_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
        echo "⚠️ 跳过 ${timestamp}: ${left_dir} 中不存在 PNG 帧"
        popd >/dev/null
        exit 1
    fi
    if ! find "$right_dir" -maxdepth 1 -name '*.png' -print -quit >/dev/null; then
        echo "⚠️ 跳过 ${timestamp}: ${right_dir} 中不存在 PNG 帧"
        popd >/dev/null
        exit 1
    fi
    input_left="$left_dir"
    input_right="$right_dir"
else
    echo "⚠️ 跳过 ${timestamp}: 未找到可用的左右目输入"
    popd >/dev/null
    exit 1
fi

out_dir="${episode_dir}"
jpg_dir="${episode_dir}/jpg"
depth_dir="${episode_dir}/depth"

echo "starting foundationstereo processing for ${timestamp}..."

mkdir -p "$depth_dir"

conda run --no-capture-output -n foundation_stereo python -u scripts/stereo2depth.py \
    --left_file "$input_left" \
    --right_file "$input_right" \
    --out_dir "$episode_dir" \
    --scale "$SCALE"

echo "starting PNG2JPG processing for ${timestamp}..."

mkdir -p "$jpg_dir"

conda run --no-capture-output -n foundation_stereo python -u scripts/png2jpg.py \
    --input_dir "${out_dir}/rgb" \
    --output_dir "$jpg_dir"

if [ -f "$left_file" ] || [ -f "$right_file" ]; then
    raw_dir="${episode_dir}/raw"
    mkdir -p "$raw_dir"
    if [ -f "$left_file" ]; then
        mv "$left_file" "$raw_dir/$(basename "$left_file")"
        echo "📦 已移动左目视频到 ${raw_dir}"
    fi
    if [ -f "$right_file" ]; then
        mv "$right_file" "$raw_dir/$(basename "$right_file")"
        echo "📦 已移动右目视频到 ${raw_dir}"
    fi
fi

echo "✅ 完成 ${timestamp}, 输出目录: $episode_dir"

popd >/dev/null

echo "processing completed."
