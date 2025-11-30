"""
Interactive SAM2 segmentation for multiple objects (e.g., 3 balls) with
user-defined ID order. Prompts on the first frame, then propagates masks.
Outputs per-frame masks named "<frame>_id<ID>.png" inside an episode-level
output directory (default: masks_multiobj).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Mitigate unsupported ops on Apple devices
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"[multi-sam] using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS."
    )


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
FS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_PATH.parents[3]

sys.path.append(str(FS_ROOT))

SAM_ROOT = None
for candidate in (FS_ROOT / "sam2_root", FS_ROOT / "sam2"):
    if candidate.exists():
        SAM_ROOT = candidate
        sys.path.append(str(candidate))
        break

if SAM_ROOT is None:
    raise FileNotFoundError("未找到 sam2_root 或 sam2 目录，请检查项目结构")

CONFIG_REL_PATH = Path("configs/sam2.1/sam2.1_hiera_l.yaml")
CONFIG_PATH = SAM_ROOT / "sam2" / CONFIG_REL_PATH
CHECKPOINT_PATH = SAM_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"未找到配置文件: {CONFIG_PATH}")
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"未找到模型权重: {CHECKPOINT_PATH}")

from sam2.build_sam import build_sam2_video_predictor


class WorkingDirectory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.previous: Optional[Path] = None

    def __enter__(self):
        self.previous = Path.cwd()
        os.chdir(self.path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous is not None:
            os.chdir(self.previous)


def load_frames_for_episode(episode_dir: Path) -> Tuple[Optional[Path], List[str]]:
    jpg_dir = episode_dir / "jpg"
    color_dir = episode_dir / "color"

    if jpg_dir.is_dir():
        frame_dir = jpg_dir
        extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    elif color_dir.is_dir():
        frame_dir = color_dir
        extensions = {".png", ".PNG"}
    else:
        return None, []

    frame_names = [p for p in os.listdir(frame_dir) if Path(p).suffix in extensions]
    try:
        frame_names.sort(key=lambda name: int(Path(name).stem))
    except ValueError:
        frame_names.sort()

    return frame_dir, frame_names


def prompt_objects(frame_path: Path, num_objects: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Collect click prompts for each object, preserving click order as object IDs."""
    prompts: List[Tuple[np.ndarray, np.ndarray]] = []
    image = Image.open(frame_path)

    for obj_id in range(1, num_objects + 1):
        while True:
            plt.figure(figsize=(9, 6))
            plt.title(f"{frame_path.name} - Object ID {obj_id}: click all points then press Enter")
            plt.imshow(image)
            clicks = plt.ginput(n=-1, timeout=0)
            plt.close()

            if not clicks:
                print(f"[multi-sam] No clicks for object {obj_id}; please try again (Esc to abort).")
                continue

            points = np.array([[pt[0], pt[1]] for pt in clicks], dtype=np.float32)
            labels = np.ones(len(points), dtype=np.int32)
            prompts.append((points, labels))
            break

    return prompts


def save_mask(mask: np.ndarray, out_path: Path, frame_idx: int, obj_id: int):
    """Persist a boolean mask as an 8-bit PNG on disk."""
    mask_np = np.asarray(mask)
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)

    if mask_np.ndim != 2:
        raise ValueError(
            f"frame {frame_idx}, obj {obj_id}: unexpected mask shape {mask_np.shape}, expected HxW"
        )

    height, width = mask_np.shape
    if height <= 0 or width <= 0:
        raise ValueError(
            f"frame {frame_idx}, obj {obj_id}: invalid mask size {mask_np.shape} (cannot save PNG)"
        )

    mask_uint8 = np.ascontiguousarray(mask_np.astype(np.uint8) * 255)
    image = Image.fromarray(mask_uint8, mode="L")
    image.save(out_path)


def segment_episode(
    predictor,
    episode_dir: Path,
    frame_dir: Path,
    frame_names: List[str],
    num_objects: int,
    output_dirname: str,
    obj_prompts: List[Tuple[np.ndarray, np.ndarray]],
):
    out_dir = episode_dir / output_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    inference_state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(inference_state)

    ann_frame_idx = 0

    # Add each object with its own ID, preserving prompt order.
    for obj_id, (points, labels) in enumerate(obj_prompts, start=1):
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

        # Save masks for the annotation frame (usually frame 0).
        for cur_obj_id, logits in zip(out_obj_ids, out_mask_logits):
            mask = (logits > 0.0).cpu().numpy()
            frame_name = Path(frame_names[ann_frame_idx]).stem
            save_mask(
                mask,
                out_dir / f"{frame_name}_id{cur_obj_id}.png",
                frame_idx=ann_frame_idx,
                obj_id=cur_obj_id,
            )

    # Propagate through the rest of the video.
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        print(f"[multi-sam] {episode_dir.name}: frame {out_frame_idx}")
        frame_name = Path(frame_names[out_frame_idx]).stem
        for cur_obj_id, logits in zip(out_obj_ids, out_mask_logits):
            mask = (logits > 0.0).cpu().numpy()
            try:
                save_mask(
                    mask,
                    out_dir / f"{frame_name}_id{cur_obj_id}.png",
                    frame_idx=out_frame_idx,
                    obj_id=cur_obj_id,
                )
            except ValueError as exc:
                print(f"[multi-sam] ⚠️  skip frame {out_frame_idx}, obj {cur_obj_id}: {exc}")

    print(f"[multi-sam] ✅ 完成 {episode_dir.name}，输出目录: {out_dir}")


def main(data_root: Path, num_objects: int, output_dirname: str):
    with WorkingDirectory(SAM_ROOT):
        predictor = build_sam2_video_predictor(
            str(CONFIG_REL_PATH),
            str(CHECKPOINT_PATH),
            device=device,
        )

    # 先收集所有 episode 的点击，再统一处理
    episodes_info: List[Tuple[Path, Path, List[str], List[Tuple[np.ndarray, np.ndarray]]]] = []
    for episode_dir in sorted(data_root.iterdir()):
        if not episode_dir.is_dir() or episode_dir.name == output_dirname:
            continue
        if (episode_dir / output_dirname).exists():
            print(f"⚠️ 跳过 {episode_dir.name}: 已存在输出目录 {output_dirname}")
            continue

        frame_dir, frame_names = load_frames_for_episode(episode_dir)
        if not frame_names:
            print(f"⚠️ 跳过 {episode_dir.name}: 未找到可用帧")
            continue

        # 仅记录点击，不立即运行推理
        first_frame = frame_dir / frame_names[0]
        obj_prompts = prompt_objects(first_frame, num_objects=num_objects)
        episodes_info.append((episode_dir, frame_dir, frame_names, obj_prompts))

    if not episodes_info:
        print(f"⚠️ 未在 {data_root} 找到可处理的 episode 目录")
        return

    processed = 0
    for episode_dir, frame_dir, frame_names, obj_prompts in episodes_info:
        segment_episode(
            predictor=predictor,
            episode_dir=episode_dir,
            frame_dir=frame_dir,
            frame_names=frame_names,
            num_objects=num_objects,
            output_dirname=output_dirname,
            obj_prompts=obj_prompts,
        )
        processed += 1

    if processed == 0:
        print(f"⚠️ 未在 {data_root} 找到可处理的 episode 目录")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prompt multiple objects (ordered IDs) on first frame, propagate SAM2 masks across the video."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="已处理数据所在目录（默认: 项目根目录下 data）",
    )
    parser.add_argument(
        "--num_objects",
        type=int,
        default=3,
        help="Number of objects to track; prompt order defines object IDs.",
    )
    parser.add_argument(
        "--output_dirname",
        type=str,
        default="masks_balls",
        help="Subdirectory name to store per-object masks.",
    )
    args = parser.parse_args()

    default_root = PROJECT_ROOT / "data"
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else default_root

    main(data_root=data_root, num_objects=args.num_objects, output_dirname=args.output_dirname)
