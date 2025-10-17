import argparse
import os
import sys
from pathlib import Path

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

print(f"using device: {device}")

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
        self.previous = None

    def __enter__(self):
        self.previous = Path.cwd()
        os.chdir(self.path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous is not None:
            os.chdir(self.previous)


def load_frames_for_episode(episode_dir: Path):
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


def prompt_points(frame_path: Path):
    image = Image.open(frame_path)
    plt.figure(figsize=(9, 6))
    plt.title(f"{frame_path}")
    plt.imshow(image)
    clicks = plt.ginput(n=-1, timeout=0)
    plt.close()

    if not clicks:
        return None

    points = np.array([[pt[0], pt[1]] for pt in clicks], dtype=np.float32)
    labels = np.ones(len(points), dtype=np.int32)
    return points, labels


def save_mask(mask: np.ndarray, out_path: Path, frame_idx: int):
    """Persist a boolean mask as an 8-bit PNG on disk."""
    mask_np = np.asarray(mask)
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)

    if mask_np.ndim != 2:
        raise ValueError(
            f"frame {frame_idx}: unexpected mask shape {mask_np.shape}, expected HxW"
        )

    height, width = mask_np.shape
    if height <= 0 or width <= 0:
        raise ValueError(
            f"frame {frame_idx}: invalid mask size {mask_np.shape} (cannot save PNG)"
        )

    mask_uint8 = np.ascontiguousarray(mask_np.astype(np.uint8) * 255)

    # Use Pillow for saving to make error handling explicit; OpenCV would emit libpng
    # warnings without surfacing a Python exception when fed invalid shapes.
    image = Image.fromarray(mask_uint8, mode="L")
    try:
        image.save(out_path)
    except (ValueError, OSError) as exc:
        raise ValueError(
            f"frame {frame_idx}: failed to save mask to {out_path}: {exc}"
        ) from exc


def main(data_root: Path):
    with WorkingDirectory(SAM_ROOT):
        predictor = build_sam2_video_predictor(
            str(CONFIG_REL_PATH),
            str(CHECKPOINT_PATH),
            device=device,
        )

    def discover_episode_dirs(root: Path):
        candidates = []
        if load_frames_for_episode(root)[1]:
            candidates.append(root)
        else:
            for child in sorted(root.iterdir()):
                if child.is_dir() and load_frames_for_episode(child)[1]:
                    candidates.append(child)
        return candidates

    episodes = discover_episode_dirs(data_root)
    if not episodes:
        print(f"⚠️ 未在 {data_root} 发现包含帧的时间戳目录")
        return

    annotations = {}
    prepared = []

    for episode_dir in sorted(episodes):
        frame_dir, frame_names = load_frames_for_episode(episode_dir)
        if not frame_names:
            print(f"⚠️ 跳过 {episode_dir.name}: 未找到可用帧")
            continue
        if (episode_dir / "masks").is_dir():
            print(f"⚠️ 跳过 {episode_dir.name}: masks 目录已存在")
            continue

        first_frame = frame_dir / frame_names[0]
        prompt = prompt_points(first_frame)
        if prompt is None:
            print(f"⚠️ 跳过 {episode_dir.name}: 未记录任何点")
            continue

        points, labels = prompt
        annotations[episode_dir] = {
            "frame_dir": frame_dir,
            "frame_names": frame_names,
            "points": points,
            "labels": labels,
        }
        prepared.append(episode_dir)

    if not annotations:
        print("⚠️ 没有有效的标注，流程结束")
        return

    for episode_dir in prepared:
        info = annotations[episode_dir]
        frame_dir = info["frame_dir"]
        frame_names = info["frame_names"]
        points = info["points"]
        labels = info["labels"]

        inference_state = predictor.init_state(video_path=str(frame_dir))
        predictor.reset_state(inference_state)

        ann_frame_idx = 0
        obj_id = 1

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

        mask_dir = episode_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        initial_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
        save_mask(
            initial_mask,
            mask_dir / f"{Path(frame_names[ann_frame_idx]).stem}.png",
            frame_idx=ann_frame_idx,
        )

        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
            print(f"processing {out_frame_idx}")
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            frame_name = frame_names[out_frame_idx]
            try:
                save_mask(
                    mask,
                    mask_dir / f"{Path(frame_name).stem}.png",
                    frame_idx=out_frame_idx,
                )
            except ValueError as exc:
                print(f"⚠️ 跳过保存 frame {out_frame_idx}: {exc}")

        print(f"✅ 完成 {episode_dir.name} 的 mask 生成，输出目录: {mask_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="已处理数据所在目录（默认: 项目根目录下 data）",
    )
    args = parser.parse_args()

    default_root = PROJECT_ROOT / "data"
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else default_root

    main(data_root)
