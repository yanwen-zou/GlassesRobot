import contextlib
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
import torch
from torchvision.transforms import functional as TF, InterpolationMode

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


EPISODE_DIR = Path("data/20251112_142342")
RGB_DIR = EPISODE_DIR / "rgb"
MASK_DIR = EPISODE_DIR / "masks"
HAND_MASK_DIR = EPISODE_DIR / "mask_hand"
OUTPUT_DIR = EPISODE_DIR / "vggt_output"
POINT_CONF_THRESH = 0.5


def gather_frame_paths() -> tuple[list[str], list[Path], list[Path]]:
    rgb_paths = sorted(RGB_DIR.glob("*.png"), key=lambda p: int(p.stem))
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB frames found in {RGB_DIR}")

    image_names: list[str] = []
    mask_paths: list[Path] = []
    hand_mask_paths: list[Path] = []
    for rgb_path in rgb_paths:
        mask_path = MASK_DIR / rgb_path.name
        hand_mask_path = HAND_MASK_DIR / rgb_path.name
        if not mask_path.exists() or not hand_mask_path.exists():
            # skip frames without both masks
            continue
        image_names.append(str(rgb_path))
        mask_paths.append(mask_path)
        hand_mask_paths.append(hand_mask_path)

    if not image_names:
        raise RuntimeError("No frames have both masks and mask_hand files.")
    stride = 10
    return image_names[::stride], mask_paths[::stride], hand_mask_paths[::stride]


def build_mask_tensor(mask_paths: list[Path], hand_paths: list[Path], target_hw: tuple[int, int]) -> torch.Tensor:
    h, w = target_hw
    mask_tensors = []
    for mask_path, hand_path in zip(mask_paths, hand_paths):
        mask = TF.to_tensor(Image.open(mask_path).convert("L"))
        hand = TF.to_tensor(Image.open(hand_path).convert("L"))
        combined = torch.clamp(mask + hand, 0.0, 1.0)
        combined = TF.resize(combined, size=[h, w], interpolation=InterpolationMode.NEAREST)
        mask_tensors.append(combined)
    return torch.stack(mask_tensors, dim=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    image_names, mask_paths, hand_mask_paths = gather_frame_paths()

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    processed = 0
    batch_size = 4
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    all_points = []
    all_colors = []
    for start in range(0, len(image_names), batch_size):
        end = start + batch_size
        names_batch = image_names[start:end]
        mask_batch = mask_paths[start:end]
        hand_batch = hand_mask_paths[start:end]

        images = load_and_preprocess_images(names_batch)
        h, w = images.shape[-2:]
        mask_tensor = build_mask_tensor(mask_batch, hand_batch, (h, w))
        images = images * (1.0 - mask_tensor)
        images = images.to(device)

        with torch.no_grad():
            amp_ctx = torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else contextlib.nullcontext()
            with amp_ctx:
                preds = model(images)
        world_points = preds["world_points"].squeeze(0).cpu()
        world_conf = preds["world_points_conf"].squeeze(0).cpu()
        valid_mask = world_conf > POINT_CONF_THRESH
        finite_mask = torch.isfinite(world_points).all(dim=-1)
        valid_mask = valid_mask & finite_mask
        points_np = world_points.reshape(-1, 3)
        valid_flat = valid_mask.reshape(-1)
        colors_np = images.cpu().permute(0, 2, 3, 1).reshape(-1, 3)
        if valid_flat.any():
            all_points.append(points_np[valid_flat])
            all_colors.append(colors_np[valid_flat])
        processed += len(names_batch)
        torch.cuda.empty_cache()

    if all_points:
        stacked_points = np.concatenate(all_points, axis=0)
        stacked_colors = np.concatenate(all_colors, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(stacked_points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.clip(stacked_colors, 0.0, 1.0).astype(np.float64))
        out_path = OUTPUT_DIR / "vggt_pointcloud.ply"
        o3d.io.write_point_cloud(str(out_path), pcd)
        print(f"Saved VGGT point cloud with {len(stacked_points)} points to {out_path}")

    print(f"Processed {processed} frames in batches of {batch_size}.")


if __name__ == "__main__":
    main()
