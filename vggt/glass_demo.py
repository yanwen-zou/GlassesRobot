import argparse
import contextlib
from pathlib import Path
from typing import Sequence

import numpy as np
import open3d as o3d
import torch

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None

from vggt.models.vggt import VGGT
from vggt.utils.geometry import closed_form_inverse_se3
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from glass_utils import gather_frame_paths, build_mask_tensor


POINT_CONF_THRESH = 0.65
ISOLATION_RADIUS = 1e-3


def tensor_dict_to_numpy(preds: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    np_preds: dict[str, np.ndarray] = {}
    for key, value in preds.items():
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
            if arr.shape[0] == 1:
                arr = arr.squeeze(0)
            np_preds[key] = arr
    return np_preds


def load_head_positions(head_dir: Path, frame_ids: Sequence[int]) -> np.ndarray:
    translations = []
    for frame_id in frame_ids:
        pose_path = head_dir / f"{frame_id:06d}.txt"
        if not pose_path.exists():
            raise FileNotFoundError(f"Missing head pose file: {pose_path}")
        values = np.loadtxt(pose_path, dtype=np.float32).reshape(-1)
        if values.size < 3:
            raise ValueError(f"Pose file {pose_path} must contain tx ty tz ...")
        translations.append(values[:3])
    return np.stack(translations, axis=0)


def compute_scale_from_head_poses(extrinsics: np.ndarray, head_positions: np.ndarray) -> float:
    if extrinsics.shape[0] != head_positions.shape[0]:
        raise ValueError("Number of extrinsics and head poses must match.")
    if extrinsics.shape[0] < 2:
        raise ValueError("Need at least two frames to compute scale.")

    cam_to_world = closed_form_inverse_se3(extrinsics)
    cam_positions = cam_to_world[:, :3, 3]
    cam_positions -= cam_positions[0]
    head_rel = head_positions - head_positions[0]

    pred_path = np.linalg.norm(np.diff(cam_positions, axis=0), axis=1).sum()
    head_path = np.linalg.norm(np.diff(head_rel, axis=0), axis=1).sum()
    if pred_path < 1e-6 or head_path < 1e-6:
        raise ValueError("Insufficient motion for reliable scale estimation.")
    return head_path / pred_path


def remove_isolated_points(
    points: np.ndarray, colors: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0 or radius <= 0:
        return points, colors

    if cKDTree is not None and len(points) >= 2:
        tree = cKDTree(points)
        neighbors = tree.query_ball_point(points, r=radius)
        mask = np.array([len(idx) > 1 for idx in neighbors], dtype=bool)
    else:  # Fallback brute-force
        mask = []
        for i, pt in enumerate(points):
            dists = np.linalg.norm(points - pt, axis=1)
            mask.append(np.count_nonzero(dists <= radius) > 1)
        mask = np.array(mask, dtype=bool)

    return points[mask], colors[mask]


def filter_points(
    preds_np: dict[str, np.ndarray],
    *,
    point_conf: float,
    depth_max: float,
    scale_factor: float,
    isolation_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    images = preds_np["images"]  # (S, 3, H, W)
    world_points = preds_np["world_points"]  # (S, H, W, 3)
    world_conf = preds_np["world_points_conf"]  # (S, H, W)
    depth_map = preds_np["depth"]
    if depth_map.ndim == 4:
        depth_map = depth_map[..., 0]

    valid_mask = (world_conf >= point_conf) & np.isfinite(world_points).all(axis=-1)
    if depth_max is not None:
        effective_depth = depth_map * scale_factor
        depth_mask = effective_depth <= depth_max
        valid_mask &= depth_mask

    points = world_points.reshape(-1, 3)
    colors = np.transpose(images, (0, 2, 3, 1)).reshape(-1, 3)
    valid_flat = valid_mask.reshape(-1)
    points = points[valid_flat]
    colors = np.clip(colors[valid_flat], 0.0, 1.0)

    non_black = np.any(colors > 1e-4, axis=1)
    points = points[non_black]
    colors = colors[non_black]

    if scale_factor != 1.0:
        points *= scale_factor

    points, colors = remove_isolated_points(points, colors, isolation_radius)
    return points, colors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VGGT point clouds with masking.")
    parser.add_argument("--episode-dir", type=Path, required=True, help="Episode directory containing rgb/masks data.")
    parser.add_argument(
        "--trans-thresh",
        type=float,
        default=0.03,
        help="Minimum translation (meters) since last keyframe to accept a new keyframe.",
    )
    parser.add_argument(
        "--rot-thresh",
        type=float,
        default=2.0,
        help="Minimum rotation (degrees) since last keyframe to accept a new keyframe.",
    )
    parser.add_argument(
        "--point-conf",
        type=float,
        default=POINT_CONF_THRESH,
        help="Confidence threshold for filtering VGGT points.",
    )
    parser.add_argument("--depth-max", type=float, default=1.0, help="Maximum depth (meters) to keep.")
    parser.add_argument(
        "--isolation-radius",
        type=float,
        default=ISOLATION_RADIUS,
        help="Radius for removing isolated points (meters).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    episode_dir = args.episode_dir
    rgb_dir = episode_dir / "rgb"
    mask_dir = episode_dir / "masks"
    hand_mask_dir = episode_dir / "mask_hand"
    head_dir = episode_dir / "head_pos"
    output_dir = episode_dir / "vggt_output"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    image_names, mask_paths, hand_mask_paths = gather_frame_paths(
        rgb_dir, mask_dir, hand_mask_dir, head_dir, args.trans_thresh, args.rot_thresh
    )

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    output_dir.mkdir(exist_ok=True, parents=True)
    frame_ids = [int(Path(p).stem) for p in image_names]

    images = load_and_preprocess_images(image_names)
    h, w = images.shape[-2:]
    mask_tensor = build_mask_tensor(mask_paths, hand_mask_paths, (h, w))
    images = images * (1.0 - mask_tensor)
    images = images.to(device)

    with torch.no_grad():
        amp_ctx = torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else contextlib.nullcontext()
        with amp_ctx:
            preds = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    preds["extrinsic"] = extrinsic
    preds["intrinsic"] = intrinsic

    preds_np = tensor_dict_to_numpy(preds)
    preds_np["images"] = images.detach().cpu().numpy()

    head_positions = load_head_positions(head_dir, frame_ids)
    scale_factor = compute_scale_from_head_poses(preds_np["extrinsic"], head_positions)
    print(f"Applying head_pos-derived scale factor: {scale_factor:.4f}")


    points, colors = filter_points(
        preds_np,
        point_conf=args.point_conf,
        depth_max=args.depth_max,
        scale_factor=scale_factor,
        isolation_radius=args.isolation_radius,
    )

    if points.size == 0:
        raise RuntimeError("No valid points after filtering; nothing to save.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    out_path = output_dir / "vggt_pointcloud.ply"
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"Saved VGGT point cloud with {len(points)} points to {out_path}")

    # Persist per-frame camera data to enable downstream reprojection tasks.
    depth_map = preds_np["depth"]
    if depth_map.ndim == 4:  # (S, H, W, 1)
        depth_map = depth_map[..., 0]
    depth_map = depth_map * scale_factor

    camera_data = {
        "extrinsic": preds_np["extrinsic"],
        "intrinsic": preds_np["intrinsic"],
        "depth": depth_map,
        "frame_ids": np.array(frame_ids, dtype=np.int32),
    }
    cam_path = output_dir / "camera_data.npz"
    np.savez(cam_path, **camera_data)
    print(f"Saved per-frame camera data to {cam_path}")


if __name__ == "__main__":
    main()
