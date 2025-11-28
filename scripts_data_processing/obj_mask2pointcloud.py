import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from PIL import Image


DEPTH_SCALE_DEFAULT = 1000.0  # RealSense depth units -> meters


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert (qx, qy, qz, qw) to a 3x3 rotation matrix."""
    qx, qy, qz, qw = quat
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def load_head_poses(head_dir: Path, T_tcp_zed: np.ndarray) -> dict[int, np.ndarray]:
    pose_files = sorted(head_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if not pose_files:
        raise FileNotFoundError(f"No pose files in {head_dir}")
    poses: dict[int, np.ndarray] = {}

    flip_y = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float32)  # convert head_pos frame to camera frame (flip Y)
    for path in pose_files:
        values = np.loadtxt(path, dtype=np.float32).reshape(-1)
        if values.size < 7:
            raise ValueError(f"Pose file {path} must contain tx ty tz qx qy qz qw.")
        t = values[:3]
        q = values[3:7]
        rot = quaternion_to_matrix(q)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = rot
        mat[:3, 3] = t
        # head_pos gives world->head. Convert to world->cam by flipping Y axis, then apply tcp->zed.
        mat = mat @ flip_y @ T_tcp_zed # if headpos is converted in ros, dont need flip_y
        poses[int(path.stem)] = mat  # world->zed (extrinsic)

    # Normalize so the first camera frame becomes world: cam_i<-cam0 = inv(Ei) @ E0
    first_key = min(poses.keys())
    E0 = poses[first_key]
    inv_E0 = np.linalg.inv(E0)
    return {k: inv_E0 @ pose for k, pose in poses.items()}  # cam_i <- cam0


def load_intrinsics(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Intrinsic matrix not found: {path}")
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    mat = np.array(rows, dtype=np.float32)
    if mat.shape != (3, 3):
        raise ValueError(f"Intrinsic matrix must be 3x3, got {mat.shape}")
    return mat


def load_mask(mask_path: Path) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    mask = np.array(Image.open(mask_path).convert("L"))
    return mask > 0


def load_rgb(rgb_path: Path) -> np.ndarray:
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    return rgb


def load_depth(depth_path: Path, depth_scale: float) -> np.ndarray:
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
    depth_m = depth_raw / depth_scale
    return depth_m


def backproject_mask_to_ref(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    cam_from_cam0: np.ndarray,
    rgb: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if depth_m.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Depth shape {depth_m.shape} and mask shape {mask.shape} must match.")

    valid_mask = mask & np.isfinite(depth_m) & (depth_m > 0)
    ys, xs = np.nonzero(valid_mask)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32), None

    z = depth_m[ys, xs].astype(np.float32)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    cam_points = np.stack([x, y, z, np.ones_like(z)], axis=1)

    # cam_from_cam0 transforms current cam points into reference cam0 frame
    ref_points = (cam_from_cam0 @ cam_points.T).T[:, :3]
    colors = rgb[ys, xs] if rgb is not None else None
    return ref_points.astype(np.float32), colors

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproject masked object points using head_pos poses for all frames.")
    parser.add_argument("--episode-dir", type=Path, required=True, help="Episode directory (contains rgb/depth/masks/head_pos).")
    parser.add_argument("--head-pos-dir", type=Path, default=None, help="Defaults to <episode-dir>/head_pos")
    parser.add_argument("--mask-dir", type=Path, default=None, help="Defaults to <episode-dir>/masks")
    parser.add_argument("--rgb-dir", type=Path, default=None, help="Defaults to <episode-dir>/rgb (for colors)")
    parser.add_argument("--depth-dir", type=Path, default=None, help="Defaults to <episode-dir>/depth")
    parser.add_argument("--intrinsic", type=Path, default=None, help="Defaults to <episode-dir>/cam_K.txt")
    parser.add_argument(
        "--T_tcp_zed",
        type=Path,
        default=Path("glasses_hardware/calib/T_tcp_zed.npy"),
        help="Path to tcp->zed transform (camera extrinsic).",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=DEPTH_SCALE_DEFAULT,
        help="Meters per depth unit (default 1000.0 for RealSense uint16).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store per-frame PLYs (defaults to <episode-dir>/vggt_output/object_masks_headpos/).",
    )
    parser.add_argument("--skip-missing", action="store_true", help="Skip frames if mask/RGB/depth is missing instead of failing.")
    return parser.parse_args()


def main():
    args = parse_args()
    episode_dir: Path = args.episode_dir
    head_dir = args.head_pos_dir or (episode_dir / "head_pos")
    mask_dir = args.mask_dir or (episode_dir / "masks")
    rgb_dir = args.rgb_dir or (episode_dir / "rgb")
    depth_dir = args.depth_dir or (episode_dir / "depth")
    intrinsic_path = args.intrinsic or (episode_dir / "cam_K.txt")
    output_dir = args.output_dir or (episode_dir / "vggt_output" / "object_masks_headpos")
    T_tcp_zed = np.load(args.T_tcp_zed).astype(np.float32)

    intr = load_intrinsics(intrinsic_path)
    poses = load_head_poses(head_dir, T_tcp_zed)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for frame_id, camref_from_cam in poses.items():
        mask_path = mask_dir / f"{frame_id:06d}.png"
        rgb_path = rgb_dir / f"{frame_id:06d}.png"
        depth_path = depth_dir / f"{frame_id:06d}.png"

        try:
            mask = load_mask(mask_path)
            depth_m = load_depth(depth_path, args.depth_scale)
            rgb = load_rgb(rgb_path)
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"[WARN] {exc}; skipping frame {frame_id:06d}")
                continue
            raise

        pts, cols = backproject_mask_to_ref(depth_m, mask, intr, camref_from_cam, rgb)
        if pts.size == 0:
            print(f"[INFO] Frame {frame_id:06d}: no masked points, skipping save.")
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        if cols is not None:
            pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0).astype(np.float64))

        out_path = output_dir / f"{frame_id:06d}.ply"
        o3d.io.write_point_cloud(str(out_path), pcd)
        written += 1

    if written == 0:
        raise RuntimeError("No masked points found across frames; nothing was saved.")

    print(f"Saved {written} per-frame object point clouds to {output_dir}")


if __name__ == "__main__":
    main()
