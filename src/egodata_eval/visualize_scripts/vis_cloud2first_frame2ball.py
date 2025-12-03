"""
Visualize three segmented balls frame-by-frame in Rerun.
Backprojects the full RGBD point cloud per frame using head poses into the
first-camera frame, then overlays ball masks with distinct colors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import rerun as rr
from PIL import Image

try:
    import open3d as o3d  # type: ignore
except Exception:  # pragma: no cover - optional
    o3d = None


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
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


def load_headposes(head_dir: Path, T_tcp_zed: np.ndarray) -> Dict[int, np.ndarray]:
    pose_files = sorted(head_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if not pose_files:
        raise FileNotFoundError(f"No head poses found in {head_dir}")
    poses: Dict[int, np.ndarray] = {}
    for path in pose_files:
        vals = np.loadtxt(path, dtype=np.float32).reshape(-1)
        if vals.size < 7:
            continue
        t = vals[:3]
        q = vals[3:7]
        R = quaternion_to_matrix(q)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t
        T = T  @ T_tcp_zed
        poses[int(path.stem)] = T
    if not poses:
        raise FileNotFoundError(f"No valid head poses found in {head_dir}")
    first_key = min(poses.keys())
    E0_inv = np.linalg.inv(poses[first_key])
    return {k: E0_inv @ pose for k, pose in poses.items()}


def load_intrinsics(path: Path) -> np.ndarray:
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    mat = np.array(rows, dtype=np.float32)
    if mat.shape != (3, 3):
        raise ValueError(f"Intrinsic matrix must be 3x3, got {mat.shape}")
    return mat


def load_mask(mask_path: Path) -> np.ndarray:
    return np.array(Image.open(mask_path).convert("L")) > 0


def load_rgb(rgb_path: Path) -> np.ndarray:
    return np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0


def load_depth(depth_path: Path, depth_scale: float) -> np.ndarray:
    depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
    return depth_raw / depth_scale


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
    ref_points = (cam_from_cam0 @ cam_points.T).T[:, :3]
    colors = rgb[ys, xs] if rgb is not None else None
    return ref_points.astype(np.float32), colors


def backproject_full_depth(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    cam_from_cam0: np.ndarray,
    rgb: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Backproject all valid depth pixels (no mask) into cam0 frame."""
    valid_mask = np.isfinite(depth_m) & (depth_m > 0)
    ys, xs = np.nonzero(valid_mask)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32), None
    z = depth_m[ys, xs].astype(np.float32)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    cam_points = np.stack([x, y, z, np.ones_like(z)], axis=1)
    ref_points = (cam_from_cam0 @ cam_points.T).T[:, :3]
    colors = rgb[ys, xs] if rgb is not None else None
    return ref_points.astype(np.float32), colors


def iter_frames(headposes: Dict[int, np.ndarray]) -> Iterable[int]:
    for fid in sorted(headposes.keys()):
        yield fid


def load_scene_cloud(path: Optional[Path]) -> tuple[np.ndarray, np.ndarray] | None:
    if not path or not path.exists() or o3d is None:
        return None
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        return None
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32) if pcd.colors else np.ones_like(pts, dtype=np.float32)
    return pts, cols


def load_cam0_to_base0(episode_dir: Path) -> np.ndarray:
    """Load base transform for frame0 from cam_to_base.[txt|npy]."""
    txt_path = episode_dir / "cam_to_base.txt"
    npy_path = episode_dir / "cam_to_base.npy"
    if txt_path.exists():
        lines = txt_path.read_text().splitlines()
        entries = []
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) != 13:
                continue
            try:
                fid = int(parts[0])
            except ValueError:
                continue
            vals = list(map(float, parts[1:]))
            R = np.array(vals[:9], dtype=np.float32).reshape(3, 3)
            t = np.array(vals[9:], dtype=np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = t
            entries.append((fid, T))
        if not entries:
            raise FileNotFoundError(f"No valid entries in {txt_path}")
        entries.sort(key=lambda x: x[0])
        return entries[0][1]
    if npy_path.exists():
        data = np.load(npy_path, allow_pickle=True).item()
        frame_ids = data.get("frame_ids")
        transforms = data.get("transforms")
        if frame_ids is None or transforms is None or len(frame_ids) == 0:
            raise FileNotFoundError(f"No valid entries in {npy_path}")
        idx0 = int(np.argmin(frame_ids))
        return transforms[idx0].astype(np.float32)
    raise FileNotFoundError(f"cam_to_base.txt/.npy not found under {episode_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stream per-frame ball point clouds with colors in Rerun.")
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=Path("data/20251125_210453"),
        help="Episode directory containing rgb/depth/mask_balls/head_pos.",
    )
    parser.add_argument("--mask-dir", type=Path, default=None, help="Defaults to <episode-dir>/mask_balls")
    parser.add_argument("--rgb-dir", type=Path, default=None, help="Defaults to <episode-dir>/rgb")
    parser.add_argument("--depth-dir", type=Path, default=None, help="Defaults to <episode-dir>/depth")
    parser.add_argument("--intrinsic", type=Path, default=None, help="Defaults to <episode-dir>/cam_K.txt")
    parser.add_argument("--head-pos-dir", type=Path, default=None, help="Defaults to <episode-dir>/head_pos")
    parser.add_argument(
        "--T_tcp_zed",
        type=Path,
        default=Path("glasses_hardware/calib/T_tcp_zed.txt"),
        help="tcp->zed extrinsic to align head poses with camera frame.",
    )
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Meters per depth unit (default 1000 for mm).")
    parser.add_argument("--num-objects", type=int, default=3, help="Number of ball IDs expected (default 3).")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback speed (frames per second).")
    parser.add_argument("--point-radius", type=float, default=0.002, help="Point radius for Rerun markers.")
    parser.add_argument("--ball-radius", type=float, default=0.004, help="Point radius for highlighted balls.")
    parser.add_argument("--no-spawn", action="store_true", help="Do not spawn a separate Rerun viewer window.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="Static scene point cloud (.ply) to show once (assumed already in the first-frame / cam0 coord). "
             "Defaults to <episode>/vggt_output/vggt_pointcloud.ply if it exists.",
    )
    args = parser.parse_args()

    episode_dir = args.episode_dir
    mask_dir = args.mask_dir or (episode_dir / "masks_balls")
    rgb_dir = args.rgb_dir or (episode_dir / "rgb")
    depth_dir = args.depth_dir or (episode_dir / "depth")
    head_dir = args.head_pos_dir or (episode_dir / "head_pos")
    intrinsic_path = args.intrinsic or (episode_dir / "cam_K.txt")
    T_base0_cam0 = load_cam0_to_base0(episode_dir)

    intr = load_intrinsics(intrinsic_path)
    # Load tcp->zed from txt (4x4). If a .npy is supplied, np.load will also work.
    T_tcp_zed = np.loadtxt(args.T_tcp_zed).astype(np.float32) if args.T_tcp_zed.suffix == ".txt" else np.load(args.T_tcp_zed).astype(np.float32)
    headposes = load_headposes(head_dir, T_tcp_zed)

    palette = {
        1: np.array([255, 0, 0, 255], dtype=np.uint8),      # red
        2: np.array([0, 200, 0, 255], dtype=np.uint8),      # green
        3: np.array([0, 120, 255, 255], dtype=np.uint8),    # blue
        4: np.array([255, 165, 0, 255], dtype=np.uint8),    # orange (extra)
    }

    scene_path = args.scene
    if scene_path is None:
        candidate = episode_dir / "vggt_output" / "vggt_pointcloud.ply"
        if candidate.exists():
            scene_path = candidate
    scene_cloud = load_scene_cloud(scene_path)

    rr.init("balls_sequence", spawn=not args.no_spawn)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass
    if scene_cloud is not None:
        pts, cols = scene_cloud
        # Scene assumed already in cam0; transform to base0.
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        homo = np.concatenate([pts, ones], axis=1)
        pts_base = (T_base0_cam0 @ homo.T).T[:, :3]
        rr.log("world/scene", rr.Points3D(pts_base, colors=cols))

    cam_path_pts: list[np.ndarray] = []

    for frame_idx, fid in enumerate(iter_frames(headposes)):
        camref_from_cam = headposes[fid]
        base_from_cam = T_base0_cam0 @ camref_from_cam
        depth_path = depth_dir / f"{fid:06d}.png"
        rgb_path = rgb_dir / f"{fid:06d}.png"
        if not depth_path.exists() or not rgb_path.exists():
            continue

        depth_m = load_depth(depth_path, args.depth_scale)
        rgb = load_rgb(rgb_path)

        rr.set_time("frame", sequence=fid)
        # Clear ball geometries each frame to avoid accumulation in the viewer.
        rr.log("world/balls", rr.Clear(recursive=True))
        # Head pose / camera frame (already in cam0 coordinates).
        translation = base_from_cam[:3, 3].astype(np.float32)
        rotation = base_from_cam[:3, :3].astype(np.float32)
        rr.log("world/headpose", rr.Transform3D(translation=translation, mat3x3=rotation))
        rr.log(
            "world/headpose/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3), dtype=np.float32),
                vectors=np.eye(3, dtype=np.float32) * 0.05,
                colors=np.array(
                    [
                        [255, 0, 0, 255],
                        [0, 255, 0, 255],
                        [0, 0, 255, 255],
                    ],
                    dtype=np.uint8,
                ),
            ),
        )
        cam_path_pts.append(translation)
        rr.log(
            "world/headpose/path",
            rr.LineStrips3D(
                [np.asarray(cam_path_pts, dtype=np.float32)],
                radii=args.point_radius,
                colors=np.array([[255, 200, 0, 255]], dtype=np.uint8),
            ),
        )

        # Full-frame cloud aligned to the first frame then into base0 coordinates.
        full_pts_cam0, full_cols = backproject_full_depth(depth_m, intr, camref_from_cam, rgb)
        if full_pts_cam0.size > 0:
            dist_full = np.linalg.norm(full_pts_cam0, axis=1)
            keep_full = dist_full <= 1.2
            full_pts_cam0 = full_pts_cam0[keep_full]
            if full_cols is not None:
                full_cols = full_cols[keep_full]
        ones_full = np.ones((full_pts_cam0.shape[0], 1), dtype=np.float32)
        homo_full = np.concatenate([full_pts_cam0, ones_full], axis=1)
        full_pts = (T_base0_cam0 @ homo_full.T).T[:, :3]
        if full_pts.size > 0:
            full_cols_u8 = np.clip(full_cols * 255.0, 0, 255).astype(np.uint8) if full_cols is not None else None
            if full_cols_u8 is not None and full_cols_u8.shape[1] == 3:
                alpha = 255 * np.ones((full_cols_u8.shape[0], 1), dtype=np.uint8)
                full_cols_u8 = np.concatenate([full_cols_u8, alpha], axis=1)
            rr.log(
                "world/frame_cloud",
                rr.Points3D(
                    positions=full_pts,
                    colors=full_cols_u8,
                    radii=args.point_radius,
                ),
            )

        for obj_id in range(1, args.num_objects + 1):
            mask_path = mask_dir / f"{fid:06d}_id{obj_id}.png"
            if not mask_path.exists():
                continue
            mask = load_mask(mask_path)
            pts_cam0, cols = backproject_mask_to_ref(depth_m, mask, intr, camref_from_cam, rgb)
            if pts_cam0.size > 0:
                dist_pts = np.linalg.norm(pts_cam0, axis=1)
                keep_pts = dist_pts <= 1.2
                pts_cam0 = pts_cam0[keep_pts]
                if cols is not None:
                    cols = cols[keep_pts]
            if pts_cam0.size == 0:
                continue
            ones_pts = np.ones((pts_cam0.shape[0], 1), dtype=np.float32)
            homo_pts = np.concatenate([pts_cam0, ones_pts], axis=1)
            pts = (T_base0_cam0 @ homo_pts.T).T[:, :3]
            if cols is not None:
                cols_uint8 = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
                if cols_uint8.shape[1] == 3:
                    alpha = 255 * np.ones((cols_uint8.shape[0], 1), dtype=np.uint8)
                    cols_uint8 = np.concatenate([cols_uint8, alpha], axis=1)
            else:
                flat = palette.get(obj_id, np.array([255, 255, 255, 255], dtype=np.uint8))
                cols_uint8 = np.broadcast_to(flat, (pts.shape[0], 4))
            rr.log(
                f"world/balls/id{obj_id}",
                rr.Points3D(
                    positions=pts,
                    # colors=cols_uint8,
                    colors=[255, 0, 0, 255] if obj_id == 1 else
                           [0, 255, 0, 255] if obj_id == 2 else
                           [0, 0, 255, 255] if obj_id == 3 else
                           [255, 255, 0, 255],
                    radii=args.ball_radius,
                ),
            )


if __name__ == "__main__":
    main()
