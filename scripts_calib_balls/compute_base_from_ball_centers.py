#!/usr/bin/env python3
"""Compute per-frame camera-to-base transforms from ball_centers.txt.

Frames that miss any ball are now tracked using head_pos relative motion
converted with T_tcp_zed, until ball detections return.

Base coordinate definition (per frame):
    - Use ball ID 2 as the origin.
    - X axis: direction from ball 2 to ball 3 (id 2 -> id 3).
    - Y axis: direction from ball 2 to ball 1 (id 2 -> id 1), orthogonalized w.r.t X.
    - Z axis: X × Y (right-handed).

All input ball centers are assumed to be in the camera coordinate system
for that frame. This script computes, for each frame, the 4x4 transform
T_base_cam that maps camera-frame points into the per-frame base frame:

    p_base = T_base_cam @ [p_cam; 1]

Output format (one line per frame after header):
    frame_id r00 r01 r02 r10 r11 r12 r20 r21 r22 tx ty tz
where R is the 3x3 rotation matrix (camera -> base) and t is the translation
of the camera origin expressed in the base frame.

Example:
    python scripts_calib_balls/compute_base_from_ball_centers.py \
        --ball-centers data/train/20251125_210453/ball_centers.txt \
        --npy-output data/train/20251125_210453/cam_to_base.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def load_ball_centers(path: Path) -> Dict[int, Dict[int, np.ndarray]]:
    """Load ball centers from text file.

    Args:
        path: Path to ball_centers.txt (format: frame_id ball_id x y z)

    Returns:
        Dictionary: frame_id (int) -> {ball_id (int): np.ndarray(3,)}
    """
    if not path.exists():
        raise FileNotFoundError(f"Ball centers file not found: {path}")

    data: Dict[int, Dict[int, np.ndarray]] = {}
    with path.open("r") as f:
        lines = f.readlines()
    # Skip header
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        frame_id = int(parts[0])
        ball_id = int(parts[1])
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        if frame_id not in data:
            data[frame_id] = {}
        data[frame_id][ball_id] = np.array([x, y, z], dtype=np.float32)
    return data


def compute_base_from_three_points(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute base frame (R_base_cam, t_base_cam) from three ball centers.

    Points are in camera coordinates:
        p1: ball 1 (id 1)
        p2: ball 2 (id 2) -> origin of base frame
        p3: ball 3 (id 3)

    Returns:
        R_base_cam: 3x3 rotation (camera -> base)
        t_base_cam: 3-vector translation of camera origin in base frame
    """
    # Origin at ball 2
    origin = p2.astype(np.float32)

    # X axis: 2 -> 3
    x_vec = (p3 - p2).astype(np.float32)
    x_norm = np.linalg.norm(x_vec)
    if x_norm < 1e-6:
        raise ValueError("Degenerate configuration: ball 2 and 3 are too close.")
    x_axis = x_vec / x_norm

    # Y axis: 2 -> 1, orthogonalized w.r.t X
    y_vec = (p1 - p2).astype(np.float32)
    # Remove projection on X
    proj = np.dot(y_vec, x_axis) * x_axis
    y_ortho = y_vec - proj
    y_norm = np.linalg.norm(y_ortho)
    if y_norm < 1e-6:
        raise ValueError("Degenerate configuration: ball 1 is colinear with balls 2 and 3.")
    y_axis = y_ortho / y_norm

    # Z axis: X × Y (right-handed)
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        raise ValueError("Degenerate configuration: cannot form valid Z axis.")
    z_axis = z_axis / z_norm

    # Re-orthogonalize Y to ensure numerical stability
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # R_cam_base has columns as base axes expressed in camera frame.
    # We need R_base_cam (camera -> base), so transpose.
    R_cam_base = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape (3, 3)
    R_base_cam = R_cam_base.T

    # Translation: t_base_cam = -R_base_cam * origin_cam
    t_base_cam = -R_base_cam @ origin

    return R_base_cam.astype(np.float32), t_base_cam.astype(np.float32)


def compute_cam_to_base_transforms(
    centers: Dict[int, Dict[int, np.ndarray]],
    head_cam_poses: Optional[Dict[int, tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[int, np.ndarray]:
    """Compute T_base_cam for all frames.

    When head_cam_poses is provided, frames that miss any ball are tracked using
    head_pos relative transforms until balls are detected again.
    """
    def build_transform_from_balls(frame_id: int, balls: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
        if not all(b in balls for b in (1, 2, 3)):
            return None
        p1, p2, p3 = balls[1], balls[2], balls[3]
        try:
            R_base_cam, t_base_cam = compute_base_from_three_points(p1, p2, p3)
        except ValueError as exc:
            print(f"[WARN] Frame {frame_id}: {exc}, skipping.")
            return None
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R_base_cam
        T[:3, 3] = t_base_cam
        return T

    if head_cam_poses is None:
        transforms: Dict[int, np.ndarray] = {}
        for frame_id, balls in centers.items():
            T = build_transform_from_balls(frame_id, balls)
            if T is not None:
                transforms[frame_id] = T
        return transforms

    # Tracking path with head_pos
    transforms: Dict[int, np.ndarray] = {}
    head_cam_poses_np: Dict[int, tuple[np.ndarray, np.ndarray]] = {
        fid: (tcp.astype(np.float64), cam.astype(np.float64)) for fid, (tcp, cam) in head_cam_poses.items()
    }

    # Bootstrap with the first frame that has both head_pos and all three balls.
    anchor_frame: Optional[int] = None
    anchor_transform: Optional[np.ndarray] = None
    sorted_frames = sorted(centers.keys())
    for frame_id in sorted_frames:
        if frame_id not in head_cam_poses_np:
            continue
        T = build_transform_from_balls(frame_id, centers[frame_id])
        if T is not None:
            anchor_frame = frame_id
            anchor_transform = T.astype(np.float64)
            break

    if anchor_frame is None or anchor_transform is None:
        raise RuntimeError("No frame has both 3 balls and head_pos; cannot initialize tracking.")

    # Keep any earlier frames with full ball detections (no tracking possible yet).
    for frame_id in sorted_frames:
        if frame_id >= anchor_frame:
            break
        T = build_transform_from_balls(frame_id, centers[frame_id])
        if T is not None:
            transforms[frame_id] = T

    transforms[anchor_frame] = anchor_transform.astype(np.float32)
    current_T_base_cam = anchor_transform
    anchor_tcp_pose, anchor_cam_pose = head_cam_poses_np[anchor_frame]
    last_tcp_pose = anchor_tcp_pose

    # Fixed glasses->ball transform from anchor frame: T_glass^ball = T_glass^anchor T_anchor^cam T_cam^ball
    T_glass_ball = anchor_cam_pose @ anchor_transform
    T_ball_glass = np.linalg.inv(T_glass_ball)

    # Track forward using head_pos when balls are missing; reset when balls return.
    all_frames = [fid for fid in sorted(set(centers.keys()) | set(head_cam_poses_np.keys())) if fid >= anchor_frame]
    for frame_id in all_frames:
        if frame_id == anchor_frame:
            continue

        balls = centers.get(frame_id)
        head_tcp_cam = head_cam_poses_np.get(frame_id)
        has_three_balls = balls is not None and all(b in balls for b in (1, 2, 3))

        if has_three_balls:
            T = build_transform_from_balls(frame_id, balls)
            if T is None:
                continue
            transforms[frame_id] = T
            current_T_base_cam = T.astype(np.float64)
            if head_tcp_cam is not None:
                last_tcp_pose, last_cam_pose = head_tcp_cam
            else:
                last_tcp_pose, last_cam_pose = None
            continue

        # Missing balls: use head_pos relative motion to propagate.
        if head_tcp_cam is None:
            print(f"[WARN] Frame {frame_id}: missing head_pos, cannot track; skipping.")
            continue
        if last_cam_pose is None:
            # We have a valid head pose now but do not know the pose corresponding to the current base estimate.
            last_tcp_pose, last_cam_pose = head_tcp_cam
            print(f"[WARN] Frame {frame_id}: no previous head_pos to track from; waiting for next frame.")
            continue

        _, curr_cam_pose = head_tcp_cam
        rel_glass = np.linalg.inv(last_cam_pose) @ curr_cam_pose  # glass_prev <- glass_curr
        rel_in_ball = T_ball_glass @ rel_glass @ T_glass_ball
        current_T_base_cam = current_T_base_cam @ rel_in_ball
        transforms[frame_id] = current_T_base_cam.astype(np.float32)
        last_tcp_pose, last_cam_pose = head_tcp_cam

    return transforms


def save_transforms(
    transforms: Dict[int, np.ndarray],
    output_path: Path,
) -> None:
    """Save per-frame T_base_cam transforms to a text file."""
    frame_ids = sorted(transforms.keys())
    with output_path.open("w") as f:
        f.write("frame_id r00 r01 r02 r10 r11 r12 r20 r21 r22 tx ty tz\n")
        for frame_id in frame_ids:
            T = transforms[frame_id]
            R = T[:3, :3]
            t = T[:3, 3]
            # Row-major R
            vals = [
                frame_id,
                R[0, 0], R[0, 1], R[0, 2],
                R[1, 0], R[1, 1], R[1, 2],
                R[2, 0], R[2, 1], R[2, 2],
                t[0], t[1], t[2],
            ]
            f.write(
                f"{vals[0]} "
                f"{vals[1]:.6f} {vals[2]:.6f} {vals[3]:.6f} "
                f"{vals[4]:.6f} {vals[5]:.6f} {vals[6]:.6f} "
                f"{vals[7]:.6f} {vals[8]:.6f} {vals[9]:.6f} "
                f"{vals[10]:.6f} {vals[11]:.6f} {vals[12]:.6f}\n"
            )


def save_transforms_npy(
    transforms: Dict[int, np.ndarray],
    output_path: Path,
) -> None:
    """Save per-frame T_base_cam transforms to a NumPy .npy file.

    The file will contain a dict with:
        - "frame_ids": int32 array of shape (N,)
        - "transforms": float32 array of shape (N, 4, 4)
    """
    frame_ids = np.array(sorted(transforms.keys()), dtype=np.int32)
    mats = np.stack([transforms[fid] for fid in frame_ids], axis=0).astype(np.float32)
    np.save(output_path, {"frame_ids": frame_ids, "transforms": mats})


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x, y, z, w) to rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    q /= norm
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_tcp_to_cam(path: Path) -> np.ndarray:
    """Load tcp->cam (glasses->zed) transform from .txt or .npy."""
    if not path.exists():
        raise FileNotFoundError(f"tcp->cam transform not found: {path}")
    if path.suffix == ".npy":
        mat = np.load(path).astype(np.float64)
    else:
        mat = np.loadtxt(path, dtype=np.float64)
    if mat.shape == (16,):
        mat = mat.reshape(4, 4)
    if mat.shape != (4, 4):
        raise ValueError(f"tcp->cam must be 4x4, got {mat.shape}")
    return mat


def load_head_cam_poses(head_dir: Path, tcp_to_cam: np.ndarray) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load head_pos tcp poses and convert to cam poses (world->cam).

    Returns dict: frame_id -> (T_world_tcp, T_world_cam)
    """
    if not head_dir.exists():
        raise FileNotFoundError(f"head_pos directory not found: {head_dir}")
    poses: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(head_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1):
        try:
            fid = int(path.stem)
        except ValueError:
            continue
        vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
        if vals.size < 7:
            print(f"[WARN] Invalid head_pos contents in {path}; expected tx ty tz qx qy qz qw.")
            continue
        t = vals[:3]
        qx, qy, qz, qw = vals[3:7]
        R_tcp = quat_to_rot(qx, qy, qz, qw)
        T_tcp = np.eye(4, dtype=np.float64)
        T_tcp[:3, :3] = R_tcp
        T_tcp[:3, 3] = t
        T_cam = T_tcp @ tcp_to_cam  # world->cam
        poses[fid] = (T_tcp.astype(np.float64), T_cam.astype(np.float64))
    if not poses:
        raise RuntimeError(f"No valid head_pos poses loaded from {head_dir}")
    return poses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-frame camera-to-base transforms from ball_centers.txt, "
            "using ID 2->3 as X axis and 2->1 as Y axis. Frames without all balls "
            "are tracked via head_pos (tcp->zed) until detections return."
        )
    )
    parser.add_argument(
        "--ball-centers",
        type=Path,
        required=True,
        help="Path to ball_centers.txt.",
    )
    parser.add_argument(
        "--head-pos-dir",
        type=Path,
        default=None,
        help="Directory containing head_pos/*.txt (tx ty tz qx qy qz qw). Default: <ball-centers-dir>/head_pos",
    )
    parser.add_argument(
        "--tcp-to-zed",
        type=Path,
        default=Path("glasses_hardware/calib/T_tcp_zed.txt"),
        help="Path to tcp->zed (camera) 4x4 transform (.txt or .npy).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output txt path for cam-to-base transforms. "
             "Default: <ball-centers-dir>/cam_to_base.txt",
    )
    parser.add_argument(
        "--npy-output",
        type=Path,
        default=None,
        help="Output .npy path for cam-to-base transforms. "
             "Default: <ball-centers-dir>/cam_to_base.npy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    centers_path: Path = args.ball_centers
    centers = load_ball_centers(centers_path)

    if not centers:
        raise RuntimeError(f"No ball centers loaded from {centers_path}")

    head_dir = args.head_pos_dir or (centers_path.parent / "head_pos")
    tcp_to_cam = load_tcp_to_cam(args.tcp_to_zed)
    head_cam_poses = load_head_cam_poses(head_dir, tcp_to_cam)

    transforms = compute_cam_to_base_transforms(centers, head_cam_poses)
    if not transforms:
        raise RuntimeError("No transforms computed.")

    # Text output
    if args.output is not None:
        output_txt = args.output
    else:
        output_txt = centers_path.parent / "cam_to_base.txt"
    save_transforms(transforms, output_txt)
    print(f"[INFO] Saved {len(transforms)} camera-to-base transforms to {output_txt}")

    # Numpy .npy output
    if args.npy_output is not None:
        output_npy = args.npy_output
    else:
        output_npy = centers_path.parent / "cam_to_base.npy"
    save_transforms_npy(transforms, output_npy)
    print(f"[INFO] Saved {len(transforms)} camera-to-base transforms to {output_npy}")


if __name__ == "__main__":
    main()
