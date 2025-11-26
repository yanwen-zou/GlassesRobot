#!/usr/bin/env python3
"""Compute per-frame camera-to-base transforms from ball_centers.txt.

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
from typing import Dict

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
    centers: Dict[int, Dict[int, np.ndarray]]
) -> Dict[int, np.ndarray]:
    """Compute T_base_cam for all frames that have balls 1, 2, 3."""
    transforms: Dict[int, np.ndarray] = {}
    for frame_id, balls in centers.items():
        if not all(b in balls for b in (1, 2, 3)):
            # Skip frames missing any ball
            continue
        p1 = balls[1]
        p2 = balls[2]
        p3 = balls[3]
        try:
            R_base_cam, t_base_cam = compute_base_from_three_points(p1, p2, p3)
        except ValueError as exc:
            print(f"[WARN] Frame {frame_id}: {exc}, skipping.")
            continue

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R_base_cam
        T[:3, 3] = t_base_cam
        transforms[frame_id] = T
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-frame camera-to-base transforms from ball_centers.txt, "
            "using ID 2->3 as X axis and 2->1 as Y axis."
        )
    )
    parser.add_argument(
        "--ball-centers",
        type=Path,
        required=True,
        help="Path to ball_centers.txt.",
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

    transforms = compute_cam_to_base_transforms(centers)
    if not transforms:
        raise RuntimeError("No valid transforms computed (missing balls 1/2/3 in all frames).")

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


