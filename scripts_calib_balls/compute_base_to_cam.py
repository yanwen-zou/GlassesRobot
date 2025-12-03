#!/usr/bin/env python3
"""Compute base->camera transforms (inverse of cam_to_base) and save per-frame quaternions.

Given a sequence directory containing cam_to_base.txt, this script inverts each
transform to obtain base_to_cam and writes one txt file per frame under
<data_path>/base_to_cam/<frame_id>.txt with the format:

    tx ty tz qx qy qz qw

where the quaternion (qx, qy, qz, qw) follows the scalar-last convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def parse_cam_to_base(path: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Load cam_to_base transforms from a txt file.

    Returns a mapping: frame_id -> (R_base_cam, t_base_cam)
    """
    if not path.exists():
        raise FileNotFoundError(f"cam_to_base.txt not found at {path}")

    transforms: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    with path.open("r") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"{path} is empty")

    for line in lines[1:]:  # skip header
        parts = line.strip().split()
        if len(parts) != 13:
            continue
        frame_id = int(parts[0])
        vals = list(map(float, parts[1:]))
        R = np.array(vals[:9], dtype=np.float32).reshape(3, 3)
        t = np.array(vals[9:], dtype=np.float32)
        transforms[frame_id] = (R, t)
    if not transforms:
        raise ValueError(f"No transforms parsed from {path}")
    return transforms


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (x, y, z, w)."""
    q = np.empty(4, dtype=np.float32)
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        idx = np.argmax(np.diag(R))
        if idx == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s
    return q


def invert_transforms(transforms: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Invert cam_to_base to base_to_cam for each frame."""
    inverted: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for fid, (R, t) in transforms.items():
        R_inv = R.T
        t_inv = -R_inv @ t
        inverted[fid] = (R_inv.astype(np.float32), t_inv.astype(np.float32))
    return inverted


def save_base_to_cam(output_dir: Path, transforms: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> None:
    """Save base_to_cam as per-frame txt files with translation + quaternion."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fid in sorted(transforms.keys()):
        R, t = transforms[fid]
        quat = matrix_to_quaternion(R)
        out_path = output_dir / f"{fid:06d}.txt"
        with out_path.open("w") as f:
            f.write(
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute base_to_cam (inverse of cam_to_base) and save per-frame quaternions."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Sequence directory containing cam_to_base.txt.",
    )
    parser.add_argument(
        "--cam_to_base",
        type=Path,
        default=None,
        help="Optional explicit path to cam_to_base.txt (default: <data_path>/cam_to_base.txt).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for base_to_cam txt files (default: <data_path>/base_to_cam).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path: Path = args.data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    cam_to_base_path = args.cam_to_base or (data_path / "cam_to_base.txt")
    base_to_cam_dir = args.output_dir or (data_path / "base_to_cam")

    transforms = parse_cam_to_base(cam_to_base_path)
    inverted = invert_transforms(transforms)
    save_base_to_cam(base_to_cam_dir, inverted)

    print(f"[INFO] Saved {len(inverted)} base_to_cam poses to {base_to_cam_dir}")


if __name__ == "__main__":
    main()
