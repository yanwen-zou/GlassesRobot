#!/usr/bin/env python3
"""
Compute an optimal (mean) transform from a stack of SE(3) transforms.

Input npy can be:
- ndarray of shape (N, 4, 4)
- dict with key "transforms" -> ndarray (N, 4, 4)

Output:
- 4x4 mean transform saved to --out (default: <input>_mean.npy)
"""

import argparse
from pathlib import Path
import numpy as np


def _load_transforms(path: Path) -> np.ndarray:
    """Load transforms from npy that is either an array or a dict with 'transforms'."""
    raw = np.load(path, allow_pickle=True)
    if raw.ndim == 0 and isinstance(raw.item(), dict):
        data = raw.item()
        if "transforms" not in data:
            raise ValueError(f"{path} dict missing 'transforms' key: keys={list(data.keys())}")
        mats = np.asarray(data["transforms"])
    else:
        mats = np.asarray(raw)

    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"{path} must contain transforms shaped (N,4,4); got {mats.shape}")
    if len(mats) == 0:
        raise ValueError(f"{path} contains zero transforms.")
    return mats


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert a single 3x3 rotation matrix to quaternion (w, x, y, z)."""
    m = mat
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        # Find the major diagonal element
        idx = np.argmax(np.diag(m))
        if idx == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def average_rotation(mats: np.ndarray) -> np.ndarray:
    """Average rotations using Markley quaternion method."""
    quats = np.stack([_mat_to_quat(m[:3, :3]) for m in mats], axis=0)
    # Keep quaternions in the same hemisphere for stability
    ref = quats[0]
    signs = np.sign(np.sum(quats * ref, axis=1, keepdims=True))
    signs[signs == 0] = 1.0
    quats *= signs

    A = quats.T @ quats
    eigvals, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    return _quat_to_mat(q_avg)


def compute_mean_transform(mats: np.ndarray) -> np.ndarray:
    """Compute mean transform: mean translation + averaged rotation."""
    R_mean = average_rotation(mats)
    t_mean = mats[:, :3, 3].mean(axis=0)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_mean
    T[:3, 3] = t_mean
    return T


def main():
    parser = argparse.ArgumentParser(description="Compute optimal (mean) transform from stacked transforms.")
    parser.add_argument("npy_path", type=Path, help="Path to .npy containing transforms (N,4,4) or dict with 'transforms'.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for the mean transform .npy (default: <input>_mean.npy).",
    )
    args = parser.parse_args()

    mats = _load_transforms(args.npy_path)
    mean_T = compute_mean_transform(mats)

    out_path = args.out or args.npy_path.with_name(args.npy_path.stem + "_mean.npy")
    np.save(out_path, mean_T.astype(np.float32))

    print(f"[OK] Loaded {mats.shape[0]} transforms from {args.npy_path}")
    print("[OK] Mean transform (robot->base):")
    print(mean_T)
    print(f"[OK] Saved mean transform to: {out_path}")


if __name__ == "__main__":
    main()
