#!/usr/bin/env python3
"""Align head_pos-based camera poses to ball/base frame using Umeyama and export per-frame quaternions.

Given an episode directory containing:
  - cam_to_base.txt   (camera->base transforms, used as reference trajectory)
  - head_pos/*.txt    (tcp poses)
  - glasses_hardware/calib/T_tcp_zed.npy (or custom path)

This script:
  1) Loads cam positions from cam_to_base.txt (base frame).
  2) Loads head_pos, converts tcp->cam via T_tcp_zed.
  3) Aligns head_cam positions to cam_base positions using Umeyama (no scale).
  4) Applies the alignment to all head_cam poses (R,t) to get camera poses in the ball/base frame.
  5) Saves each pose as tx ty tz qx qy qz qw to <episode_dir>/cam_pose_in_ball/<frame_id>.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
UMEYAMA_PATH = SCRIPT_DIR / "umeyama.py"


def load_umeyama():
    import importlib.util

    spec = importlib.util.spec_from_file_location("umeyama", UMEYAMA_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load umeyama from {UMEYAMA_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod.umeyama  # type: ignore


umeyama = load_umeyama()


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x, y, z, w) to rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    q /= norm
    x, y, z, w = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion (x, y, z, w)."""
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        idx = int(np.argmax(np.diag(R)))
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


def load_cam_base_traj(cam_to_base_txt: Path) -> Dict[str, np.ndarray]:
    """Load camera positions (base frame) from cam_to_base.txt."""
    if not cam_to_base_txt.exists():
        raise FileNotFoundError(f"cam_to_base.txt not found at {cam_to_base_txt}")
    traj: Dict[str, np.ndarray] = {}
    lines = cam_to_base_txt.read_text().splitlines()
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) != 13:
            continue
        fid = f"{int(parts[0]):06d}"
        t = np.array(list(map(float, parts[10:13])), dtype=np.float64)
        traj[fid] = t
    if not traj:
        raise RuntimeError(f"No valid entries in {cam_to_base_txt}")
    return traj


def load_head_cam_traj(head_dir: Path, tcp_to_zed: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load head_pos tcp poses and convert to cam poses (R,t)."""
    if not head_dir.exists():
        raise FileNotFoundError(f"head_pos directory not found: {head_dir}")
    traj: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(head_dir.glob("*.txt")):
        try:
            fid = f"{int(path.stem):06d}"
        except ValueError:
            continue
        vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
        if vals.size < 7:
            continue
        t = vals[:3]
        qx, qy, qz, qw = vals[3:7]
        R_tcp = quat_to_rot(qx, qy, qz, qw)
        T_tcp = np.eye(4, dtype=np.float64)
        T_tcp[:3, :3] = R_tcp
        T_tcp[:3, 3] = t
        T_cam = T_tcp @ tcp_to_zed
        traj[fid] = (T_cam[:3, :3].astype(np.float64), T_cam[:3, 3].astype(np.float64))
    if not traj:
        raise RuntimeError(f"No valid head poses found in {head_dir}")
    return traj


def align_head_to_base(
    cam_base_traj: Dict[str, np.ndarray],
    head_cam_traj: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Align head_cam trajectory to cam_base using Umeyama on positions."""
    common = sorted(set(cam_base_traj.keys()) & set(head_cam_traj.keys()))
    if len(common) < 3:
        raise RuntimeError(f"Not enough overlapping frames for alignment (found {len(common)}).")
    base_pts = np.vstack([cam_base_traj[fid] for fid in common])
    head_pts = np.vstack([head_cam_traj[fid][1] for fid in common])

    _, R_align, t_align = umeyama(base_pts, head_pts, estimate_scale=False)
    R_inv = R_align.T

    aligned_positions: Dict[str, np.ndarray] = {}
    aligned_poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for fid, (R_head, t_head) in head_cam_traj.items():
        t_aligned = R_inv @ (t_head - t_align)
        R_aligned = R_inv @ R_head
        aligned_positions[fid] = t_aligned.astype(np.float32)
        aligned_poses[fid] = (R_aligned.astype(np.float32), t_aligned.astype(np.float32))

    return aligned_positions, aligned_poses


def save_poses(output_dir: Path, poses: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    """Save poses as tx ty tz qx qy qz qw per frame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fid in sorted(poses.keys()):
        R, t = poses[fid]
        q = rot_to_quat(R)
        out_path = output_dir / f"{fid}.txt"
        with out_path.open("w") as f:
            f.write(
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align head_pos cam trajectory to ball/base frame using Umeyama and export per-frame poses."
    )
    parser.add_argument(
        "--episode_dir",
        type=Path,
        required=True,
        help="Episode directory containing cam_to_base.txt and head_pos/.",
    )
    parser.add_argument(
        "--tcp_to_zed",
        type=Path,
        default=Path("glasses_hardware/calib/T_tcp_zed.npy"),
        help="Path to tcp->zed transform (4x4 npy).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for aligned poses (default: <episode_dir>/cam_pose_in_ball).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_dir: Path = args.episode_dir
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory does not exist: {episode_dir}")

    cam_to_base_txt = episode_dir / "cam_to_base.txt"
    head_dir = episode_dir / "head_pos"
    out_dir = args.output_dir or (episode_dir / "cam_pose_in_ball")

    tcp_to_zed = np.load(args.tcp_to_zed).astype(np.float64)
    if tcp_to_zed.shape != (4, 4):
        raise ValueError(f"tcp_to_zed must be 4x4, got {tcp_to_zed.shape}")

    cam_base_traj = load_cam_base_traj(cam_to_base_txt)
    head_cam_traj = load_head_cam_traj(head_dir, tcp_to_zed)

    aligned_positions, aligned_poses = align_head_to_base(cam_base_traj, head_cam_traj)

    save_poses(out_dir, aligned_poses)
    print(f"[INFO] Saved {len(aligned_poses)} aligned cam poses to {out_dir}")


if __name__ == "__main__":
    main()
