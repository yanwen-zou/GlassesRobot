#!/usr/bin/env python3
"""Align camera trajectory from base_to_cam with head_pos (tcp) using Umeyama and visualize.

Given a sequence directory, this script:
1) Loads base_to_cam/*.txt (format: tx ty tz qx qy qz qw) giving camera pose in base frame.
2) Loads head_pos/*.txt (format: tx ty tz qx qy qz qw) and converts tcp->cam via T_tcp_zed.npy.
3) Uses Umeyama (no scale) to align the camera trajectories (positions only).
4) Prints the transform (base->head/world) and visualizes both trajectories in Rerun.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from umeyama import umeyama


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


def load_base_to_cam(base_to_cam_dir: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Load base_to_cam txts and return poses per frame (R_base_cam, t_base_cam)."""
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(base_to_cam_dir.glob("*.txt")):
        try:
            fid = int(path.stem)
        except ValueError:
            continue
        vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
        if vals.size < 7:
            continue
        t_bc = vals[:3]
        qx, qy, qz, qw = vals[3:7]
        R_bc = quat_to_rot(qx, qy, qz, qw)
        poses[fid] = (R_bc.astype(np.float32), t_bc.astype(np.float32))
    return poses


def load_head_cam(head_dir: Path, tcp_to_zed: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Load head_pos (tcp) and convert to cam poses via tcp->zed."""
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(head_dir.glob("*.txt")):
        try:
            fid = int(path.stem)
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
        poses[fid] = (T_cam[:3, :3].astype(np.float32), T_cam[:3, 3].astype(np.float32))
    return poses


def build_arrays(
    base_poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    head_poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Collect matched positions for Umeyama."""
    common = sorted(set(base_poses.keys()) & set(head_poses.keys()))
    if len(common) < 3:
        raise RuntimeError(f"Not enough overlapping frames ({len(common)}) to run Umeyama.")
    base_pts = []
    head_pts = []
    for fid in common:
        _, t_base = base_poses[fid]
        _, t_head = head_poses[fid]
        base_pts.append(t_base)
        head_pts.append(t_head)
    return np.vstack(base_pts), np.vstack(head_pts), common


def _import_rerun():
    try:
        import rerun as rr  # type: ignore
    except Exception as exc:
        raise RuntimeError("Rerun is required for visualization. Install with `pip install rerun-sdk`.") from exc
    return rr


def visualize(rr, aligned: np.ndarray, target: np.ndarray) -> None:
    rr.init("align_base_to_head", spawn=True)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    # Static full paths for context
    rr.log(
        "traj/aligned",
        rr.LineStrips3D([aligned.astype(np.float32)], colors=np.array([[0, 200, 255, 255]], dtype=np.uint8), radii=0.005),
    )
    rr.log(
        "traj/head",
        rr.LineStrips3D([target.astype(np.float32)], colors=np.array([[255, 120, 0, 255]], dtype=np.uint8), radii=0.005),
    )

    # Per-frame visualization (time series)
    n = min(len(aligned), len(target))
    for idx in range(n):
        rr.set_time("frame", sequence=idx)
        rr.log(
            "traj/aligned/pose",
            rr.Transform3D(translation=aligned[idx].astype(np.float32)),
        )
        rr.log(
            "traj/head/pose",
            rr.Transform3D(translation=target[idx].astype(np.float32)),
        )
        rr.log(
            "traj/aligned/point",
            rr.Points3D(aligned[idx : idx + 1].astype(np.float32), colors=np.array([[0, 200, 255, 255]], dtype=np.uint8), radii=0.01),
        )
        rr.log(
            "traj/head/point",
            rr.Points3D(target[idx : idx + 1].astype(np.float32), colors=np.array([[255, 120, 0, 255]], dtype=np.uint8), radii=0.01),
        )
        # Accumulated path up to current frame
        rr.log(
            "traj/aligned/path",
            rr.LineStrips3D([aligned[: idx + 1].astype(np.float32)], colors=np.array([[0, 200, 255, 120]], dtype=np.uint8), radii=0.004),
        )
        rr.log(
            "traj/head/path",
            rr.LineStrips3D([target[: idx + 1].astype(np.float32)], colors=np.array([[255, 120, 0, 120]], dtype=np.uint8), radii=0.004),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align base_to_cam trajectory with head_pos (tcp->cam) using Umeyama (no scale) and visualize."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Sequence directory containing base_to_cam/ and head_pos/.",
    )
    parser.add_argument(
        "--tcp_to_zed",
        type=Path,
        default=Path("glasses_hardware/calib/T_tcp_zed.npy"),
        help="Path to tcp->zed transform (4x4 npy).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path: Path = args.data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    base_to_cam_dir = data_path / "base_to_cam"
    head_dir = data_path / "head_pos"
    if not base_to_cam_dir.exists():
        raise FileNotFoundError(f"base_to_cam directory not found: {base_to_cam_dir}")
    if not head_dir.exists():
        raise FileNotFoundError(f"head_pos directory not found: {head_dir}")

    tcp_to_zed = np.load(args.tcp_to_zed).astype(np.float64)
    if tcp_to_zed.shape != (4, 4):
        raise ValueError(f"tcp_to_zed must be 4x4, got {tcp_to_zed.shape}")

    base_poses = load_base_to_cam(base_to_cam_dir)
    head_poses = load_head_cam(head_dir, tcp_to_zed)
    if not base_poses:
        raise RuntimeError("No base_to_cam poses loaded.")
    if not head_poses:
        raise RuntimeError("No head_pos poses loaded.")

    base_pts, head_pts, common = build_arrays(base_poses, head_poses)
    s, R, t = umeyama(base_pts, head_pts, estimate_scale=False)
    T_align = np.eye(4, dtype=np.float32)
    T_align[:3, :3] = R.astype(np.float32)
    T_align[:3, 3] = t.astype(np.float32)

    aligned_pts = (R @ base_pts.T).T + t  # already s=1

    print(f"[INFO] Used {len(common)} overlapping frames for alignment.")
    print(f"[INFO] Transform (base -> head/world):")
    print(f"R:\n{R}")
    print(f"t: {t}")

    rr = _import_rerun()
    visualize(rr, aligned_pts, head_pts)


if __name__ == "__main__":
    main()
