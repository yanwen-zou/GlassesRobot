#!/usr/bin/env python3
"""
Visualize a trajectory file using Rerun.

Input format: outputs/delta_eval_book_traj.txt
  - Lines: "frame_id x y z (absolute ref frame)" in camera frame. Lines starting with '#' are ignored.

Behavior:
  - Streams points from the first frame to the last, one by one, in order of frame_id.
  - Logs each point at its frame time and maintains a line strip of the path so far.

Usage:
  python src/egodata_eval/visualize_scripts/vis_traj_rerun.py \
      --file outputs/delta_eval_book_traj.txt --fps 30

Should be run in vis.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np


def _load_transform(path: Path) -> np.ndarray:
    T = np.load(path)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 SE3 matrix at {path}, got {T.shape}")
    return T


def load_traj_xyz(traj_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ids: list[int] = []
    xyz: list[tuple[float, float, float]] = []
    with open(traj_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                fid = int(float(parts[0]))
                x, y, z = map(float, parts[1:4])
            except ValueError:
                continue
            ids.append(fid)
            xyz.append((x, y, z))
    if not xyz:
        raise FileNotFoundError(f"No valid waypoints loaded from {traj_path}")
    # Sort by frame id to ensure forward playback
    order = np.argsort(np.asarray(ids))
    ids_arr = np.asarray(ids, dtype=np.int64)[order]
    xyz_arr = np.asarray(xyz, dtype=np.float32)[order]
    return ids_arr, xyz_arr


def main():
    parser = argparse.ArgumentParser(description="Visualize trajectory using Rerun")
    parser.add_argument("--file", type=Path, default=Path("outputs/delta_eval_book_traj.txt"))
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS")
    parser.add_argument("--T_zed_aruco", type=Path, default=Path("T_zed_aruco.npy"))
    parser.add_argument(
        "--T_base_aruco", type=Path, default=Path("glasses_hardware/calib/T_base_aruco.npy")
    )
    args = parser.parse_args()

    ids, xyz = load_traj_xyz(args.file)
    # Normalize trajectory to be relative to the first frame's position
    origin = xyz[0].copy()
    xyz = xyz - origin
    # Conjugate relative trajectory into base frame:
    #   T_rel_base = T_base_zed @ T_rel_zed @ inv(T_base_zed)
    # which preserves translation-only deltas but keeps logic consistent with eval/replay.
    T_zed_aruco = _load_transform(args.T_zed_aruco)
    T_base_aruco = _load_transform(args.T_base_aruco)
    T_base_zed = T_base_aruco @ np.linalg.inv(T_zed_aruco)
    T_zed_base = np.linalg.inv(T_base_zed)
    xyz_base = []
    for p in xyz:
        T_rel = np.eye(4, dtype=np.float32)
        T_rel[:3, 3] = p.astype(np.float32)
        T_conj = T_base_zed @ T_rel @ T_zed_base
        xyz_base.append(T_conj[:3, 3].astype(np.float32))
    xyz_base = np.asarray(xyz_base, dtype=np.float32)
    xyz = xyz.astype(np.float32)

    try:
        import rerun as rr
    except Exception as e:
        raise RuntimeError(
            "Rerun package is required. Install with `pip install rerun-sdk`."
        ) from e

    rr.init("Trajectory Playback", spawn=True)
    # Set a world coordinate convention (Right-Down-Forward typical for cameras)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    # Log the full planned path once (optional visualization)
    rr.set_time("frame", sequence=int(ids[0]))
    rr.log("traj/path_full_zed", rr.LineStrips3D([xyz]))
    rr.log("traj/path_full_base", rr.LineStrips3D([xyz_base]))

    # Stream point-by-point with an accumulating strip
    dt = 1.0 / max(args.fps, 1e-6)
    acc_zed: list[np.ndarray] = []
    acc_base: list[np.ndarray] = []
    for i, (fid, p_zed, p_base) in enumerate(zip(ids, xyz, xyz_base)):
        rr.set_time("frame", sequence=int(fid))
        # Points for ZED and base frames
        rr.log("traj/zed_point", rr.Points3D(p_zed[np.newaxis, :]))
        rr.log("traj/base_point", rr.Points3D(p_base[np.newaxis, :]))
        # Accumulated line strips
        acc_zed.append(p_zed)
        acc_base.append(p_base)
        rr.log("traj/path_so_far_zed", rr.LineStrips3D([np.asarray(acc_zed, dtype=np.float32)]))
        rr.log("traj/path_so_far_base", rr.LineStrips3D([np.asarray(acc_base, dtype=np.float32)]))
        time.sleep(dt)


if __name__ == "__main__":
    main()
