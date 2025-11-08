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
    parser.add_argument("--spawn", action="store_true", help="Spawn Rerun viewer window")
    args = parser.parse_args()

    ids, xyz = load_traj_xyz(args.file)

    try:
        import rerun as rr
    except Exception as e:
        raise RuntimeError(
            "Rerun package is required. Install with `pip install rerun-sdk`."
        ) from e

    rr.init("Trajectory Playback", spawn=args.spawn)
    # Set a world coordinate convention (Right-Down-Forward typical for cameras)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    # Log the full planned path once (optional visualization)
    rr.set_time("frame", sequence=int(ids[0]))
    rr.log("traj/path_full", rr.LineStrips3D([xyz.astype(np.float32)]))

    # Stream point-by-point with an accumulating strip
    dt = 1.0 / max(args.fps, 1e-6)
    acc = []
    for i, (fid, p) in enumerate(zip(ids, xyz)):
        rr.set_time("frame", sequence=int(fid))
        # Point at current frame
        rr.log("traj/point", rr.Points3D(p[np.newaxis, :].astype(np.float32)))
        # Accumulated line strip so far
        acc.append(p.astype(np.float32))
        rr.log("traj/path_so_far", rr.LineStrips3D([np.asarray(acc, dtype=np.float32)]))
        time.sleep(dt)


if __name__ == "__main__":
    main()
