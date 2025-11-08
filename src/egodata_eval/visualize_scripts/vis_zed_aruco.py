#!/usr/bin/env python3
"""
Visualize ZED, ArUco, and robot base coordinate frames in Rerun.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def _load_transform(path: Path) -> np.ndarray:
    T = np.load(path)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 SE3 at {path}, got {T.shape}")
    return T.astype(np.float32)


def _log_frame(rr, name: str, T: np.ndarray, axis_len: float) -> None:
    origin = T[:3, 3]
    R = T[:3, :3]
    rr.log(
        f"frames/{name}",
        rr.Transform3D(
            translation=origin,
            mat3x3=R,
        ),
    )
    origins = np.repeat(origin[None, :], 3, axis=0)
    vectors = (R.T * axis_len).astype(np.float32)
    colors = np.array(
        [
            [255, 0, 0, 255],   # +X red
            [0, 255, 0, 255],   # +Y green
            [0, 0, 255, 255],   # +Z blue
        ],
        dtype=np.uint8,
    )
    rr.log(
        f"frames/{name}/axes",
        rr.Arrows3D(
            origins=origins,
            vectors=vectors,
            colors=colors,
            radii=np.full(3, axis_len * 0.05, dtype=np.float32),
        ),
    )


def _load_traj(traj_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ids: list[int] = []
    xyz: list[Tuple[float, float, float]] = []
    with open(traj_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
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
    order = np.argsort(np.asarray(ids))
    ids_arr = np.asarray(ids, dtype=np.int64)[order]
    xyz_arr = np.asarray(xyz, dtype=np.float32)[order]
    return ids_arr, xyz_arr


def main():
    parser = argparse.ArgumentParser(description="Visualize ZED/Aruco/Base frames in Rerun")
    parser.add_argument("--axis_len", type=float, default=0.2, help="Axis length in meters")
    parser.add_argument(
        "--traj",
        type=Path,
        default=Path("outputs/delta_eval_book_traj.txt"),
        help="Absolute trajectory text file (frame_id x y z)",
    )
    parser.add_argument("--T_zed_aruco", type=Path, default=Path("T_zed_aruco.npy"))
    parser.add_argument(
        "--T_base_aruco",
        type=Path,
        default=Path("glasses_hardware/calib/T_base_aruco.npy"),
    )
    args = parser.parse_args()

    T_zed_aruco = _load_transform(args.T_zed_aruco)

    # Base frame is the visualization root; express ZED and ArUco relative to base.
    T_zed = np.eye(4, dtype=np.float32)
    T_aruco = T_zed_aruco.astype(np.float32)

    import rerun as rr

    rr.init("Frame Visualization", spawn=True)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    _log_frame(rr, "zed", T_zed, args.axis_len)
    _log_frame(rr, "aruco", T_aruco, args.axis_len)


    traj_path = args.traj
    if traj_path and traj_path.exists():
        ids, xyz_abs_zed = _load_traj(traj_path)
        N = ids.shape[0]
        hom = np.concatenate(
            [xyz_abs_zed.astype(np.float32), np.ones((N, 1), dtype=np.float32)],
            axis=1,
        )
        rr.set_time("frame", sequence=int(ids[0]))
        rr.log("traj_abs/zed/path", rr.LineStrips3D([xyz_abs_zed.astype(np.float32)]))
        rr.log("traj_abs/zed/points", rr.Points3D(xyz_abs_zed.astype(np.float32)))
    else:
        print(f"[WARN] Trajectory file not found: {traj_path}")


if __name__ == "__main__":
    main()
