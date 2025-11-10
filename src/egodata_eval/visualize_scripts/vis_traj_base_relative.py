#!/usr/bin/env python3
"""
Stream a ZED-frame absolute trajectory, convert it to base-frame relative motion,
and visualize the playback in Rerun.

Pipeline:
1. Load absolute poses (frame_id, x, y, z) expressed in the ZED camera frame.
2. Subtract the first sample so the path becomes relative to the initial position.
3. Conjugate each relative transform into the robot base frame using cached
   calibration (`T_zed_aruco.npy`, `T_base_aruco.npy`).
4. Stream the relative base-frame displacement per frame while accumulating a line strip.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np


RDF_TO_FRU = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)
RDF_TO_FRU_H = np.eye(4, dtype=np.float32)
RDF_TO_FRU_H[:3, :3] = RDF_TO_FRU


def _rdf_to_fru_transform(T_rdf: np.ndarray) -> np.ndarray:
    return (RDF_TO_FRU_H @ T_rdf).astype(np.float32)


def _rdf_to_fru_points(xyz_rdf: np.ndarray) -> np.ndarray:
    return (RDF_TO_FRU @ xyz_rdf.T).T.astype(np.float32)


def _load_transform(path: Path) -> np.ndarray:
    T = np.load(path)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 SE3 matrix at {path}, got {T.shape}")
    return T.astype(np.float32)


def _load_traj(traj_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ids: list[int] = []
    xyz: list[tuple[float, float, float]] = []
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
    parser = argparse.ArgumentParser(
        description="Visualize base-relative trajectory playback derived from ZED absolute poses."
    )
    parser.add_argument("--file", type=Path, default=Path("outputs/delta_eval_book_traj.txt"))
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS")
    parser.add_argument("--spawn", action="store_true", help="Spawn a standalone Rerun viewer window")
    parser.add_argument("--T_zed_aruco", type=Path, default=Path("T_zed_aruco.npy"))
    parser.add_argument(
        "--T_base_aruco", type=Path, default=Path("glasses_hardware/calib/T_base_aruco.npy")
    )
    args = parser.parse_args()

    ids, xyz_abs_zed = _load_traj(args.file)
    xyz_abs_zed_fru = _rdf_to_fru_points(xyz_abs_zed)
    origin = xyz_abs_zed_fru[0].copy()
    xyz_rel_zed = xyz_abs_zed_fru - origin

    T_zed_aruco_rdf = _load_transform(args.T_zed_aruco)
    T_zed_aruco = _rdf_to_fru_transform(T_zed_aruco_rdf)
    T_base_aruco = _load_transform(args.T_base_aruco)
    T_base_zed = T_base_aruco @ np.linalg.inv(T_zed_aruco)
    T_zed_base = np.linalg.inv(T_base_zed)

    xyz_rel_base: list[np.ndarray] = []
    for p in xyz_rel_zed:
        T_rel = np.eye(4, dtype=np.float32)
        T_rel[:3, 3] = p.astype(np.float32)
        T_conj = T_base_zed @ T_rel @ T_zed_base
        xyz_rel_base.append(T_conj[:3, 3].astype(np.float32))
    xyz_rel_base = np.asarray(xyz_rel_base, dtype=np.float32)

    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("Rerun package is required. Install with `pip install rerun-sdk`.") from exc

    rr.init("Base Relative Trajectory", spawn=args.spawn)
    try:
        rr.log("world", rr.ViewCoordinates.FRU)
    except Exception:
        pass

    rr.set_time("frame", sequence=int(ids[0]))
    rr.log("traj/base/path_full", rr.LineStrips3D([xyz_rel_base]))

    dt = 1.0 / max(args.fps, 1e-6)
    acc_base: list[np.ndarray] = []
    for fid, p in zip(ids, xyz_rel_base):
        rr.set_time("frame", sequence=int(fid))
        rr.log("traj/base_point", rr.Points3D(p[np.newaxis, :]))
        acc_base.append(p)
        rr.log("traj/base_path_so_far", rr.LineStrips3D([np.asarray(acc_base, dtype=np.float32)]))
        time.sleep(dt)


if __name__ == "__main__":
    main()
