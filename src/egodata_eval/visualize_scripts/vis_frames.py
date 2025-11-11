#!/usr/bin/env python3
"""
Visualize ZED, ArUco, and robot base coordinate frames in Rerun and add trajectory display.

Inputs:
- Trajectory: outputs/delta_eval_book_traj.txt, each line "frame_id x y z" in ZED (RDF) frame, meters.
- Transforms: T_zed_aruco.npy (zed<-aruco), glasses_hardware/calib/T_base_aruco.npy (base<-aruco, no offset).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
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
    # Axes should live in the frame's local coordinates; Rerun applies the transform we just logged.
    origins = np.zeros((3, 3), dtype=np.float32)
    vectors = (np.eye(3, dtype=np.float32) * axis_len).astype(np.float32)
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
    parser = argparse.ArgumentParser(description="Visualize ZED/Aruco/Base frames in Rerun and plot ZED trajectory")
    parser.add_argument("--spawn", action="store_true", help="Spawn Rerun viewer window")
    parser.add_argument("--axis_len", type=float, default=0.2, help="Axis length in meters")
    parser.add_argument("--sleep", type=float, default=0.02, help="Sleep seconds between frames when streaming")
    parser.add_argument("--rel-to-first", action="store_true", help="Display trajectory relative to first frame (subtract first XYZ)")
    parser.add_argument(
        "--traj",
        type=Path,
        default=Path("outputs/delta_eval_book_traj.txt"),
        help="Absolute trajectory text file (frame_id x y z in ZED frame)",
    )
    parser.add_argument("--T_zed_aruco", type=Path, default=Path("T_zed_aruco.npy"), help="4x4 SE3 zed<-aruco")
    parser.add_argument(
        "--T_base_aruco",
        type=Path,
        default=Path("glasses_hardware/calib/T_base_aruco.npy"),
        help="4x4 SE3 base<-aruco (no offset)",
    )
    args = parser.parse_args()

    T_zed_aruco = _load_transform(args.T_zed_aruco)
    T_base_aruco = _load_transform(args.T_base_aruco)

    # Derive base<-zed: T_base_zed = T_base_aruco @ inv(T_zed_aruco)
    T_base_zed = T_base_aruco @ np.linalg.inv(T_zed_aruco)
    T_base = np.eye(4, dtype=np.float32)

    import rerun as rr

    rr.init("Frame Visualization", spawn=args.spawn)
    try:
        rr.log("frames", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    # Log frames: base, aruco, zed
    _log_frame(rr, "base", T_base, args.axis_len)
    _log_frame(rr, "aruco", T_base_aruco, args.axis_len)
    _log_frame(rr, "zed", T_base_zed.astype(np.float32), args.axis_len)

    # Plot trajectory in ZED frame under frames/zed
    traj_path = args.traj
    if traj_path and traj_path.exists():
        ids, xyz_abs_zed = _load_traj(traj_path)
        # Base<-cam rotation for conjugation
        R_base_cam = T_base_zed[:3, :3].astype(np.float32)
        # Relative-to-first base-frame trajectory via conjugation: R * (p_cam - p0_cam)
        p0_zed = xyz_abs_zed[0].astype(np.float32)
        path_points: list[list[float]] = []
        base_rel_points: list[list[float]] = []
        for i, p in enumerate(xyz_abs_zed):
            rr.set_time_sequence("frame", int(ids[i]))
            path_points.append(p.tolist())
            rr.log(
                "frames/zed/traj/current",
                rr.Points3D(
                    np.asarray([p], dtype=np.float32),
                    colors=np.array([[255, 255, 0, 255]], dtype=np.uint8),
                ),
            )
            rr.log(
                "frames/zed/traj/path",
                rr.LineStrips3D(
                    [np.asarray(path_points, dtype=np.float32)],
                    colors=np.array([[0, 200, 255, 255]], dtype=np.uint8),
                ),
            )
            # Compute and plot base-frame relative trajectory
            p_rel_base = (R_base_cam @ (p.astype(np.float32) - p0_zed)).astype(np.float32)
            base_rel_points.append(p_rel_base.tolist())
            rr.log(
                "frames/base/traj_base_rel/current",
                rr.Points3D(
                    np.asarray([p_rel_base], dtype=np.float32),
                    colors=np.array([[255, 128, 0, 255]], dtype=np.uint8),
                ),
            )
            rr.log(
                "frames/base/traj_base_rel/path",
                rr.LineStrips3D(
                    [np.asarray(base_rel_points, dtype=np.float32)],
                    colors=np.array([[255, 128, 0, 255]], dtype=np.uint8),
                ),
            )
            if args.sleep and args.sleep > 0:
                time.sleep(float(args.sleep))
    else:
        print(f"[WARN] Trajectory file not found: {traj_path}")


if __name__ == "__main__":
    main()
