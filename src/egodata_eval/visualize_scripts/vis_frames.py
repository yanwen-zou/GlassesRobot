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
FRU_TO_RDF = RDF_TO_FRU.T
FRU_TO_RDF_H = np.eye(4, dtype=np.float32)
FRU_TO_RDF_H[:3, :3] = FRU_TO_RDF


def _rdf_to_fru_transform(T_rdf: np.ndarray) -> np.ndarray:
    """Change-of-basis from ZED's default RDF convention into FRU."""
    return (RDF_TO_FRU_H @ T_rdf).astype(np.float32)


def _rdf_to_fru_points(xyz_rdf: np.ndarray) -> np.ndarray:
    return (RDF_TO_FRU @ xyz_rdf.T).T.astype(np.float32)


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
    parser = argparse.ArgumentParser(description="Visualize ZED/Aruco/Base frames in Rerun")
    parser.add_argument("--spawn", action="store_true", help="Spawn Rerun viewer window")
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

    T_zed_aruco_rdf = _load_transform(args.T_zed_aruco)
    T_zed_aruco = _rdf_to_fru_transform(T_zed_aruco_rdf)
    T_base_aruco = _load_transform(args.T_base_aruco)

    # Convert ZED data into FRU first, then apply the rest of the transform chain.
    T_origin = np.eye(4, dtype=np.float32)
    T_origin_aruco = T_zed_aruco.astype(np.float32)
    T_zed_base = T_zed_aruco @ np.linalg.inv(T_base_aruco)

    import rerun as rr

    rr.init("Frame Visualization", spawn=args.spawn)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    _log_frame(rr, "origin", T_origin, args.axis_len)
    _log_frame(rr, "aruco", T_origin_aruco, args.axis_len)
    _log_frame(rr, "base", T_zed_base.astype(np.float32), args.axis_len)

    arrow_radius = args.axis_len * 0.03
    translation_aruco = T_origin_aruco[:3, 3].astype(np.float32)
    translation_base = T_zed_base[:3, 3].astype(np.float32)
    rr.log(
        "frames/origin/aruco_offset",
        rr.Arrows3D(
            origins=np.zeros((1, 3), dtype=np.float32),
            vectors=translation_aruco[None, :],
            colors=np.array([[255, 255, 0, 255]], dtype=np.uint8),
            radii=np.full(1, arrow_radius, dtype=np.float32),
        ),
    )
    rr.log(
        "frames/origin/base_offset",
        rr.Arrows3D(
            origins=np.zeros((1, 3), dtype=np.float32),
            vectors=translation_base[None, :],
            colors=np.array([[0, 255, 255, 255]], dtype=np.uint8),
            radii=np.full(1, arrow_radius, dtype=np.float32),
        ),
    )


    traj_path = args.traj
    if traj_path and traj_path.exists():
        ids, xyz_abs_zed = _load_traj(traj_path)
        xyz_abs_fru = _rdf_to_fru_points(xyz_abs_zed)

        rr.set_time("frame", sequence=int(ids[0]))
        rr.log("traj_abs/origin/path", rr.LineStrips3D([xyz_abs_fru.astype(np.float32)]))
        rr.log("traj_abs/origin/points", rr.Points3D(xyz_abs_fru.astype(np.float32)))
    else:
        print(f"[WARN] Trajectory file not found: {traj_path}")


if __name__ == "__main__":
    main()
