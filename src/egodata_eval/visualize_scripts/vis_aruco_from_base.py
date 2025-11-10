#!/usr/bin/env python3
"""
Visualize the ArUco marker pose relative to the robot base frame.

This script assumes you already have calibration files produced via `piper_calib`:
  - `T_base_aruco.npy`: 4x4 SE3 transform (base <- aruco)

It logs the base frame as the world origin and displays the ArUco frame plus a
translation arrow that highlights the marker offset in base coordinates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

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


def _load_transform(path: Path) -> np.ndarray:
    T = np.load(path)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 SE3 matrix at {path}, got {T.shape}")
    return T.astype(np.float32)


def _rdf_to_fru_transform(T_rdf: np.ndarray) -> np.ndarray:
    return (RDF_TO_FRU_H @ T_rdf @ np.linalg.inv(RDF_TO_FRU_H)).astype(np.float32)


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


def main():
    parser = argparse.ArgumentParser(description="Visualize ArUco pose with the robot base as origin")
    parser.add_argument("--spawn", action="store_true", help="Spawn a standalone Rerun viewer window")
    parser.add_argument("--axis_len", type=float, default=0.2, help="Axis length in meters")
    parser.add_argument(
        "--T_base_aruco",
        type=Path,
        default=Path("glasses_hardware/calib/T_base_aruco.npy"),
        help="Cached 4x4 transform from ArUco to robot base",
    )
    parser.add_argument(
        "--T_zed_aruco",
        type=Path,
        default=Path("T_zed_aruco.npy"),
        help="Cached 4x4 transform from ArUco to ZED",
    )
    args = parser.parse_args()

    T_base_aruco = _load_transform(args.T_base_aruco)
    T_zed_aruco_rdf = _load_transform(args.T_zed_aruco)
    T_zed_aruco_fru = _rdf_to_fru_transform(T_zed_aruco_rdf)

    T_zed_aruco = (T_zed_aruco_fru).astype(np.float32)
    T_base_zed = T_base_aruco @ np.linalg.inv(T_zed_aruco)
    T_base = np.eye(4, dtype=np.float32)

    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("Rerun package is required. Install with `pip install rerun-sdk`.") from exc

    rr.init("Base/Aruco Frames", spawn=args.spawn)
    try:
        rr.log("world", rr.ViewCoordinates.FRU)
    except Exception:
        pass

    _log_frame(rr, "base_origin", T_base, args.axis_len)
    _log_frame(rr, "aruco", T_base_aruco, args.axis_len)
    _log_frame(rr, "zed", T_base_zed.astype(np.float32), args.axis_len)

    translation = T_base_aruco[:3, 3].astype(np.float32)
    rr.log(
        "frames/base_origin/aruco_offset",
        rr.Arrows3D(
            origins=np.zeros((1, 3), dtype=np.float32),
            vectors=translation[None, :],
            colors=np.array([[255, 255, 0, 255]], dtype=np.uint8),
            radii=np.full(1, args.axis_len * 0.03, dtype=np.float32),
        ),
    )

    rr.log(
        "frames/base_origin/zed_offset",
        rr.Arrows3D(
            origins=np.zeros((1, 3), dtype=np.float32),
            vectors=T_base_zed[:3, 3][None, :].astype(np.float32),
            colors=np.array([[0, 255, 255, 255]], dtype=np.uint8),
            radii=np.full(1, args.axis_len * 0.03, dtype=np.float32),
        ),
    )

    print("[INFO] Logged base origin, ZED, and ArUco frames to Rerun.")


if __name__ == "__main__":
    main()
