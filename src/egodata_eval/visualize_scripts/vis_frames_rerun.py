#!/usr/bin/env python3
"""
Visualize ZED, ArUco, and robot base coordinate frames in Rerun.
"""
from __future__ import annotations

import argparse
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="Visualize ZED/Aruco/Base frames in Rerun")
    parser.add_argument("--spawn", action="store_true", help="Spawn Rerun viewer window")
    parser.add_argument("--axis_len", type=float, default=0.2, help="Axis length in meters")
    parser.add_argument("--T_zed_aruco", type=Path, default=Path("T_zed_aruco.npy"))
    parser.add_argument(
        "--T_base_aruco",
        type=Path,
        default=Path("glasses_hardware/calib/T_base_aruco.npy"),
    )
    args = parser.parse_args()

    T_zed_aruco = _load_transform(args.T_zed_aruco)
    T_base_aruco = _load_transform(args.T_base_aruco)

    # Base frame is the origin; express other frames relative to base.
    T_base = np.eye(4, dtype=np.float32)
    T_aruco = T_base_aruco.astype(np.float32)
    T_base_zed = T_base_aruco @ np.linalg.inv(T_zed_aruco)
    T_zed = T_base_zed.astype(np.float32)

    try:
        import rerun as rr
    except Exception as e:
        raise RuntimeError("Rerun package is required. Install with `pip install rerun-sdk`.") from e

    rr.init("Frame Visualization", spawn=args.spawn)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    _log_frame(rr, "zed", T_zed, args.axis_len)
    _log_frame(rr, "aruco", T_aruco, args.axis_len)
    _log_frame(rr, "base", T_base, args.axis_len)

if __name__ == "__main__":
    main()
