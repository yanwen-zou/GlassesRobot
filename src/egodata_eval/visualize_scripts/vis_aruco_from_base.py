#!/usr/bin/env python3
"""
Load a single transform T_robot_base (4x4) and visualize it with rerun.

Interpretation:
    - Input matrix maps points from base frame to robot frame.
    - We display both frames: base at the origin, and the robot frame placed by T_robot_base.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_transform(path: Path) -> np.ndarray:
    T = np.load(path)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 SE3 matrix at {path}, got {T.shape}")
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
    parser = argparse.ArgumentParser(description="Visualize T_robot_base with rerun (base at origin).")
    parser.add_argument(
        "--T_robot_base",
        type=Path,
        default=Path("glasses_hardware/calib/T_robot_base.npy"),
        help="4x4 SE3 mapping base frame -> robot frame.",
    )
    parser.add_argument("--axis_len", type=float, default=0.2, help="Axis length in meters")
    parser.add_argument("--spawn", action="store_true", help="Spawn a standalone Rerun viewer window")
    args = parser.parse_args()

    T_robot_base = _load_transform(args.T_robot_base)
    T_base = np.eye(4, dtype=np.float32)

    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("Rerun package is required. Install with `pip install rerun-sdk`.") from exc

    rr.init("T_robot_base Visualization", spawn=args.spawn)
    try:
        rr.log("world", rr.ViewCoordinates.FRU)
    except Exception:
        pass

    _log_frame(rr, "base", T_base, args.axis_len)
    _log_frame(rr, "robot", T_robot_base, args.axis_len)

    # Arrow from base to robot origin for quick spatial cue
    rr.log(
        "frames/base/robot_offset",
        rr.Arrows3D(
            origins=np.zeros((1, 3), dtype=np.float32),
            vectors=T_robot_base[:3, 3][None, :].astype(np.float32),
            colors=np.array([[255, 255, 0, 255]], dtype=np.uint8),
            radii=np.full(1, args.axis_len * 0.03, dtype=np.float32),
        ),
    )

    print(f"[OK] Loaded {args.T_robot_base} and logged frames 'base' and 'robot'.")


if __name__ == "__main__":
    main()
