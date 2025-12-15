#!/usr/bin/env python3
"""Simple rerun visualization for a single T_base_cam transform."""

import argparse
import numpy as np
from pathlib import Path

def log_frame(rr, name: str, T: np.ndarray, axis_len: float = 0.2) -> None:
    """Log a coordinate frame with explicit arrows at the transform origin."""
    origins = np.repeat(T[np.newaxis, :3, 3], 3, axis=0)
    axes = T[:3, :3] @ (np.eye(3, dtype=np.float32) * axis_len)
    rr.log(
        f"{name}/axes",
        rr.Arrows3D(
            origins=origins,
            vectors=axes.T,
            colors=np.array(
                [
                    [255, 0, 0, 255],
                    [0, 255, 0, 255],
                    [0, 0, 255, 255],
                ],
                dtype=np.uint8,
            ),
            radii=np.full(3, axis_len * 0.2, dtype=np.float32),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize a sample T_base_cam transform in rerun.")
    parser.add_argument("--spawn", action="store_true", help="Spawn rerun viewer.")
    parser.add_argument("--axis-len", type=float, default=0.2, help="Axis length for frames.")
    parser.add_argument(
        "--T_robot_base",
        type=Path,
        default=Path("glasses_hardware/calib/T_robot_base.npy"),
        help="Path to T_robot_base.npy (base->robot).",
    )
    args = parser.parse_args()

    T_base_cam = np.array(
        [
            [0.38992298, 0.21825397, -0.8946090, 0.880444],
            [0.9175130, -0.00948099, 0.3975930, -0.315775],
            [0.078295, -0.975846, -0.203948, 0.287336],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("rerun package required. `pip install rerun-sdk`.") from exc

    rr.init("vis_test T_base_cam", spawn=args.spawn)
    try:
        rr.log("world", rr.ViewCoordinates.FRU)
    except Exception:
        pass

    T_robot_base = np.load(args.T_robot_base).astype(np.float32)
    log_frame(rr, "robot_base", T_robot_base, axis_len=args.axis_len)
    log_frame(rr, "base_cam", T_robot_base @ T_base_cam, axis_len=args.axis_len)
    print("Logged provided T_base_cam transform. Open rerun to inspect.")


if __name__ == "__main__":
    main()
