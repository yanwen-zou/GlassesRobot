#!/usr/bin/env python3
"""Visualize i2rt as base frame with i2rt->glasses and i2rt->zed transforms."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _import_rerun():
    try:
        import rerun as rr  # type: ignore
    except Exception as exc:
        raise RuntimeError("Rerun is required for visualization. Install with `pip install rerun-sdk`.") from exc
    return rr


def _load_se3(path: Path) -> np.ndarray:
    mat = np.loadtxt(path, dtype=np.float32)
    if mat.shape == (3, 4):
        mat = np.vstack([mat, np.array([0, 0, 0, 1], dtype=np.float32)])
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 SE3 in {path}, got {mat.shape}")
    return mat


def _rot_to_quat(rot: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to (x, y, z, w) quaternion."""
    r = rot.astype(np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    else:
        if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = (1.0 + r[0, 0] - r[1, 1] - r[2, 2]) ** 0.5 * 2.0
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = (1.0 + r[1, 1] - r[0, 0] - r[2, 2]) ** 0.5 * 2.0
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = (1.0 + r[2, 2] - r[0, 0] - r[1, 1]) ** 0.5 * 2.0
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def _log_axes(rr, path: str, T: np.ndarray, length: float = 0.05) -> None:
    origin = T[:3, 3].astype(np.float32)
    rot = T[:3, :3].astype(np.float32)
    axes = rot @ (length * np.eye(3, dtype=np.float32))
    rr.log(
        path,
        rr.Arrows3D(
            origins=np.repeat(origin[None, :], 3, axis=0),
            vectors=axes.T,
            colors=np.array(
                [
                    [255, 0, 0, 255],
                    [0, 255, 0, 255],
                    [0, 0, 255, 255],
                ],
                dtype=np.uint8,
            ),
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize i2rt base with i2rt->glasses and i2rt->zed frames.")
    ap.add_argument(
        "--calib-dir",
        type=Path,
        default=Path("glasses_hardware/calib"),
        help="Calibration directory containing T_i2rt_glasses.txt and T_i2rt_zed.txt.",
    )
    ap.add_argument(
        "--i2rt-glasses",
        type=Path,
        default=None,
        help="Override path to T_i2rt_glasses.txt.",
    )
    ap.add_argument(
        "--i2rt-zed",
        type=Path,
        default=None,
        help="Override path to T_i2rt_zed.txt.",
    )
    args = ap.parse_args()

    i2rt_glasses = args.i2rt_glasses or (args.calib_dir / "T_i2rt_glasses.txt")
    i2rt_zed = args.i2rt_zed or (args.calib_dir / "T_i2rt_zed.txt")

    T_i2rt_glasses = _load_se3(i2rt_glasses)
    T_i2rt_zed = _load_se3(i2rt_zed)

    rr = _import_rerun()
    rr.init("vis_i2rt_frames", spawn=True)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    rr.log(
        "world/i2rt",
        rr.Transform3D(
            translation=[0.0, 0.0, 0.0],
            rotation=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0]),
        ),
    )
    _log_axes(rr, "world/i2rt/axes", np.eye(4, dtype=np.float32))
    rr.log(
        "world/i2rt/glasses",
        rr.Transform3D(
            translation=T_i2rt_glasses[:3, 3],
            rotation=rr.Quaternion(xyzw=_rot_to_quat(T_i2rt_glasses[:3, :3])),
        ),
    )
    _log_axes(rr, "world/i2rt/glasses/axes", T_i2rt_glasses)
    rr.log(
        "world/i2rt/zed",
        rr.Transform3D(
            translation=T_i2rt_zed[:3, 3],
            rotation=rr.Quaternion(xyzw=_rot_to_quat(T_i2rt_zed[:3, :3])),
        ),
    )
    _log_axes(rr, "world/i2rt/zed/axes", T_i2rt_zed)


if __name__ == "__main__":
    main()
