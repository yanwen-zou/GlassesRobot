#!/usr/bin/env python3
"""Visualize saved base transforms with axis arrows using Rerun."""

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


def _log_axes(rr, path: str, length: float) -> None:
    axes = length * np.eye(3, dtype=np.float32)
    rr.log(
        path,
        rr.Arrows3D(
            origins=np.zeros((3, 3), dtype=np.float32),
            vectors=axes,
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


def _log_frame(rr, name: str, T: np.ndarray, length: float) -> None:
    rr.log(
        name,
        rr.Transform3D(
            translation=T[:3, 3],
            rotation=rr.Quaternion(xyzw=_rot_to_quat(T[:3, :3])),
        ),
    )
    _log_axes(rr, f"{name}/axes", length)


def _log_origin_arrow(rr, name: str, T: np.ndarray) -> None:
    origin = np.zeros((1, 3), dtype=np.float32)
    target = T[:3, 3].astype(np.float32)
    rr.log(
        name,
        rr.Arrows3D(
            origins=origin,
            vectors=target[None, :],
            colors=np.array([[255, 255, 0, 255]], dtype=np.uint8),
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize base-frame transforms saved in test_i2rt_calib.")
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing T_base_cam.txt/T_base_tcp.txt/T_base_glasses.txt/T_base_i2rt.txt/T_i2rt_tcp.txt.",
    )
    ap.add_argument("--axis-length", type=float, default=0.05, help="Axis arrow length in meters.")
    args = ap.parse_args()

    base_dir = args.dir
    T_base_cam = _load_se3(base_dir / "T_base_cam.txt")
    T_base_tcp = _load_se3(base_dir / "T_base_tcp.txt")
    T_base_glasses = _load_se3(base_dir / "T_base_glasses.txt")
    T_base_i2rt = _load_se3(base_dir / "T_base_i2rt.txt")
    T_i2rt_tcp = _load_se3(base_dir / "T_i2rt_tcp.txt")

    rr = _import_rerun()
    rr.init("test_i2rt_calib_visualize", spawn=True)
    try:
        rr.log("world", rr.ViewCoordinates.FRU)
    except Exception:
        pass

    T_base = np.eye(4, dtype=np.float32)
    _log_frame(rr, "world/base", T_base, args.axis_length)
    _log_frame(rr, "world/base/cam", T_base_cam, args.axis_length)
    _log_frame(rr, "world/base/tcp", T_base_tcp, args.axis_length)
    _log_frame(rr, "world/base/glasses", T_base_glasses, args.axis_length)
    _log_frame(rr, "world/base/i2rt", T_base_i2rt, args.axis_length)
    _log_frame(rr, "world/i2rt/tcp", T_i2rt_tcp, args.axis_length)
    _log_origin_arrow(rr, "world/base/arrow_i2rt_tcp", T_i2rt_tcp)
    # _log_origin_arrow(rr, "world/base/arrow_cam", T_base_cam)
    # _log_origin_arrow(rr, "world/base/arrow_tcp", T_base_tcp)
    # _log_origin_arrow(rr, "world/base/arrow_glasses", T_base_glasses)
    # _log_origin_arrow(rr, "world/base/arrow_i2rt", T_base_i2rt)


if __name__ == "__main__":
    main()
