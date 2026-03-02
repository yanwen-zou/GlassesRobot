#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_transform(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        return np.vstack([arr, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)])
    raise ValueError(f"Unsupported matrix shape {arr.shape} in {path}")


def _log_axis(rr, path: str, T: np.ndarray, scale: float) -> None:
    rr.log(
        path,
        rr.Transform3D(
            translation=T[:3, 3],
            mat3x3=T[:3, :3],
        ),
    )
    rr.log(
        f"{path}/axes",
        rr.Arrows3D(
            origins=np.zeros((3, 3), dtype=np.float32),
            vectors=np.eye(3, dtype=np.float32) * scale,
            colors=np.array(
                [
                    [255, 0, 0, 255],   # x
                    [0, 255, 0, 255],   # y
                    [0, 0, 255, 255],   # z
                ],
                dtype=np.uint8,
            ),
            radii=np.full(3, scale * 0.05, dtype=np.float32),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize T_i2rt_zed and T_glasses_zed in rerun with zed as base frame."
    )
    parser.add_argument(
        "--T-i2rt-zed",
        type=Path,
        default=Path("glasses_hardware/calib/T_i2rt_zed.txt"),
        help="Path to T_i2rt_zed.txt",
    )
    parser.add_argument(
        "--T-glasses-zed",
        type=Path,
        default=Path("glasses_hardware/calib/T_glasses_zed.txt"),
        help="Path to T_glasses_zed.txt",
    )
    parser.add_argument("--axis-len", type=float, default=0.2, help="Axis length for frame arrows.")
    parser.add_argument("--spawn", action="store_true", help="Spawn rerun viewer.")
    args = parser.parse_args()

    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("rerun package required. Install with `pip install rerun-sdk`.") from exc

    T_i2rt_zed = _load_transform(args.T_i2rt_zed.resolve())
    T_glasses_zed = _load_transform(args.T_glasses_zed.resolve())

    rr.init("ZED Calibration Frames", spawn=args.spawn)
    rr.log("world", rr.ViewCoordinates.FRU)

    T_zed = np.eye(4, dtype=np.float32)
    _log_axis(rr, "zed", T_zed, args.axis_len)
    _log_axis(rr, "zed/i2rt", np.linalg.inv(T_i2rt_zed), args.axis_len * 0.9)
    _log_axis(rr, "zed/glasses", np.linalg.inv(T_glasses_zed), args.axis_len * 0.9)

    print(f"[OK] zed base frame at identity")
    print(f"[OK] loaded: {args.T_i2rt_zed.resolve()}")
    print(f"[OK] loaded: {args.T_glasses_zed.resolve()}")
    print("[OK] logged frames: zed, zed/i2rt, zed/glasses")


if __name__ == "__main__":
    main()

