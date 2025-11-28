#!/usr/bin/env python3
"""
Visualize T_zed_tcp.npy with Rerun.

Conventions:
- ZED is the root frame and uses OpenCV/ZED RDF coordinates.
- T_zed_tcp is the 4x4 transform zed -> tcp (homogeneous).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _import_rerun():
    try:
        import rerun as rr  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Please install rerun-sdk: pip install rerun-sdk") from exc
    return rr


def _load_transform(path: Path) -> np.ndarray:
    T = np.load(str(path))
    if T.shape != (4, 4):
        raise ValueError(f"{path} is not a 4x4 transform; got {T.shape}")
    return T.astype(np.float32)


def _log_frame(rr, name: str, T: np.ndarray, axis_len: float) -> None:
    """Log a coordinate frame plus RGB axes."""
    rr.log(
        f"frames/{name}",
        rr.Transform3D(
            translation=T[:3, 3],
            mat3x3=T[:3, :3],
        ),
    )
    origins = np.zeros((3, 3), dtype=np.float32)
    vectors = np.eye(3, dtype=np.float32) * axis_len
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize T_zed_tcp.npy with Rerun")
    parser.add_argument(
        "--T_zed_tcp",
        type=Path,
        default=Path(__file__).resolve().parent / "T_tcp_zed.npy",
        help="Path to the 4x4 transform zed -> tcp",
    )
    parser.add_argument("--axis-len", type=float, default=0.2, help="Axis length in meters")
    parser.add_argument("--spawn", action="store_true", help="Spawn a separate Rerun Viewer window")
    args = parser.parse_args()

    T_zed_tcp = _load_transform(args.T_zed_tcp)
    T_tcp_zed = np.linalg.inv(T_zed_tcp)

    print(f"[Check] zed -> tcp translation: {T_zed_tcp[:3, 3]}")
    print(f"[Check] tcp -> zed translation: {T_tcp_zed[:3, 3]}")
    print(f"[Check] camera to TCP distance: {np.linalg.norm(T_zed_tcp[:3, 3]):.3f} m")

    rr = _import_rerun()
    rr.init("ZED <-> TCP", spawn=args.spawn)
    rr.log("world", rr.ViewCoordinates.RDF)  # matches ZED/OpenCV

    _log_frame(rr, "tcp", np.eye(4, dtype=np.float32), args.axis_len)
    _log_frame(rr, "zed", T_zed_tcp, args.axis_len)

    # Visualize the translation vector for quick distance/direction sanity checks.
    rr.log(
        "frames/zed/tcp_offset",
        rr.Arrows3D(
            origins=np.zeros((1, 3), dtype=np.float32),
            vectors=T_zed_tcp[:3, 3][None, :],
            colors=np.array([[255, 255, 0, 255]], dtype=np.uint8),
            radii=np.full(1, args.axis_len * 0.04, dtype=np.float32),
        ),
    )


if __name__ == "__main__":
    main()
