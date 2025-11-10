#!/usr/bin/env python3
"""Visualize the eih_camT transform stored in eih_camT.npy.bak."""

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
    """Log a coordinate frame with RGB axes into Rerun."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize eih_camT.npy.bak transform with Rerun.")
    parser.add_argument("--spawn", action="store_true", help="Spawn a standalone Rerun viewer window")
    parser.add_argument("--axis_len", type=float, default=0.2, help="Axis length in meters")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("glasses_hardware/calib/eih_camT.npy.bak"),
        help="Path to eih_camT.npy.bak (4x4 SE3)",
    )
    args = parser.parse_args()

    T_eih_cam = _load_transform(args.path)
    print(f"[INFO] Loaded transform from {args.path}:\n{T_eih_cam}")

    T_world = np.eye(4, dtype=np.float32)

    try:
        import rerun as rr
    except Exception as exc:  # pragma: no cover - visualization helper
        raise RuntimeError("Rerun package is required. Install with `pip install rerun-sdk`.") from exc

    rr.init("eih_camT Visualization", spawn=args.spawn)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    _log_frame(rr, "world", T_world, args.axis_len)
    _log_frame(rr, "eih_cam", T_eih_cam, args.axis_len)

    translation = T_eih_cam[:3, 3].astype(np.float32)
    rr.log(
        "frames/world/eih_cam_offset",
        rr.Arrows3D(
            origins=np.zeros((1, 3), dtype=np.float32),
            vectors=translation[None, :],
            colors=np.array([[255, 255, 0, 255]], dtype=np.uint8),
            radii=np.full(1, args.axis_len * 0.03, dtype=np.float32),
        ),
    )

    print("[INFO] Logged world and eih_cam frames to Rerun.")


if __name__ == "__main__":
    main()
