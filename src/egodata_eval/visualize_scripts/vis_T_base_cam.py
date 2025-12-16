#!/usr/bin/env python3
"""Visualize per-frame T_base_cam transforms recorded by eval_dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_transforms(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"Expected [N,4,4] array in {path}, got {arr.shape}")
    return arr.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize T_base_cam sequence with rerun.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory produced by eval_dataset.py")
    parser.add_argument("--axis-len", type=float, default=0.5, help="Axis length for frames.")
    parser.add_argument("--spawn", action="store_true", help="Spawn rerun viewer window.")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    T_path = data_dir / "T_base_cam.npy"
    if not T_path.exists():
        raise FileNotFoundError(f"{T_path} not found; run eval_dataset.py with latest changes.")
    transforms = load_transforms(T_path)

    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("rerun package required: pip install rerun-sdk") from exc

    rr.init(f"T_base_cam ({data_dir.name})", spawn=args.spawn)
    try:
        rr.log("world", rr.ViewCoordinates.FRU)
    except Exception:
        pass

    for idx, T in enumerate(transforms):
        rr.set_time_sequence("frame", idx)
        rr.log(
            f"T_base_cam/frame_{idx}",
            rr.Transform3D(
                translation=T[:3, 3],
                mat3x3=T[:3, :3],
            ),
        )
        rr.log(
            f"T_base_cam/frame_{idx}/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3), dtype=np.float32),
                vectors=np.eye(3, dtype=np.float32) * args.axis_len,
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
    print(f"[OK] Visualized {transforms.shape[0]} T_base_cam frames from {data_dir}")


if __name__ == "__main__":
    main()
