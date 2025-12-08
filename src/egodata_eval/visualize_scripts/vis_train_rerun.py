#!/usr/bin/env python3
"""
Visualize precomputed training trajectories (object poses) using rerun.

Consumes the temp data produced by vis_train.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Temp file not found: {path}")
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected payload format in {path}")
    return data


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
            vectors=(np.eye(3, dtype=np.float32) * scale),
            colors=np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8),
            radii=np.full(3, scale * 0.2, dtype=np.float32),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize training trajectories from a temp file using rerun.")
    parser.add_argument("--temp-file", type=Path, required=True, help="Temp .npy produced by vis_train.py.")
    parser.add_argument("--spawn", action="store_true", help="Spawn rerun viewer window.")
    args = parser.parse_args()

    payload = _load_payload(args.temp_file)
    axis_len = float(payload.get("axis_len", 0.25))
    T_robot_base = np.asarray(payload["T_robot_base"], dtype=np.float32)
    sequences: List[Dict[str, Any]] = payload.get("sequences", [])
    data_root = payload.get("data_root", args.temp_file.parent)

    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("rerun package not found. Install with `pip install rerun-sdk`.") from exc

    rr.init(f"Training Traj Visualization ({Path(data_root).name})", spawn=args.spawn)
    try:
        rr.log("world", rr.ViewCoordinates.FRU)
    except Exception:
        pass

    T_base_robot = np.linalg.inv(T_robot_base)
    _log_axis(rr, "frames/robot", np.eye(4, dtype=np.float32), axis_len * 1.2)
    _log_axis(rr, "frames/base", T_base_robot, axis_len)

    for seq in sequences:
        seq_name = seq.get("seq", "unknown")
        pts_robot = np.asarray(seq.get("pts_robot", []), dtype=np.float32)
        pts_base = np.asarray(seq.get("pts_base", []), dtype=np.float32)
        if pts_robot.size == 0:
            continue
        seq_path = f"sequences/{seq_name}"
        if pts_robot.shape[0] >= 2:
            rr.log(f"{seq_path}/traj_robot", rr.LineStrips3D([pts_robot]))
            rr.log(f"{seq_path}/traj_base", rr.LineStrips3D([pts_base]))
        else:
            rr.log(f"{seq_path}/traj_robot", rr.Points3D(pts_robot))
            rr.log(f"{seq_path}/traj_base", rr.Points3D(pts_base))
    print(f"[OK] Visualized {len(sequences)} sequences from {data_root}")


if __name__ == "__main__":
    main()
