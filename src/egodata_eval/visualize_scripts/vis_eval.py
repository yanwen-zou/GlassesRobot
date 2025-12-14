#!/usr/bin/env python3
"""
Visualize evaluation logs (object pose records, executed poses, TCP history) using rerun.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    arr = np.load(path, allow_pickle=True)
    return [dict(item) for item in arr]


def load_array(path: Path) -> np.ndarray | None:
    if not path or not path.exists():
        return None
    try:
        return np.load(path).astype(np.float32)
    except Exception as exc:
        print(f"[WARN] Failed to load {path}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Visualize eval logs in rerun.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing eval outputs.")
    parser.add_argument(
        "--T_robot_base",
        type=Path,
        default=Path("glasses_hardware/calib/T_robot_base.npy"),
        help="Path to T_robot_base.npy (base->robot).",
    )
    parser.add_argument(
        "--T_base_cam",
        type=Path,
        default=Path("glasses_hardware/calib/T_base_cam_runtime.npy"),
        help="Fallback path to T_base_cam.npy if per-episode file is missing.",
    )
    parser.add_argument("--axis_len", type=float, default=0.2, help="Axis length for frames.")
    parser.add_argument("--spawn", action="store_true", help="Spawn rerun viewer.")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    pose_records = load_records(data_dir / "robot_pose_records.npy")
    executed = load_array(data_dir / "robot_executed_poses.npy")
    tcp_hist = load_array(data_dir / "robot_tcp_history.npy")

    T_robot_base = np.load(args.T_robot_base).astype(np.float32)
    runtime_cam_path = data_dir / "T_base_cam_runtime.npy"
    if runtime_cam_path.exists():
        T_base_cam_seq = load_array(runtime_cam_path)
    else:
        T_base_cam_seq = load_array(args.T_base_cam)

    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("rerun package required. `pip install rerun-sdk`.") from exc

    rr.init(f"Eval Visualization ({data_dir.name})", spawn=args.spawn)
    try:
        rr.log("world", rr.ViewCoordinates.FRU)
    except Exception:
        pass

    def log_axis(path: str, T: np.ndarray, scale: float) -> None:
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

    def _cam_transform_for_frame(idx: int) -> np.ndarray | None:
        if T_base_cam_seq is None:
            return None
        if T_base_cam_seq.ndim == 2:
            return T_base_cam_seq
        if idx < T_base_cam_seq.shape[0]:
            return T_base_cam_seq[idx]
        return T_base_cam_seq[-1]

    for rec in pose_records:
        frame_idx = rec.get("frame_idx", -1)
        pose_robot = rec.get("object_pose_robot")
        pred_seq = rec.get("pred_seq_robot")
        if pose_robot is None:
            continue
        pose_robot = np.asarray(pose_robot, dtype=np.float32).reshape(4, 4)
        rr.log(
            f"frames/frame_{frame_idx}/object_pose",
            rr.Points3D(
                positions=pose_robot[:3, 3],
                colors=np.array([0, 255, 0, 255], dtype=np.uint8),
                radii=args.axis_len * 0.05,
            ),
        )
        if pred_seq is not None:
            pred_seq = np.asarray(pred_seq, dtype=np.float32)
            rr.log(
                f"frames/frame_{frame_idx}/pred_points",
                rr.Points3D(
                    positions=pred_seq[:, :3, 3],
                    colors=np.array([[255, 255, 0, 255]], dtype=np.uint8),
                    radii=np.full(pred_seq.shape[0], args.axis_len * 0.03, dtype=np.float32),
                ),
            )
        log_axis(f"frames/frame_{frame_idx}/robot_base", T_robot_base, args.axis_len * 0.5)
        cam_tf = _cam_transform_for_frame(frame_idx)
        if cam_tf is not None:
            log_axis(f"frames/frame_{frame_idx}/base_cam", (T_robot_base @ cam_tf), args.axis_len * 0.4)

    if executed is not None and executed.size > 0:
        rr.log(
            "executed/points",
            rr.Points3D(
                positions=executed[:, :3],
                colors=np.array([[0, 200, 255, 255]], dtype=np.uint8),
                radii=np.full(executed.shape[0], args.axis_len * 0.04, dtype=np.float32),
            ),
        )
    if tcp_hist is not None and tcp_hist.size > 0:
        rr.log(
            "tcp_history/points",
            rr.Points3D(
                positions=tcp_hist[:, :3],
                colors=np.array([[255, 0, 255, 255]], dtype=np.uint8),
                radii=np.full(tcp_hist.shape[0], args.axis_len * 0.04, dtype=np.float32),
            ),
        )

    print(f"[OK] Visualized {len(pose_records)} pose records from {data_dir}")


if __name__ == "__main__":
    main()
