#!/usr/bin/env python3
"""
Visualize evaluation logs (object pose records, executed poses, TCP history) using rerun.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    parser.add_argument(
        "--ball-centroids",
        type=Path,
        default=None,
        help="Optional path to ball_centroids_*.txt for visualizing calibration balls.",
    )
    parser.add_argument("--spawn", action="store_true", help="Spawn rerun viewer.")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    pose_records = load_records(data_dir / "robot_pose_records.npy")
    executed = load_array(data_dir / "robot_executed_poses.npy")
    tcp_hist = load_array(data_dir / "robot_tcp_history.npy")

    T_robot_base = np.load(args.T_robot_base).astype(np.float32)
    runtime_cam_path = data_dir / "T_base_cam_runtime.npy"
    T_base_cam_seq = load_array(runtime_cam_path)
    if T_base_cam_seq is None:
        T_base_cam_seq = load_array(args.T_base_cam)

    def _load_ball_centroids(path: Optional[Path], search_root: Path) -> tuple[np.ndarray | None, Optional[Path]]:
        search_path = path
        if search_path is None:
            candidates = sorted(search_root.glob("ball_centroids_*.txt"))
            if candidates:
                search_path = candidates[-1]
        if search_path is None or not search_path.exists():
            return None, None
        points: list[list[float]] = []
        with open(search_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.lower().startswith("ball_id"):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                except ValueError:
                    continue
                points.append(coords)
        if not points:
            return None, None
        return np.array(points, dtype=np.float32), search_path

    ball_centroids, centroid_source_path = _load_ball_centroids(args.ball_centroids, data_dir)

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
                radii=np.full(3, scale * 0.05, dtype=np.float32),
            ),
        )

    def _cam_transform_for_frame(idx: int) -> np.ndarray | None:
        return T_base_cam_seq

    def _transform_points_cam_to_robot(points_cam: np.ndarray) -> np.ndarray | None:
        if points_cam is None or T_base_cam_seq is None:
            return None
        homog = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float32)], axis=1)
        pts_base = (T_base_cam_seq @ homog.T).T
        pts_robot = (T_robot_base @ pts_base.T).T
        return pts_robot[:, :3]

    ball_centroids_robot = _transform_points_cam_to_robot(ball_centroids) if ball_centroids is not None else None

    if ball_centroids_robot is not None:
        rr.log(
            "calibration/ball_centroids",
            rr.Points3D(
                positions=ball_centroids_robot,
                colors=np.array([[255, 105, 180, 255]] * ball_centroids_robot.shape[0], dtype=np.uint8),
                radii=np.full(ball_centroids_robot.shape[0], args.axis_len * 0.06, dtype=np.float32),
            ),
        )
        if centroid_source_path is not None:
            print(f"[INFO] Visualizing ball centroids from {centroid_source_path}")

    for rec in pose_records:
        frame_idx = rec.get("frame_idx", -1)
        pose_robot = rec.get("object_pose_robot")
        pred_seq = rec.get("pred_seq_robot")
        if pose_robot is None:
            continue
        pose_robot = np.asarray(pose_robot, dtype=np.float32).reshape(4, 4)
        obj_position = pose_robot[:3, 3]
        rr.log(
            f"frames/frame_{frame_idx}/object_pose",
            rr.Points3D(
                positions=obj_position.reshape(1, 3),
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
        print(f"cam_tf: {cam_tf}")
        if cam_tf is not None:
            cam_world = T_robot_base @ cam_tf
            log_axis(f"frames/frame_{frame_idx}/robot_cam", cam_world, args.axis_len * 0.4)
            cam_position = cam_world[:3, 3]
            rr.log(
                f"frames/frame_{frame_idx}/robot_base_to_cam",
                rr.Arrows3D(
                    origins=T_robot_base[:3, 3].reshape(1, 3),
                    vectors=(cam_position - T_robot_base[:3, 3]).reshape(1, 3),
                    colors=np.array([[0, 191, 255, 255]], dtype=np.uint8),
                    radii=np.full(1, args.axis_len * 0.02, dtype=np.float32),
                ),
            )
            rr.log(
                f"frames/frame_{frame_idx}/object_to_cam",
                rr.Arrows3D(
                    origins=obj_position.reshape(1, 3),
                    vectors=(cam_position - obj_position).reshape(1, 3),
                    colors=np.array([[255, 140, 0, 255]], dtype=np.uint8),
                    radii=np.full(1, args.axis_len * 0.05, dtype=np.float32),
                ),
            )

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
