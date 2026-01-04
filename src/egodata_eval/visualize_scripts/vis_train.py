#!/usr/bin/env python3
"""
Visualize evaluation logs (object pose records, executed poses, TCP history) using rerun.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

import numpy as np

here = Path(__file__).resolve()
project_root = here.parents[3]
src_root = project_root / "src"
mba_root = project_root / "MBA"
for path in (src_root, mba_root, project_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

def _rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - proj * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _build_pose_mats(translation: np.ndarray, rotation_6d: np.ndarray) -> np.ndarray:
    mats = np.repeat(np.eye(4, dtype=np.float32)[None, ...], len(translation), axis=0)
    rot_mats = _rotation_6d_to_matrix(rotation_6d).astype(np.float32)
    mats[:, :3, :3] = rot_mats
    mats[:, :3, 3] = translation
    return mats


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    arr = np.load(path, allow_pickle=True)
    return [dict(item) for item in arr]


def load_array(path: Path) -> np.ndarray | None:
    if not path or not path.exists():
        raise RuntimeError(f"Path does not exist: {path}")
    try:
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object:
            return arr
        return arr.astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to load array from {path}") from exc


def main():
    parser = argparse.ArgumentParser(description="Visualize eval logs in rerun.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing eval outputs.")
    parser.add_argument(
        "--T_robot_base",
        type=Path,
        default=Path("glasses_hardware/calib/T_robot_base.txt"),
        help="Path to T_robot_base.txt (base->robot).",
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
    try:
        headpose_preds = load_array(data_dir / "headpose_pred.npy")
    except Exception:
        headpose_preds = None

    T_robot_base = np.loadtxt(args.T_robot_base, dtype=np.float32)
    runtime_cam_path = data_dir / "T_base_cam_runtime.npy"
    T_base_cam_seq = load_array(runtime_cam_path)

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

    def _coerce_matrix(T: np.ndarray) -> np.ndarray | None:
        if T is None:
            return None
        T = np.asarray(T, dtype=np.float32)
        if T.shape == (4, 4):
            return T
        if T.shape == (3, 4):
            pad = np.array([[0, 0, 0, 1]], dtype=np.float32)
            return np.vstack([T, pad])
        return None

    def _cam_transform_for_frame(idx: int) -> np.ndarray | None:
        if T_base_cam_seq is None:
            return None
        if T_base_cam_seq.ndim == 3:
            if idx < 0 or idx >= T_base_cam_seq.shape[0]:
                return None
            return _coerce_matrix(T_base_cam_seq[idx])
        return _coerce_matrix(T_base_cam_seq)

    def _transform_points_cam_to_robot(points_cam: np.ndarray) -> np.ndarray | None:
        if T_base_cam_seq.ndim == 3:
            T_base_cam = _coerce_matrix(T_base_cam_seq[0])
        else:
            T_base_cam = _coerce_matrix(T_base_cam_seq)
        homog = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float32)], axis=1)
        pts_base = (T_base_cam @ homog.T).T
        pts_robot = (T_robot_base @ pts_base.T).T
        return pts_robot[:, :3]

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
        if headpose_preds is not None:  # headpose_pred:[frames, num_actions, 9]
            headpose_entry = None
            if 0 <= frame_idx < headpose_preds.shape[0]:
                headpose_entry = headpose_preds[frame_idx]
            if headpose_entry is not None:
                headpose_entry = np.asarray(headpose_entry, dtype=np.float32)
                headpose_mats = _build_pose_mats(headpose_entry[:, :3], headpose_entry[:, 3:9])
                headpose_points = []
                for headpose_T in headpose_mats:
                    headpose_robot = T_robot_base @ headpose_T.astype(np.float32)
                    headpose_points.append(headpose_robot[:3, 3])
                rr.log(
                    f"frames/frame_{frame_idx}/headpose_pred/points",
                    rr.Points3D(
                        positions=np.asarray(headpose_points, dtype=np.float32),
                        colors=np.array([[255, 120, 0, 255]], dtype=np.uint8),
                        radii=np.full(len(headpose_points), args.axis_len * 0.04, dtype=np.float32),
                    ),
                )
        log_axis(f"frames/frame_{frame_idx}/robot_base", T_robot_base, args.axis_len * 0.5)
        cam_tf = _cam_transform_for_frame(frame_idx) # load T_base_cam for this frame
        if cam_tf is not None:
            cam_robot = T_robot_base @ cam_tf 
            log_axis(f"frames/frame_{frame_idx}/robot_cam", cam_robot, args.axis_len * 0.4)

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
