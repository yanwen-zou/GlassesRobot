#!/usr/bin/env python3
"""
Visualize evaluation logs (object pose records, executed poses, TCP history) using rerun.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import re
import numpy as np

here = Path(__file__).resolve()
project_root = here.parents[3]
src_root = project_root / "src"
mba_root = project_root / "MBA"
for path in (src_root, mba_root, project_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
from egodata_eval.eval_constant import UPDATE_INTERVAL
from MBA.utils.constants import IMG_MEAN, IMG_STD  # type: ignore

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
        raise RuntimeError(f"Path does not exist: {path}")
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
        "--pointcloud-dir",
        type=Path,
        default=None,
        help="Directory containing per-frame pointcloud npz (default: <data-dir>/pointcloud).",
    )
    parser.add_argument("--point-radius", type=float, default=0.003, help="Rerun point radius for pointcloud.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=200_000,
        help="Max points to log per frame (0 disables downsample).",
    )
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
    tcp_hist = None
    headpose_preds = load_array(data_dir / "headpose_pred.npy")

    T_robot_base = np.loadtxt(args.T_robot_base, dtype=np.float32)
    runtime_cam_path = data_dir / "T_base_cam_runtime.npy"
    T_base_cam_seq = load_array(runtime_cam_path)
    pointcloud_dir = (data_dir / "pointcloud") if args.pointcloud_dir is None else args.pointcloud_dir.resolve()

    def _load_ball_centroids(path: Optional[Path]) -> tuple[np.ndarray | None, Optional[Path]]:
        if path is None:
            return None, None
        search_path = path
        if not search_path.exists():
            raise RuntimeError(f"Ball centroids path does not exist: {search_path}")
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

    ball_centroids, centroid_source_path = _load_ball_centroids(args.ball_centroids) # in cam coordinate

    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("rerun package required. `pip install rerun-sdk`.") from exc
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV required. `pip install opencv-python`.") from exc

    rr.init(f"Eval Visualization ({data_dir.name})", spawn=args.spawn)
    rr.log("world", rr.ViewCoordinates.FRU)

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

    def _coerce_matrix(T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=np.float32)
        if T.shape == (4, 4):
            return T
        raise RuntimeError(f"Expected (4,4) transform, got {T.shape}")

    def _cam_transform_for_frame(idx: int) -> np.ndarray:
        if idx < 0 or idx >= T_base_cam_seq.shape[0]:
            raise RuntimeError(f"Camera transform index out of range: idx={idx}, len={T_base_cam_seq.shape[0]}")
        return _coerce_matrix(T_base_cam_seq[idx])

    def _load_pointcloud_npz(frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if not pointcloud_dir.exists():
            raise RuntimeError(f"Pointcloud directory does not exist: {pointcloud_dir}")
        pcd_path = pointcloud_dir / f"{frame_idx:06d}.npz"
        if not pcd_path.exists():
            raise RuntimeError(f"Missing pointcloud npz: {pcd_path}")
        data = np.load(pcd_path, allow_pickle=True)
        if data.get("cloud") is None:
            raise RuntimeError(f"Missing 'cloud' key in pointcloud npz: {pcd_path}")
        cloud = np.asarray(data["cloud"], dtype=np.float32)
        if cloud.ndim != 2 or cloud.shape[1] < 6:
            raise RuntimeError(f"Invalid cloud shape in {pcd_path}: {cloud.shape}")
        xyz_base = cloud[:, :3]
        colors_norm = cloud[:, 3:6]
        colors = np.clip(colors_norm * IMG_STD + IMG_MEAN, 0.0, 1.0)
        rgb_u8 = (colors * 255).astype(np.uint8)
        return xyz_base, rgb_u8

    def _transform_points_cam_to_robot(points_cam: np.ndarray, cam_tf: np.ndarray) -> np.ndarray:
        homog = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float32)], axis=1)
        pts_base = (cam_tf @ homog.T).T
        pts_robot = (T_robot_base @ pts_base.T).T
        return pts_robot[:, :3]

    def _transform_points_base_to_robot(points_base: np.ndarray) -> np.ndarray:
        homog = np.concatenate([points_base, np.ones((points_base.shape[0], 1), dtype=np.float32)], axis=1)
        pts_robot = (T_robot_base @ homog.T).T
        return pts_robot[:, :3]

    def _transform_points_cam_to_robot_first(points_cam: np.ndarray) -> np.ndarray:
        if T_base_cam_seq.ndim == 3:
            T_base_cam = _coerce_matrix(T_base_cam_seq[0])
        else:
            T_base_cam = _coerce_matrix(T_base_cam_seq)
        homog = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float32)], axis=1)
        pts_base = (T_base_cam @ homog.T).T
        pts_robot = (T_robot_base @ pts_base.T).T
        return pts_robot[:, :3]

    ball_centroids_robot = _transform_points_cam_to_robot_first(ball_centroids) if ball_centroids is not None else None

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

    video_path = data_dir / "stream.mp4"
    if not video_path.exists():
        raise RuntimeError(f"Missing video file: {video_path}")
    video_cap = cv2.VideoCapture(str(video_path))
    if not video_cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frame_count <= 0:
        raise RuntimeError(f"Invalid frame count for video: {video_path}")

    pred_window = 5
    for rec_idx, rec in enumerate(pose_records):
        pose_robot = rec["object_pose_robot"]
        pred_seq = rec["pred_obj_seq_robot"]
        frame_idx = int(rec["frame_idx"])
        rr.set_time_sequence("frame", frame_idx)

        # Pointcloud (saved per update step from eval.py).
        cloud_xyz, rgb = _load_pointcloud_npz(frame_idx)
        if cloud_xyz.size == 0:
            print(f"[pcd] frame_idx={frame_idx} rec_idx={rec_idx} empty pointcloud")
            pts_robot = cloud_xyz.reshape(0, 3).astype(np.float32)
        else:
            if cloud_xyz.shape[1] != 3:
                raise RuntimeError(f"Unexpected point xyz shape: {cloud_xyz.shape}")
            pts_robot = _transform_points_base_to_robot(cloud_xyz)

        if args.max_points and pts_robot.shape[0] > args.max_points:
            step = int(np.ceil(pts_robot.shape[0] / args.max_points))
            pts_robot = pts_robot[::step]
            rgb = rgb[::step]
        rr.log("frame/pointcloud", rr.Clear(recursive=True))
        rr.log(
            "frame/pointcloud",
            rr.Points3D(
                positions=pts_robot,
                colors=rgb,
                radii=np.full(pts_robot.shape[0], args.point_radius, dtype=np.float32),
            ),
        )

        if len(pose_records) <= 1 or video_frame_count <= 1:
            raise RuntimeError("pose_records and video must both contain at least 2 frames.")
        video_idx = int(round((frame_idx / (len(pose_records) - 1)) * (video_frame_count - 1)))
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, video_idx)
        ok, frame = video_cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {video_idx} from video.")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rr.log("video/stream", rr.Image(frame_rgb))

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
        pred_seq = np.asarray(pred_seq, dtype=np.float32)
        rr.log(
            f"frames/frame_{frame_idx}/pred_points",
            rr.Points3D(
                positions=pred_seq[:, :3, 3],
                colors=np.array([[255, 255, 0, 255]], dtype=np.uint8),
                radii=np.full(pred_seq.shape[0], args.axis_len * 0.03, dtype=np.float32),
            ),
        )
        expired_idx = frame_idx - pred_window
        if expired_idx >= 0:
            rr.log(f"frames/frame_{expired_idx}/pred_points", rr.Clear(recursive=True))
        if rec_idx >= headpose_preds.shape[0]:
            raise RuntimeError(
                f"headpose_pred length mismatch: rec_idx={rec_idx}, headpose_len={headpose_preds.shape[0]}"
            )
        headpose_entry = np.asarray(headpose_preds[rec_idx], dtype=np.float32)
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
        expired_idx = frame_idx - pred_window
        if expired_idx >= 0:
            rr.log(f"frames/frame_{expired_idx}/headpose_pred/points", rr.Clear(recursive=True))
        log_axis(f"frames/frame_{frame_idx}/robot_base", T_robot_base, args.axis_len * 0.5)
        # Prefer aligning camera transforms to the record index (they are saved per update step).
        cam_tf = _cam_transform_for_frame(rec_idx)
        cam_robot = T_robot_base @ cam_tf
        log_axis(f"frames/frame_{frame_idx}/robot_cam", cam_robot, args.axis_len * 0.4)

    video_cap.release()

    if executed.size > 0:
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
