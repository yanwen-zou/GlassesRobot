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
    try:
        executed = load_array(data_dir / "robot_executed_poses.npy")
        # tcp_hist = load_array(data_dir / "robot_tcp_history.npy")
    except Exception:
        executed = None
    tcp_hist = None
    try:
        headpose_preds = load_array(data_dir / "headpose_pred.npy")
    except Exception:
        print("No headpose predictions found.")
        headpose_preds = None

    T_robot_base = np.loadtxt(args.T_robot_base, dtype=np.float32)
    runtime_cam_path = data_dir / "T_base_cam_runtime.npy"
    T_base_cam_seq = load_array(runtime_cam_path)
    pointcloud_dir = (data_dir / "pointcloud") if args.pointcloud_dir is None else args.pointcloud_dir.resolve()

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

    ball_centroids, centroid_source_path = _load_ball_centroids(args.ball_centroids, data_dir) # in cam coordinate

    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("rerun package required. `pip install rerun-sdk`.") from exc
    try:
        import cv2
    except Exception:
        cv2 = None

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

        if idx < 0 or idx >= T_base_cam_seq.shape[0]:
            return None
        return _coerce_matrix(T_base_cam_seq[idx])

    def _load_pointcloud_npz(frame_idx: int, fallback_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not pointcloud_dir.exists():
            return None, None
        candidates = [
            pointcloud_dir / f"{frame_idx:06d}.npz",
            pointcloud_dir / f"{fallback_idx:06d}.npz",
        ]
        pcd_path = next((p for p in candidates if p.exists()), None)
        if pcd_path is None:
            return None, None
        data = np.load(pcd_path, allow_pickle=True)
        # New format from eval.py: `cloud` is (N,6) with xyz in base frame + normalized rgb (like dataset).
        if data.get("cloud") is not None:
            cloud = np.asarray(data["cloud"], dtype=np.float32)
            if cloud.ndim != 2 or cloud.shape[1] < 6:
                raise RuntimeError(f"Invalid cloud shape in {pcd_path}: {cloud.shape}")
            xyz_base = cloud[:, :3]
            colors_norm = cloud[:, 3:6]
            colors = np.clip(colors_norm * IMG_STD + IMG_MEAN, 0.0, 1.0)
            rgb_u8 = (colors * 255).astype(np.uint8)
            return xyz_base, rgb_u8

        # Backward compatibility: older format stored camera-frame xyz + uint8 rgb.
        xyz_cam = np.asarray(data.get("xyz_cam"), dtype=np.float32) if data.get("xyz_cam") is not None else None
        rgb = np.asarray(data.get("rgb"), dtype=np.uint8) if data.get("rgb") is not None else None
        return xyz_cam, rgb

    def _transform_points_cam_to_robot(points_cam: np.ndarray, cam_tf: np.ndarray) -> np.ndarray:
        homog = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float32)], axis=1)
        pts_base = (cam_tf @ homog.T).T
        pts_robot = (T_robot_base @ pts_base.T).T
        return pts_robot[:, :3]

    def _transform_points_base_to_robot(points_base: np.ndarray) -> np.ndarray:
        homog = np.concatenate([points_base, np.ones((points_base.shape[0], 1), dtype=np.float32)], axis=1)
        pts_robot = (T_robot_base @ homog.T).T
        return pts_robot[:, :3]

    def _transform_points_cam_to_robot_first(points_cam: np.ndarray) -> np.ndarray | None:
        if T_base_cam_seq.ndim == 3:
            T_base_cam = _coerce_matrix(T_base_cam_seq[0])
        else:
            T_base_cam = _coerce_matrix(T_base_cam_seq)
        if T_base_cam is None:
            return None
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
    video_cap = None
    video_frame_count = None
    if video_path.exists():
        if cv2 is None:
            raise RuntimeError("OpenCV required to load stream.mp4. `pip install opencv-python`.") 
        video_cap = cv2.VideoCapture(str(video_path))
        if not video_cap.isOpened():
            print(f"[WARN] Failed to open video: {video_path}")
            video_cap = None
        else:
            video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if video_frame_count <= 0:
                print(f"[WARN] Invalid frame count for video: {video_path}")
                video_frame_count = None

    pred_window = 5
    for rec_idx, rec in enumerate(pose_records):
        pose_robot = rec.get("object_pose_robot")
        pred_seq = rec.get("pred_seq_robot")

        frame_idx = int(rec.get("frame_idx", rec_idx))
        rr.set_time_sequence("frame", frame_idx)

        # Pointcloud (saved per update step from eval.py).
        cloud_xyz, rgb = _load_pointcloud_npz(frame_idx, rec_idx)
        if cloud_xyz is not None:
            if cloud_xyz.size == 0:
                print(f"[pcd] frame_idx={frame_idx} rec_idx={rec_idx} empty pointcloud")
                pts_robot = cloud_xyz.reshape(0, 3).astype(np.float32)
            else:
                # New format stores xyz in base frame; transform directly to robot frame.
                if cloud_xyz.shape[1] != 3:
                    raise RuntimeError(f"Unexpected point xyz shape: {cloud_xyz.shape}")
                pts_robot = _transform_points_base_to_robot(cloud_xyz)

            if args.max_points and pts_robot.shape[0] > args.max_points:
                step = int(np.ceil(pts_robot.shape[0] / args.max_points))
                pts_robot = pts_robot[::step]
                if rgb is not None:
                    rgb = rgb[::step]
            # Log to a fixed path so playback only shows the current frame's pointcloud.
            rr.log("frame/pointcloud", rr.Clear(recursive=True))
            rr.log(
                "frame/pointcloud",
                rr.Points3D(
                    positions=pts_robot,
                    colors=rgb if rgb is not None else None,
                    radii=np.full(pts_robot.shape[0], args.point_radius, dtype=np.float32),
                ),
            )
        else:
            print(f"[pcd] frame_idx={frame_idx} rec_idx={rec_idx} missing pointcloud npz under {pointcloud_dir}")

        if video_cap is not None and video_frame_count:
            if len(pose_records) > 1 and video_frame_count > 1:
                video_idx = int(round((frame_idx / (len(pose_records) - 1)) * (video_frame_count - 1)))
            else:
                video_idx = 0
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, video_idx)
            ok, frame = video_cap.read()
            if ok:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rr.log("video/stream", rr.Image(frame_rgb))
            else:
                video_cap.release()
                video_cap = None

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
            expired_idx = frame_idx - pred_window
            if expired_idx >= 0:
                rr.log(f"frames/frame_{expired_idx}/pred_points", rr.Clear(recursive=True))
        if headpose_preds is not None:  # headpose_pred:[frames, num_actions, 9]
            headpose_entry = None
            if "headpose_pred_cursor" not in locals():
                headpose_pred_cursor = 0
            cursor = headpose_pred_cursor
            if 0 <= cursor < headpose_preds.shape[0]:
                headpose_entry = headpose_preds[cursor]
                headpose_pred_cursor = cursor + 1
            if headpose_entry is not None:
                headpose_entry = np.asarray(headpose_entry, dtype=np.float32)
                # In eval output, headpose_pred.npy is already absolute in base frame.
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
        if cam_tf is not None:
            cam_robot = T_robot_base @ cam_tf 
            log_axis(f"frames/frame_{frame_idx}/robot_cam", cam_robot, args.axis_len * 0.4)

    if video_cap is not None:
        video_cap.release()

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
