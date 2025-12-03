#!/usr/bin/env python3
"""
Visualize a sequence of RGB-D frames in a common robot frame.

For each frame id, this script:
  - Loads RGB/Depth (if available) from the data directory
  - Backprojects to a point cloud in camera coordinates
  - Transforms the cloud into the robot frame using robot_to_cam (robot->cam)
  - Logs robot and camera coordinate frames plus the point cloud in Rerun

Frame ids are taken from robot_to_cam.npy (dict with frame_ids/transforms,
single 4x4, or a sequence). Sequence indices are treated as 1-based frame ids.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

# Add project root to sys.path to import RerunVisualizer if needed
import sys

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.rerun_visualizer import RerunVisualizer  # noqa: E402


DEPTH_SCALE_DEFAULT = 1000.0  # Depth units -> meters


def load_intrinsics(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Intrinsic matrix not found: {path}")
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    mat = np.array(rows, dtype=np.float32)
    if mat.shape != (3, 3):
        raise ValueError(f"Intrinsic matrix must be 3x3, got {mat.shape}")
    return mat


def _find_rgb_image(data_dir: Path, frame_id: str) -> Path | None:
    candidates = [
        data_dir / "rgb" / f"{frame_id}.png",
        data_dir / "rgb" / f"{frame_id}.jpg",
        data_dir / "jpg" / f"{frame_id}.png",
        data_dir / "jpg" / f"{frame_id}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_depth_image(data_dir: Path, frame_id: str) -> Path | None:
    candidates = [
        data_dir / "depth" / f"{frame_id}.png",
        data_dir / "depth" / f"{frame_id}.jpg",
        data_dir / "depth_vis" / f"{frame_id}.png",
        data_dir / "depth_vis" / f"{frame_id}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def backproject_depth_to_points(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    rgb_img: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    if stride < 1:
        stride = 1

    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    ys = ys.reshape(-1)
    xs = xs.reshape(-1)

    z = depth_m[ys, xs]
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), None

    xs = xs[valid]
    ys = ys[valid]
    z = z[valid].astype(np.float32)

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    colors = None
    if rgb_img is not None:
        rgb = rgb_img[ys, xs].astype(np.uint8)
        colors = rgb

    return points, colors


def load_transform_map(path: Path, frame_start: int = 1) -> Dict[int, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    tf_map: Dict[int, np.ndarray] = {}
    if data.ndim == 0 and isinstance(data.item(), dict):
        obj = data.item()
        fids = obj.get("frame_ids")
        mats = obj.get("transforms")
        if fids is None or mats is None:
            raise ValueError(f"{path} dict missing frame_ids/transforms")
        fids = np.asarray(fids, dtype=int)
        mats = np.asarray(mats, dtype=np.float32)
        if fids.shape[0] != mats.shape[0]:
            raise ValueError(f"{path} frame_ids and transforms length mismatch")
        for fid, mat in zip(fids, mats):
            tf_map[int(fid)] = mat.astype(np.float32)
    elif data.shape == (4, 4):
        tf_map[frame_start] = data.astype(np.float32)
    elif data.ndim == 3 and data.shape[1:] == (4, 4):
        for i in range(data.shape[0]):
            fid = frame_start + i
            tf_map[fid] = data[i].astype(np.float32)
    else:
        raise ValueError(f"Unsupported transform format at {path}: shape {data.shape}")
    return tf_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize all frames in a common robot frame.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Sequence directory containing rgb/, depth/, cam_K.txt, etc.",
    )
    parser.add_argument(
        "--robot-to-cam-npy",
        type=Path,
        required=True,
        help="Path to robot_to_cam.npy (dict, sequence, or single 4x4).",
    )
    parser.add_argument(
        "--cam-to-base-npy",
        type=Path,
        default=None,
        help="Optional path to cam_to_base.npy (cam->base); will be inverted to base->cam.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=DEPTH_SCALE_DEFAULT,
        help="Meters per depth unit (default 1000.0 for RealSense uint16).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Pixel stride for subsampling when building the point cloud (>=1).",
    )
    parser.add_argument(
        "--point-radius",
        type=float,
        default=0.002,
        help="Point radius (meters) for Rerun markers.",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not spawn a separate Rerun viewer window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    intrinsic_path = data_dir / "cam_K.txt"
    if not intrinsic_path.exists():
        intrinsic_path = data_dir / "camera_intrinsics.txt"
    intrinsic = load_intrinsics(intrinsic_path)

    robot_to_cam = load_transform_map(args.robot_to_cam_npy, frame_start=1)
    frame_ids = sorted(robot_to_cam.keys())
    if not frame_ids:
        raise RuntimeError("No transforms found in robot_to_cam.npy")

    cam_to_base = load_transform_map(args.cam_to_base_npy, frame_start=1) if args.cam_to_base_npy is not None else None

    vis = RerunVisualizer(
        name=f"SequenceRobotFrame[{data_dir.name}]",
        spawn=not args.no_spawn,
        fps=None,
    )

    for fid in frame_ids:
        frame_str = f"{int(fid):06d}"
        depth_path = _find_depth_image(data_dir, frame_str)
        if depth_path is None:
            print(f"[WARN] Skipping frame {frame_str}: no depth image.")
            continue
        rgb_path = _find_rgb_image(data_dir, frame_str)

        depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
        depth_m = depth_raw / args.depth_scale

        rgb_img = None
        if rgb_path is not None:
            rgb_img = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)

        points_cam, colors = backproject_depth_to_points(depth_m, intrinsic, rgb_img, stride=args.stride)
        if points_cam.shape[0] == 0:
            print(f"[WARN] Skipping frame {frame_str}: no valid depth points.")
            continue

        T_robot_cam = robot_to_cam[int(fid)]
        T_cam_base = cam_to_base.get(int(fid)) if cam_to_base is not None else None
        # T_cam_robot = np.linalg.inv(T_robot_cam).astype(np.float32)
        R = T_robot_cam[:3, :3]
        t = T_robot_cam[:3, 3]
        points_robot = (R @ points_cam.T + t.reshape(3, 1)).T

        # Default colors if none
        if colors is None:
            colors = np.full((points_robot.shape[0], 3), 200, dtype=np.uint8)

        vis.set_frame(frame_str)

        # Robot frame at origin, camera frame relative to robot
        vis.log_coordinate_frame(
            entity="robot_frame",
            pose=np.eye(4, dtype=np.float32),
            axis_len=max(args.point_radius * 50.0, 0.05),
        )

        vis.log_coordinate_frame(
            entity="camera_frame",
            pose=T_robot_cam,
            axis_len=max(args.point_radius * 50.0, 0.05),
        )

        if cam_to_base is not None and int(fid) in cam_to_base:
            vis.log_coordinate_frame(
                entity="base_frame",
                pose=T_robot_cam @ np.linalg.inv(T_cam_base),
                axis_len=max(args.point_radius * 50.0, 0.05),
            )

        vis.log_point_cloud(
            entity="cloud/points",
            points=points_robot,
            colors=colors,
            radius=args.point_radius,
            clear=True,
        )

        print(f"[INFO] Visualized frame {frame_str} in robot frame.")


if __name__ == "__main__":
    main()
