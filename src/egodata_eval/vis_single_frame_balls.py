#!/usr/bin/env python3
"""Visualize a single frame point cloud with ball masks in Rerun.

This script reads RGB, depth, and mask_balls from a given sequence folder
and visualizes:
    - The full point cloud reconstructed from RGB + depth
    - The three balls (id1, id2, id3) as colored subsets of the point cloud

Coordinates are in the current frame camera coordinate system.

Example:
    python src/egodata_eval/vis_single_frame_balls.py \
        --data-dir data/20251128_143254 \
        --frame-id 0
    
    python src/egodata_eval/vis_single_frame_balls.py \
        --data-dir data/20251128_143254 \
        --cam-to-base-npy data/20251128_143254/cam_to_base.npy \
        --frame-id 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# Add project root to sys.path to import utilities
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.rerun_visualizer import RerunVisualizer  # noqa: E402


DEPTH_SCALE_DEFAULT = 1000.0  # Depth units -> meters


def load_intrinsics(path: Path) -> np.ndarray:
    """Load camera intrinsics from a text file (3x3 matrix)."""
    if not path.exists():
        raise FileNotFoundError(f"Intrinsic matrix not found: {path}")
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    mat = np.array(rows, dtype=np.float32)
    if mat.shape != (3, 3):
        raise ValueError(f"Intrinsic matrix must be 3x3, got {mat.shape}")
    return mat


def _find_rgb_image(data_dir: Path, frame_id: str) -> Path | None:
    """Find an RGB image for the given frame."""
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
    """Find a depth image for the given frame."""
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


def _load_mask(mask_path: Path) -> np.ndarray:
    """Load mask image and return boolean array."""
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    mask = np.array(Image.open(mask_path).convert("L"))
    return mask > 0


def backproject_depth_to_points(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    rgb_img: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Backproject depth (and optional RGB) to 3D points in camera coordinates.

    Args:
        depth_m: Depth in meters (H, W)
        intrinsic: Camera intrinsic matrix (3x3)
        rgb_img: Optional RGB image (H, W, 3) in uint8
        stride: Subsampling stride for pixels (default 1 = use all pixels)

    Returns:
        points: (N, 3) 3D points in camera coordinates
        colors: (N, 3) colors in 0-255 uint8, or None if rgb_img is None
    """
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


def compute_ball_points(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute 3D points for masked ball pixels in camera coordinates."""
    if depth_m.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Depth shape {depth_m.shape} and mask shape {mask.shape} must match.")

    valid_mask = mask & np.isfinite(depth_m) & (depth_m > 0)
    ys, xs = np.nonzero(valid_mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth_m[ys, xs].astype(np.float32)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    cam_points = np.stack([x, y, z], axis=1)
    return cam_points.astype(np.float32)


def visualize_single_frame(
    data_dir: Path,
    frame_id: str,
    depth_scale: float,
    point_radius: float,
    spawn_viewer: bool,
    stride: int,
    robot_to_cam: dict[int, np.ndarray] | None = None,
    cam_to_base: dict[int, np.ndarray] | None = None,
) -> None:
    """Visualize a single frame point cloud and ball masks in Rerun."""
    # Intrinsics
    intrinsic_path = data_dir / "cam_K.txt"
    if not intrinsic_path.exists():
        intrinsic_path = data_dir / "camera_intrinsics.txt"
    if not intrinsic_path.exists():
        raise FileNotFoundError(f"Camera intrinsic file not found in {data_dir}")
    intrinsic = load_intrinsics(intrinsic_path)

    # Images
    rgb_path = _find_rgb_image(data_dir, frame_id)
    depth_path = _find_depth_image(data_dir, frame_id)

    if depth_path is None:
        raise FileNotFoundError(f"No depth image found for frame {frame_id} in {data_dir}")

    if rgb_path is None:
        print(f"[WARN] No RGB image found for frame {frame_id}, point cloud will be uncolored.")

    depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
    depth_m = depth_raw / depth_scale

    rgb_img = None
    if rgb_path is not None:
        rgb_img = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)

    # Backproject full point cloud
    points, colors = backproject_depth_to_points(depth_m, intrinsic, rgb_img, stride=stride)
    if points.shape[0] == 0:
        raise RuntimeError(f"No valid depth points for frame {frame_id}")

    # Fallback colors if RGB is missing
    if colors is None:
        colors = np.full((points.shape[0], 3), 200, dtype=np.uint8)

    # Load ball masks and compute their 3D points
    mask_balls_dir = data_dir / "masks_balls"
    if not mask_balls_dir.exists():
        print(f"[WARN] mask_balls directory not found at {mask_balls_dir}, no balls will be visualized.")
        ball_points = {}
    else:
        ball_points: dict[int, np.ndarray] = {}
        for ball_id in (1, 2, 3):
            mask_path = mask_balls_dir / f"{frame_id}_id{ball_id}.png"
            if not mask_path.exists():
                continue
            try:
                mask = _load_mask(mask_path)
            except Exception as exc:
                print(f"[WARN] Failed to load mask {mask_path}: {exc}")
                continue
            pts = compute_ball_points(depth_m, intrinsic, mask)
            if pts.shape[0] > 0:
                ball_points[ball_id] = pts

    # Compute ball centers from 3D points
    ball_centers: dict[int, np.ndarray] = {}
    for ball_id, pts in ball_points.items():
        if pts.shape[0] == 0:
            continue
        center = np.mean(pts, axis=0).astype(np.float32)
        ball_centers[ball_id] = center

    # Colors for balls (RGBA)
    ball_colors = {
        1: np.array([[255, 0, 0, 255]], dtype=np.uint8),   # Red
        2: np.array([[0, 255, 0, 255]], dtype=np.uint8),   # Green
        3: np.array([[0, 0, 255, 255]], dtype=np.uint8),   # Blue
    }

    # Visualization
    vis = RerunVisualizer(
        name=f"SingleFrameCloud[{data_dir.name}/{frame_id}]",
        spawn=spawn_viewer,
        fps=None,
    )

    vis.set_frame(frame_id)

    # Always log camera frame at origin (point cloud is in camera coordinates)
    vis.log_coordinate_frame(
        entity="camera_frame",
        pose=np.eye(4, dtype=np.float32),
        axis_len=max(point_radius * 50.0, 0.05),
    )

    # Log robot frame if provided
    if robot_to_cam is not None:
        try:
            frame_id_int = int(frame_id)
            T_robot_cam = robot_to_cam.get(frame_id_int)
        except ValueError:
            T_robot_cam = None
        if T_robot_cam is not None:
            # Stored as robot->cam; invert to place robot frame in camera coordinates.
            T_cam_robot = np.linalg.inv(T_robot_cam).astype(np.float32)
            vis.log_coordinate_frame(
                entity="robot_frame",
                pose=T_cam_robot,
                axis_len=max(point_radius * 50.0, 0.05),
            )

    # Log base coordinate frame if provided
    if cam_to_base is not None:
        try:
            frame_id_int = int(frame_id)
            T_cam_base = cam_to_base.get(frame_id_int)
        except ValueError:
            T_cam_base = None
        if T_cam_base is not None:
            # Stored as base->cam; invert to place base frame in camera coordinates.
            vis.log_coordinate_frame(
                entity="base_frame",
                pose=T_cam_base,
                axis_len=max(point_radius * 50.0, 0.05),
            )

    # Log base point cloud
    vis.log_point_cloud(
        entity="cloud/points",
        points=points,
        colors=colors,
        radius=point_radius,
        clear=True,
    )

    # Log balls as separate colored point sets
    for ball_id, pts in ball_points.items():
        color = ball_colors.get(ball_id, np.array([[255, 255, 255, 255]], dtype=np.uint8))
        # Repeat color to match number of points
        color_rep = np.repeat(color, pts.shape[0], axis=0)
        vis.log_point_cloud(
            entity=f"balls/ball_{ball_id}",
            points=pts,
            colors=color_rep,
            radius=point_radius * 2.0,
            clear=False,
        )

    # Log ball centers as single larger points
    for ball_id, center in ball_centers.items():
        color = ball_colors.get(ball_id, np.array([[255, 255, 255, 255]], dtype=np.uint8))
        vis.log_point_cloud(
            entity=f"balls_centers/ball_{ball_id}",
            points=center.reshape(1, 3),
            colors=np.repeat(color, 1, axis=0),
            radius=point_radius * 4.0,
            clear=False,
        )
        print(f"Ball {ball_id} center: {center}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a single frame point cloud and ball masks in Rerun."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Sequence directory containing rgb/, depth/, mask_balls/, cam_K.txt, etc.",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        required=True,
        help="Frame ID (e.g., '000000').",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=DEPTH_SCALE_DEFAULT,
        help="Meters per depth unit (default 1000.0 for RealSense uint16).",
    )
    parser.add_argument(
        "--point-radius",
        type=float,
        default=0.002,
        help="Point radius (meters) for Rerun markers.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Pixel stride for subsampling when building the point cloud (>=1).",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not spawn a separate Rerun viewer window.",
    )
    parser.add_argument(
        "--robot-to-cam-npy",
        type=Path,
        default=None,
        help="Optional path to robot_to_cam.npy to visualize robot frame.",
    )
    parser.add_argument(
        "--cam-to-base-npy",
        type=Path,
        default=None,
        help="Optional path to cam_to_base.npy (cam->base); will be inverted to base->cam.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    frame_id = args.frame_id
    # Normalize frame id to 6 digits if it is numeric
    if frame_id.isdigit():
        frame_id = f"{int(frame_id):06d}"

    def _load_transform_map(path: Path, invert: bool = False, frame_start: int = 1) -> dict[int, np.ndarray]:
        """Load transform map from .npy (dict with frame_ids/transforms, single 4x4, or sequence)."""
        data_raw = np.load(path, allow_pickle=True)
        transform_map: dict[int, np.ndarray] = {}
        if data_raw.ndim == 0 and isinstance(data_raw.item(), dict):
            data = data_raw.item()
            frame_ids_arr = data.get("frame_ids", None)
            transforms_arr = data.get("transforms", None)
            if frame_ids_arr is None or transforms_arr is None:
                raise ValueError(f"{path} does not contain expected keys.")
            if frame_ids_arr.shape[0] != transforms_arr.shape[0]:
                raise ValueError(f"Mismatch between number of frame_ids and transforms in {path}.")
            for idx, fid in enumerate(frame_ids_arr):
                T = transforms_arr[idx].astype(np.float32)
                transform_map[int(fid)] = np.linalg.inv(T) if invert else T
        elif data_raw.shape == (4, 4):
            fid = frame_start
            T = data_raw.astype(np.float32)
            transform_map[fid] = np.linalg.inv(T) if invert else T
        elif data_raw.ndim == 3 and data_raw.shape[1:] == (4, 4):
            for i in range(data_raw.shape[0]):
                fid = frame_start + i
                T = data_raw[i].astype(np.float32)
                transform_map[fid] = np.linalg.inv(T) if invert else T
        else:
            raise ValueError(f"Unexpected shape for {path}: {data_raw.shape}")
        return transform_map

    # Load robot_to_cam transforms if provided
    robot_to_cam: dict[int, np.ndarray] | None = None
    if args.robot_to_cam_npy is not None:
        robot_to_cam_path: Path = args.robot_to_cam_npy
        if not robot_to_cam_path.exists():
            raise FileNotFoundError(f"robot_to_cam.npy not found at {robot_to_cam_path}")
        robot_to_cam = _load_transform_map(robot_to_cam_path, invert=False, frame_start=1)

    # Load base->cam either directly or by inverting cam->base
    if args.cam_to_base_npy is not None:
        cam_to_base_path: Path = args.cam_to_base_npy
        if not cam_to_base_path.exists():
            raise FileNotFoundError(f"cam_to_base.npy not found at {cam_to_base_path}")
        # invert cam->base to base->cam
        cam_to_base = _load_transform_map(cam_to_base_path, invert=True, frame_start=1)

    visualize_single_frame(
        data_dir=data_dir,
        frame_id=frame_id,
        depth_scale=args.depth_scale,
        point_radius=args.point_radius,
        spawn_viewer=not args.no_spawn,
        stride=args.stride,
        robot_to_cam=robot_to_cam,
        cam_to_base=cam_to_base if args.cam_to_base_npy is not None else None,
    )


if __name__ == "__main__":
    main()
