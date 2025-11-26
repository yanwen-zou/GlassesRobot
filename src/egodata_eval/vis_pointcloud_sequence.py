#!/usr/bin/env python3
"""Stream point clouds frame-by-frame in Rerun.

Example:
    python src/egodata_eval/vis_pointcloud_sequence.py \
        --data_path data/moving --seq_id 20251125_210453 --fps 5
    
    # Transform point clouds from camera to base coordinates no need of head pose
    python src/egodata_eval/vis_pointcloud_sequence.py \
    --data_path /home/akihi/code/GlassesRobot/data \
    --split train \
    --seq_id 20251125_210453 \
    --cam_to_base_npy /home/akihi/code/GlassesRobot/data/train/20251125_210453/cam_to_base.npy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import open3d as o3d
from PIL import Image


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEPTH_SCALE_DEFAULT = 1000.0  # Depth units -> meters


def _import_rerun():
    try:
        import rerun as rr  # type: ignore
    except Exception as exc:  # pragma: no cover - rerun is optional at install time
        raise RuntimeError("Rerun package is required. Install with `pip install rerun-sdk`.") from exc
    return rr


def list_sequences(data_path: Path, split: str) -> List[str]:
    """List all sequence IDs in the data path.
    
    Args:
        data_path: Root data directory
        split: Dataset split ('train', 'eval', or 'all')
        
    Returns:
        Sorted list of sequence IDs (directory names)
    """
    if split == 'all':
        search_path = data_path
    else:
        search_path = data_path / split
    
    if not search_path.exists():
        raise FileNotFoundError(f"Data path {search_path} does not exist.")
    
    sequences = sorted([
        d.name for d in search_path.iterdir()
        if d.is_dir() and (d / "rgb").exists() and (d / "depth").exists()
    ])
    
    if not sequences:
        raise RuntimeError(f"No valid sequences found under {search_path}.")
    
    return sequences


def list_frames(seq_path: Path) -> List[str]:
    """List all frame IDs in a sequence.
    
    Args:
        seq_path: Sequence directory path
        
    Returns:
        Sorted list of frame IDs (without extension)
    """
    rgb_dir = seq_path / "rgb"
    depth_dir = seq_path / "depth"
    
    # Try rgb directory first
    if rgb_dir.exists():
        frame_files = sorted(set([
            f.stem for f in list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg"))
        ]))
    elif depth_dir.exists():
        frame_files = sorted(set([
            f.stem for f in list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.jpg"))
        ]))
    else:
        raise FileNotFoundError(f"Neither rgb nor depth directory found in {seq_path}")
    
    return frame_files


def _load_head_pose_from_file(head_pos_dir: Path, frame_id: int) -> np.ndarray | None:
    """Load head pose from file directly.
    
    Args:
        head_pos_dir: Directory containing head pose files
        frame_id: Frame ID (integer)
        
    Returns:
        4x4 transformation matrix (world->cam) or None if file not found
    """
    pose_file = head_pos_dir / f"{frame_id:06d}.txt"
    if not pose_file.exists():
        return None
    try:
        values = np.loadtxt(pose_file, dtype=np.float32).reshape(-1)
        if values.size < 7:
            return None
        t = values[:3]
        q = values[3:7]
        # Normalize quaternion
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            return None
        q = q / q_norm
        qx, qy, qz, qw = q
        
        # Convert quaternion to rotation matrix
        rot = np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=np.float32,
        )
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = rot
        mat[:3, 3] = t
        return mat
    except Exception:
        return None


def load_camera_intrinsics(seq_path: Path) -> np.ndarray:
    """Load camera intrinsics from file.
    
    Args:
        seq_path: Sequence directory path
        
    Returns:
        3x3 camera intrinsic matrix
    """
    intrinsic_path = seq_path / "cam_K.txt"
    if not intrinsic_path.exists():
        intrinsic_path = seq_path / "camera_intrinsics.txt"
    
    if not intrinsic_path.exists():
        raise FileNotFoundError(f"Camera intrinsic file not found in {seq_path}")
    
    rows = [list(map(float, line.split())) for line in intrinsic_path.read_text().splitlines() if line.strip()]
    mat = np.array(rows, dtype=np.float32)
    if mat.shape != (3, 3):
        raise ValueError(f"Intrinsic matrix must be 3x3, got {mat.shape}")
    return mat


def load_camera_extrinsics(seq_path: Path) -> Dict[int, np.ndarray]:
    """Load all camera extrinsics from head_pos directory.
    
    Args:
        seq_path: Sequence directory path
        
    Returns:
        Dictionary mapping frame_id (int) to 4x4 transformation matrix (world->cam)
    """
    head_pos_dir = seq_path / "head_pos"
    if not head_pos_dir.exists():
        return {}
    
    extrinsics: Dict[int, np.ndarray] = {}
    pose_files = sorted(head_pos_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    
    for pose_file in pose_files:
        try:
            frame_id = int(pose_file.stem)
            pose = _load_head_pose_from_file(head_pos_dir, frame_id)
            if pose is not None:
                extrinsics[frame_id] = pose
        except (ValueError, Exception):
            continue
    
    return extrinsics


def load_point_cloud_from_files(
    rgb_path: Path,
    depth_path: Path,
    intrinsic: np.ndarray,
    depth_scale: float,
    voxel_size: float,
) -> np.ndarray:
    """Load point cloud from RGB and depth images.
    
    Args:
        rgb_path: Path to RGB image
        depth_path: Path to depth image
        intrinsic: Camera intrinsic matrix (3x3)
        depth_scale: Depth scale factor (meters per depth unit)
        voxel_size: Voxel size for downsampling
        
    Returns:
        Point cloud array [N, 6] where first 3 columns are xyz and last 3 are rgb (0-1 range)
    """
    # Load images
    rgb_img = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    depth_img = np.array(Image.open(depth_path), dtype=np.float32)
    
    h, w = depth_img.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # Create point cloud using open3d
    colors = o3d.geometry.Image(rgb_img.astype(np.uint8))
    depths = o3d.geometry.Image(depth_img.astype(np.float32))
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, depth_scale, convert_rgb_to_intensity=False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    cloud = cloud.voxel_down_sample(voxel_size)
    
    points = np.array(cloud.points, dtype=np.float32)
    colors = np.array(cloud.colors, dtype=np.float32)
    
    # Concatenate points and colors (colors are in 0-1 range)
    cloud_array = np.concatenate([points, colors], axis=1)
    return cloud_array.astype(np.float32)


def _log_coordinate_frame(rr,
                          entity: str,
                          pose: np.ndarray,
                          axis_len: float = 0.05,
                          color_triplets: np.ndarray | None = None) -> None:
    colors = (
        color_triplets
        if color_triplets is not None
        else np.array(
            [
                [255, 0, 0, 255],
                [0, 255, 0, 255],
                [0, 0, 255, 255],
            ],
            dtype=np.uint8,
        )
    )
    rr.log(
        entity,
        rr.Transform3D(
            translation=pose[:3, 3].astype(np.float32),
            mat3x3=pose[:3, :3].astype(np.float32),
        ),
    )
    origins = np.zeros((3, 3), dtype=np.float32)
    vectors = (np.eye(3, dtype=np.float32) * axis_len).astype(np.float32)
    rr.log(
        f"{entity}/axes",
        rr.Arrows3D(
            origins=origins,
            vectors=vectors,
            colors=colors,
            radii=np.full(3, axis_len * 0.05, dtype=np.float32),
        ),
    )


def _pose_from_seven(values: Sequence[float]) -> np.ndarray:
    """Convert 7-value pose [x, y, z, qx, qy, qz, qw] to 4x4 transformation matrix."""
    if len(values) != 7:
        raise ValueError(f"Expected 7 values for pose, got {len(values)}")
    x, y, z, qx, qy, qz, qw = values
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = np.array([x, y, z], dtype=np.float32)
    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        raise ValueError("Quaternion norm too small")
    quat /= norm
    qx, qy, qz, qw = quat
    T[:3, :3] = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )
    return T


def _load_matrix4x4(path: Path) -> np.ndarray:
    """Load a 4x4 transformation matrix from file, handling various formats."""
    mat = np.loadtxt(path).astype(np.float32)
    if mat.ndim == 1:
        if mat.size == 16:
            mat = mat.reshape(4, 4)
        elif mat.size == 12:
            mat = np.vstack([mat.reshape(3, 4), np.array([0, 0, 0, 1], dtype=np.float32)])
        else:
            raise ValueError(f"Unexpected matrix length {mat.size} in {path}")
    if mat.shape == (3, 4):
        mat = np.vstack([mat, np.array([0, 0, 0, 1], dtype=np.float32)])
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix in {path}, got {mat.shape}")
    return mat.astype(np.float32)


def _load_tcp_to_zed_transform(transform_path: Path) -> np.ndarray:
    """Load TCP to ZED camera transform matrix from file.
    
    Args:
        transform_path: Path to the transform matrix file (4x4 matrix)
    
    Returns:
        4x4 transformation matrix (tcp_T_zed format)
    
    Raises:
        FileNotFoundError: If the transform file does not exist
        ValueError: If the file does not contain a valid 4x4 matrix
    """
    if not transform_path.exists():
        raise FileNotFoundError(f"TCP to ZED transform file not found: {transform_path}")
    return _load_matrix4x4(transform_path)


def _load_calibrated_head_pose(seq_path: Path) -> np.ndarray:
    """Load calibrated head pose from calibrated_head_pose.txt."""
    head_pose_path = seq_path / "calibrated_head_pose.txt"
    if not head_pose_path.exists():
        raise FileNotFoundError(f"Missing calibrated head pose at {head_pose_path}")
    head_vals = np.loadtxt(head_pose_path).astype(np.float32)
    if head_vals.ndim != 1 or head_vals.size != 7:
        raise ValueError(f"Invalid contents in {head_pose_path}")
    world_T_cam = _pose_from_seven(head_vals)
    return world_T_cam.astype(np.float32)


def transform_clouds_to_coordinate_system(
    extrinsics: Dict[int, np.ndarray],
    frame_clouds: Sequence[Tuple[str, np.ndarray]],
    target_pose: np.ndarray | None = None,
    ref_frame_id: str | None = None,
    warn_prefix: str = "vis_pointcloud",
) -> List[Tuple[str, np.ndarray]]:
    """Transform point clouds to a target coordinate system.
    
    Args:
        extrinsics: Dictionary mapping frame_id (int) to 4x4 transformation matrix (world->cam)
        frame_clouds: List of (frame_id, cloud) tuples
        target_pose: Target pose matrix (4x4). If None, uses first frame as reference.
        ref_frame_id: Reference frame ID (used if target_pose is None)
        warn_prefix: Prefix for warning messages
    
    Returns:
        Transformed point clouds
    """
    if not frame_clouds:
        return []
    
    # Determine target pose
    if target_pose is not None:
        target_inv = np.linalg.inv(target_pose).astype(np.float32)
    elif ref_frame_id is not None:
        ref_frame_id_int = int(ref_frame_id) if isinstance(ref_frame_id, str) else ref_frame_id
        ref_extr = extrinsics.get(ref_frame_id_int)
        if ref_extr is None:
            if warn_prefix:
                print(f"[{warn_prefix}] Missing camera extrinsic for ref_frame={ref_frame_id}, using identity")
            ref_extr = np.eye(4, dtype=np.float32)
        target_inv = np.linalg.inv(ref_extr).astype(np.float32)
    else:
        # No transformation
        return list(frame_clouds)
    
    # Transform each cloud
    transformed: List[Tuple[str, np.ndarray]] = []
    for frame_id, cloud in frame_clouds:
        points = cloud[:, :3].astype(np.float32)
        if points.size == 0:
            transformed.append((frame_id, cloud))
            continue
        
        frame_id_int = int(frame_id) if isinstance(frame_id, str) else frame_id
        cam_extr = extrinsics.get(frame_id_int)
        if cam_extr is None:
            if warn_prefix:
                print(f"[{warn_prefix}] Missing camera extrinsic for frame={frame_id}, using identity")
            cam_extr = np.eye(4, dtype=np.float32)
        
        T = target_inv @ cam_extr
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        homo = np.concatenate([points, ones], axis=1)
        points_target = (T @ homo.T).T[:, :3]
        cloud_new = cloud.copy()
        cloud_new[:, :3] = points_target
        transformed.append((frame_id, cloud_new))
    
    return transformed


def get_camera_poses(
    extrinsics: Dict[int, np.ndarray],
    frame_ids: Sequence[str],
    target_pose: np.ndarray | None = None,
    ref_frame_id: str | None = None,
    warn_prefix: str = "vis_pointcloud",
    tcp_to_zed: np.ndarray | None = None,
) -> List[Tuple[str, np.ndarray]]:
    """Get camera poses in a target coordinate system.
    
    Args:
        extrinsics: Dictionary mapping frame_id (int) to 4x4 transformation matrix (world->cam)
        frame_ids: List of frame IDs
        target_pose: Target pose matrix (4x4). If None, uses first frame as reference.
        ref_frame_id: Reference frame ID (used if target_pose is None)
        warn_prefix: Prefix for warning messages
        tcp_to_zed: Optional TCP to ZED camera transform matrix (tcp_T_zed). If provided,
                    converts head_pos (TCP poses) to camera poses: world_T_zed = world_T_tcp @ tcp_T_zed
    
    Returns:
        List of (frame_id, pose) tuples
    """
    # Determine reference inverse
    ref_inv = None
    if target_pose is not None:
        ref_inv = np.linalg.inv(target_pose)
    elif ref_frame_id is not None:
        ref_frame_id_int = int(ref_frame_id) if isinstance(ref_frame_id, str) else ref_frame_id
        ref_extr = extrinsics.get(ref_frame_id_int)
        if ref_extr is None:
            if warn_prefix:
                print(f"[{warn_prefix}] Missing camera extrinsic for ref_frame={ref_frame_id}, using identity")
            ref_extr = np.eye(4, dtype=np.float32)
        # Apply TCP to ZED transform if provided
        if tcp_to_zed is not None:
            ref_extr = ref_extr @ tcp_to_zed
        ref_inv = np.linalg.inv(ref_extr)
    
    poses: List[Tuple[str, np.ndarray]] = []
    for fid in frame_ids:
        fid_int = int(fid) if isinstance(fid, str) else fid
        cam_extr = extrinsics.get(fid_int)
        if cam_extr is None:
            if warn_prefix:
                print(f"[{warn_prefix}] Missing camera extrinsic for frame={fid}, using identity")
            cam_extr = np.eye(4, dtype=np.float32)
        # Apply TCP to ZED transform if provided: world_T_zed = world_T_tcp @ tcp_T_zed
        if tcp_to_zed is not None:
            cam_extr = cam_extr @ tcp_to_zed
        pose = ref_inv @ cam_extr if ref_inv is not None else cam_extr
        poses.append((fid, pose.astype(np.float32)))
    return poses


def visualize_sequence(
    frame_clouds: Sequence[Tuple[str, np.ndarray]],
    seq_id: str,
    fps: float,
    point_radius: float,
    spawn_viewer: bool,
    camera_poses: Sequence[Tuple[str, np.ndarray]] | None = None,
    anchor_frames: Sequence[Tuple[str, np.ndarray]] | None = None,
    ball_centers: Dict[str, Dict[int, np.ndarray]] | None = None,
    head_pos_dir: Path | None = None,
    target_pose: np.ndarray | None = None,
    ref_frame_id: str | None = None,
) -> None:
    """Visualize point cloud sequence in Rerun."""
    if not frame_clouds:
        raise ValueError("cloud_sequence is empty; nothing to visualize.")

    rr = _import_rerun()
    rr.init(f"DatasetClouds[{seq_id}]", spawn=spawn_viewer)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    dt = 1.0 / fps if fps > 1e-6 else 0.0
    entity_path = f"sample/{seq_id}/cloud"
    cam_entity = f"sample/{seq_id}/camera_pose"
    cam_path_entity = f"sample/{seq_id}/camera_path"
    ball_centers_entity = f"sample/{seq_id}/ball_centers"
    cam_pose_map = {fid: pose for fid, pose in camera_poses} if camera_poses else {}
    cam_points: list[np.ndarray] = []
    
    # Ball colors: red, green, blue for ball_id 1, 2, 3
    ball_colors = {
        1: np.array([255, 0, 0, 255], dtype=np.uint8),  # Red
        2: np.array([0, 255, 0, 255], dtype=np.uint8),  # Green
        3: np.array([0, 0, 255, 255], dtype=np.uint8),  # Blue
    }

    rr.log(f"sample/{seq_id}", rr.Transform3D())
    rr.log(entity_path, rr.Transform3D())

    # Log anchor frames (coordinate systems)
    if anchor_frames:
        for anchor_name, pose in anchor_frames:
            _log_coordinate_frame(
                rr,
                f"sample/{seq_id}/{anchor_name}",
                pose,
                axis_len=max(point_radius * 50.0, 0.05),
            )

    # Visualize each frame
    for idx, (frame_id, cloud) in enumerate(frame_clouds):
        if cloud.size == 0:
            continue
        positions = cloud[:, :3].astype(np.float32)
        
        # Handle colors: convert from 0-1 range to 0-255
        if cloud.shape[1] >= 6:
            color_data = cloud[:, 3:6]
            # Colors are in 0-1 range, convert to 0-255
            colors = (color_data * 255.0).astype(np.uint8)
        else:
            # No colors, use white
            colors = np.full((positions.shape[0], 3), 255, dtype=np.uint8)
        
        try:
            time_idx = int(frame_id)
        except ValueError:
            time_idx = idx
        rr.set_time("frame", sequence=time_idx)
        rr.log(entity_path, rr.Clear(recursive=False))
        rr.log(
            entity_path,
            rr.Points3D(
                positions=positions,
                colors=colors,
                radii=point_radius,
            ),
        )
        
        # Clear ball centers for this frame
        if ball_centers is not None:
            rr.log(ball_centers_entity, rr.Clear(recursive=False))
        
        # Log camera pose
        pose = cam_pose_map.get(frame_id)
        if pose is not None:
            rr.log(cam_entity, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]))
            rr.log(
                f"{cam_entity}/axes",
                rr.Arrows3D(
                    origins=np.zeros((3, 3), dtype=np.float32),
                    vectors=np.eye(3, dtype=np.float32) * 0.05,
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
            cam_points.append(pose[:3, 3].astype(np.float32))
            rr.log(
                cam_path_entity,
                rr.LineStrips3D(
                    [np.asarray(cam_points, dtype=np.float32)],
                    radii=point_radius,
                    colors=np.array([[255, 200, 0, 255]], dtype=np.uint8),
                ),
            )
        
        # Visualize ball centers for this frame
        if ball_centers is not None:
            frame_balls = ball_centers.get(frame_id)
            if frame_balls is not None and len(frame_balls) > 0:
                ball_positions = []
                ball_colors_list = []
                
                # Calculate transformation from current frame camera to target coordinate system
                transform_to_target = None
                if head_pos_dir is not None:
                    try:
                        cam_extr = _load_head_pose_from_file(head_pos_dir, int(frame_id))
                        if cam_extr is not None:
                            if target_pose is not None:
                                target_inv = np.linalg.inv(target_pose).astype(np.float32)
                                transform_to_target = target_inv @ cam_extr
                            elif ref_frame_id is not None:
                                ref_extr = _load_head_pose_from_file(head_pos_dir, int(ref_frame_id))
                                if ref_extr is not None:
                                    target_inv = np.linalg.inv(ref_extr).astype(np.float32)
                                    transform_to_target = target_inv @ cam_extr
                    except Exception as e:
                        print(f"[WARN] Failed to get transform for ball centers in frame {frame_id}: {e}")
                
                for ball_id in sorted(frame_balls.keys()):
                    pos = frame_balls[ball_id]  # Position in current frame camera coordinate system
                    
                    # Transform to target coordinate system if transform is available
                    if transform_to_target is not None:
                        pos_homo = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float32)
                        pos_target = (transform_to_target @ pos_homo)[:3]
                        ball_positions.append(pos_target)
                    else:
                        # No transformation, use as-is
                        ball_positions.append(pos)
                    
                    ball_colors_list.append(ball_colors.get(ball_id, np.array([255, 255, 255, 255], dtype=np.uint8)))
                
                if ball_positions:
                    rr.log(
                        ball_centers_entity,
                        rr.Points3D(
                            positions=np.array(ball_positions, dtype=np.float32),
                            colors=np.array(ball_colors_list, dtype=np.uint8),
                            radii=point_radius * 5.0,  # Make ball centers more visible
                        ),
                    )
        
        if dt > 0:
            time.sleep(dt)


def load_ball_centers(ball_centers_path: Path) -> Dict[str, Dict[int, np.ndarray]]:
    """Load ball centers from text file.
    
    Args:
        ball_centers_path: Path to ball_centers.txt file (format: frame_id ball_id x y z)
        
    Returns:
        Dictionary mapping frame_id (as string, with leading zeros) to dict of ball_id -> [x, y, z] position
    """
    if not ball_centers_path.exists():
        raise FileNotFoundError(f"Ball centers file not found: {ball_centers_path}")
    
    ball_centers: Dict[str, Dict[int, np.ndarray]] = {}
    with open(ball_centers_path, "r") as f:
        lines = f.readlines()
        # Skip header
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            frame_id_int = int(parts[0])
            # Convert to string with leading zeros (6 digits) to match dataset format
            frame_id = f"{frame_id_int:06d}"
            ball_id = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            
            if frame_id not in ball_centers:
                ball_centers[frame_id] = {}
            ball_centers[frame_id][ball_id] = np.array([x, y, z], dtype=np.float32)
    
    return ball_centers


def _frame_sort_key(frame_id: str) -> Tuple[int, int | str]:
    """Sort key for frame IDs: numeric frames first, then string frames."""
    try:
        return (0, int(frame_id))
    except ValueError:
        return (1, frame_id)


def gather_sequence_frames_from_files(
    seq_path: Path,
    voxel_size: float,
    depth_scale: float = DEPTH_SCALE_DEFAULT,
) -> Tuple[List[Tuple[str, np.ndarray]], Dict]:
    """Gather all frames for a sequence by loading from files.
    
    Args:
        seq_path: Sequence directory path
        voxel_size: Voxel size for point cloud downsampling
        depth_scale: Depth scale factor (meters per depth unit)
        
    Returns:
        Tuple of (frames list, metadata dict)
    """
    seq_id = seq_path.name
    
    # List all frames
    frame_ids = list_frames(seq_path)
    if not frame_ids:
        raise RuntimeError(f"No frames found in {seq_path}")
    
    # Load intrinsics
    intrinsic = load_camera_intrinsics(seq_path)
    
    # Load point clouds for each frame
    rgb_dir = seq_path / "rgb"
    depth_dir = seq_path / "depth"
    frames: List[Tuple[str, np.ndarray]] = []
    
    for frame_id in frame_ids:
        # Find RGB and depth files
        rgb_path = rgb_dir / f"{frame_id}.png"
        if not rgb_path.exists():
            rgb_path = rgb_dir / f"{frame_id}.jpg"
        
        depth_path = depth_dir / f"{frame_id}.png"
        if not depth_path.exists():
            depth_path = depth_dir / f"{frame_id}.jpg"
        
        if not rgb_path.exists() or not depth_path.exists():
            print(f"[WARN] Missing RGB or depth for frame {frame_id}, skipping")
            continue
        
        try:
            cloud = load_point_cloud_from_files(
                rgb_path, depth_path, intrinsic, depth_scale, voxel_size
            )
            frames.append((frame_id, cloud))
        except Exception as e:
            print(f"[WARN] Failed to load point cloud for frame {frame_id}: {e}")
            continue
    
    frames.sort(key=lambda tup: _frame_sort_key(tup[0]))
    meta = {
        "seq_id": seq_id,
        "data_path": str(seq_path),
        "total_frames": len(frames),
    }
    return frames, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize point clouds per frame using Rerun.")
    parser.add_argument("--data_path", type=str, default="data/moving", help="Dataset root path.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval", "all"], help="Dataset split.")
    parser.add_argument("--seq_id", type=str, default=None,
                        help="Sequence folder name to visualize (overrides --seq_index).")
    parser.add_argument("--seq_index", type=int, default=0,
                        help="Sequence index from listing when --seq_id is not provided.")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="Voxel size for point cloud downsampling.")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback speed for stepping through frames.")
    parser.add_argument("--point_radius", type=float, default=0.002, help="Point radius (meters) for Rerun markers.")
    parser.add_argument("--no_spawn", action="store_true", help="Do not spawn a separate Rerun viewer window.")
    parser.add_argument("--no_align_ref", action="store_true",
                        help="Disable transforming each frame into the first-frame coordinate system.")
    parser.add_argument("--align_calib_head", action="store_true",
                        help="Align clouds and poses to the calibrated head pose coordinate system.")
    parser.add_argument("--raw_frames", action="store_true",
                        help="Display point clouds and ball centers in each frame's original camera coordinate system without any transformation.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to display (from the beginning).")
    parser.add_argument("--end_frame_id", type=str, default=None,
                        help="Display frames up to and including this frame ID (e.g., '000100').")
    parser.add_argument("--tcp_to_zed", type=str, default=None,
                        help="Path to TCP to ZED camera transform matrix file (4x4 matrix). If provided, converts head_pos (TCP poses) to camera poses.")
    parser.add_argument("--ball_centers", type=str, default=None,
                        help="Path to ball_centers.txt file. If provided, visualizes ball centers in each frame.")
    parser.add_argument("--cam_to_base_npy", type=str, default=None,
                        help="Optional path to cam_to_base.npy to transform point clouds from camera to base coordinates.")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load TCP to ZED transform if provided
    tcp_to_zed_transform = None
    if args.tcp_to_zed is not None:
        tcp_to_zed_path = Path(args.tcp_to_zed)
        try:
            tcp_to_zed_transform = _load_tcp_to_zed_transform(tcp_to_zed_path)
            print(f"[INFO] Loaded TCP to ZED transform from {tcp_to_zed_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load TCP to ZED transform from {tcp_to_zed_path}: {e}") from e
    
    # Get data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist.")
    
    # Get sequence ID
    if args.seq_id is not None:
        seq_id = args.seq_id
    else:
        sequences = list_sequences(data_path, args.split)
        if args.seq_index < 0 or args.seq_index >= len(sequences):
            raise IndexError(f"seq_index {args.seq_index} out of range (found {len(sequences)} sequences)")
        seq_id = sequences[args.seq_index]
    
    # Get sequence path
    if args.split == 'all':
        seq_path = data_path / seq_id
    else:
        seq_path = data_path / args.split / seq_id
    
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequence path {seq_path} does not exist.")
    
    # Load frames and point clouds
    print(f"[INFO] Loading frames from {seq_path}")
    frame_clouds, meta = gather_sequence_frames_from_files(
        seq_path, args.voxel_size, DEPTH_SCALE_DEFAULT
    )
    if not frame_clouds:
        raise RuntimeError(f"No frames collected for sequence {seq_id}.")
    
    # Filter frames
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max_frames must be positive")
        frame_clouds = frame_clouds[:args.max_frames]
        print(f"[INFO] Limiting display to first {args.max_frames} frames")
    elif args.end_frame_id is not None:
        frame_ids_list = [fid for fid, _ in frame_clouds]
        try:
            end_idx = frame_ids_list.index(args.end_frame_id)
            frame_clouds = frame_clouds[:end_idx + 1]
            print(f"[INFO] Displaying frames up to and including {args.end_frame_id} (total: {len(frame_clouds)} frames)")
        except ValueError:
            available_frames = frame_ids_list[:10] if len(frame_ids_list) > 10 else frame_ids_list
            raise ValueError(
                f"Frame ID '{args.end_frame_id}' not found. "
                f"Available frames: {available_frames}..."
                if len(frame_ids_list) > 10
                else f"Frame ID '{args.end_frame_id}' not found. Available frames: {frame_ids_list}"
            )
    
    if not frame_clouds:
        raise RuntimeError("No frames to display after filtering.")
    
    frame_ids_ordered = [fid for fid, _ in frame_clouds]
    
    # Optional: load camera-to-base transforms (camera -> base) from .npy
    cam_to_base: dict[str, np.ndarray] | None = None
    if args.cam_to_base_npy is not None:
        cam_to_base_path = Path(args.cam_to_base_npy)
        if not cam_to_base_path.exists():
            raise FileNotFoundError(f"cam_to_base.npy not found at {cam_to_base_path}")
        cam_base_data = np.load(cam_to_base_path, allow_pickle=True).item()
        frame_ids_arr = cam_base_data.get("frame_ids", None)
        transforms_arr = cam_base_data.get("transforms", None)
        if frame_ids_arr is None or transforms_arr is None:
            raise ValueError(f"cam_to_base.npy at {cam_to_base_path} does not contain expected keys 'frame_ids' and 'transforms'.")
        if frame_ids_arr.shape[0] != transforms_arr.shape[0]:
            raise ValueError("Mismatch between number of frame_ids and transforms in cam_to_base.npy.")
        # Store using 6-digit string frame ids to match frame_clouds
        cam_to_base = {
            f"{int(fid):06d}": transforms_arr[idx].astype(np.float32)
            for idx, fid in enumerate(frame_ids_arr)
        }
        print(f"[INFO] Loaded cam_to_base transforms from {cam_to_base_path} ({len(cam_to_base)} frames)")
    
    # If cam_to_base is provided, transform point clouds from camera to base coordinates
    if cam_to_base is not None:
        print("[INFO] Transforming point clouds from camera to base coordinate system using cam_to_base.npy")
        frame_clouds_proc: List[Tuple[str, np.ndarray]] = []
        camera_poses: List[Tuple[str, np.ndarray]] = []
        for fid, cloud in frame_clouds:
            T_base_cam = cam_to_base.get(fid)
            if T_base_cam is None:
                print(f"[WARN] No cam_to_base transform for frame {fid}, leaving point cloud in camera coordinates")
                frame_clouds_proc.append((fid, cloud))
                continue
            points = cloud[:, :3].astype(np.float32)
            if points.size == 0:
                frame_clouds_proc.append((fid, cloud))
                continue
            ones = np.ones((points.shape[0], 1), dtype=np.float32)
            homo = np.concatenate([points, ones], axis=1)
            points_base = (T_base_cam @ homo.T).T[:, :3]
            cloud_new = cloud.copy()
            cloud_new[:, :3] = points_base
            frame_clouds_proc.append((fid, cloud_new))
            # Camera pose in base coordinates: base_T_cam
            camera_poses.append((fid, T_base_cam.astype(np.float32)))
        # In this mode, we are already in the base coordinate system
        target_pose: np.ndarray | None = None
        anchor_frames: List[Tuple[str, np.ndarray]] = [("base_frame", np.eye(4, dtype=np.float32))]
    else:
        # Load camera extrinsics
        extrinsics = load_camera_extrinsics(seq_path)
        
        # Get reference frame ID (first frame)
        ref_frame_id = None
        if frame_ids_ordered:
            try:
                ref_frame_id_int = int(frame_ids_ordered[0])
                ref_frame_id = frame_ids_ordered[0]
            except ValueError:
                pass
        
        ref_pose = None
        if ref_frame_id is not None:
            ref_frame_id_int = int(ref_frame_id)
            ref_pose = extrinsics.get(ref_frame_id_int)
        
        # Handle raw_frames mode: no transformation, display in original camera coordinate system
        if args.raw_frames:
            print("[INFO] Using raw_frames mode: displaying point clouds and ball centers in each frame's original camera coordinate system")
            # Use original point clouds without transformation
            frame_clouds_proc = list(frame_clouds)
            # Camera poses in original coordinate system (identity for each frame's camera)
            camera_poses = [(fid, np.eye(4, dtype=np.float32)) for fid in frame_ids_ordered]
            target_pose = None
            ref_frame_id = None
            anchor_frames = []
        else:
            align_to_ref = not args.no_align_ref and not args.align_calib_head
            
            # Determine target coordinate system and transform clouds
            target_pose: np.ndarray | None = None
            anchor_frames: List[Tuple[str, np.ndarray]] = []
            
            if args.align_calib_head:
                calib_world_pose = _load_calibrated_head_pose(seq_path)
                target_pose = calib_world_pose
                anchor_frames.append(("calib_head_frame", np.eye(4, dtype=np.float32)))
            elif align_to_ref:
                if ref_pose is None:
                    raise RuntimeError("Reference frame extrinsic is required for alignment.")
                target_pose = None  # Will use ref_frame_id
                anchor_frames.append(("ref_frame", np.eye(4, dtype=np.float32)))
            
            # Transform clouds to target coordinate system
            frame_clouds_proc = transform_clouds_to_coordinate_system(
                extrinsics,
                frame_clouds,
                target_pose=target_pose,
                ref_frame_id=ref_frame_id if align_to_ref else None,
                warn_prefix="vis_pointcloud",
            )
            
            # Get camera poses in target coordinate system
            camera_poses = get_camera_poses(
                extrinsics,
                frame_ids_ordered,
                target_pose=target_pose,
                ref_frame_id=ref_frame_id if align_to_ref else None,
                warn_prefix="vis_pointcloud",
                tcp_to_zed=tcp_to_zed_transform,
            )
    
    # Load ball centers if provided
    ball_centers_data = None
    if args.ball_centers is not None:
        ball_centers_path = Path(args.ball_centers)
        try:
            ball_centers_data = load_ball_centers(ball_centers_path)
            print(f"[INFO] Loaded ball centers from {ball_centers_path} ({len(ball_centers_data)} frames)")
        except Exception as e:
            print(f"[WARN] Failed to load ball centers from {ball_centers_path}: {e}")
    
    print(f"[INFO] Visualizing entire seq={seq_id} ({meta['total_frames']} unique frames) from {meta['data_path']}")

    # Get head_pos directory path for ball centers transformation
    # In raw_frames mode, don't transform ball centers (they're already in current frame camera coordinate system)
    head_pos_dir = None
    if ball_centers_data is not None and not args.raw_frames:
        seq_path = Path(meta["data_path"])
        head_pos_dir = seq_path / "head_pos"
        if not head_pos_dir.exists():
            head_pos_dir = None
            print(f"[WARN] head_pos directory not found at {seq_path / 'head_pos'}, ball centers will not be transformed")

    # Determine ref_frame_id for visualization (only used if cam_to_base is None, not raw_frames and align_to_ref)
    vis_ref_frame_id = None
    if cam_to_base is None and not args.raw_frames:
        align_to_ref = not args.no_align_ref and not args.align_calib_head
        if align_to_ref:
            vis_ref_frame_id = ref_frame_id
    
    visualize_sequence(
        frame_clouds=frame_clouds_proc,
        seq_id=meta["seq_id"],
        fps=args.fps,
        point_radius=args.point_radius,
        spawn_viewer=not args.no_spawn,
        camera_poses=camera_poses,
        anchor_frames=anchor_frames,
        ball_centers=ball_centers_data,
        head_pos_dir=head_pos_dir,
        target_pose=target_pose,
        ref_frame_id=vis_ref_frame_id,
    )


if __name__ == "__main__":
    main()
