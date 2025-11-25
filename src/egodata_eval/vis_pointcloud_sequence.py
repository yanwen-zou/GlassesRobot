#!/usr/bin/env python3
"""Stream RealWorldDataset point clouds frame-by-frame in Rerun.

Example:
    python src/egodata_eval/vis_pointcloud_sequence.py \
        --data_path data/moving --sample_index 0 --fps 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MBA.dataset.realworld import RealWorldDataset, collate_fn  # type: ignore
from MBA.utils.constants import IMG_MEAN, IMG_STD  # type: ignore

AXIS_MIRROR = np.diag([1.0, -1.0, 1.0]).astype(np.float32)


def _import_rerun():
    try:
        import rerun as rr  # type: ignore
    except Exception as exc:  # pragma: no cover - rerun is optional at install time
        raise RuntimeError("Rerun package is required. Install with `pip install rerun-sdk`.") from exc
    return rr


def _denormalize_colors(norm_colors: np.ndarray) -> np.ndarray:
    colors = norm_colors * IMG_STD + IMG_MEAN
    colors = np.clip(colors, 0.0, 1.0)
    return (colors * 255).astype(np.uint8)


def _build_dataloader(dataset: RealWorldDataset,
                      num_workers: int = 0) -> torch.utils.data.DataLoader:
    sampler = torch.utils.data.SequentialSampler(dataset)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )


def load_data_point(dataset: RealWorldDataset,
                    sample_index: int,
                    num_workers: int = 0) -> Tuple[Dict, Dict]:
    """Load a single sample via DataLoader to honor collate_fn behavior."""
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index {sample_index} out of range (dataset has {len(dataset)} samples)")

    dataloader = _build_dataloader(dataset, num_workers)
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == sample_index:
            meta = {
                "seq_id": dataset.seq_ids[sample_index],
                "obs_frame_ids": dataset.obs_frame_ids[sample_index],
                "action_frame_ids": dataset.action_frame_ids[sample_index],
                "data_path": dataset.data_paths[sample_index],
            }
            return batch, meta
    raise RuntimeError(f"Failed to load sample_index {sample_index} via DataLoader.")


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


def _mirror_pose_y(T: np.ndarray, ctx: str = "") -> np.ndarray:
    """Mirror pose along Y-axis (Y-up to Y-down coordinate system conversion)."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_new = AXIS_MIRROR @ R @ AXIS_MIRROR
    t_new = AXIS_MIRROR @ t
    T_new = np.eye(4, dtype=np.float32)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = t_new
    if ctx:
        print(f"[DEBUG] mirror_pose({ctx})=\n{T_new}")
    return T_new


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
    """Load calibrated head pose from calibrated_head_pose.txt and convert to standard camera coordinate system."""
    head_pose_path = seq_path / "calibrated_head_pose.txt"
    if not head_pose_path.exists():
        raise FileNotFoundError(f"Missing calibrated head pose at {head_pose_path}")
    head_vals = np.loadtxt(head_pose_path).astype(np.float32)
    if head_vals.ndim != 1 or head_vals.size != 7:
        raise ValueError(f"Invalid contents in {head_pose_path}")
    world_T_cam = _pose_from_seven(head_vals)
    world_T_cam = _mirror_pose_y(world_T_cam, ctx="calib_head_pose")
    return world_T_cam.astype(np.float32)


def _load_aruco_transform(seq_path: Path) -> np.ndarray:
    """Load ArUco transform from calibrated_transform.txt (cam_T_aruco, NOT Y-mirrored)."""
    aruco_path = seq_path / "calibrated_transform.txt"
    if not aruco_path.exists():
        raise FileNotFoundError(f"Missing calibrated transform at {aruco_path}")
    return _load_matrix4x4(aruco_path)


def _load_aruco_poses(seq_path: Path, ref_pose: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    """Load ArUco poses in different coordinate systems.
    
    Returns:
        Dictionary with keys:
        - 'world': ArUco pose in world coordinate system
        - 'calib_head': ArUco pose in calibrated head pose coordinate system
        - 'first': ArUco pose in first frame coordinate system (if ref_pose provided)
        - 'camera_at_aruco_first': Camera pose when seeing ArUco, in first frame coordinate system
    """
    result: Dict[str, np.ndarray] = {}
    
    # Load base data
    head_pose_path = seq_path / "calibrated_head_pose.txt"
    if not head_pose_path.exists():
        return result
    
    head_vals = np.loadtxt(head_pose_path).astype(np.float32)
    if head_vals.ndim != 1 or head_vals.size != 7:
        return result
    
    world_T_cam = _pose_from_seven(head_vals)
    world_T_cam = _mirror_pose_y(world_T_cam, ctx="aruco_head_pose")
    
    # Load ArUco transform (cam_T_aruco, NOT Y-mirrored)
    cam_T_aruco = _load_aruco_transform(seq_path)
    
    # World coordinate system: world_T_aruco = world_T_cam @ cam_T_aruco
    result['world'] = (world_T_cam @ cam_T_aruco).astype(np.float32)
    
    # Calib head coordinate system: In calib_head frame, camera is at identity, so ArUco is directly cam_T_aruco
    result['calib_head'] = cam_T_aruco.astype(np.float32)
    
    # First frame coordinate system (if ref_pose provided)
    if ref_pose is not None:
        first_inv = np.linalg.inv(ref_pose.astype(np.float32))
        first_T_cam = first_inv @ world_T_cam
        result['camera_at_aruco_first'] = first_T_cam.astype(np.float32)
        result['first'] = (first_T_cam @ cam_T_aruco).astype(np.float32)
    
    return result


def transform_clouds_to_coordinate_system(
    dataset: RealWorldDataset,
    seq_id: str,
    frame_clouds: Sequence[Tuple[str, np.ndarray]],
    target_pose: np.ndarray | None = None,
    ref_frame_id: str | None = None,
    warn_prefix: str = "vis_pointcloud",
) -> List[Tuple[str, np.ndarray]]:
    """Transform point clouds to a target coordinate system.
    
    Args:
        dataset: RealWorldDataset instance
        seq_id: Sequence ID
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
        ref_extr = dataset.get_camera_extrinsic(seq_id, int(ref_frame_id), warn_prefix=f"{warn_prefix}(ref)")
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
        
        cam_extr = dataset.get_camera_extrinsic(seq_id, int(frame_id), warn_prefix=warn_prefix)
        T = target_inv @ cam_extr
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        homo = np.concatenate([points, ones], axis=1)
        points_target = (T @ homo.T).T[:, :3]
        cloud_new = cloud.copy()
        cloud_new[:, :3] = points_target
        transformed.append((frame_id, cloud_new))
    
    return transformed


def get_camera_poses(
    dataset: RealWorldDataset,
    seq_id: str,
    frame_ids: Sequence[str],
    target_pose: np.ndarray | None = None,
    ref_frame_id: str | None = None,
    warn_prefix: str = "vis_pointcloud",
    tcp_to_zed: np.ndarray | None = None,
) -> List[Tuple[str, np.ndarray]]:
    """Get camera poses in a target coordinate system.
    
    Args:
        dataset: RealWorldDataset instance
        seq_id: Sequence ID
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
        ref_extr = dataset.get_camera_extrinsic(seq_id, int(ref_frame_id), warn_prefix=f"{warn_prefix}(cam_ref)")
        # Apply TCP to ZED transform if provided
        if tcp_to_zed is not None:
            ref_extr = ref_extr @ tcp_to_zed
        ref_inv = np.linalg.inv(ref_extr)
    
    poses: List[Tuple[str, np.ndarray]] = []
    for fid in frame_ids:
        cam_extr = dataset.get_camera_extrinsic(seq_id, int(fid), warn_prefix=f"{warn_prefix}(cam)")
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
    aruco_pose: np.ndarray | None = None,
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
    cam_pose_map = {fid: pose for fid, pose in camera_poses} if camera_poses else {}
    cam_points: list[np.ndarray] = []

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
        
        # Handle colors: check if they need denormalization
        if cloud.shape[1] >= 6:
            color_data = cloud[:, 3:6]
            # Check if colors are in normalized range (0-1) or already denormalized
            if color_data.max() <= 1.0:
                # Colors are in 0-1 range, convert to 0-255
                colors = (color_data * 255.0).astype(np.uint8)
            else:
                # Colors are already denormalized (RealWorldDataset format)
                colors = _denormalize_colors(color_data)
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
            
            # Draw line from camera to ArUco if both poses are available
            if aruco_pose is not None:
                cam_pos = pose[:3, 3].astype(np.float32)
                aruco_pos = aruco_pose[:3, 3].astype(np.float32)
                line_entity = f"sample/{seq_id}/camera_to_aruco"
                rr.log(
                    line_entity,
                    rr.LineStrips3D(
                        [np.array([cam_pos, aruco_pos], dtype=np.float32)],
                        radii=point_radius * 2.0,
                        colors=np.array([[255, 0, 255, 255]], dtype=np.uint8),
                    ),
                )
        if dt > 0:
            time.sleep(dt)


def _frame_sort_key(frame_id: str) -> Tuple[int, int | str]:
    """Sort key for frame IDs: numeric frames first, then string frames."""
    try:
        return (0, int(frame_id))
    except ValueError:
        return (1, frame_id)


def gather_sequence_frames(
    dataset: RealWorldDataset,
    seq_id: str,
    num_workers: int = 0,
) -> Tuple[List[Tuple[str, np.ndarray]], Dict]:
    """Gather all frames for a sequence."""
    indices = [i for i, sid in enumerate(dataset.seq_ids) if sid == seq_id]
    if not indices:
        raise ValueError(f"Sequence id '{seq_id}' not found in dataset.")

    dataloader = _build_dataloader(dataset, num_workers)
    target_set = set(indices)
    frames: List[Tuple[str, np.ndarray]] = []
    seen_frames: set[str] = set()
    for idx, batch in enumerate(dataloader):
        if idx not in target_set:
            continue
        clouds_batch = batch.get("clouds_list")
        if not clouds_batch:
            continue
        frame_ids = dataset.obs_frame_ids[idx]
        cloud_sequence = clouds_batch[0]
        for fid, cloud in zip(frame_ids, cloud_sequence):
            if fid in seen_frames:
                continue
            frames.append((fid, cloud))
            seen_frames.add(fid)
        target_set.remove(idx)
        if not target_set:
            break

    frames.sort(key=lambda tup: _frame_sort_key(tup[0]))
    meta = {
        "seq_id": seq_id,
        "data_path": dataset.data_paths[indices[0]],
        "total_frames": len(frames),
    }
    return frames, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dataset point clouds per frame using Rerun.")
    parser.add_argument("--data_path", type=str, default="data/moving", help="Dataset root path.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval", "all"], help="Dataset split.")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="Dataset sample index (used when --mode sample).")
    parser.add_argument("--seq_id", type=str, default=None,
                        help="Sequence folder name to visualize (overrides --seq_index) when --mode sequence.")
    parser.add_argument("--seq_index", type=int, default=0,
                        help="Sequence index from dataset listing when --seq_id is not provided.")
    parser.add_argument("--num_obs", type=int, default=1, help="Number of observation frames per sample.")
    parser.add_argument("--num_action", type=int, default=20, help="Number of action frames per sample (affects padding).")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="Voxel size passed to RealWorldDataset.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback speed for steppqiting through frames.")
    parser.add_argument("--point_radius", type=float, default=0.002, help="Point radius (meters) for Rerun markers.")
    parser.add_argument("--no_spawn", action="store_true", help="Do not spawn a separate Rerun viewer window.")
    parser.add_argument("--no_align_ref", action="store_true",
                        help="Disable transforming each frame into the first-frame coordinate system.")
    parser.add_argument("--align_calib_head", action="store_true",
                        help="Align clouds and poses to the calibrated head pose coordinate system.")
    parser.add_argument("--show_aruco_frame", action="store_true",
                        help="Display the calibrated ArUco frame (requires calibrated files).")
    parser.add_argument("--show_aruco_in_calib", action="store_true",
                        help="When aligning to calibrated head pose, also show ArUco frame in that coordinate system.")
    parser.add_argument("--align_aruco", action="store_true",
                        help="Align clouds and poses to the ArUco coordinate system.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to display (from the beginning).")
    parser.add_argument("--end_frame_id", type=str, default=None,
                        help="Display frames up to and including this frame ID (e.g., '000100').")
    parser.add_argument("--tcp_to_zed", type=str, default=None,
                        help="Path to TCP to ZED camera transform matrix file (4x4 matrix). If provided, converts head_pos (TCP poses) to camera poses.")
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
    
    dataset = RealWorldDataset(
        path=args.data_path,
        split=args.split,
        num_obs=args.num_obs,
        num_action=args.num_action,
        voxel_size=args.voxel_size,
        with_cloud=True,
        with_obj_action=False,
        aug=False,
        aug_jitter=False,
    )

    # Get sequence ID
    if args.seq_id is not None:
        seq_id = args.seq_id
    else:
        if args.seq_index < 0 or args.seq_index >= len(dataset.all_demos):
            raise IndexError(f"seq_index {args.seq_index} out of range (dataset has {len(dataset.all_demos)} demos)")
        seq_id = dataset.all_demos[args.seq_index]
    frame_clouds, meta = gather_sequence_frames(dataset, seq_id, args.num_workers)
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
    ref_frame_id = dataset.seq_ref_frame.get(meta["seq_id"])
    ref_pose = (
        dataset.get_camera_extrinsic(meta["seq_id"], int(ref_frame_id), warn_prefix="vis_pointcloud(ref_base)")
        if ref_frame_id is not None
        else None
    )
    
    seq_path = Path(meta["data_path"])
    align_to_ref = not args.no_align_ref and not args.align_calib_head and not args.align_aruco
    
    # Load ArUco poses
    aruco_poses = _load_aruco_poses(seq_path, ref_pose)
    
    # Determine target coordinate system and transform clouds
    target_pose: np.ndarray | None = None
    anchor_frames: List[Tuple[str, np.ndarray]] = []
    aruco_pose_for_line: np.ndarray | None = None
    
    if args.align_aruco:
        target_pose = aruco_poses.get('world')
        if target_pose is None:
            raise RuntimeError("ArUco pose in world is required for --align_aruco.")
        anchor_frames.append(("aruco_frame", np.eye(4, dtype=np.float32)))
        aruco_pose_for_line = np.eye(4, dtype=np.float32)
    elif args.align_calib_head:
        calib_world_pose = _load_calibrated_head_pose(seq_path)
        target_pose = calib_world_pose
        anchor_frames.append(("calib_head_frame", np.eye(4, dtype=np.float32)))
        if args.show_aruco_frame or args.show_aruco_in_calib:
            aruco_pose_calib = aruco_poses.get('calib_head')
            if aruco_pose_calib is not None:
                anchor_frames.append(("aruco_frame", aruco_pose_calib))
                aruco_pose_for_line = aruco_pose_calib
    elif align_to_ref:
        if ref_pose is None:
            raise RuntimeError("Reference frame extrinsic is required for alignment.")
        target_pose = None  # Will use ref_frame_id
        anchor_frames.append(("ref_frame", np.eye(4, dtype=np.float32)))
        if args.show_aruco_frame:
            aruco_cam_first = aruco_poses.get('camera_at_aruco_first')
            aruco_pose_first = aruco_poses.get('first')
            if aruco_cam_first is not None:
                anchor_frames.append(("camera_at_aruco", aruco_cam_first))
            if aruco_pose_first is not None:
                anchor_frames.append(("aruco_frame", aruco_pose_first))
                aruco_pose_for_line = aruco_pose_first
    elif args.show_aruco_frame:
        aruco_cam_first = aruco_poses.get('camera_at_aruco_first')
        aruco_pose_first = aruco_poses.get('first')
        if aruco_cam_first is not None:
            anchor_frames.append(("camera_at_aruco", aruco_cam_first))
        if aruco_pose_first is not None:
            anchor_frames.append(("aruco_frame", aruco_pose_first))
            aruco_pose_for_line = aruco_pose_first
    
    # Transform clouds to target coordinate system
    frame_clouds_proc = transform_clouds_to_coordinate_system(
        dataset,
        meta["seq_id"],
        frame_clouds,
        target_pose=target_pose,
        ref_frame_id=ref_frame_id if align_to_ref else None,
        warn_prefix="vis_pointcloud",
    )
    
    # Get camera poses in target coordinate system
    camera_poses = get_camera_poses(
        dataset,
        meta["seq_id"],
        frame_ids_ordered,
        target_pose=target_pose,
        ref_frame_id=ref_frame_id if align_to_ref else None,
        warn_prefix="vis_pointcloud",
        tcp_to_zed=tcp_to_zed_transform,
    )
    
    print(f"[INFO] Visualizing entire seq={seq_id} ({meta['total_frames']} unique frames) from {meta['data_path']}")

    visualize_sequence(
        frame_clouds=frame_clouds_proc,
        seq_id=meta["seq_id"],
        fps=args.fps,
        point_radius=args.point_radius,
        spawn_viewer=not args.no_spawn,
        camera_poses=camera_poses,
        anchor_frames=anchor_frames,
        aruco_pose=aruco_pose_for_line,
    )


if __name__ == "__main__":
    main()
