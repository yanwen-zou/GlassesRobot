#!/usr/bin/env python3
"""Visualize RealWorldDataset point clouds (already in ball/base frame) in Rerun.

The script mirrors the flow of vis_pointcloud_sequence: load frames, transform
points to the base frame, and stream them with camera poses.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Project paths
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MBA.dataset.realworld import RealWorldDataset  # noqa: E402
from MBA.utils.constants import IMG_MEAN, IMG_STD  # noqa: E402


def _import_rerun():
    try:
        import rerun as rr  # type: ignore
    except Exception as exc:
        raise RuntimeError("Rerun package is required. Install with `pip install rerun-sdk`.") from exc
    return rr


def _unnormalize_colors(colors_norm: np.ndarray) -> np.ndarray:
    """Convert normalized colors back to 0-255 uint8."""
    colors = (colors_norm * IMG_STD + IMG_MEAN) * 255.0
    colors = np.clip(colors, 0.0, 255.0).astype(np.uint8)
    return colors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize RealWorldDataset point clouds in Rerun.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to realworld dataset root.")
    parser.add_argument("--demo", type=str, required=True, help="Demo/episode directory name to visualize the full sequence.")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback speed.")
    parser.add_argument("--point-radius", type=float, default=0.002, help="Point radius for visualization.")
    parser.add_argument("--no-spawn", action="store_true", help="Do not spawn a separate Rerun viewer window.")
    parser.add_argument("--show-objects", action="store_true", help="Also visualize object poses if ob_in_cam exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = RealWorldDataset(
        path=args.data_path,
        split="all",
        with_cloud=True,
        with_obj_action=False,
        cam_to_base_rot_noise_std=0.0,  # visualization/inference: no extrinsic noise
    )

    demo_name = args.demo
    if demo_name not in dataset.seq_intrinsics:
        raise ValueError(f"Demo '{demo_name}' not found under {dataset.data_path}.")
    demo_path = Path(dataset.data_path) / demo_name
    rgb_dir = demo_path / "rgb"
    depth_dir = demo_path / "depth"
    if not rgb_dir.exists() or not depth_dir.exists():
        raise FileNotFoundError(f"rgb/depth folders missing in {demo_path}")
    frame_files = sorted(
        f for f in rgb_dir.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    )
    frame_ids = [f.stem for f in frame_files]
    cam_intrinsic = dataset.seq_intrinsics[demo_name]
    clouds_list = []
    for fid in frame_ids:
        rgb_path = rgb_dir / f"{fid}.png"
        if not rgb_path.exists():
            rgb_path = rgb_dir / f"{fid}.jpg"
        depth_path = depth_dir / f"{fid}.png"
        if not depth_path.exists():
            depth_path = depth_dir / f"{fid}.jpg"
        if not rgb_path.exists() or not depth_path.exists():
            print(f"[WARN] Missing rgb/depth for frame {fid}, skipping.")
            continue
        colors_np = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        depth_np = np.array(Image.open(depth_path), dtype=np.float32)
        points, colors = dataset.load_point_cloud(colors_np, depth_np, cam_intrinsic)
        T_base_cam = dataset._get_cam_to_base(demo_name, fid)  # noqa: SLF001
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        homo = np.concatenate([points, ones], axis=1)
        points_base = (T_base_cam @ homo.T).T[:, :3]
        colors_uint8 = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
        colors_norm = (colors_uint8.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
        cloud = np.concatenate([points_base, colors_norm], axis=1)
        clouds_list.append((fid, cloud))
    # Sort by frame id numeric order where possible
    def _sort_key(fid: str):
        try:
            return int(fid)
        except ValueError:
            return fid
    clouds_list.sort(key=lambda tup: _sort_key(tup[0]))
    frame_ids = [fid for fid, _ in clouds_list]
    clouds = [c for _, c in clouds_list]
    seq_id = demo_name
    obj_dir = demo_path / "ob_in_cam"

    rr = _import_rerun()
    rr.init(f"RealWorld[{seq_id}]", spawn=not args.no_spawn)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    entity_path = f"sample/{seq_id}/cloud"
    cam_entity = f"sample/{seq_id}/camera_pose"
    cam_path_entity = f"sample/{seq_id}/camera_path"
    obj_entity = f"sample/{seq_id}/object"
    rr.log(f"sample/{seq_id}", rr.Transform3D())
    rr.log(entity_path, rr.Transform3D())
    rr.log(f"sample/{seq_id}/base_frame", rr.Transform3D())

    cam_points: list[np.ndarray] = []
    dt = 1.0 / args.fps if args.fps > 1e-6 else 0.0

    for idx, (frame_id, cloud) in enumerate(zip(frame_ids, clouds)):
        if cloud.size == 0:
            continue

        positions = cloud[:, :3].astype(np.float32)
        color_data = cloud[:, 3:6] if cloud.shape[1] >= 6 else None
        if color_data is not None:
            colors = _unnormalize_colors(color_data)
        else:
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
                radii=args.point_radius,
            ),
        )

        # Log camera pose if available
        T_base_cam = dataset._get_cam_to_base(seq_id, frame_id)  # noqa: SLF001 (intentional use)
        trans = T_base_cam[:3, 3].astype(np.float32)
        rot = T_base_cam[:3, :3].astype(np.float32)
        rr.log(cam_entity, rr.Transform3D(translation=trans, mat3x3=rot))
        cam_points.append(trans)
        rr.log(
            cam_path_entity,
            rr.LineStrips3D(
                [np.asarray(cam_points, dtype=np.float32)],
                radii=args.point_radius,
                colors=np.array([[255, 200, 0, 255]], dtype=np.uint8),
            ),
        )
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

        # Log object pose if available and requested
        if args.show_objects and obj_dir.exists():
            obj_path = obj_dir / f"{frame_id}.txt"
            if obj_path.exists():
                pose_values = np.loadtxt(obj_path).astype(np.float32).reshape(-1)
                if pose_values.size == 16:
                    pose_mat = pose_values.reshape(4, 4)
                elif pose_values.size == 12:
                    pose_mat = np.vstack([pose_values.reshape(3, 4), np.array([0, 0, 0, 1], dtype=np.float32)])
                else:
                    pose_mat = None
                if pose_mat is not None:
                    pose_base = T_base_cam @ pose_mat
                    rr.log(obj_entity, rr.Transform3D(translation=pose_base[:3, 3], mat3x3=pose_base[:3, :3]))
                    rr.log(
                        f"{obj_entity}/axes",
                        rr.Arrows3D(
                            origins=np.zeros((3, 3), dtype=np.float32),
                            vectors=np.eye(3, dtype=np.float32) * 0.03,
                            colors=np.array(
                                [
                                    [200, 50, 50, 255],
                                    [50, 200, 50, 255],
                                    [50, 50, 200, 255],
                                ],
                                dtype=np.uint8,
                            ),
                        ),
                    )

        if dt > 0:
            time.sleep(dt)


if __name__ == "__main__":
    main()
