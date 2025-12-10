"""
Visualize per-frame point clouds using RealWorldDataset cam_to_base.

Loads RGB and depth from an episode, converts each frame's points into the base
frame via cam_to_base computed in MBA/dataset/realworld.py, and streams to
Rerun. The first cam_to_base entry anchors the base frame.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import rerun as rr
from PIL import Image

from dataset.realworld import RealWorldDataset


def load_rgb(rgb_path: Path) -> np.ndarray:
    return np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0


def load_depth(depth_path: Path) -> np.ndarray:
    depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
    return depth_raw


def iter_frames(frame_ids: Iterable[str]) -> Iterable[int]:
    for fid in sorted(frame_ids, key=lambda x: int(x)):
        yield int(fid)


def _load_pose_matrix(path: Path) -> np.ndarray | None:
    """Load a 4x4 pose matrix from txt/npy supporting 16/12 vector forms."""
    if not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        vals = np.load(path).astype(np.float32)
    else:
        vals = np.loadtxt(path).astype(np.float32)
    if vals.ndim == 1:
        if vals.size == 16:
            mat = vals.reshape(4, 4)
        elif vals.size == 12:
            mat = np.vstack([vals.reshape(3, 4), np.array([0, 0, 0, 1], dtype=np.float32)])
        else:
            return None
    else:
        mat = vals
    if mat.shape == (3, 4):
        mat = np.vstack([mat, np.array([0, 0, 0, 1], dtype=np.float32)])
    if mat.shape != (4, 4):
        return None
    return mat


def main():
    parser = argparse.ArgumentParser(description="Stream per-frame point clouds (base frame) using RealWorldDataset.")
    parser.add_argument("--data_path", type=Path, default=Path("data"), help="Root dataset path (contains train/eval splits).")
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval", "all"], help="Dataset split to use.")
    parser.add_argument("--seq-index", type=int, default=0, help="Episode index within the split.")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Meters per depth unit (default 1000 for mm).")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback speed (frames per second).")
    parser.add_argument("--point-radius", type=float, default=0.002, help="Point radius for Rerun markers.")
    parser.add_argument("--no-spawn", action="store_true", help="Do not spawn a separate Rerun viewer window.")
    parser.add_argument("--show-objects", action="store_true", help="Visualize ob_in_cam poses transformed to base frame.")

    args = parser.parse_args()

    ds = RealWorldDataset(
        args.data_path,
        split=args.split,
        num_obs=1,
        num_action=1,
        with_obj_action=False,
        cam_to_base_rot_noise_std=0.0,
    )
    if args.seq_index < 0 or args.seq_index >= len(ds.all_demos):
        raise IndexError(f"seq-index {args.seq_index} out of range (0..{len(ds.all_demos)-1}).")
    seq_id = ds.all_demos[args.seq_index]
    episode_dir = Path(ds.data_paths[ds.seq_ids.index(seq_id)])  # first occurrence

    rgb_dir = episode_dir / "rgb"
    depth_dir = episode_dir / "depth"
    intrinsic = ds.seq_intrinsics[seq_id]

    frame_files = [f for f in os.listdir(rgb_dir) if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg"]]
    frame_ids = [os.path.splitext(f)[0] for f in frame_files]
    if not frame_ids:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")

    rr.init("points_sequence_realworld", spawn=not args.no_spawn)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    cam_path_pts: list[np.ndarray] = []
    T_base_cam0: np.ndarray | None = None
    obj_dir = episode_dir / "ob_in_cam"
    obj_positions_base: list[np.ndarray] = []

    for fid in iter_frames(frame_ids):
        fid_key = f"{fid:06d}"
        depth_path = depth_dir / f"{fid_key}.png"
        rgb_path = rgb_dir / f"{fid_key}.png"
        if not depth_path.exists() or not rgb_path.exists():
            continue

        depth_m = load_depth(depth_path)
        rgb = load_rgb(rgb_path)

        T_base_cam = ds._get_cam_to_base(seq_id, fid_key)
        if T_base_cam0 is None:
            T_base_cam0 = T_base_cam
        translation = T_base_cam[:3, 3].astype(np.float32)
        rotation = T_base_cam[:3, :3].astype(np.float32)

        # Base frame (ball frame)
        rr.log("world/base", rr.Transform3D())
        rr.log(
            "world/base/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3), dtype=np.float32),
                vectors=np.eye(3, dtype=np.float32) * 0.05,
                colors=np.array(
                    [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
                    dtype=np.uint8,
                ),
            ),
        )

        rr.log("world/cam_current", rr.Transform3D(translation=translation, mat3x3=rotation))
        rr.log(
            "world/cam_current/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3), dtype=np.float32),
                vectors=np.eye(3, dtype=np.float32) * 0.05,
                colors=np.array(
                    [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
                    dtype=np.uint8,
                ),
            ),
        )
        rr.set_time("frame", sequence=fid)

        full_pts, full_cols = ds.load_point_cloud(
            (rgb * 255).astype(np.uint8),
            depth_m,
            intrinsic,
            depth_scale=args.depth_scale,
            T_base_cam=T_base_cam,
        )
        if full_pts.size > 0:
            dist_full = np.linalg.norm(full_pts, axis=1)
            keep_full = dist_full <= 1.2
            full_pts = full_pts[keep_full]
            if full_cols is not None:
                full_cols = full_cols[keep_full]
        if full_pts.size > 0:
            full_cols_u8 = np.clip(full_cols * 255.0, 0, 255).astype(np.uint8) if full_cols is not None else None
            if full_cols_u8 is not None and full_cols_u8.shape[1] == 3:
                alpha = 255 * np.ones((full_cols_u8.shape[0], 1), dtype=np.uint8)
                full_cols_u8 = np.concatenate([full_cols_u8, alpha], axis=1)
            rr.log(
                "world/frame_cloud",
                rr.Points3D(
                    positions=full_pts,
                    colors=full_cols_u8,
                    radii=args.point_radius,
                ),
            )

        # Visualize ob_in_cam transformed into base frame
        if args.show_objects and obj_dir.exists():
            pose_path_txt = obj_dir / f"{fid_key}.txt"
            pose_path_npy = obj_dir / f"{fid_key}.npy"
            pose_cam = _load_pose_matrix(pose_path_txt)
            if pose_cam is None:
                pose_cam = _load_pose_matrix(pose_path_npy)
            if pose_cam is not None:
                pose_base = T_base_cam @ pose_cam
                obj_positions_base.append(pose_base[:3, 3].astype(np.float32))
                rr.log(
                    "world/object",
                    rr.Transform3D(translation=pose_base[:3, 3].astype(np.float32), mat3x3=pose_base[:3, :3].astype(np.float32)),
                )
                rr.log(
                    "world/object/axes",
                    rr.Arrows3D(
                        origins=np.zeros((3, 3), dtype=np.float32),
                        vectors=np.eye(3, dtype=np.float32) * 0.03,
                        colors=np.array(
                            [[200, 50, 50, 255], [50, 200, 50, 255], [50, 50, 200, 255]],
                            dtype=np.uint8,
                        ),
                    ),
                )
                if obj_positions_base:
                    rr.log(
                        "world/object_path",
                        rr.LineStrips3D(
                            [np.asarray(obj_positions_base, dtype=np.float32)],
                            radii=args.point_radius * 0.8,
                            colors=np.array([[180, 100, 255, 255]], dtype=np.uint8),
                        ),
                    )


if __name__ == "__main__":
    main()
