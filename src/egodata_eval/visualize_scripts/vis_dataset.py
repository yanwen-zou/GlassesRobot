#!/usr/bin/env python3
"""
Visualize cached dataset samples frame-by-frame using rerun.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import numpy as np

here = Path(__file__).resolve()
project_root = here.parents[3]
mba_root = project_root / "MBA"
for path in (project_root, mba_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from MBA.utils.constants import IMG_MEAN, IMG_STD


def _rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - proj * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _pose_from_vec(pose_vec: np.ndarray) -> np.ndarray | None:
    if pose_vec is None:
        return None
    pose_vec = np.asarray(pose_vec, dtype=np.float32).reshape(-1)
    if pose_vec.size < 9:
        return None
    translation = pose_vec[:3]
    rot_6d = pose_vec[3:9]
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = _rotation_6d_to_matrix(rot_6d)
    mat[:3, 3] = translation
    return mat


def _parse_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        return 0
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize cached dataset samples.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Directory that holds cached samples (default: visualize_scripts/tmp/<data-dir-name>).",
    )
    parser.add_argument("--axis-len", type=float, default=0.15)
    parser.add_argument("--spawn", action="store_true", help="Spawn rerun viewer.")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    tmp_dir = args.tmp_dir
    if tmp_dir is None:
        tmp_dir = here.parent / "tmp" / data_dir.name
    if not tmp_dir.exists():
        raise FileNotFoundError(f"Cached samples not found: {tmp_dir}")

    sample_paths = sorted(tmp_dir.glob("sample_*.npz"), key=_parse_index)
    if args.max_frames is not None:
        sample_paths = sample_paths[: args.max_frames]
    if not sample_paths:
        raise RuntimeError(f"No cached samples found under {tmp_dir}")

    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("rerun package required. `pip install rerun-sdk`.") from exc

    rr.init(f"Dataset Visualization ({data_dir.name})", spawn=args.spawn)
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

    for frame_idx, sample_path in enumerate(sample_paths):
        data = np.load(sample_path, allow_pickle=True)
        clouds_list = data["clouds_list"]
        if isinstance(clouds_list, np.ndarray) and clouds_list.dtype == object:
            cloud = clouds_list[-1]
        else:
            cloud = clouds_list

        points = np.asarray(cloud[:, :3], dtype=np.float32)
        colors_norm = np.asarray(cloud[:, 3:6], dtype=np.float32)
        colors = np.clip(colors_norm * IMG_STD + IMG_MEAN, 0.0, 1.0)
        colors_u8 = (colors * 255).astype(np.uint8)

        rr.set_time_sequence("frame", frame_idx)
        rr.log(
            "frame/points",
            rr.Points3D(
                positions=points,
                colors=colors_u8,
                radii=np.full(points.shape[0], 0.003, dtype=np.float32),
            ),
        )

        obj_pose = _pose_from_vec(data.get("current_obj_pose"))
        if obj_pose is not None:
            log_axis("frame/object_pose", obj_pose, args.axis_len)

        head_pose = _pose_from_vec(data.get("current_headpose"))
        if head_pose is not None:
            log_axis("frame/head_pose", head_pose, args.axis_len * 0.8)

    print(f"[INFO] Visualized {len(sample_paths)} frames from {tmp_dir}")


if __name__ == "__main__":
    main()
