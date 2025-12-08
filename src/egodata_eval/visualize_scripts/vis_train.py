#!/usr/bin/env python3
"""
Pre-compute training-set object pose trajectories (base + robot frames).

Results are saved to a temporary file for visualization via vis_train_rerun.py.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MBA_ROOT = REPO_ROOT / "MBA"
if MBA_ROOT.exists() and str(MBA_ROOT) not in sys.path:
    sys.path.insert(0, str(MBA_ROOT))

from MBA.dataset.realworld import RealWorldDataset


def _load_transform_like_realworld(path: Path) -> np.ndarray:
    data = np.loadtxt(path).astype(np.float32)
    if data.ndim == 1:
        if data.size == 16:
            mat = data.reshape(4, 4)
        elif data.size == 12:
            mat = np.vstack([data.reshape(3, 4), np.array([0, 0, 0, 1], dtype=np.float32)])
        elif data.size == 7:
            x, y, z, qx, qy, qz, qw = data
            mat = np.eye(4, dtype=np.float32)
            mat[:3, 3] = np.array([x, y, z], dtype=np.float32)
            q = np.array([qx, qy, qz, qw], dtype=np.float32)
            norm = np.linalg.norm(q)
            if norm < 1e-8:
                raise ValueError(f"Quaternion norm too small in {path}")
            q /= norm
            qx, qy, qz, qw = q
            mat[:3, :3] = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unexpected vector length {data.size} when parsing {path}")
    else:
        mat = data
    if mat.shape == (3, 4):
        mat = np.vstack([mat, np.array([0, 0, 0, 1], dtype=np.float32)])
    if mat.shape != (4, 4):
        raise ValueError(f"Invalid SE3 matrix shape {mat.shape} in {path}")
    return mat.astype(np.float32)


def _load_object_pose(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Object pose file missing: {path}")
    return _load_transform_like_realworld(path)


def _gather_frame_ids(obj_dir: Path) -> List[str]:
    files = sorted(obj_dir.glob("*.txt"))
    if not files:
        return []
    ids = []
    for path_obj in files:
        ids.append(path_obj.stem)
    return ids


def main():
    parser = argparse.ArgumentParser(description="Pre-compute training trajectories (object poses).")
    parser.add_argument("--data-path", type=Path, required=True, help="Directory containing sequence folders.")
    parser.add_argument("--T_robot_base", type=Path, default=Path("glasses_hardware/calib/T_robot_base.npy"))
    parser.add_argument("--head-to-zed", type=Path, default=Path("glasses_hardware/calib/T_tcp_zed.npy"))
    parser.add_argument("--axis-len", type=float, default=0.25)
    parser.add_argument("--max-seqs", type=int, default=None, help="Optional limit on number of sequences visualized.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vis_train_temp.npy"),
        help="Where to save the temporary trajectory data (.npy).",
    )
    args = parser.parse_args()

    T_robot_base = np.load(args.T_robot_base).astype(np.float32)
    if T_robot_base.shape != (4, 4):
        raise ValueError(f"T_robot_base must be 4x4, got {T_robot_base.shape}")

    data_path = args.data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist.")
    dataset = RealWorldDataset(
        path=str(data_path),
        split="all",
        num_obs=1,
        num_action=1,
        with_cloud=False,
        with_obj_action=False,
        aug=False,
        head_to_zed_path=str(args.head_to_zed),
    )
    seq_dirs = [Path(dataset.data_path) / seq for seq in dataset.all_demos]
    if args.max_seqs is not None:
        seq_dirs = seq_dirs[: max(args.max_seqs, 0)]

    seq_count = 0
    seq_entries = []
    for seq_dir in seq_dirs:
        obj_dir = seq_dir / "ob_in_cam"
        if not obj_dir.exists():
            warnings.warn(f"[vis_train] Skipping {seq_dir.name}: ob_in_cam missing.")
            continue
        frame_ids = _gather_frame_ids(obj_dir)
        if not frame_ids:
            warnings.warn(f"[vis_train] Skipping {seq_dir.name}: no object pose files.")
            continue
        seq_id = seq_dir.name
        pts_base: List[np.ndarray] = []
        pts_robot: List[np.ndarray] = []
        for fid in frame_ids:
            pose_path = obj_dir / f"{fid}.txt"
            pose_cam = _load_object_pose(pose_path)
            T_base_cam = dataset._get_cam_to_base(seq_id, fid)
            T_base_obj = T_base_cam @ pose_cam
            T_robot_obj = T_robot_base @ T_base_obj
            pts_base.append(T_base_obj[:3, 3].astype(np.float32))
            pts_robot.append(T_robot_obj[:3, 3].astype(np.float32))
        if not pts_robot:
            continue
        seq_entries.append(
            {
                "seq": seq_dir.name,
                "pts_robot": np.asarray(pts_robot, dtype=np.float32),
                "pts_base": np.asarray(pts_base, dtype=np.float32),
            }
        )
        seq_count += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_root": str(Path(dataset.data_path).resolve()),
        "T_robot_base": T_robot_base.astype(np.float32),
        "axis_len": float(args.axis_len),
        "sequences": seq_entries,
    }
    np.save(args.output, payload, allow_pickle=True)
    print(f"[OK] Saved {seq_count} sequences to {args.output}")


if __name__ == "__main__":
    main()
