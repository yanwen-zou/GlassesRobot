#!/usr/bin/env python3
"""
Offline evaluation script that replays a dataset of RGB/depth/object-pose frames,
feeds them into the trajectory predictor, and stores results in train_output/episode/*.

Expected dataset layout under --data-path:
    rgb/000000.png, ...
    depth/000000.npy (meters) or png (depth-in-mm)
    ob_in_cam/000000.npy (4x4 pose matrices)
    cam_to_base/000000.npy (4x4, camera->base for first frame; reused for all frames)
    optional: intrinsics.npy (3x3)

Outputs (for vis_eval.py):
    train_output/episode/<dataset_name>_<timestamp>/
        robot_pose_records.npy
        robot_executed_poses.npy (empty placeholder)
        robot_tcp_history.npy (empty placeholder)
        T_base_cam_runtime.npy (copied from dataset cam_to_base)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from types import SimpleNamespace

import cv2
import MinkowskiEngine as ME  # type: ignore
import numpy as np
import torch
import sys

here = Path(__file__).resolve()
project_root = here.parents[2]
src_root = project_root / "src"
mba_root = project_root / "MBA"
# Ensure project root paths are on sys.path, preferring MBA utils over src/utils.
for path in (src_root, mba_root, project_root):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
for path in reversed([project_root, mba_root, src_root]):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from egodata_eval.eval import TrajectoryPredictor
from egodata_eval.eval_utils import _build_pose_mats, _denormalize_obj_traj
from MBA.dataset.realworld import RealWorldDataset


def _load_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".txt", ".csv"}:
        data = np.loadtxt(str(path), dtype=np.float32, delimiter=None)
    else:
        data = np.load(str(path)).astype(np.float32)
    if data.ndim == 1:
        raise ValueError(f"Matrix file {path} should contain at least 3 rows.")
    if data.shape == (4, 4):
        return data
    if data.shape == (3, 4):
        pad = np.array([[0, 0, 0, 1]], dtype=np.float32)
        return np.vstack([data, pad])
    raise ValueError(f"Unsupported matrix shape {data.shape} in {path}")


def _load_depth(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read depth image {path}")
    depth = image.astype(np.float32)
    if depth.max() > 20.0:  # likely stored in millimeters
        depth /= 1000.0
    return depth


def _load_intrinsics(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".txt", ".csv"}:
        K = np.loadtxt(str(path), dtype=np.float32)
    else:
        K = np.load(str(path)).astype(np.float32)
    if K.size == 9:
        K = K.reshape(3, 3)
    if K.shape != (3, 3):
        raise ValueError(f"Intrinsics must be 3x3, got {K.shape} from {path}")
    return K


def _load_cam_to_base_map(
    seq_path: Path,
    frame_ids: Sequence[str],
    head_to_zed: Path,
) -> Dict[str, np.ndarray]:
    helper = SimpleNamespace(cam_to_base_rot_noise_std=0.0)
    head_pos_dir = seq_path / "head_pos"
    extr_map: Dict[int, np.ndarray] = {}
    extr_map = RealWorldDataset._load_camera_extrinsics_from_dir(helper, str(head_pos_dir), str(head_to_zed))
    cam_map = RealWorldDataset._load_cam_to_base(helper, str(seq_path), list(frame_ids), extr_map)
    return cam_map


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read RGB image {path}")
    return image


def _list_frames(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Missing directory: {directory}")
    return sorted([p for p in directory.iterdir() if p.is_file()])


def _match_frame_paths(
    rgb_dir: Path,
    depth_dir: Path,
    pose_dir: Path,
) -> List[Tuple[Path, Path, Path]]:
    rgb_files = _list_frames(rgb_dir)
    depth_map: Dict[str, Path] = {p.stem: p for p in _list_frames(depth_dir)}
    pose_map: Dict[str, Path] = {p.stem: p for p in _list_frames(pose_dir)}

    matched: List[Tuple[Path, Path, Path]] = []
    for rgb in rgb_files:
        stem = rgb.stem
        if stem not in depth_map or stem not in pose_map:
            print(f"[WARN] Missing depth/pose for frame {stem}; skipping.")
            continue
        matched.append((rgb, depth_map[stem], pose_map[stem]))
    return matched


def _predict_sequence(
    predictor: TrajectoryPredictor,
    image_bgr: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    pose_cam_ob: np.ndarray,
    T_base_cam: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray | None]:
    pose_base_ob = T_base_cam @ pose_cam_ob
    feats, coords = predictor._make_sparse_input(image_bgr, depth_m, K, T_base_cam=T_base_cam)
    st = ME.SparseTensor(feats, coords)
    cur_obj = predictor._current_obj_vec(pose_base_ob)

    with torch.no_grad():
        outputs = predictor.model(
            st,
            actions=None,
            batch_size=1,
            current_obj=torch.from_numpy(cur_obj[None, :]).to(predictor.device),
        )
    if "obj_pred" not in outputs:
        return pose_base_ob, None
    obj_traj_norm = outputs["obj_pred"].squeeze(0).detach().cpu().numpy()
    obj_traj_ref = _denormalize_obj_traj(obj_traj_norm)
    pose_mats_ref = _build_pose_mats(obj_traj_ref[:, :3], obj_traj_ref[:, 3:9])
    return pose_base_ob, pose_mats_ref


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation from saved dataset frames.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to dataset sequence directory.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Trajectory predictor checkpoint.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("src/egodata_eval/train_output/episode"),
        help="Directory where episode outputs are stored.",
    )
    parser.add_argument(
        "--T_robot_base",
        type=Path,
        default=Path("glasses_hardware/calib/T_robot_base.npy"),
        help="Path to T_robot_base transform (base->robot).",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=None,
        help="Optional path to 3x3 intrinsics matrix. If omitted, look for data-path/cam_K.txt (or K.npy).",
    )
    parser.add_argument(
        "--head-to-zed",
        type=Path,
        default=Path("glasses_hardware/calib/T_tcp_zed.txt"),
        help="Path to tcp->zed calibration used when deriving cam_to_base.",
    )
    args = parser.parse_args()

    data_path = args.data_path.resolve()
    rgb_dir = data_path / "rgb"
    depth_dir = data_path / "depth"
    pose_dir = data_path / "ob_in_cam"

    frame_triples = _match_frame_paths(rgb_dir, depth_dir, pose_dir)
    if not frame_triples:
        raise RuntimeError(f"No matching frames found under {data_path}")

    rgb_frame_ids = [path.stem for path in _list_frames(rgb_dir)]
    cam2base_map = _load_cam_to_base_map(data_path, rgb_frame_ids, args.head_to_zed)

    if args.intrinsics:
        K = _load_intrinsics(args.intrinsics)
    else:
        K_txt = data_path / "cam_K.txt"
        K_npy = data_path / "K.npy"
        if K_txt.exists():
            K = _load_intrinsics(K_txt)
        elif K_npy.exists():
            K = _load_intrinsics(K_npy)
        else:
            raise FileNotFoundError(f"Intrinsics file missing. Provide --intrinsics or {K_txt} (or {K_npy}).")

    T_robot_base = _load_matrix(args.T_robot_base)

    predictor = TrajectoryPredictor(args.ckpt)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    episode_dir = args.output_root.resolve() / f"{data_path.name}_{timestamp}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    pose_records: List[Dict[str, object]] = []
    cam_runtime: List[np.ndarray] = []

    for frame_idx, (rgb_path, depth_path, pose_path) in enumerate(frame_triples):
        image_bgr = _load_rgb(rgb_path)
        depth_m = _load_depth(depth_path)
        pose_cam_ob = _load_matrix(pose_path)
        frame_key = rgb_path.stem
        T_base_cam = cam2base_map[frame_key]
        T_robot_cam = T_robot_base @ T_base_cam

        cam_runtime.append(T_robot_cam.astype(np.float32))

        pose_base_ob, seq_base = _predict_sequence(
            predictor,
            image_bgr,
            depth_m,
            K,
            pose_cam_ob,
            T_base_cam,
        )
        pose_robot_ob = T_robot_base @ pose_base_ob
        pose_seq_robot = None
        if seq_base is not None and seq_base.size > 0:
            pose_seq_robot = np.einsum(
                "ij,njk->nik",
                T_robot_base.astype(np.float32),
                seq_base.astype(np.float32),
            )

        pose_records.append(
            {
                "timestamp": float(time.time()),
                "frame_idx": int(frame_idx),
                "object_pose_robot": pose_robot_ob.astype(np.float32),
                "pred_seq_robot": pose_seq_robot.astype(np.float32),
            }
        )
        print(f"[INFO] processed frame {frame_idx:04d} -> pose record saved.")

    np.save(episode_dir / "robot_pose_records.npy", np.array(pose_records, dtype=object))
    np.save(episode_dir / "T_base_cam_runtime.npy", np.stack(cam_runtime, axis=0))
    np.save(episode_dir / "robot_executed_poses.npy", np.zeros((0, 3), dtype=np.float32))
    np.save(episode_dir / "robot_tcp_history.npy", np.zeros((0, 3), dtype=np.float32))
    print(f"[OK] Saved {len(pose_records)} pose records to {episode_dir}")


if __name__ == "__main__":
    main()
