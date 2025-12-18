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
from MBA.utils.constants import TRANS_MIN, TRANS_MAX
from MBA.utils.transformation import xyz_rot_transform


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


def _normalize_obj_np(traj: np.ndarray) -> np.ndarray:
    norm = traj.copy()
    norm[:, :3] = (norm[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
    return norm


def _build_future_obj_traj(
    frame_idx: int,
    frame_ids: Sequence[str],
    pose_paths: Dict[str, Path],
    cam_to_base_map: Dict[str, np.ndarray],
    terminal_ids: set[int],
    horizon: int,
    clamp_to_last: bool = False,
) -> np.ndarray:
    last_idx = len(frame_ids) - 1
    if clamp_to_last:
        max_steps = horizon
    else:
        available = max(0, last_idx - frame_idx)
        max_steps = min(horizon, available)
        if max_steps == 0:
            return np.zeros((0, 10), dtype=np.float32)
    traj = []
    for step in range(1, max_steps + 1):
        idx = frame_idx + step
        if idx > last_idx:
            if clamp_to_last:
                idx = last_idx
            else:
                break
        frame_key = frame_ids[idx]
        pose_path = pose_paths.get(frame_key)
        if pose_path is None:
            raise KeyError(f"Pose file missing for frame {frame_key}")
        pose_cam = _load_matrix(pose_path)
        T_base_cam = cam_to_base_map[frame_key]
        pose_base = T_base_cam @ pose_cam
        xyz_rot = xyz_rot_transform(pose_base, from_rep="matrix", to_rep="rotation_6d").astype(np.float32)
        term = np.array([1.0 if int(frame_key) in terminal_ids else 0.0], dtype=np.float32)
        traj.append(np.concatenate([xyz_rot, term], axis=0))
    traj_np = np.stack(traj, axis=0)
    return _normalize_obj_np(traj_np)


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
) -> Tuple[np.ndarray, Dict[str, np.ndarray] | None]:
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
    return pose_base_ob, {"pose_mats": pose_mats_ref, "traj_norm": obj_traj_norm}


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
        "--num_action",
        type=int,
        default=20,
        help="Number of action/object steps predicted by the checkpoint (must match training).",
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
    parser.add_argument(
        "--clamp_future_loss",
        action="store_true",
        help="When set, clamp GT future indices to the last frame when computing loss.",
    )
    args = parser.parse_args()

    data_path = args.data_path.resolve()
    rgb_dir = data_path / "rgb"
    depth_dir = data_path / "depth"
    pose_dir = data_path / "ob_in_cam"

    frame_triples = _match_frame_paths(rgb_dir, depth_dir, pose_dir)
    if not frame_triples:
        raise RuntimeError(f"No matching frames found under {data_path}")
    frame_ids = [rgb.stem for rgb, _, _ in frame_triples]
    pose_path_map = {pose.stem: pose for _, _, pose in frame_triples}
    if len(frame_ids) >= 5:
        terminal_ids = set(int(fid) for fid in frame_ids[-5:])
    else:
        terminal_ids = set(int(fid) for fid in frame_ids)

    cam2base_map = _load_cam_to_base_map(data_path, frame_ids, args.head_to_zed)

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

    predictor = TrajectoryPredictor(args.ckpt, num_action=args.num_action)
    horizon = getattr(predictor.model.action_decoder, "horizon", args.num_action)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    episode_dir = args.output_root.resolve() / f"{data_path.name}_{timestamp}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    pose_records: List[Dict[str, object]] = []
    cam_runtime: List[np.ndarray] = []
    base_cam_raw: List[np.ndarray] = []
    loss_sum = 0.0
    loss_count = 0

    for frame_idx, (rgb_path, depth_path, pose_path) in enumerate(frame_triples):
        image_bgr = _load_rgb(rgb_path)
        depth_m = _load_depth(depth_path)
        pose_cam_ob = _load_matrix(pose_path)
        frame_key = rgb_path.stem
        T_base_cam = cam2base_map[frame_key]
        T_robot_cam = T_robot_base @ T_base_cam

        cam_runtime.append(T_base_cam.astype(np.float32))
        base_cam_raw.append(T_base_cam.astype(np.float32))

        pose_base_ob, seq_pred = _predict_sequence(
            predictor,
            image_bgr,
            depth_m,
            K,
            pose_cam_ob,
            T_base_cam,
        )
        pose_robot_ob = T_robot_base @ pose_base_ob
        pose_seq_robot = None
        obj_traj_norm = None
        if seq_pred is not None:
            pose_mats = seq_pred.get("pose_mats")
            obj_traj_norm = seq_pred.get("traj_norm")
            if pose_mats is not None and pose_mats.size > 0:
                pose_seq_robot = np.einsum(
                    "ij,njk->nik",
                    T_robot_base.astype(np.float32),
                    pose_mats.astype(np.float32),
                )

        if obj_traj_norm is not None and horizon > 0:
            gt_traj = _build_future_obj_traj(
                frame_idx,
                frame_ids,
                pose_path_map,
                cam2base_map,
                terminal_ids,
                min(horizon, len(obj_traj_norm)),
                clamp_to_last=args.clamp_future_loss,
            )
            steps = min(obj_traj_norm.shape[0], gt_traj.shape[0])
            if steps > 0:
                diff = obj_traj_norm[:steps] - gt_traj[:steps]
                mse_per_step = np.mean(diff * diff, axis=1)
                for step, loss in enumerate(mse_per_step):
                    print(f"[LOSS] frame {frame_idx:04d} step {step:02d} mse {loss:.6f}")
                    loss_sum += float(loss)
                    loss_count += 1

        pose_records.append(
            {
                "timestamp": float(time.time()),
                "frame_idx": int(frame_idx),
                "object_pose_robot": pose_robot_ob.astype(np.float32),
                "pred_seq_robot": None if pose_seq_robot is None else pose_seq_robot.astype(np.float32),
            }
        )
        print(f"[INFO] processed frame {frame_idx:04d} -> pose record saved.")

    np.save(episode_dir / "robot_pose_records.npy", np.array(pose_records, dtype=object))
    np.save(episode_dir / "T_base_cam_runtime.npy", np.stack(cam_runtime, axis=0))
    np.save(episode_dir / "T_base_cam.npy", np.stack(base_cam_raw, axis=0))
    np.save(episode_dir / "robot_executed_poses.npy", np.zeros((0, 3), dtype=np.float32))
    np.save(episode_dir / "robot_tcp_history.npy", np.zeros((0, 3), dtype=np.float32))
    print(f"[OK] Saved {len(pose_records)} pose records to {episode_dir}")
    if loss_count > 0:
        avg_loss = loss_sum / loss_count
        print(f"[LOSS] average mse over {loss_count} steps: {avg_loss:.6f}")
    else:
        print("[LOSS] average mse: N/A (no predictions)")


if __name__ == "__main__":
    main()
