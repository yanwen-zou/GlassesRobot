#!/usr/bin/env python3
"""
Offline evaluation script that replays a dataset of RGB/depth/object-pose frames,
feeds them into the trajectory predictor, and stores results in train_output/episode/*.

Expected dataset layout under --data-path:
    rgb/000000.png, ...
    depth/000000.npy (meters) or png (depth-in-mm)
    ob_in_cam.npy (Nx4x4 pose matrices) OR ob_in_cam/000000.npy (4x4 per frame)
    cam_to_base.txt (anchor cam->base per RealWorldDataset) + head_pos.txt or head_pos/
    optional: intrinsics.npy (3x3), cam_K.txt
Legacy (fallback) cam_to_base/000000.npy is still supported.

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
import warnings

import cv2
import MinkowskiEngine as ME  # type: ignore
import numpy as np
import torch
import sys
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

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

from egodata_eval.traj_predictor import TrajectoryPredictor
from egodata_eval.eval_utils import _build_pose_mats, _denormalize_obj_traj, _normalize_obj_pose
from egodata_eval.eval_constant import TASK_CHOICES, DEFAULT_MESH_NAME, VIDEO_FPS
from egodata_eval.get_pose import PoseEstimatorFP
from MBA.dataset.realworld import RealWorldDataset
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
    cam_to_base_rot_noise_std: float = 0.0,
) -> Dict[str, np.ndarray]:
    helper = SimpleNamespace(cam_to_base_rot_noise_std=float(cam_to_base_rot_noise_std))
    head_pos_file = seq_path / "head_pos.txt"
    head_pos_dir = seq_path / "head_pos"
    if head_pos_file.is_file():
        extr_map = RealWorldDataset._load_camera_extrinsics_from_dir(
            helper, str(head_pos_file), str(head_to_zed)
        )
    elif head_pos_dir.is_dir():
        extr_map = RealWorldDataset._load_camera_extrinsics_from_dir(
            helper, str(head_pos_dir), str(head_to_zed)
        )
    else:
        extr_map = {}
        warnings.warn(
            f"[eval_dataset] Missing head_pos.txt or head_pos dir in {seq_path}; using identity extrinsics."
        )

    try:
        return RealWorldDataset._load_cam_to_base(helper, str(seq_path), list(frame_ids), extr_map)
    except FileNotFoundError as exc:
        warnings.warn(f"[eval_dataset] {exc} Falling back to legacy cam_to_base loader.")
        return _load_cam_to_base_legacy(seq_path, frame_ids)


def _load_cam_to_base_legacy(seq_path: Path, frame_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    cam_dir = seq_path / "cam_to_base"
    if not cam_dir.exists():
        raise FileNotFoundError(f"Missing cam_to_base.txt or legacy cam_to_base dir in {seq_path}.")
    cam_files = {p.stem: p for p in _list_frames(cam_dir)}
    if not cam_files:
        raise FileNotFoundError(f"Empty cam_to_base dir in {seq_path}.")
    if len(cam_files) == 1:
        single_mat = _load_matrix(next(iter(cam_files.values())))
        return {fid: single_mat.copy() for fid in frame_ids}
    cam_map: Dict[str, np.ndarray] = {}
    for fid in frame_ids:
        path = cam_files.get(fid)
        if path is None:
            raise FileNotFoundError(f"Missing cam_to_base for frame {fid} in {seq_path}.")
        cam_map[fid] = _load_matrix(path)
    return cam_map


def _absolute_to_delta(pose_seq: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    """Convert absolute pose sequence to delta representation relative to base_pose."""
    pose_seq = pose_seq.copy()
    pose = torch.from_numpy(pose_seq[..., :9]).float()
    base = torch.from_numpy(base_pose[..., :9]).float()
    if base.dim() == 1:
        base = base.unsqueeze(0).expand(pose.shape[0], -1)

    result = pose.clone()
    result[..., :3] = pose[..., :3] - base[..., :3]

    target_rot6 = pose[..., 3:9].contiguous()
    base_rot6 = base[..., 3:9].contiguous()
    target_rot_mat = rotation_6d_to_matrix(target_rot6.reshape(-1, 6))
    base_rot_mat = rotation_6d_to_matrix(base_rot6.reshape(-1, 6))
    delta_rot_mat = target_rot_mat @ base_rot_mat.transpose(-1, -2)
    delta_rot6 = matrix_to_rotation_6d(delta_rot_mat).reshape_as(target_rot6)
    result[..., 3:9] = delta_rot6

    pose_seq[..., :9] = result.numpy()
    return pose_seq


def _normalize_pose_seq(traj: np.ndarray, obj_pose_mode: str) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim != 2:
        raise ValueError(f"Expected (T, D) pose sequence, got shape {traj.shape}")
    return np.stack(
        [_normalize_obj_pose(pose, obj_pose_mode=obj_pose_mode) for pose in traj],
        axis=0,
    )


def _build_future_obj_traj(
    frame_idx: int,
    frame_ids: Sequence[str],
    pose_list: Sequence[np.ndarray],
    cam_to_base_map: Dict[str, np.ndarray],
    terminal_ids: set[int],
    horizon: int,
    obj_pose_mode: str,
    current_obj: np.ndarray | None = None,
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
        if idx >= len(pose_list):
            raise KeyError(f"Pose list missing for frame index {idx} ({frame_key})")
        pose_cam = pose_list[idx]
        T_base_cam = cam_to_base_map[frame_key]
        pose_base = T_base_cam @ pose_cam
        xyz_rot = xyz_rot_transform(pose_base, from_rep="matrix", to_rep="rotation_6d").astype(np.float32)
        term = np.array([1.0 if int(frame_key) in terminal_ids else 0.0], dtype=np.float32)
        traj.append(np.concatenate([xyz_rot, term], axis=0))
    traj_np = np.stack(traj, axis=0)
    if obj_pose_mode == "delta":
        if current_obj is None:
            raise ValueError("current_obj is required when obj_pose_mode is 'delta'.")
        traj_np = _absolute_to_delta(traj_np, current_obj)
    return _normalize_pose_seq(traj_np, obj_pose_mode=obj_pose_mode)


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
) -> List[Tuple[Path, Path]]:
    rgb_files = _list_frames(rgb_dir)
    depth_map: Dict[str, Path] = {p.stem: p for p in _list_frames(depth_dir)}

    matched: List[Tuple[Path, Path]] = []
    for rgb in rgb_files:
        stem = rgb.stem
        if stem not in depth_map:
            print(f"[WARN] Missing depth for frame {stem}; skipping.")
            continue
        matched.append((rgb, depth_map[stem]))
    return matched


def _predict_sequence(
    predictor: TrajectoryPredictor,
    image_bgr: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    pose_cam_ob: np.ndarray,
    T_base_cam: np.ndarray | None,
    headpose_cond: np.ndarray | None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray] | None, np.ndarray]:
    pose_base_ob = T_base_cam @ pose_cam_ob
    overlay, _ = predictor.predict_and_overlay(
        image_bgr,
        depth_m,
        K,
        pose_cam_ob,
        T_base_cam=T_base_cam,
        headpose_cond=headpose_cond,
    )
    outputs = {}
    if predictor.last_traj_pred is not None:
        obj_traj_abs = predictor.last_traj_pred.astype(np.float32)
        # print(f"obj_traj_abs: {obj_traj_abs}")
        if predictor.obj_pose_mode == "delta":
            base_xyz6d = xyz_rot_transform(
                pose_base_ob, from_rep="matrix", to_rep="rotation_6d"
            ).astype(np.float32)
            obj_traj_delta = _absolute_to_delta(obj_traj_abs, base_xyz6d)
            outputs["obj_pred"] = _normalize_pose_seq(
                obj_traj_delta, obj_pose_mode=predictor.obj_pose_mode
            )
        else:
            outputs["obj_pred"] = _normalize_pose_seq(
                obj_traj_abs, obj_pose_mode=predictor.obj_pose_mode
            )
    if predictor.last_headpose_pred is not None:
        outputs["headpose_pred"] = _normalize_pose_seq(
            predictor.last_headpose_pred.astype(np.float32),
            obj_pose_mode=predictor.obj_pose_mode,
        )
    pred: Dict[str, np.ndarray] = {}
    if outputs is not None and "obj_pred" in outputs:
        obj_pred = outputs["obj_pred"]
        if torch.is_tensor(obj_pred):
            obj_traj_norm = obj_pred.squeeze(0).detach().cpu().numpy()
        else:
            obj_traj_norm = np.asarray(obj_pred, dtype=np.float32)
        obj_traj_ref = _denormalize_obj_traj(obj_traj_norm, obj_pose_mode=predictor.obj_pose_mode)
        pose_mats_ref = _build_pose_mats(obj_traj_ref[:, :3], obj_traj_ref[:, 3:9])
        pred["pose_mats"] = pose_mats_ref
        pred["traj_norm"] = obj_traj_norm
    if outputs is not None and "headpose_pred" in outputs:
        headpose_pred = outputs["headpose_pred"]
        if torch.is_tensor(headpose_pred):
            headpose_pred_norm = headpose_pred.squeeze(0).detach().cpu().numpy()
        else:
            headpose_pred_norm = np.asarray(headpose_pred, dtype=np.float32)
        pred["headpose_pred"] = _denormalize_obj_traj(headpose_pred_norm, obj_pose_mode=predictor.obj_pose_mode)
        pred["headpose_pred_norm"] = headpose_pred_norm
    if not pred:
        raise RuntimeError("No predictions were made by the model.")
    return pose_base_ob, pred, overlay


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
        default=10,
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
        default=Path("glasses_hardware/calib/T_glasses_zed.txt"),
        help="Path to tcp->zed calibration used when deriving cam_to_base.",
    )
    parser.add_argument(
        "--cam-to-base-rot-noise-std",
        type=float,
        default=0.0,
        help="Std-dev (radians) of optional rotation noise for cam_to_base (matches RealWorldDataset).",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=TASK_CHOICES,
        default=DEFAULT_MESH_NAME,
        help="Task name (also used as mesh-name under data/<task>/mesh.obj).",
    )
    parser.add_argument(
        "--clamp_future_loss",
        action="store_true",
        help="When set, clamp GT future indices to the last frame when computing loss.",
    )
    parser.add_argument(
        "--enable-headpose-head",
        action="store_true",
        help="Enable headpose diffusion head and conditioning.",
    )
    parser.add_argument(
        "--obj-pose-mode",
        type=str, choices=["abs", "delta"],
        default="delta",
        help="Model output pose mode: abs or delta.",
    )
    parser.add_argument(
        "--add_curr_cond",
        required=True,
        choices=["true", "false"],
        help="Whether to add current obj pose as extra cond for diffusion head (true/false).",
    )

    args = parser.parse_args()

    data_path = args.data_path.resolve()
    rgb_dir = data_path / "rgb"
    depth_dir = data_path / "depth"
    pose_dir = data_path / "ob_in_cam"
    pose_npy = data_path / "ob_in_cam.npy"

    frame_pairs = _match_frame_paths(rgb_dir, depth_dir)
    if not frame_pairs:
        raise RuntimeError(f"No matching frames found under {data_path}")
    frame_ids = [rgb.stem for rgb, _ in frame_pairs]

    pose_list: List[np.ndarray] = []
    if pose_npy.exists():
        pose_arr = np.load(str(pose_npy)).astype(np.float32)
        if pose_arr.ndim != 3 or pose_arr.shape[1:] != (4, 4):
            raise ValueError(f"Invalid ob_in_cam.npy shape {pose_arr.shape}; expected (N,4,4).")
        if pose_arr.shape[0] < len(frame_ids):
            raise ValueError(
                f"ob_in_cam.npy has {pose_arr.shape[0]} poses but {len(frame_ids)} frames found."
            )
        pose_list = [pose_arr[i] for i in range(len(frame_ids))]
    else:
        if not pose_dir.exists():
            raise FileNotFoundError(f"Missing pose data: {pose_npy} or {pose_dir}")
        pose_map: Dict[str, Path] = {p.stem: p for p in _list_frames(pose_dir)}
        for fid in frame_ids:
            pose_path = pose_map.get(fid)
            if pose_path is None:
                raise KeyError(f"Pose file missing for frame {fid}")
            pose_list.append(_load_matrix(pose_path))
    if len(frame_ids) >= 5:
        terminal_ids = set(int(fid) for fid in frame_ids[-5:])
    else:
        terminal_ids = set(int(fid) for fid in frame_ids)

    cam2base_map = _load_cam_to_base_map(
        data_path,
        frame_ids,
        args.head_to_zed,
        cam_to_base_rot_noise_std=args.cam_to_base_rot_noise_std,
    )

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

    add_curr_cond = str(args.add_curr_cond).lower() == "true"
    predictor = TrajectoryPredictor(
        args.ckpt,
        num_action=args.num_action,
        obj_pose_mode=args.obj_pose_mode,
        enable_headpose_head=args.enable_headpose_head,
        add_curr_cond=add_curr_cond,
    )
    horizon = getattr(predictor.model.action_decoder, "horizon", args.num_action)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    episode_dir = args.output_root.resolve() / f"{data_path.name}_{timestamp}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    video_path = episode_dir / "stream.mp4"

    mesh_path = project_root / "data" / args.task / "mesh.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    pose_drawer = PoseEstimatorFP(mesh_path)
    writer: cv2.VideoWriter | None = None

    pose_records: List[Dict[str, object]] = []
    cam_runtime: List[np.ndarray] = []
    headpose_preds: List[np.ndarray | None] = []
    headpose_preds_norm: List[np.ndarray | None] = []
    loss_sum = 0.0
    loss_count = 0

    for frame_idx, (rgb_path, depth_path) in enumerate(frame_pairs):
        image_bgr = _load_rgb(rgb_path)
        depth_m = _load_depth(depth_path)
        pose_cam_ob = pose_list[frame_idx]
        frame_key = rgb_path.stem
        T_base_cam = cam2base_map[frame_key]
        pose_base_ob_gt = T_base_cam @ pose_cam_ob
        base_xyz = pose_base_ob_gt[:3, 3].astype(np.float32)
        print(f"[GT] frame {frame_idx:04d} base_xyz: {base_xyz.tolist()}")

        cam_runtime.append(T_base_cam.astype(np.float32))
        headpose_raw = xyz_rot_transform(T_base_cam, from_rep="matrix", to_rep="rotation_6d").astype(np.float32)
        headpose_cond = (
            _normalize_obj_pose(headpose_raw, obj_pose_mode="abs")
            if args.enable_headpose_head
            else None
        )

        pose_base_ob, seq_pred, overlay = _predict_sequence(
            predictor,
            image_bgr,
            depth_m,
            K,
            pose_cam_ob,
            T_base_cam,
            headpose_cond,
        )
        # print(f"[DEBUG] pose_base_ob:{pose_base_ob}")
        # print(f"[PRED] frame {frame_idx:04d} cam_xyz: {pose_cam_ob[:3, 3].tolist()}")
        overlay = pose_drawer.draw_overlay(overlay, K, pose_cam_ob=pose_cam_ob)
        if writer is None:
            h, w = overlay.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (w, h))
        if writer is not None and writer.isOpened():
            writer.write(overlay)
        pose_robot_ob = T_robot_base @ pose_base_ob
        pose_seq_robot = None
        obj_traj_norm = None
        headpose_pred = None
        headpose_pred_norm = None
        if seq_pred is not None:
            pose_mats = seq_pred.get("pose_mats")
            obj_traj_norm = seq_pred.get("traj_norm")
            headpose_pred = seq_pred.get("headpose_pred")
            headpose_pred_norm = seq_pred.get("headpose_pred_norm")
            if pose_mats is not None and pose_mats.size > 0:
                pose_seq_robot = np.einsum(
                    "ij,njk->nik",
                    T_robot_base.astype(np.float32),
                    pose_mats.astype(np.float32),
                )

        if obj_traj_norm is not None and horizon > 0:
            current_obj_raw = xyz_rot_transform(
                pose_base_ob, from_rep="matrix", to_rep="rotation_6d"
            ).astype(np.float32)
            current_term = np.array(
                [1.0 if int(frame_key) in terminal_ids else 0.0], dtype=np.float32
            )
            current_obj = np.concatenate([current_obj_raw, current_term], axis=0)
            gt_traj = _build_future_obj_traj(
                frame_idx,
                frame_ids,
                pose_list,
                cam2base_map,
                terminal_ids,
                min(horizon, len(obj_traj_norm)),
                predictor.obj_pose_mode,
                current_obj=current_obj,
                clamp_to_last=args.clamp_future_loss,
            )
            steps = min(obj_traj_norm.shape[0], gt_traj.shape[0])
            if steps > 0:
                obj_traj_denorm = _denormalize_obj_traj(obj_traj_norm[:steps], obj_pose_mode=predictor.obj_pose_mode)
                gt_traj_denorm = _denormalize_obj_traj(gt_traj[:steps], obj_pose_mode=predictor.obj_pose_mode)
                diff = obj_traj_denorm - gt_traj_denorm
                mse_per_step = np.mean(diff * diff, axis=1)
                for step, loss in enumerate(mse_per_step):
                    #print(f"[LOSS] frame {frame_idx:04d} step {step:02d} mse {loss:.6f}")
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
        headpose_preds.append(None if headpose_pred is None else headpose_pred.astype(np.float32))
        headpose_preds_norm.append(None if headpose_pred_norm is None else headpose_pred_norm.astype(np.float32))
        print(f"[INFO] processed frame {frame_idx:04d} -> pose record saved.")

    np.save(episode_dir / "robot_pose_records.npy", np.array(pose_records, dtype=object))
    np.save(episode_dir / "T_base_cam_runtime.npy", np.stack(cam_runtime, axis=0))
    np.save(episode_dir / "robot_executed_poses.npy", np.zeros((0, 3), dtype=np.float32))
    np.save(episode_dir / "robot_tcp_history.npy", np.zeros((0, 3), dtype=np.float32))
    np.save(episode_dir / "headpose_pred.npy", np.array(headpose_preds, dtype=object))
    np.save(episode_dir / "headpose_pred_norm.npy", np.array(headpose_preds_norm, dtype=object))
    if writer is not None:
        writer.release()
        print(f"[OK] Saved video to {video_path}")
    print(f"[OK] Saved {len(pose_records)} pose records to {episode_dir}")
    if loss_count > 0:
        avg_loss = loss_sum / loss_count
        print(f"[LOSS] average mse over {loss_count} steps: {avg_loss:.6f}")
    else:
        print("[LOSS] average mse: N/A (no predictions)")


if __name__ == "__main__":
    main()
