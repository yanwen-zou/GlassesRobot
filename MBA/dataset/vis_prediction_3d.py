"""
Minimal 3D prediction visualization in Rerun.

Loads a RealWorldDataset sample, runs the RISE policy, and streams predicted
(and optional GT) object trajectories in the base frame to Rerun as 3D lines
and poses. This is a framework to build richer views on top of.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import MinkowskiEngine as ME
import numpy as np
import rerun as rr
import torch

from dataset.realworld import RealWorldDataset
from policy import RISE
from utils.constants import TRANS_MAX, TRANS_MIN
from utils.transformation import rotation_transform


def denormalize_obj_traj(obj_traj: np.ndarray) -> np.ndarray:
    obj_out = obj_traj.copy()
    obj_out[:, :3] = (obj_out[:, :3] + 1) * 0.5 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    return obj_out


def build_pose_mats(translation: np.ndarray, rotation_6d: np.ndarray) -> np.ndarray:
    mats = np.repeat(np.eye(4)[None, ...], len(translation), axis=0)
    rot_mats = rotation_transform(rotation_6d, "rotation_6d", "matrix")
    mats[:, :3, :3] = rot_mats
    mats[:, :3, 3] = translation
    return mats


def delta_to_absolute_traj(delta_traj: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    abs_traj = delta_traj.copy()
    abs_traj[:, :3] = delta_traj[:, :3] + base_pose[:3]
    delta_rot_mats = rotation_transform(delta_traj[:, 3:3 + 6], "rotation_6d", "matrix")
    base_rot_mat = rotation_transform(base_pose[None, 3:3 + 6], "rotation_6d", "matrix")[0]
    abs_rot_mats = delta_rot_mats @ base_rot_mat
    abs_rot_6d = rotation_transform(abs_rot_mats, "matrix", "rotation_6d")
    abs_traj[:, 3:3 + 6] = abs_rot_6d
    return abs_traj


def run_model(ds: RealWorldDataset, idx: int, model: RISE, obj_pose_mode: str):
    item = ds[idx]
    coords, feats = ME.utils.sparse_collate(item["input_coords_list"],
                                            item["input_feats_list"])
    st = ME.SparseTensor(feats.cuda(), coords.cuda())

    current_obj = item.get("current_obj_pose_normalized")
    current_obj_np = None
    if current_obj is not None:
        current_obj_np = current_obj.numpy()
        current_obj = current_obj.unsqueeze(0).cuda()

    with torch.no_grad():
        outputs = model(st, actions=None, batch_size=1, current_obj=current_obj)
    if "obj_pred" not in outputs:
        raise RuntimeError("Model did not return object predictions.")

    obj_traj_norm = outputs["obj_pred"].squeeze(0).cpu().numpy()
    if obj_pose_mode == "delta":
        if current_obj_np is None:
            raise RuntimeError("Current object pose is required to convert delta predictions.")
        if getattr(model.action_decoder, "returns_absolute_pose", True):
            pass
        else:
            obj_traj_norm = delta_to_absolute_traj(obj_traj_norm, current_obj_np)

    obj_traj_base = denormalize_obj_traj(obj_traj_norm)
    pose_mats_pred = build_pose_mats(obj_traj_base[:, :3], obj_traj_base[:, 3:3 + 6])

    pose_mats_gt = None
    if "action_obj" in item:
        gt_traj_base = item["action_obj"].numpy()
        pose_mats_gt = build_pose_mats(gt_traj_base[:, :3], gt_traj_base[:, 3:3 + 6])

    return pose_mats_pred, pose_mats_gt, item


def log_trajectory(name: str, pose_mats: np.ndarray, color: Tuple[int, int, int, int]) -> None:
    pts = pose_mats[:, :3, 3].astype(np.float32)
    rr.log(
        f"world/{name}/line",
        rr.LineStrips3D([pts], colors=np.array([color], dtype=np.uint8)),
    )


def main():
    ap = argparse.ArgumentParser(description="3D prediction visualization framework (Rerun).")
    ap.add_argument("--data_path", default="data")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num_action", type=int, default=20)
    ap.add_argument("--demo_index", type=int, default=0, help="Episode index")
    ap.add_argument("--demo_step", type=int, default=0, help="Window inside episode")
    ap.add_argument("--split", type=str, default="train", choices=["train", "eval", "all"], help="Dataset split")
    ap.add_argument("--obj_pose_mode", type=str, default="delta", choices=["abs", "delta"],
                    help="Object pose prediction target type used by the checkpoint.")
    ap.add_argument("--no-spawn", action="store_true", help="Do not spawn a separate Rerun viewer window")
    args = ap.parse_args()

    ds = RealWorldDataset(
        args.data_path,
        split=args.split,
        num_obs=1,
        num_action=args.num_action,
        with_obj_action=True,
        cam_to_base_rot_noise_std=0.0,  # inference: disable rot noise
    )
    available_seq_ids = getattr(ds, "all_demos", sorted(set(ds.seq_ids)))
    if args.demo_index < 0 or args.demo_index >= len(available_seq_ids):
        raise IndexError(f"demo_index {args.demo_index} out of range for {len(available_seq_ids)} episodes.")
    seq_id = available_seq_ids[args.demo_index]
    seq_indices_all = [i for i, sid in enumerate(ds.seq_ids) if sid == seq_id]
    if not seq_indices_all:
        raise RuntimeError(f"No samples found for episode id {seq_id}.")

    if args.demo_step < 0 or args.demo_step >= len(seq_indices_all):
        demo_step = 0
    else:
        demo_step = args.demo_step
    seq_idx = seq_indices_all[demo_step]

    model = RISE(num_action=args.num_action,
                 input_dim=6,
                 obs_feature_dim=512,
                 action_dim=10,
                 hidden_dim=512,
                 enable_mba=True,
                 obj_dim=10,
                 obj_pose_mode=args.obj_pose_mode).cuda().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"), strict=False)

    rr.init("prediction_3d", spawn=not args.no_spawn)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    pose_pred, pose_gt, item = run_model(ds, seq_idx, model, args.obj_pose_mode)

    rr.log("world/base", rr.Transform3D())
    rr.log(
        "world/base/axes",
        rr.Arrows3D(
            origins=np.zeros((3, 3), dtype=np.float32),
            vectors=np.eye(3, dtype=np.float32) * 0.05,
            colors=np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8),
        ),
    )

    log_trajectory("pred", pose_pred, color=(0, 120, 255, 255))
    if pose_gt is not None:
        log_trajectory("gt", pose_gt, color=(0, 220, 0, 255))

    # Camera pose for the current frame
    obs_frame_ids = ds.obs_frame_ids[seq_idx]
    cur_frame_id = obs_frame_ids[-1]
    T_base_cam = ds._get_cam_to_base(seq_id, cur_frame_id)  # noqa: SLF001
    rr.log(
        "world/cam",
        rr.Transform3D(translation=T_base_cam[:3, 3].astype(np.float32), mat3x3=T_base_cam[:3, :3].astype(np.float32)),
    )

    print("Rerun logging complete; open the viewer to inspect trajectories.")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
