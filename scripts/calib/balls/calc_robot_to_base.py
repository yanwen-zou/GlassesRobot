#!/usr/bin/env python3
"""
Compute robot->base transforms from robot->cam sequence and a cam->base transform.

Assumptions:
- robot_to_cam.npy stores a sequence (or dict with frame_ids+transforms) of robot->cam 4x4 matrices
- cam_to_base.npy stores cam->base as a single 4x4 or a sequence/dict with frame_ids

Output:
- robot_to_base.npy stored alongside robot_to_cam.npy (or at a user-specified path)
"""

import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compute robot->base transforms from robot->cam and cam->base.")
    parser.add_argument(
        "--cam_pose",
        "--robot-to-cam",
        type=Path,
        required=True,
        dest="cam_pose",
        help="Path to robot_to_cam.npy containing robot->cam transforms (shape: N,4,4 or dict).",
    )
    parser.add_argument(
        "--cam_to_base",
        type=Path,
        required=True,
        help="Path to cam_to_base.npy containing cam->base transform (shape: 4,4).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for robot_to_base.npy (default: cam_pose parent / robot_to_base.npy).",
    )
    args = parser.parse_args()

    cam_pose_path = args.cam_pose
    cam_to_base_path = args.cam_to_base
    out_path = args.out or cam_pose_path.parent / "robot_to_base.npy"

    robot_cam_raw = np.load(cam_pose_path, allow_pickle=True)
    cam_base_raw = np.load(cam_to_base_path, allow_pickle=True)

    # cam_pose may be raw array or dict with transforms + frame_ids
    robot_cam_frame_ids = None
    if robot_cam_raw.ndim == 0 and isinstance(robot_cam_raw.item(), dict):
        data = robot_cam_raw.item()
        if "transforms" not in data:
            raise ValueError(f"cam_pose dict missing 'transforms' key: keys={list(data.keys())}")
        robot_cam_all = np.asarray(data["transforms"])
        robot_cam_frame_ids = np.asarray(data.get("frame_ids")) if "frame_ids" in data else None
    else:
        robot_cam_all = robot_cam_raw

    # cam_to_base may be a single 4x4 or a dict with per-frame transforms
    cam_base_frame_ids = None
    if cam_base_raw.ndim == 0 and isinstance(cam_base_raw.item(), dict):
        data = cam_base_raw.item()
        if "transforms" not in data:
            raise ValueError(f"cam_to_base dict missing 'transforms' key: keys={list(data.keys())}")
        cam_base = np.asarray(data["transforms"])
        cam_base_frame_ids = np.asarray(data.get("frame_ids")) if "frame_ids" in data else None
    else:
        cam_base = cam_base_raw

    if robot_cam_all.ndim != 3 or robot_cam_all.shape[1:] != (4, 4):
        raise ValueError(f"robot_to_cam.npy must have shape (N,4,4); got {robot_cam_all.shape}")
    if cam_base.shape == (4, 4):
        cam_base_all = np.broadcast_to(cam_base, (robot_cam_all.shape[0], 4, 4))
    elif cam_base.ndim == 3 and cam_base.shape[1:] == (4, 4):
        cam_base_all = cam_base
        if cam_base_all.shape[0] != robot_cam_all.shape[0]:
            print(f"cam_to_base.npy has {cam_base_all.shape[0]} transforms, but robot_to_cam.npy has {robot_cam_all.shape[0]}.")
            # Try frame-id alignment first
            if cam_base_frame_ids is not None and robot_cam_frame_ids is not None:
                cam_idx = {int(fid): i for i, fid in enumerate(cam_base_frame_ids)}
                common_ids = [fid for fid in robot_cam_frame_ids if int(fid) in cam_idx]
                if not common_ids:
                    raise ValueError("Length mismatch and no overlapping frame_ids to align.")
                cam_indices = [cam_idx[int(fid)] for fid in common_ids]
                robot_indices = [i for i, fid in enumerate(robot_cam_frame_ids) if int(fid) in cam_idx]
                robot_cam_all = robot_cam_all[robot_indices]
                cam_base_all = cam_base_all[cam_indices]
                # Preserve aligned frame_ids
                robot_cam_frame_ids = np.asarray(common_ids)
                cam_base_frame_ids = np.asarray(common_ids)
                print(f"[WARN] Length mismatch; aligned {len(common_ids)} frames by frame_ids.")
            elif cam_base_frame_ids is not None and robot_cam_frame_ids is None:
                # Align using cam_to_base frame ids as frame indices for robot_cam_all
                ids = [int(fid) for fid in cam_base_frame_ids]
                cam_idx = {int(fid): i for i, fid in enumerate(cam_base_frame_ids)}
                # Try 0-based and 1-based indexing; prefer 0-based if valid
                zero_based = [i for i in ids if 0 <= i < len(robot_cam_all)]
                one_based = [i - 1 for i in ids if 1 <= i <= len(robot_cam_all)]
                indices = zero_based if len(zero_based) >= len(one_based) else one_based
                if not indices:
                    raise ValueError("Length mismatch and cannot align using cam_to_base frame_ids as indices.")
                robot_cam_all = robot_cam_all[indices]
                # Reorder cam_base_all to match the original cam_base_frame_ids ordering
                cam_order = [cam_idx[fid] for fid in ids if fid in cam_idx]
                cam_base_all = cam_base_all[cam_order][: len(indices)]
                cam_base_frame_ids = np.asarray(cam_base_frame_ids)[: len(indices)]
                robot_cam_frame_ids = np.asarray(ids[: len(indices)])
                which = "0-based" if indices == zero_based else "1-based"
                print(f"[WARN] Length mismatch; aligned using cam_to_base frame_ids as indices ({which}).")
            else:
                common = min(cam_base_all.shape[0], robot_cam_all.shape[0])
                print(f"[WARN] Length mismatch robot->cam {robot_cam_all.shape[0]} vs cam->base {cam_base_all.shape[0]}; using first {common}")
                robot_cam_all = robot_cam_all[:common]
                cam_base_all = cam_base_all[:common]
                if cam_base_frame_ids is not None:
                    cam_base_frame_ids = cam_base_frame_ids[:common]
                if robot_cam_frame_ids is not None:
                    robot_cam_frame_ids = robot_cam_frame_ids[:common]
    else:
        raise ValueError(f"cam_to_base.npy must be (4,4) or (N,4,4); got {cam_base.shape}")

    robot_base_all = robot_cam_all @ np.linalg.inv(cam_base_all) # caution
    np.save(out_path, robot_base_all.astype(np.float32))

    print(f"[OK] Loaded robot->cam: {robot_cam_all.shape}, cam->base: {cam_base_all.shape}")
    print(f"[OK] Saved robot->base to: {out_path} with shape {robot_base_all.shape}")
    for idx, T in enumerate(robot_base_all):
        label = None
        if robot_cam_frame_ids is not None and idx < len(robot_cam_frame_ids):
            label = robot_cam_frame_ids[idx]
        elif cam_base_frame_ids is not None and idx < len(cam_base_frame_ids):
            label = cam_base_frame_ids[idx]
        else:
            label = idx
        print(f"[frame {label}] robot->base:\n{T}")


if __name__ == "__main__":
    main()
