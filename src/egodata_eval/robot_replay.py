#!/usr/bin/env python3
"""
Replay a ZED-frame absolute trajectory on the Flexiv robot as base-frame relative deltas.

Logic matches vis_frames.py:
- Load absolute points [frame_id x y z] in ZED camera frame (meters)
- Load T_zed_aruco (zed<-aruco) and T_base_aruco (base<-aruco)
- Derive base<-zed: T_base_cam = T_base_aruco @ inv(T_zed_aruco)
- Convert to base-frame relative-to-first points: p_base_rel[i] = R_base_cam @ (p_cam[i] - p0_cam)
- Convert p_base_rel to incremental deltas: d = p_base_rel[i] - p_base_rel[i-1] (with p_base_rel[-1]=0)
- Execute relative deltas on the robot TCP, keeping rotation identity (rot6d)

Current TCP pose is treated as the starting pose by execute_relative_traj.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def load_T(path: Path) -> np.ndarray:
    T = np.load(str(path))
    if T.shape != (4, 4):
        raise ValueError(f"Invalid SE3 at {path}, expected (4,4), got {T.shape}")
    return T.astype(np.float32)


def load_traj_xyz(traj_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ids: list[int] = []
    xyz: list[Tuple[float, float, float]] = []
    with open(traj_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                fid = int(float(parts[0]))
                x, y, z = map(float, parts[1:4])
            except ValueError:
                continue
            ids.append(fid)
            xyz.append((x, y, z))
    if not xyz:
        raise FileNotFoundError(f"No valid waypoints loaded from {traj_path}")
    return np.asarray(ids, dtype=np.int64), np.asarray(xyz, dtype=np.float32)


def build_base_relative_deltas(xyz_abs_cam: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """Compute base-frame relative deltas (dx,dy,dz,rot6d) from absolute ZED points.

    - xyz_abs_cam: (N,3) absolute points in ZED
    - T_base_cam: 4x4 base<-zed transform
    Returns: (N-1, 9) deltas with rot6d identity for each step
    """
    if xyz_abs_cam.shape[0] < 2:
        return np.zeros((0, 9), dtype=np.float32)
    R_base_cam = T_base_cam[:3, :3].astype(np.float32)
    p0 = xyz_abs_cam[0].astype(np.float32)
    # Relative-to-first in base frame for each subsequent point (i=1..N-1)
    base_rel = (R_base_cam @ (xyz_abs_cam.astype(np.float32) - p0).T).T
    base_rel = base_rel[1:, :]  # drop the zero for first frame
    # Incremental deltas between consecutive base_rel points
    dxyz = np.vstack([base_rel[0:1, :], base_rel[1:, :] - base_rel[:-1, :]]).astype(np.float32)
    I_r6 = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    rot6 = np.repeat(I_r6[None, :], dxyz.shape[0], axis=0)
    return np.concatenate([dxyz, rot6], axis=1).astype(np.float32)


def execute_base_relative_points_with_send(
    base_rel_points: np.ndarray,
    sleep_s: float = 0.05,
    steps: int | None = None,
) -> None:
    """Execute a base-frame relative trajectory by directly calling robot.send_tcp_pose.

    - base_rel_points: (N,3) positions relative to the starting TCP, expressed in base frame.
    - Keeps the initial TCP orientation fixed for all steps.
    """
    from glasses_hardware.hardware.my_device.robot import FlexivRobot
    from glasses_hardware.hardware.my_device.robot import _mat_to_pose7  # reuse conversion helper
    from MBA.utils.transformation import rotation_transform  # type: ignore

    robot = FlexivRobot()
    curr_pose7 = robot.get_tcp_pose().astype(np.float32)
    start_xyz = curr_pose7[:3].astype(np.float32)
    start_quat = curr_pose7[3:7].astype(np.float32)

    # Build target pose7 for each relative waypoint: xyz = start_xyz + p_rel, quat = start_quat
    n_total = base_rel_points.shape[0]
    n = n_total if steps is None else min(int(steps), n_total)
    for i in range(n):
        xyz = start_xyz + base_rel_points[i].astype(np.float32)
        pose7 = np.concatenate([xyz, start_quat], axis=0).astype(np.float32)
        print(f"[cmd] send_tcp_pose: step={i+1}/{n}, pose7={np.round(pose7, 6)}")
        robot.send_tcp_pose(pose7)
        import time as _t
        _t.sleep(sleep_s)


def main():
    parser = argparse.ArgumentParser(description="Replay ZED traj on Flexiv as base-frame relative deltas")
    parser.add_argument(
        "--traj",
        type=Path,
        default=Path("outputs/delta_eval_book_traj.txt"),
        help="Input trajectory text file (frame_id x y z in ZED frame)",
    )
    parser.add_argument(
        "--T_zed_aruco",
        type=Path,
        default=Path("T_zed_aruco.npy"),
        help="Path to 4x4 SE3 numpy file for zed<-aruco",
    )
    parser.add_argument(
        "--T_base_aruco",
        type=Path,
        default=Path("glasses_hardware/calib/T_base_aruco.npy"),
        help="Path to 4x4 SE3 numpy file for base<-aruco (no offset)",
    )
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep seconds between steps")
    parser.add_argument("--max-steps", type=int, default=0, help="If >0, limit number of steps executed")
    parser.add_argument("--save", action="store_true", help="Save converted base trajectories to outputs/")

    args = parser.parse_args()

    ids, xyz_abs_cam = load_traj_xyz(args.traj)
    T_zed_aruco = load_T(args.T_zed_aruco)
    T_base_aruco = load_T(args.T_base_aruco)
    T_base_cam = T_base_aruco @ np.linalg.inv(T_zed_aruco)

    # Build base-frame relative-to-first points per vis_frames logic
    if xyz_abs_cam.shape[0] < 2:
        print("[warn] Trajectory has <2 points; nothing to execute.")
        return
    R_base_cam = T_base_cam[:3, :3].astype(np.float32)
    p0_cam = xyz_abs_cam[0].astype(np.float32)
    base_rel_points = (R_base_cam @ (xyz_abs_cam.astype(np.float32) - p0_cam).T).T  # shape (N,3), first is [0,0,0]
    base_rel_points = base_rel_points[1:, :]  # drop the first zero to start moving

    # Optional save
    if args.save:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.traj).stem
        base_rel_path = out_dir / f"{stem}_base_rel_points_visframes.npy"
        np.save(str(base_rel_path), base_rel_points.astype(np.float32))
        print(f"[info] Saved base relative points to {base_rel_path}")

    max_steps = int(args.max_steps) if args.max_steps and args.max_steps > 0 else None
    print(f"[info] Executing {len(base_rel_points) if max_steps is None else min(max_steps, len(base_rel_points))} steps from current TCP (base frame) using send_tcp_pose")
    execute_base_relative_points_with_send(base_rel_points, sleep_s=args.sleep, steps=max_steps)


if __name__ == "__main__":
    main()
