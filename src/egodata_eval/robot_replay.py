#!/usr/bin/env python3
"""
Replay a 3D trajectory with relative deltas using the same transform logic as eval.py.

Trajectory source: outputs/delta_eval_book_traj.txt
  - Text lines: "frame_id x y z (absolute ref frame)" in ZED camera frame.
  - We convert absolute XYZ to relative deltas (incremental), set rotation delta to identity,
    then conjugate each delta from camera to base using:
        T_delta_base = T_base_cam @ T_delta_cam @ inv(T_base_cam)
    where T_base_cam = T_base_aruco @ inv(T_zed_aruco).
  - Finally, we execute the resulting relative trajectory via execute_relative_traj.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def _add_project_root_to_path():
    import sys
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def load_T(path: Path) -> np.ndarray:
    T = np.load(str(path))
    if T.shape != (4, 4):
        raise ValueError(f"Invalid SE3 at {path}, expected (4,4), got {T.shape}")
    return T.astype(np.float32)


def load_traj_xyz(traj_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load trajectory as arrays of frame_ids and xyz (meters) from a text file.

    Returns:
      ids: int array shape (N,)
      xyz_cam: float array shape (N, 3)
    """
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


def _absolute_xyz_to_relative_cam_eval_style(xyz_abs_cam: np.ndarray) -> np.ndarray:
    """Convert absolute camera xyz to pairwise relative deltas using eval.py logic.

    Mirrors TrajectoryPredictor._absolute_to_delta_np but with a moving base pose
    (previous absolute pose). Rotations are assumed identity for input, and kept via
    delta computation: R_delta = R_abs @ R_base^T.

    Returns: (T-1, 10) array [dxyz, drot6d, grip(0)] in camera frame.
    """
    from MBA.utils.transformation import rotation_transform  # type: ignore

    N = xyz_abs_cam.shape[0]
    if N < 2:
        return np.zeros((0, 10), dtype=np.float32)

    # Identity rotation_6d for each absolute pose
    I_r6 = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    abs_r6_all = np.repeat(I_r6[None, :], N, axis=0)

    # Pairwise base (prev) and abs (curr)
    abs_xyz = xyz_abs_cam[1:, :].astype(np.float32)
    base_xyz = xyz_abs_cam[:-1, :].astype(np.float32)
    delta_xyz = abs_xyz - base_xyz

    R_abs = rotation_transform(abs_r6_all[1:, :], "rotation_6d", "matrix")
    R_base = rotation_transform(abs_r6_all[:-1, :], "rotation_6d", "matrix")
    # For identity inputs, this yields identity; kept for consistency with eval.py
    R_delta = np.einsum('nij,njk->nik', R_abs, np.transpose(R_base, (0, 2, 1)))
    delta_r6 = rotation_transform(R_delta, "matrix", "rotation_6d").astype(np.float32)

    grip = np.zeros((N - 1, 1), dtype=np.float32)
    delta_full = np.concatenate([delta_xyz.astype(np.float32), delta_r6, grip], axis=1)
    return delta_full


def _traj_cam_to_base(traj_cam: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """Match eval.py: conjugate each relative delta (xyz+rot6d[+grip]) camera->base.

    T_delta_base = T_base_cam @ T_delta_cam @ inv(T_base_cam)
    """
    from MBA.utils.transformation import mat_to_xyz_rot  # type: ignore
    from src.egodata_eval.eval_utils import _build_pose_mats  # type: ignore

    if traj_cam is None or traj_cam.size == 0:
        return traj_cam
    out = []
    Tbc_inv = np.linalg.inv(T_base_cam)
    for i in range(traj_cam.shape[0]):
        step = traj_cam[i]
        xyz = step[:3].astype(np.float32)
        r6 = step[3:9].astype(np.float32)
        grip = step[9:10] if step.shape[0] > 9 else None
        T_cam = _build_pose_mats(xyz[None, :], r6[None, :])[0]
        T_base = T_base_cam @ T_cam @ Tbc_inv
        xyzr6 = mat_to_xyz_rot(T_base.astype(np.float32), rotation_rep="rotation_6d").astype(np.float32)
        if grip is not None:
            out.append(np.concatenate([xyzr6[:3], xyzr6[3:9], grip.astype(np.float32)], axis=0))
        else:
            out.append(np.concatenate([xyzr6[:3], xyzr6[3:9]], axis=0))
    return np.stack(out, axis=0).astype(np.float32)


def _transform_points(T: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    N = pts_cam.shape[0]
    hom = np.concatenate([pts_cam.astype(np.float32), np.ones((N, 1), dtype=np.float32)], axis=1)
    out = (T.astype(np.float32) @ hom.T).T
    return out[:, :3].astype(np.float32)


def replay_relative_traj_base(
    traj_base_delta: np.ndarray,
    sleep_s: float = 0.05,
    steps: int | None = None,
) -> None:
    """Execute relative trajectory in base frame using execute_relative_traj.

    traj_base_delta: (T, 10) or (T, 9) with [dx,dy,dz, rot6d, (optional grip)]
    """
    from glasses_hardware.hardware.my_device.robot import FlexivRobot, FlexivGripper, execute_relative_traj
    robot = FlexivRobot()
    gripper = FlexivGripper(robot)
    n = traj_base_delta.shape[0] if steps is None else min(int(steps), traj_base_delta.shape[0])
    _ = execute_relative_traj(robot, gripper, traj_base_delta, steps=n, step_sleep=sleep_s)


def main():
    _add_project_root_to_path()

    parser = argparse.ArgumentParser(description="Replay camera-frame trajectory in robot base frame")
    parser.add_argument(
        "--traj",
        type=Path,
        default=Path("outputs/delta_eval_book_traj.txt"),
        help="Input trajectory text file (frame_id x y z)",
    )
    parser.add_argument(
        "--T_zed_aruco",
        type=Path,
        default=Path("T_zed_aruco.npy"),
        help="Path to 4x4 SE3 numpy file for camera<-aruco",
    )
    parser.add_argument(
        "--T_base_aruco",
        type=Path,
        default=Path("glasses_hardware/calib/T_base_aruco.npy"),
        help="Path to 4x4 SE3 numpy file for base<-aruco",
    )
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep seconds between steps")
    parser.add_argument("--max-steps", type=int, default=0, help="If >0, limit number of steps executed")
    parser.add_argument("--save", action="store_true", help="Save converted base trajectories to outputs/")

    args = parser.parse_args()

    ids, xyz_abs_cam = load_traj_xyz(args.traj)
    T_zed_aruco = load_T(args.T_zed_aruco)
    T_base_aruco = load_T(args.T_base_aruco)
    T_base_cam = T_base_aruco @ np.linalg.inv(T_zed_aruco)

    # 1) Make relative deltas in camera frame
    traj_cam_delta = _absolute_xyz_to_relative_cam_eval_style(xyz_abs_cam)
    # If fewer than 2 points, nothing to execute
    if traj_cam_delta.shape[0] == 0:
        print("[warn] Trajectory has <2 points; nothing to execute.")
        return

    # 3) Conjugate deltas to base frame
    traj_base_delta = _traj_cam_to_base(traj_cam_delta, T_base_cam)

    # Also compute absolute base path by transforming absolute cam points per-point
    xyz_abs_base = _transform_points(T_base_cam, xyz_abs_cam)

    # Optionally save outputs
    if args.save:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.traj).stem
        base_delta_path = out_dir / f"{stem}_base_delta.npy"
        base_abs_path = out_dir / f"{stem}_base_abs.npy"
        np.save(str(base_delta_path), traj_base_delta.astype(np.float32))
        np.save(str(base_abs_path), xyz_abs_base.astype(np.float32))
        print(f"[info] Saved base delta to {base_delta_path}")
        print(f"[info] Saved base absolute to {base_abs_path}")

    # 4) Execute relative steps
    max_steps = int(args.max_steps) if args.max_steps and args.max_steps > 0 else None
    print(f"[info] Executing {len(traj_base_delta) if max_steps is None else min(max_steps, len(traj_base_delta))} relative steps")
    replay_relative_traj_base(traj_base_delta, sleep_s=args.sleep, steps=max_steps)


if __name__ == "__main__":
    main()
