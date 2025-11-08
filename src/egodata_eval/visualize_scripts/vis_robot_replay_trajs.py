#!/usr/bin/env python3
"""
Visualize three trajectories together for robot_replay:
  1) Absolute trajectory converted to base frame (per-point transform)
  2) Relative-trajectory converted to base and integrated back to absolute
  3) Original camera-frame absolute trajectory (also shown transformed to base for comparison)

Usage:
  python src/egodata_eval/visualize_scripts/vis_robot_replay_trajs.py \
      --traj outputs/delta_eval_book_traj.txt \
      --T_zed_aruco T_zed_aruco.npy \
      --T_base_aruco glasses_hardware/calib/T_base_aruco.npy \
      --spawn

Notes:
  - Requires `rerun` (pip install rerun-sdk).
  - Supports streaming playback with `--fps`, logging accumulating paths per frame.
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
    ids: list[int] = []
    xyz: list[tuple[float, float, float]] = []
    with open(traj_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
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
    ids_arr = np.asarray(ids, dtype=np.int64)
    xyz_arr = np.asarray(xyz, dtype=np.float32)
    return ids_arr, xyz_arr


def transform_points(T: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    N = pts_cam.shape[0]
    hom = np.concatenate([pts_cam.astype(np.float32), np.ones((N, 1), dtype=np.float32)], axis=1)
    out = (T.astype(np.float32) @ hom.T).T
    return out[:, :3].astype(np.float32)


def integrate_base_deltas_to_abs(traj_base_delta: np.ndarray, start_xyz_base: np.ndarray) -> np.ndarray:
    """Integrate only translations from base deltas to reconstruct absolute positions.

    This avoids heavy dependencies; it ignores rotation for visualization of path.
    """
    if traj_base_delta is None or traj_base_delta.size == 0:
        return start_xyz_base[None, :].astype(np.float32)
    pts = [start_xyz_base.astype(np.float32)]
    cur = start_xyz_base.astype(np.float32)
    for d in traj_base_delta:
        cur = (cur + d[:3].astype(np.float32))
        pts.append(cur.copy())
    return np.asarray(pts, dtype=np.float32)


def main():
    _add_project_root_to_path()
    parser = argparse.ArgumentParser(description="Visualize robot_replay trajectories together")
    parser.add_argument("--traj", type=Path, default=Path("outputs/delta_eval_book_traj.txt"))
    parser.add_argument("--T_zed_aruco", type=Path, default=Path("T_zed_aruco.npy"))
    parser.add_argument("--T_base_aruco", type=Path, default=Path("glasses_hardware/calib/T_base_aruco.npy"))
    parser.add_argument("--spawn", action="store_true", help="Spawn Rerun viewer window")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS")
    args = parser.parse_args()

    ids, xyz_abs_cam = load_traj_xyz(args.traj)
    stem = Path(args.traj).stem
    base_delta_path = Path("outputs") / f"{stem}_base_delta.npy"
    base_abs_path = Path("outputs") / f"{stem}_base_abs.npy"
    if not base_delta_path.exists() or not base_abs_path.exists():
        raise FileNotFoundError(
            f"Missing saved files: {base_delta_path} or {base_abs_path}.\n"
            f"Run robot_replay.py with --traj {args.traj} --save first."
        )
    traj_base_delta = np.load(str(base_delta_path)).astype(np.float32)
    xyz_abs_base_direct = np.load(str(base_abs_path)).astype(np.float32)
    xyz_abs_base_from_rel = integrate_base_deltas_to_abs(traj_base_delta, xyz_abs_base_direct[0])
    xyz_abs_cam_path = xyz_abs_cam.astype(np.float32)

    try:
        import rerun as rr
    except Exception as e:
        raise RuntimeError("Rerun package is required. Install with `pip install rerun-sdk`.") from e

    rr.init("Robot Replay Trajectories", spawn=args.spawn)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    # Colors as uint8 RGB
    col_red = np.array([[255, 0, 0]], dtype=np.uint8)
    col_green = np.array([[0, 200, 0]], dtype=np.uint8)
    col_blue = np.array([[30, 144, 255]], dtype=np.uint8)  # dodger blue

    # Stream per-frame with accumulating strips
    import time
    dt = 1.0 / max(float(args.fps), 1e-6)

    # Ensure monotonic increasing sequence/time based on provided frame ids
    order = np.argsort(ids)
    ids_sorted = ids[order]
    base_abs_direct_sorted = xyz_abs_base_direct[order]
    base_abs_from_rel_sorted = xyz_abs_base_from_rel[order]
    cam_abs_sorted = xyz_abs_cam_path[order]

    acc_abs = []
    acc_rel = []
    acc_cam = []
    # Log full planned paths once for reference (optional)
    rr.set_time("frame", sequence=int(ids_sorted[0]))
    rr.log("traj_full/base_abs_from_absolute", rr.LineStrips3D([base_abs_direct_sorted.astype(np.float32)], colors=col_red))
    rr.log("traj_full/base_abs_from_relative", rr.LineStrips3D([base_abs_from_rel_sorted.astype(np.float32)], colors=col_green))
    rr.log("traj_full/cam_abs", rr.LineStrips3D([cam_abs_sorted.astype(np.float32)], colors=col_blue))

    for fid, p_abs, p_rel, p_cam in zip(ids_sorted, base_abs_direct_sorted, base_abs_from_rel_sorted, cam_abs_sorted):
        rr.set_time("frame", sequence=int(fid))
        # Points at current frame
        rr.log("traj_step/base_abs_from_absolute_point", rr.Points3D(p_abs[np.newaxis, :].astype(np.float32), colors=col_red))
        rr.log("traj_step/base_abs_from_relative_point", rr.Points3D(p_rel[np.newaxis, :].astype(np.float32), colors=col_green))
        rr.log("traj_step/cam_abs_point", rr.Points3D(p_cam[np.newaxis, :].astype(np.float32), colors=col_blue))

        # Accumulated line strips so far
        acc_abs.append(p_abs.astype(np.float32))
        acc_rel.append(p_rel.astype(np.float32))
        acc_cam.append(p_cam.astype(np.float32))
        rr.log("traj/base_abs_from_absolute", rr.LineStrips3D([np.asarray(acc_abs, dtype=np.float32)], colors=col_red))
        rr.log("traj/base_abs_from_relative", rr.LineStrips3D([np.asarray(acc_rel, dtype=np.float32)], colors=col_green))
        rr.log("traj/cam_abs", rr.LineStrips3D([np.asarray(acc_cam, dtype=np.float32)], colors=col_blue))
        time.sleep(dt)


if __name__ == "__main__":
    main()
