import argparse
import os
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
from pathlib import Path
import sys
import subprocess

# Reuse pose loading utilities similar to vis_gt
from MBA.utils.transformation import xyz_rot_transform, rotation_transform, mat_to_xyz_rot, xyz_rot_to_mat
import matplotlib.pyplot as plt

# Robot control
from glasses_hardware.hardware.my_device.robot import (
    FlexivRobot,
    FlexivGripper,
    execute_relative_traj,
)


def _sorted_frame_stems(dir_path: str) -> List[str]:
    files = [f for f in os.listdir(dir_path) if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg", ".txt", ".npy"]]
    stems = [os.path.splitext(f)[0] for f in files]
    def key_fn(s: str):
        try:
            return int(s)
        except Exception:
            return s
    return sorted(list(set(stems)), key=key_fn)


def _load_pose_matrix(path: str) -> np.ndarray:
    vals = np.loadtxt(path).astype(np.float32)
    if vals.ndim == 1:
        if vals.size == 16:
            mat = vals.reshape(4, 4)
        elif vals.size == 12:
            mat = np.vstack([vals.reshape(3, 4), np.array([0, 0, 0, 1], dtype=np.float32)])
        else:
            raise ValueError(f"Invalid pose vector length {vals.size} in {path}")
    else:
        mat = vals
    if mat.shape == (3, 4):
        mat = np.vstack([mat, np.array([0, 0, 0, 1], dtype=np.float32)])
    if mat.shape != (4, 4):
        raise ValueError(f"Invalid pose matrix shape {mat.shape} in {path}")
    return mat


def _load_extrinsic_from_file(path: str) -> np.ndarray:
    vals = np.loadtxt(path).astype(np.float32)
    if vals.ndim == 1:
        if vals.size == 16:
            mat = vals.reshape(4, 4)
        elif vals.size == 12:
            mat = np.vstack([vals.reshape(3, 4), np.array([0, 0, 0, 1], dtype=np.float32)])
        elif vals.size == 7:
            mat = xyz_rot_transform(vals, from_rep="quaternion", to_rep="matrix").astype(np.float32)
        else:
            raise ValueError(f"Invalid extrinsic vector length {vals.size} in {path}")
    else:
        mat = vals
    if mat.shape == (3, 4):
        mat = np.vstack([mat, np.array([0, 0, 0, 1], dtype=np.float32)])
    if mat.shape != (4, 4):
        raise ValueError(f"Invalid extrinsic matrix shape {mat.shape} in {path}")
    return mat


def load_abs_traj(data_root: str, split: str, seq: str) -> Tuple[List[str], List[np.ndarray]]:
    base = data_root if split == 'all' else os.path.join(data_root, split)
    seq_dir = os.path.join(base, seq)
    ob_dir = os.path.join(seq_dir, 'ob_in_cam')
    head_dir = os.path.join(seq_dir, 'head_pos')
    if not os.path.isdir(ob_dir):
        raise FileNotFoundError(f"Missing ob_in_cam: {ob_dir}")
    if not os.path.isdir(head_dir):
        raise FileNotFoundError(f"Missing head_pos: {head_dir}")

    stems = _sorted_frame_stems(ob_dir)
    # build extrinsic map
    extr_map: Dict[int, np.ndarray] = {}
    if os.path.isdir(head_dir):
        for f in os.listdir(head_dir):
            if not f.lower().endswith('.txt'):
                continue
            stem = os.path.splitext(f)[0]
            try:
                key = int(stem)
            except Exception:
                key = stem
            try:
                extr_map[key] = _load_extrinsic_from_file(os.path.join(head_dir, f))
            except Exception as e:
                raise FileNotFoundError(f"Failed to load head_pos {f}: {e}")

    ref_key = int(stems[0]) if stems[0].isdigit() else stems[0]
    ref_extr = extr_map.get(ref_key, np.eye(4, dtype=np.float32))
    ref_extr_inv = np.linalg.inv(ref_extr)

    pose_abs_list: List[np.ndarray] = []
    for s in stems:
        pth = os.path.join(ob_dir, f"{s}.txt")
        if not os.path.exists(pth):
            warnings.warn(f"Missing pose for {s} in {ob_dir}")
            continue
        pose_cam = _load_pose_matrix(pth)
        key = int(s) if s.isdigit() else s
        cam_extr = extr_map.get(key, np.eye(4, dtype=np.float32))
        pose_abs = ref_extr_inv @ cam_extr @ pose_cam
        pose_abs_list.append(pose_abs)
    return stems, pose_abs_list


# Note: no helper conversion function per request; compute deltas inline in main.


def main():
    ap = argparse.ArgumentParser(description='Replay dataset absolute trajectory on robot as relative motions.')
    ap.add_argument('--data_path', type=str, default='data', help='Dataset root')
    ap.add_argument('--split', type=str, default='train', choices=['train', 'val', 'all'])
    ap.add_argument('--seq', type=str, required=True, help='Sequence id (timestamp)')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of steps to execute (0=all)')
    ap.add_argument('--sleep', type=float, default=0.05, help='Sleep per step for motion pacing')
    args = ap.parse_args()

    stems, pose_abs = load_abs_traj(args.data_path, args.split, args.seq)
    if not pose_abs:
        print('No poses loaded.')
        return

    # Initialize ZED->ArUco->Base mapping if calibration is available
    def _load_calib_mat(calib_dir: Path, stem: str) -> np.ndarray | None:
        """Search calib_dir for a file matching stem (e.g., 'T_zed_aruco', 'T_base_aruco', 'eih_camT').
        Prefer exact '<stem>.npy', otherwise pick the first file whose name starts with stem.
        Returns np.ndarray if found and valid 4x4, else None.
        """
        exact = calib_dir / f"{stem}.npy"
        cand = None
        if exact.exists():
            cand = exact
        else:
            for p in calib_dir.glob(f"{stem}*.npy"):
                cand = p
                break
        if cand is None:
            return None
        try:
            arr = np.load(str(cand)).astype(np.float32)
            if arr.shape == (4, 4):
                return arr
            # allow 3x4 by padding
            if arr.shape == (3, 4):
                arr = np.vstack([arr, np.array([0, 0, 0, 1], dtype=np.float32)])
                return arr
        except Exception:
            return None
        return None

    project_root = Path(__file__).resolve().parents[2]
    calib_dir = project_root / 'glasses_hardware' / 'calib'
    T_zed_aruco = _load_calib_mat(calib_dir, 'T_zed_aruco')
    T_base_aruco = _load_calib_mat(calib_dir, 'T_base_aruco')
    # eih_camT is not required for this mapping, but load in case future logic needs it
    _ = _load_calib_mat(calib_dir, 'eih_camT')

    # If missing T_zed_aruco, compute it live via piper_calib
    try:
        script = project_root / 'src' / 'egodata_eval' / 'piper_calib.py'
        out_path = calib_dir / 'T_zed_aruco.npy'
        calib_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(script), '--marker-length', '0.045', '--out', str(out_path)]
        print(f"[INFO] Computing T_zed_aruco live via: {' '.join(cmd)}")
        ret = subprocess.run(cmd, check=False)
        if ret.returncode == 0 and out_path.exists():
            T_zed_aruco = _load_calib_mat(calib_dir, 'T_zed_aruco')
            print(f"[OK] Loaded T_zed_aruco from {out_path}")
        else:
            FileNotFoundError(f"Failed to compute T_zed_aruco via piper_calib.")
    except Exception as e:
        print(f"[WARN] Exception when computing T_zed_aruco live: {e}")
        
    print("T_zed_aruco =\n", T_zed_aruco)
    print("T_base_aruco =\n", T_base_aruco)

    #T_list = pose_abs
    T_list = [T_base_aruco @ (np.linalg.inv(T_zed_aruco) @ Tz) for Tz in pose_abs]
    print(f"[INFO] Applied calibration from {calib_dir} (T_zed_aruco & T_base_aruco)")


    # Assume robot is already at the trajectory start. Build deltas relative to the first pose.
    # Base-relative transforms: B_i = inv(T0) @ Ti; incremental command: C_i = inv(B_{i-1}) @ B_i
    if len(T_list) <= 1:
        print('Not enough poses to execute.')
        return
    T0 = T_list[0]
    B_prev = np.eye(4, dtype=np.float32)
    cmd_list = []
    for i in range(1, len(T_list)):
        Bi = np.linalg.inv(T0) @ T_list[i]
        Ci = np.linalg.inv(B_prev) @ Bi
        xyz_r6 = mat_to_xyz_rot(Ci.astype(np.float32), 'rotation_6d')
        d = xyz_r6[:3]
        rot6 = xyz_r6[3:3 + 6]
        grip = np.array([0.0], dtype=np.float32)
        cmd_list.append(np.concatenate([d.astype(np.float32), rot6.astype(np.float32), grip], axis=0))
        B_prev = Bi

    rel_traj = np.stack(cmd_list, axis=0) if cmd_list else np.empty((0, 10), dtype=np.float32)
    if args.limit and args.limit > 0:
        rel_traj = rel_traj[:args.limit]

    # === Preview 3D trajectory and wait for user decision ===
    def _accumulate_positions(traj: np.ndarray) -> np.ndarray:
        if traj.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        pts = [T[:3, 3].copy()]
        for i in range(len(traj)):
            d = traj[i, :9].astype(np.float32)
            Td = xyz_rot_to_mat(d, rotation_rep='rotation_6d')
            T = T @ Td
            pts.append(T[:3, 3].copy())
        return np.stack(pts, axis=0)

    pts = _accumulate_positions(rel_traj)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    if pts.size:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-o', markersize=3)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'Relative Trajectory Preview (steps={len(rel_traj)})')
        # set roughly equal aspect
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        xmin, xmax = float(np.min(xs)), float(np.max(xs))
        ymin, ymax = float(np.min(ys)), float(np.max(ys))
        zmin, zmax = float(np.min(zs)), float(np.max(zs))
        cx = 0.5 * (xmin + xmax); cy = 0.5 * (ymin + ymax); cz = 0.5 * (zmin + zmax)
        r = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.6 + 1e-6
        ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r); ax.set_zlim(cz - r, cz + r)
    else:
        ax.set_title('No trajectory to execute')
    print("Preview window: press 'p' to execute, 'q' to quit.")
    decision = {'run': False}
    def on_key(event):
        if event.key is None:
            return
        k = event.key.lower()
        if k == 'p':
            decision['run'] = True
            plt.close(event.canvas.figure)
        elif k == 'q':
            decision['run'] = False
            plt.close(event.canvas.figure)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if not decision['run']:
        print('User canceled execution.')
        return

    robot = FlexivRobot()
    gripper = FlexivGripper(robot)
    print(f'Executing {len(rel_traj)} relative steps based on start-relative trajectory...')
    executed_targets = execute_relative_traj(
        robot,
        gripper,
        rel_traj,
        steps=len(rel_traj),
        step_sleep=args.sleep,
        scale_factor=2,
    )
    try:
        # Visualize executed vs preview trajectory (normalized to respective starts)
        exec_pts = np.asarray(executed_targets, dtype=np.float32)
        if exec_pts.size:
            exec_xyz = exec_pts[:, :3]
            exec_xyz = exec_xyz - exec_xyz[0:1]
            fig2 = plt.figure(figsize=(6, 6))
            ax2 = fig2.add_subplot(111, projection='3d')
            if pts.size:
                ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-o', color='tab:blue', markersize=3, label='Preview rel path')
            ax2.plot(exec_xyz[:, 0], exec_xyz[:, 1], exec_xyz[:, 2], '-o', color='tab:orange', markersize=3, label='Executed rel path')
            ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
            ax2.set_title('Executed vs Preview (both start-normalized)')
            ax2.legend()
            # set approximate equal aspect
            allp = []
            if pts.size: allp.append(pts)
            allp.append(exec_xyz)
            cat = np.vstack(allp)
            xs, ys, zs = cat[:, 0], cat[:, 1], cat[:, 2]
            xmin, xmax = float(np.min(xs)), float(np.max(xs))
            ymin, ymax = float(np.min(ys)), float(np.max(ys))
            zmin, zmax = float(np.min(zs)), float(np.max(zs))
            cx = 0.5 * (xmin + xmax); cy = 0.5 * (ymin + ymax); cz = 0.5 * (zmin + zmax)
            r = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.6 + 1e-6
            ax2.set_xlim(cx - r, cx + r); ax2.set_ylim(cy - r, cy + r); ax2.set_zlim(cz - r, cz + r)
            plt.show()
        else:
            print('No executed target poses returned for visualization.')
    except Exception as e:
        warnings.warn(f'Failed to visualize executed trajectory: {e}')
    print('Done.')


if __name__ == '__main__':
    main()
