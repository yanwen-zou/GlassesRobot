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


def _load_extrinsic_from_vec(vals: np.ndarray) -> np.ndarray:
    if vals.ndim != 1:
        raise ValueError(f"Extrinsic vector must be 1D, got {vals.shape}")
    if vals.size == 16:
        mat = vals.reshape(4, 4)
    elif vals.size == 12:
        mat = np.vstack([vals.reshape(3, 4), np.array([0, 0, 0, 1], dtype=np.float32)])
    elif vals.size == 7:
        mat = xyz_rot_transform(vals, from_rep="quaternion", to_rep="matrix").astype(np.float32)
    else:
        raise ValueError(f"Invalid extrinsic vector length {vals.size}")
    if mat.shape == (3, 4):
        mat = np.vstack([mat, np.array([0, 0, 0, 1], dtype=np.float32)])
    if mat.shape != (4, 4):
        raise ValueError(f"Invalid extrinsic matrix shape {mat.shape}")
    return mat


def load_abs_traj(data_root: str, split: str, seq: str) -> Tuple[List[str], List[np.ndarray]]:
    base = data_root if split == 'all' else os.path.join(data_root, split)
    seq_dir = os.path.join(base, seq)
    ob_dir = os.path.join(seq_dir, 'ob_in_cam')
    head_path = os.path.join(seq_dir, 'head_pos.txt')
    if not os.path.isdir(ob_dir):
        raise FileNotFoundError(f"Missing ob_in_cam: {ob_dir}")
    if not os.path.isfile(head_path):
        raise FileNotFoundError(f"Missing head_pos.txt: {head_path}")

    stems = _sorted_frame_stems(ob_dir)
    # build extrinsic map
    extr_map: Dict[int, np.ndarray] = {}
    rows = np.loadtxt(head_path, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows[None, :]
    for row_idx, row in enumerate(rows):
        if row.size in (8, 13, 17):
            fid = int(round(row[0]))
            data = row[1:]
        elif row.size in (7, 12, 16):
            fid = row_idx
            data = row
        else:
            raise ValueError(f"Invalid head_pos row length {row.size} in {head_path}")
        try:
            extr_map[fid] = _load_extrinsic_from_vec(data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load head_pos row {row_idx}: {e}")

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
    ap = argparse.ArgumentParser(description='Replay absolute T_list on robot: align initial xyz, send tcp pose.')
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
        """Search calib_dir for a file matching stem (e.g.,'T_base_aruco', 'eih_camT').
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
    T_base_aruco = _load_calib_mat(calib_dir, 'T_base_aruco')
    # eih_camT is not required for this mapping, but load in case future logic needs it
    _ = _load_calib_mat(calib_dir, 'eih_camT')

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

    # Build absolute trajectory in robot base frame
    T_list = [T_base_aruco @ (np.linalg.inv(T_zed_aruco) @ Tz) for Tz in pose_abs]
    for T in T_list: 
        T[1,3] =-T[1,3]
        

    print(f"[INFO] Applied calibration from {calib_dir} (T_zed_aruco & T_base_aruco)")

    if len(T_list) <= 1:
        print("Not enough poses to execute.")
        return

    # Initialize robot now to get current TCP for initial xyz alignment
    robot = FlexivRobot()
    gripper = FlexivGripper(robot)
    tcp0 = robot.get_tcp_pose().astype(np.float32)
    tcp0_xyz = tcp0[:3]

    # Align only initial xyz: compute translation offset so that T_list[0] matches current tcp xyz
    T0_xyz = T_list[0][:3, 3].astype(np.float32)
    trans_delta = tcp0_xyz - T0_xyz

    # Apply translation offset to all poses, keep rotation as in T_list
    T_list_aligned = []
    for T in T_list:
        T_adj = T.copy().astype(np.float32)
        T_adj[:3, 3] = T_adj[:3, 3].astype(np.float32) + trans_delta
        
        T_list_aligned.append(T_adj)

    # Optionally limit number of poses
    if args.limit and args.limit > 0:
        T_list_aligned = T_list_aligned[:max(1, args.limit)]
        T_list = T_list[:max(1, args.limit)]

    print(f"[INFO] Absolute poses to execute: {len(T_list_aligned)} (aligned xyz only)")

    # Derive base-frame relative transforms between consecutive absolute poses
    Ci_mats = []
    for i in range(1, len(T_list_aligned)):
        Bi_prev = T_list_aligned[i - 1]
        Bi = T_list_aligned[i]
        Ci = np.linalg.inv(Bi_prev) @ Bi
        Ci_mats.append(Ci.astype(np.float32))


    # Preview: absolute path after alignment
    abs_pts = np.stack([T[:3, 3] for T in T_list_aligned], axis=0) if len(T_list_aligned) else np.empty((0, 3), dtype=np.float32)
    # Also preview the path if executing via relative TCP-local deltas equivalent to abs path
    def _accumulate_positions(C_list: List[np.ndarray], T0: np.ndarray) -> np.ndarray:
        if len(C_list) == 0:
            return np.empty((0, 3), dtype=np.float32)
        T = T0.astype(np.float32).copy()
        pts = [T[:3, 3].copy()]
        for Ci in C_list:
            Td_local = np.linalg.inv(T) @ Ci @ T
            T = T @ Td_local
            pts.append(T[:3, 3].copy())
        return np.stack(pts, axis=0)
    rel_pts = _accumulate_positions(Ci_mats, T_list_aligned[0]) if len(Ci_mats) else np.empty((0,3), dtype=np.float32)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    if abs_pts.size:
        ax.plot(abs_pts[:, 0], abs_pts[:, 1], abs_pts[:, 2], '-o', markersize=3, label='Absolute aligned (T_list)')
    if rel_pts.size:
        ax.plot(rel_pts[:, 0], rel_pts[:, 1], rel_pts[:, 2], '-o', markersize=3, label='Relative (TCP-local) preview')
    if abs_pts.size:
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'Trajectory Preview (abs poses={len(T_list_aligned)})')
        ax.legend(loc='best')
        # set axes strictly by data min/max
        xs = abs_pts[:, 0]; ys = abs_pts[:, 1]; zs = abs_pts[:, 2]
        xmin, xmax = float(np.min(xs)), float(np.max(xs))
        ymin, ymax = float(np.min(ys)), float(np.max(ys))
        zmin, zmax = float(np.min(zs)), float(np.max(zs))
        # Set axes strictly by data min/max; guard zero ranges slightly
        eps = 1e-6
        if xmax - xmin < eps: xmax = xmin + eps
        if ymax - ymin < eps: ymax = ymin + eps
        if zmax - zmin < eps: zmax = zmin + eps
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
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

    # Execute by converting base-frame deltas to TCP-local deltas per step
    print(f'Executing {len(Ci_mats)} steps with on-the-fly conjugation to TCP frame (equivalent to abs path)...')
    executed = []
    for Ci in Ci_mats:
        # Read current tcp pose and build T_world_tcp
        curr_pose7 = robot.get_tcp_pose()
        T_world_tcp = np.eye(4, dtype=np.float32)
        T_world_tcp[:3, 3] = curr_pose7[:3]
        R_world_tcp = rotation_transform(curr_pose7[3:7][None, :], 'quaternion', 'matrix').squeeze(0)
        T_world_tcp[:3, :3] = R_world_tcp

        # Conjugate base delta into current TCP local frame
        T_delta_tcp = np.linalg.inv(T_world_tcp) @ Ci @ T_world_tcp
        d_local = mat_to_xyz_rot(T_delta_tcp, 'rotation_6d').astype(np.float32)
        step_vec = np.concatenate([d_local, np.array([0.0], dtype=np.float32)], axis=0)[None, :]
        tgt_arr = execute_relative_traj(robot, gripper, step_vec, steps=1, step_sleep=args.sleep, scale_factor=1.0)
        if tgt_arr is not None and len(tgt_arr):
            executed.append(tgt_arr[-1])

    # Optional quick visualize executed xyz path
    try:
        exec_pts = np.asarray(executed, dtype=np.float32)
        if exec_pts.size:
            exec_xyz = exec_pts[:, :3]
            fig2 = plt.figure(figsize=(6, 6))
            ax2 = fig2.add_subplot(111, projection='3d')
            ax2.plot(abs_pts[:, 0], abs_pts[:, 1], abs_pts[:, 2], '-o', color='tab:blue', markersize=3, label='Planned abs path')
            ax2.plot(exec_xyz[:, 0], exec_xyz[:, 1], exec_xyz[:, 2], '-o', color='tab:orange', markersize=3, label='Executed path')
            ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
            ax2.set_title('Executed vs Planned (absolute)')
            ax2.legend()
            plt.show()
        else:
            print('No executed poses recorded for visualization.')
    except Exception as e:
        warnings.warn(f'Failed to visualize executed absolute trajectory: {e}')
    print('Done.')


if __name__ == '__main__':
    main()
