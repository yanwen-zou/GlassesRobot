import argparse
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.transformation import xyz_rot_transform


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
    """
    Load camera extrinsic from file. Supports:
    - 4x4 / 3x4 matrix
    - 16 / 12 length vector (row-major)
    - 7D vector [x, y, z, qx, qy, qz, qw] (xyz + quaternion)
    """
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


def load_sequence_paths(data_root: str, split: str, seq_id: str) -> Dict[str, str]:
    base = data_root if split == 'all' else os.path.join(data_root, split)
    seq_dir = os.path.join(base, seq_id)
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
    ob_dir = os.path.join(seq_dir, 'ob_in_cam')
    head_dir = os.path.join(seq_dir, 'head_pos')
    if not os.path.isdir(ob_dir):
        raise FileNotFoundError(f"Missing ob_in_cam directory: {ob_dir}")
    if not os.path.isdir(head_dir):
        warnings.warn(f"Missing head_pos directory: {head_dir}; using identity extrinsics.")
    return {"seq": seq_dir, "ob": ob_dir, "head": head_dir}


def compute_trajectories(ob_dir: str, head_dir: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    frame_stems = _sorted_frame_stems(ob_dir)
    if not frame_stems:
        raise RuntimeError(f"No frames found under {ob_dir}")

    # Load extrinsics map
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
                warnings.warn(f"Failed to load head_pos {f}: {e}")

    # Reference extrinsic (first visible frame)
    ref_key = int(frame_stems[0]) if frame_stems[0].isdigit() else frame_stems[0]
    ref_extr = extr_map.get(ref_key, np.eye(4, dtype=np.float32))
    ref_extr_inv = np.linalg.inv(ref_extr)

    pts_in_cam_list = []
    pts_in_ref_list = []

    for stem in frame_stems:
        pose_path = os.path.join(ob_dir, f"{stem}.txt")
        if not os.path.exists(pose_path):
            # allow .npy optionally
            pose_path_npy = os.path.join(ob_dir, f"{stem}.npy")
            if os.path.exists(pose_path_npy):
                pose_mat = np.load(pose_path_npy).astype(np.float32)
                if pose_mat.shape == (3, 4):
                    pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1], dtype=np.float32)])
            else:
                warnings.warn(f"Missing object pose for frame {stem} in {ob_dir}")
                continue
        else:
            pose_mat = _load_pose_matrix(pose_path)

        # translation in the current cam frame
        t_cam = pose_mat[:3, 3]
        pts_in_cam_list.append(t_cam)

        # map to absolute (first-frame cam) frame: ref_inv @ cam_extr @ ob_in_cam
        key = int(stem) if stem.isdigit() else stem
        cam_extr = extr_map.get(key, np.eye(4, dtype=np.float32))
        pose_world = ref_extr_inv @ cam_extr @ pose_mat
        t_ref = pose_world[:3, 3]
        pts_in_ref_list.append(t_ref)

    pts_in_cam = np.stack(pts_in_cam_list, axis=0) if pts_in_cam_list else np.empty((0, 3), dtype=np.float32)
    pts_in_ref = np.stack(pts_in_ref_list, axis=0) if pts_in_ref_list else np.empty((0, 3), dtype=np.float32)
    return frame_stems, pts_in_cam, pts_in_ref


def plot_3d_traj(points: np.ndarray, title: str, out_path: str) -> None:
    if points.size == 0:
        warnings.warn(f"No points to plot for {title}")
        return
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], '-o', markersize=3)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # set equal aspect-ish
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    zmin, zmax = float(np.min(zs)), float(np.max(zs))
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    r = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.6 + 1e-6
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_side_by_side(native_pts: np.ndarray, abs_pts: np.ndarray, title: str, out_path: str) -> None:
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    if native_pts.size:
        ax1.plot(native_pts[:, 0], native_pts[:, 1], native_pts[:, 2], '-o', markersize=3)
    ax1.set_title('ob_in_cam (per-frame camera)')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    if abs_pts.size:
        ax2.plot(abs_pts[:, 0], abs_pts[:, 1], abs_pts[:, 2], '-o', markersize=3)
    ax2.set_title('Transformed to first-frame cam (absolute)')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

    # match axis ranges for visual comparability on the right subplot
    def _set_bounds(ax, pts):
        if not pts.size:
            return
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        xmin, xmax = float(np.min(xs)), float(np.max(xs))
        ymin, ymax = float(np.min(ys)), float(np.max(ys))
        zmin, zmax = float(np.min(zs)), float(np.max(zs))
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        r = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.6 + 1e-6
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(cz - r, cz + r)
    _set_bounds(ax1, native_pts)
    _set_bounds(ax2, abs_pts)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Visualize GT object trajectory in cam frame and absolute (first-frame) frame.')
    ap.add_argument('--data_path', type=str, default='data', help='Dataset root')
    ap.add_argument('--split', type=str, default='train', choices=['train', 'val', 'all'])
    ap.add_argument('--seq', type=str, required=True, help='Sequence id (timestamp folder name) under split')
    ap.add_argument('--out_dir', type=str, default='outputs/vis_gt', help='Directory to write plots')
    args = ap.parse_args()

    paths = load_sequence_paths(args.data_path, args.split, args.seq)
    stems, pts_cam, pts_abs = compute_trajectories(paths['ob'], paths['head'])

    base_out = os.path.join(args.out_dir, args.split, args.seq)
    os.makedirs(base_out, exist_ok=True)

    # Individual plots
    plot_3d_traj(pts_cam, f'{args.seq}: ob_in_cam (native)', os.path.join(base_out, 'traj_ob_in_cam_3d.png'))
    plot_3d_traj(pts_abs, f'{args.seq}: absolute (first-frame cam)', os.path.join(base_out, 'traj_absolute_3d.png'))

    # Side-by-side
    plot_side_by_side(pts_cam, pts_abs, f'{args.seq}: Trajectory Comparison', os.path.join(base_out, 'traj_compare_3d.png'))

    print(f"Saved 3D trajectory visualizations to {base_out}")


if __name__ == '__main__':
    main()
