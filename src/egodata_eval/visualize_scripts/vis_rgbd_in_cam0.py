#!/usr/bin/env python3
"""
可视化：逐帧将 RGB-D 点云转换到第一帧相机坐标系下，并显示 ArUco 坐标系。

输入：
- RGB 序列：<episode>/rgb/000000.png ...
- 深度序列：<episode>/depth/000000.png （米或 depth_scale 转换）
- head_pos：<episode>/head_pos/000000.txt (tx ty tz qx qy qz qw)，左手系，需 flip Y，再乘 tcp->zed
- 内参：<episode>/cam_K.txt (3x3)
- T_tcp_zed：glasses_hardware/calib/T_tcp_zed.npy
- T_base_aruco：glasses_hardware/calib/T_base_aruco.npy

渲染：Rerun 中显示 cam0 坐标系、Aruco 坐标系和累积点云（每帧加粗不同颜色）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rerun as rr
from PIL import Image


def load_intrinsics(path: Path) -> np.ndarray:
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    mat = np.array(rows, dtype=np.float32)
    if mat.shape != (3, 3):
        raise ValueError(f"Intrinsic matrix must be 3x3, got {mat.shape}")
    return mat


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quat
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def load_headposes(head_dir: Path, T_tcp_zed: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    pose_files = sorted(head_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if not pose_files:
        raise FileNotFoundError(f"No head poses found in {head_dir}")

    poses_cam: Dict[int, np.ndarray] = {}
    poses_raw: Dict[int, np.ndarray] = {}

    flip_y = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float32)

    for path in pose_files:
        vals = np.loadtxt(path, dtype=np.float32).reshape(-1)
        if vals.size < 7:
            continue
        t = vals[:3]
        q = vals[3:7]
        R = quaternion_to_matrix(q)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t
        poses_raw[int(path.stem)] = T.copy()  # world->head (left-handed, unflipped)

        # world->head (left-handed) -> world->cam (flip Y) -> world->zed
        T_cam = T @ flip_y @ T_tcp_zed
        poses_cam[int(path.stem)] = T_cam

    if not poses_cam:
        raise FileNotFoundError(f"No valid head poses found in {head_dir}")

    first_key = min(poses_cam.keys())
    E0_cam = poses_cam[first_key]
    inv_E0_cam = np.linalg.inv(E0_cam)
    poses_cam0 = {k: inv_E0_cam @ pose for k, pose in poses_cam.items()}  # cam_i (flipped) in cam0

    # Raw poses relative to first frame (no flip/tcp->zed): cam_i_raw in cam0_raw
    E0_raw = poses_raw[first_key]
    inv_E0_raw = np.linalg.inv(E0_raw)
    poses_raw_rel = {k: inv_E0_raw @ pose for k, pose in poses_raw.items()}

    return poses_cam0, poses_raw_rel


def rgbd_to_pointcloud(color: np.ndarray, depth_m: np.ndarray, intrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Open3D backprojection matching MBA.realworld load_point_cloud."""
    import open3d as o3d

    h, w = depth_m.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    colors_o3d = o3d.geometry.Image((color * 255).astype(np.uint8))
    depths_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
    cam_intr = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors_o3d, depths_o3d, 1.0, convert_rgb_to_intensity=False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam_intr)
    points = np.asarray(cloud.points, dtype=np.float32)
    colors = np.asarray(cloud.colors, dtype=np.float32)
    return points, colors


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_depth(path: Path, depth_scale: float) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.float32) / depth_scale


def main():
    parser = argparse.ArgumentParser(description="Visualize per-frame RGB-D point clouds in cam0 frame with ArUco axis.")
    parser.add_argument("--episode-dir", type=Path, required=True, help="Episode root containing rgb/depth/head_pos.")
    parser.add_argument("--intrinsic", type=Path, default=None, help="Defaults to <episode>/cam_K.txt")
    parser.add_argument("--head-pos-dir", type=Path, default=None, help="Defaults to <episode>/head_pos")
    parser.add_argument("--rgb-dir", type=Path, default=None, help="Defaults to <episode>/rgb")
    parser.add_argument("--depth-dir", type=Path, default=None, help="Defaults to <episode>/depth")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Meters per depth unit.")
    parser.add_argument("--T_tcp_zed", type=Path, default=Path("glasses_hardware/calib/T_tcp_zed.npy"))
    parser.add_argument("--T_base_aruco", type=Path, default=Path("glasses_hardware/calib/T_base_aruco.npy"))
    parser.add_argument("--spawn", action="store_true", help="Spawn rerun viewer.")
    parser.add_argument("--point_size", type=float, default=0.002, help="Point radius in rerun.")
    parser.add_argument("--plot-headpos-2d", action="store_true", help="Plot head_pos xyz over frames in a Matplotlib window.")
    args = parser.parse_args()

    episode = args.episode_dir
    rgb_dir = args.rgb_dir or (episode / "rgb")
    depth_dir = args.depth_dir or (episode / "depth")
    head_dir = args.head_pos_dir or (episode / "head_pos")

    K = load_intrinsics(args.intrinsic or (episode / "cam_K.txt"))
    T_tcp_zed = np.load(args.T_tcp_zed).astype(np.float32)

    headposes_cam0, headposes_raw_rel = load_headposes(head_dir, T_tcp_zed)
    T_base_aruco = np.load(args.T_base_aruco).astype(np.float32)

    rr.init("rgbd_in_cam0", spawn=args.spawn)

    rr.log(
        "frames/base_aruco",
        rr.Transform3D(
            translation=T_base_aruco[:3, 3],
            mat3x3=T_base_aruco[:3, :3],
        ),
    )

    rr.log(
        "frames/cam0",
        rr.Transform3D(
            translation=np.zeros(3, dtype=np.float32),
            mat3x3=np.eye(3, dtype=np.float32),
        ),
    )

    files = sorted(rgb_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"No RGB frames in {rgb_dir}")

    head_xyz_cam0 = []        # flipped+tcp->zed, cam0 frame
    head_xyz_raw = []         # raw head_pos relative to first frame

    for idx, rgb_path in enumerate(files):
        fid = int(rgb_path.stem)
        depth_path = depth_dir / f"{fid:06d}.png"
        if not depth_path.exists() or fid not in headposes_cam0:
            continue

        depth_m = load_depth(depth_path, args.depth_scale)
        rgb = load_rgb(rgb_path)
        pts_cam, colors = rgbd_to_pointcloud(rgb, depth_m, K)

        if pts_cam.size == 0:
            continue

        colors = (colors * 255).astype(np.uint8)

        # Transform cam_i points into cam0 frame
        T_cam_from_cam0 = headposes_cam0[fid]
        pts_ref = (T_cam_from_cam0[:3, :3] @ pts_cam.T + T_cam_from_cam0[:3, 3:4]).T.astype(np.float32)

        rr.set_time_sequence("frame", fid)

        rr.log(
            "frames/cam0/points",
            rr.Points3D(
                pts_ref,
                colors=colors,
                radii=np.full(len(pts_ref), args.point_size, dtype=np.float32),
            ),
        )

        # Log cam_i frame in cam0 for reference
        rr.log(
            "frames/cam_i",
            rr.Transform3D(
                translation=T_cam_from_cam0[:3, 3],
                mat3x3=T_cam_from_cam0[:3, :3],
            ),
        )

        rr.log(
            "frames/cam_i/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3), dtype=np.float32),
                vectors=np.eye(3, dtype=np.float32) * 0.05,
                colors=np.array(
                    [
                        [255, 0, 0, 255],
                        [0, 255, 0, 255],
                        [0, 0, 255, 255],
                    ],
                    dtype=np.uint8,
                ),
                radii=np.full(3, 0.0025, dtype=np.float32),
            ),
        )

        head_xyz_cam0.append((fid, T_cam_from_cam0[:3, 3].copy()))

        if fid in headposes_raw_rel:
            head_xyz_raw.append((fid, headposes_raw_rel[fid][:3, 3].copy()))

    if args.plot_headpos_2d and head_xyz_cam0:
        import matplotlib.pyplot as plt

        head_xyz_cam0.sort(key=lambda x: x[0])
        fids_cam0 = np.array([f for f, _ in head_xyz_cam0], dtype=np.int32)
        xyz_cam0 = np.stack([p for _, p in head_xyz_cam0], axis=0)
        t_cam0 = fids_cam0 - fids_cam0[0]

        labels = ["x", "y", "z"]
        colors = ["r", "g", "b"]

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        for i in range(3):
            axes[i].plot(t_cam0, xyz_cam0[:, i], color=colors[i], label=f"cam0_{labels[i]}")

            if head_xyz_raw:
                head_xyz_raw.sort(key=lambda x: x[0])
                fids_raw = np.array([f for f, _ in head_xyz_raw], dtype=np.int32)
                xyz_raw = np.stack([p for _, p in head_xyz_raw], axis=0)
                t_raw = fids_raw - fids_raw[0]

                axes[i].plot(
                    t_raw,
                    xyz_raw[:, i],
                    color=colors[i],
                    linestyle="--",
                    alpha=0.6,
                    label=f"raw_{labels[i]}",
                )

            axes[i].set_ylabel(labels[i])
            axes[i].grid(True)

        axes[-1].set_xlabel("frame index offset")
        axes[0].set_title("Head pose: cam0 (flip+tcp->zed) vs raw (relative to first)")

        for ax in axes:
            ax.legend(loc="upper right")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
