import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d
import rerun as rr


def load_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise RuntimeError(f"Point cloud at {path} is empty.")
    points = np.asarray(pcd.points, dtype=np.float32)
    if pcd.colors:
        colors = np.asarray(pcd.colors, dtype=np.float32)
    else:
        colors = np.ones_like(points, dtype=np.float32)
    return points, colors


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize TSDF/scene point cloud and per-frame object clouds with Rerun.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("data/20251112_142342/vggt_output/vggt_pointcloud.ply"),
        help="Static scene/TSDF point cloud (.ply) to show once.",
    )
    parser.add_argument(
        "--object-dir",
        type=Path,
        default=Path("data/20251112_142342/vggt_output/object_masks_headpos"),
        help="Directory containing per-frame object point clouds (PLY).",
    )
    parser.add_argument(
        "--head-pos-dir",
        type=Path,
        default=None,
        help="Optional directory of head_pos poses (txt, tx ty tz qx qy qz qw). If provided, will plot trajectory relative to first frame.",
    )
    parser.add_argument(
        "--T_tcp_zed",
        type=Path,
        default=Path("glasses_hardware/calib/T_tcp_zed.npy"),
        help="tcp->zed extrinsic to align head poses with camera frame.",
    )
    parser.add_argument("--recording", type=str, default="tsdf_pointcloud", help="Rerun recording name.")
    parser.add_argument("--spawn", action="store_true", help="Spawn Rerun viewer window.")
    return parser.parse_args()


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert [qx,qy,qz,qw] to 3x3 rotation matrix."""
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


def load_headposes(head_dir: Path, T_tcp_zed: np.ndarray) -> dict[int, np.ndarray]:
    """Load world->head poses, flip Y to right-handed, apply tcp->zed, return cam0<-cami."""
    pose_files = sorted(head_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if not pose_files:
        raise FileNotFoundError(f"No head poses found in {head_dir}")
    poses = {}
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
        # world->head (left-handed) -> world->cam (right-handed) -> world->zed
        T = T  @ flip_y @ T_tcp_zed
        poses[int(path.stem)] = T
    if not poses:
        raise FileNotFoundError(f"No valid head poses found in {head_dir}")
    first_key = min(poses.keys())
    E0 = poses[first_key]
    E0_inv = np.linalg.inv(E0)
    return {k: E0_inv @ pose for k, pose in poses.items()}  # cam_i origin expressed in cam0


def iter_frame_clouds(directory: Path) -> Iterable[tuple[int, Path]]:
    paths = sorted(directory.glob("*.ply"), key=lambda p: int(p.stem))
    for path in paths:
        try:
            fid = int(path.stem)
        except ValueError:
            continue
        yield fid, path


def main():
    args = parse_args()
    rr.init(args.recording, spawn=args.spawn)

    scene_points: np.ndarray | None = None
    scene_colors: np.ndarray | None = None
    if args.scene and args.scene.exists():
        scene_points, scene_colors = load_point_cloud(args.scene)
    else:
        print(f"[WARN] Scene point cloud not found: {args.scene}")

    obj_dir = args.object_dir
    if not obj_dir.is_dir():
        raise FileNotFoundError(f"Object cloud directory not found: {obj_dir}")

    headposes = None
    if args.head_pos_dir:
        head_dir = args.head_pos_dir
    else:
        # try sibling head_pos next to object dir
        head_dir = obj_dir.parent.parent / "head_pos"
    if head_dir.exists():
        try:
            T_tcp_zed = np.load(args.T_tcp_zed).astype(np.float32)
            headposes = load_headposes(head_dir, T_tcp_zed)
        except Exception as exc:
            print(f"[WARN] Failed to load head poses from {head_dir}: {exc}")

    any_logged = False
    for fid, path in iter_frame_clouds(obj_dir):
        rr.set_time_sequence("frame", fid)
        if scene_points is not None:
            rr.log("world/scene", rr.Points3D(scene_points, colors=scene_colors))
        pts, cols = load_point_cloud(path)
        rr.log("world/object", rr.Points3D(pts, colors=cols))
        if headposes and fid in headposes:
            T = headposes[fid]
            rr.log(
                "world/headpose",
                rr.Transform3D(
                    translation=T[:3, 3],
                    mat3x3=T[:3, :3],
                ),
            )
            axes = (np.eye(3, dtype=np.float32) * 0.05).astype(np.float32)
            rr.log(
                "world/headpose/axes",
                rr.Arrows3D(
                    origins=np.zeros((3, 3), dtype=np.float32),
                    vectors=axes,
                    colors=np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8),
                    radii=np.full(3, 0.0025, dtype=np.float32),
                ),
            )
        any_logged = True

    if not any_logged:
        raise RuntimeError(f"No per-frame object clouds found in {obj_dir}")


if __name__ == "__main__":
    main()
