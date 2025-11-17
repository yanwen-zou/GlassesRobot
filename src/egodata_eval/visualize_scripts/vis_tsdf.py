import argparse
from pathlib import Path

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


def visualize_tsdf_point_cloud(cloud_path: Path, recording: str):
    points, colors = load_point_cloud(cloud_path)
    rr.init(recording, spawn=True)
    rr.log("world/tsdf_point_cloud", rr.Points3D(points, colors=colors))


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize VGGT reconstruction with Rerun.")
    parser.add_argument(
        "--path",
        type=str,
        default="data/20251112_142342/vggt_output/vggt_pointcloud.ply",
        help="Path to the TSDF point cloud (.ply) file.",
    )
    parser.add_argument(
        "--recording",
        type=str,
        default="tsdf_pointcloud",
        help="Rerun recording name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cloud_path = Path(args.path)
    if not cloud_path.is_file():
        raise FileNotFoundError(f"Point cloud file not found: {cloud_path}")

    visualize_tsdf_point_cloud(cloud_path, args.recording)


if __name__ == "__main__":
    main()
