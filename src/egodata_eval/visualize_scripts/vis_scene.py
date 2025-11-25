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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and visualize a scene point cloud in Rerun.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/20251112_142342/vggt_output/vggt_pointcloud.ply"),
        help="Path to the scene point cloud (.ply).",
    )
    parser.add_argument("--recording", type=str, default="scene_pointcloud", help="Rerun recording name.")
    parser.add_argument("--spawn", action="store_true", help="Spawn the Rerun viewer window.")
    return parser.parse_args()


def main():
    args = parse_args()
    cloud_path = args.path
    book_cloud_path = Path("data/20251112_142342/vggt_output/object_masks_headpos/000000.ply")   
    if not cloud_path.is_file():
        raise FileNotFoundError(f"Point cloud file not found: {cloud_path}")

    pts, cols = load_point_cloud(cloud_path)

    pts_book, cols_book = load_point_cloud(book_cloud_path)

    rr.init(args.recording, spawn=args.spawn)
    rr.log("world/scene", rr.Points3D(pts, colors=cols))
    rr.log("world/object_book", rr.Points3D(pts_book, colors=cols_book))


if __name__ == "__main__":
    main()
