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
    parser.add_argument("--recording", type=str, default="tsdf_pointcloud", help="Rerun recording name.")
    parser.add_argument("--spawn", action="store_true", help="Spawn Rerun viewer window.")
    return parser.parse_args()


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

    any_logged = False
    for fid, path in iter_frame_clouds(obj_dir):
        rr.set_time_sequence("frame", fid)
        if scene_points is not None:
            rr.log("world/scene", rr.Points3D(scene_points, colors=scene_colors))
        pts, cols = load_point_cloud(path)
        rr.log("world/object", rr.Points3D(pts, colors=cols))
        any_logged = True

    if not any_logged:
        raise RuntimeError(f"No per-frame object clouds found in {obj_dir}")


if __name__ == "__main__":
    main()
