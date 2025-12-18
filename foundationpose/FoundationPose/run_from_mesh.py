#!/usr/bin/env python3
"""
Run FoundationPose on a recorded demo while automatically locating mesh.obj under data/<mesh_name>/.

Usage example:
    python foundationpose/FoundationPose/run_from_mesh.py \\
        --demo-name small_book \\
        --mesh-name book
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import trimesh

from estimater import *  # type: ignore
from zed_datareader import *  # type: ignore


def parse_args() -> argparse.Namespace:
    code_dir = Path(__file__).resolve().parent
    default_data_root = code_dir.parents[1] / "data"
    parser = argparse.ArgumentParser(description="FoundationPose runner that auto-loads mesh.obj from data/<mesh>/")
    parser.add_argument("--demo-name", type=str, required=True, help="Name of dataset directory under data/.")
    parser.add_argument("--mesh-name", type=str, required=True, help="Folder name under data/ containing mesh.obj.")
    parser.add_argument("--data-root", type=Path, default=default_data_root, help="Root directory containing data/.")
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug-dir", type=Path, default=code_dir / "debug")
    return parser.parse_args()


def locate_mesh(data_root: Path, mesh_name: str) -> Path:
    mesh_path = data_root / mesh_name / "mesh.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(f"未找到 mesh: {mesh_path}")
    return mesh_path


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    demo_dir = data_root / args.demo_name
    mesh_file = locate_mesh(data_root, args.mesh_name)

    if not demo_dir.exists():
        raise FileNotFoundError(f"未找到 demo 数据: {demo_dir}")

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(mesh_file)
    logging.info("Loaded mesh from %s", mesh_file)

    debug_dir = Path(args.debug_dir)
    os.system(f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis")

    ob_in_cam_dir = demo_dir / "ob_in_cam"
    ob_in_cam_dir.mkdir(parents=True, exist_ok=True)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=args.debug,
        glctx=glctx,
    )
    logging.info("Estimator ready.")

    reader = YcbineoatReader(video_dir=str(demo_dir), shorter_side=None, zfar=np.inf)

    video_path = demo_dir / "foundationpose_vis.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_height, frame_width = reader.get_color(0).shape[:2]
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 20, (frame_width, frame_height))

    for i in range(len(reader.color_files)):
        logging.info("frame %d", i)
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

            if args.debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(debug_dir / "model_tf.obj")
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(str(debug_dir / "scene_complete.ply"), pcd)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

        np.savetxt(ob_in_cam_dir / f"{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        if args.debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow("FoundationPose", vis[..., ::-1])
            cv2.waitKey(1)
            video_writer.write(vis[..., ::-1])

        if args.debug >= 2:
            (debug_dir / "track_vis").mkdir(parents=True, exist_ok=True)
            imageio.imwrite(debug_dir / f"track_vis/{reader.id_strs[i]}.png", vis)

    video_writer.release()
    print(f"[OK] 视频保存至: {video_path}")


if __name__ == "__main__":
    main()
