import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MBA_DIR = os.path.dirname(CURRENT_DIR)
if MBA_DIR not in sys.path:
    sys.path.insert(0, MBA_DIR)

import open3d as o3d
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset.constants import IMG_MEAN, IMG_STD
from dataset.realworld import RealWorldDataset, collate_fn
from utils.transformation import rotation_transform


def build_point_cloud(cloud_feats):
    points = cloud_feats[:, :3]
    colors_norm = cloud_feats[:, 3:]
    colors = np.clip(colors_norm * IMG_STD + IMG_MEAN, 0.0, 1.0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def build_traj_geometry(action_obj, frame_size=0.05):
    translations = action_obj[:, :3]
    rot6d = action_obj[:, 3:3 + 6]
    rot_mats = rotation_transform(rot6d, from_rep="rotation_6d", to_rep="matrix")
    geometries = []
    line_points = []
    for idx, (trans, rot_mat) in enumerate(zip(translations, rot_mats)):
        tf = np.eye(4, dtype=np.float32)
        tf[:3, :3] = rot_mat
        tf[:3, 3] = trans
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame.transform(tf)
        frame.paint_uniform_color([0.2, 0.6, 0.8])
        geometries.append(frame)
        line_points.append(trans)
    if len(line_points) > 1:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        lines = [[i, i + 1] for i in range(len(line_points) - 1)]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.2, 0.2] for _ in lines])
        geometries.append(line_set)
    return geometries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data_back", help="Root directory of dataset")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "all"])
    parser.add_argument("--num_obs", type=int, default=1)
    parser.add_argument("--num_action", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--visualize", action="store_true", help="Visualize point cloud and trajectory")
    parser.add_argument("--index", type=int, default=0, help="Dataset index to inspect")
    parser.add_argument("--show_pose_matrix", action="store_true", help="Print SE(3) matrix for object pose inputs")
    args = parser.parse_args()

    dataset = RealWorldDataset(
        path=args.data_path,
        split=args.split,
        num_obs=args.num_obs,
        num_action=args.num_action,
        with_obj_action=True,
    )

    print(f"Dataset length: {len(dataset)}")
    idx = max(0, min(args.index, len(dataset) - 1))
    sample = dataset[idx]
    print(f"Inspecting sample index: {idx}")
    print("Single sample:")
    print(f"  input_feats_list[0].shape: {sample['input_feats_list'][0].shape}")
    print(f"  action_obj.shape: {sample['action_obj'].shape}")
    print(f"  action_obj_normalized.shape: {sample['action_obj_normalized'].shape}")
    if "current_obj_pose" in sample:
        cur_pose = sample["current_obj_pose"]
        cur_pose_norm = sample["current_obj_pose_normalized"]
        if isinstance(cur_pose, torch.Tensor):
            cur_pose = cur_pose.cpu().numpy()
        if isinstance(cur_pose_norm, torch.Tensor):
            cur_pose_norm = cur_pose_norm.cpu().numpy()
        print("  current_obj_pose:", np.array2string(cur_pose, precision=4))
        print("  current_obj_pose_normalized:", np.array2string(cur_pose_norm, precision=4))
        if args.show_pose_matrix:
            cur_mat = rotation_transform(cur_pose[None, 3:3 + 6], from_rep="rotation_6d", to_rep="matrix")[0]
            tf_mat = np.eye(4, dtype=np.float32)
            tf_mat[:3, :3] = cur_mat
            tf_mat[:3, 3] = cur_pose[:3]
            print("  current_obj_pose matrix:\n", tf_mat)

    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    batch = next(iter(loader))
    print("Batch:")
    print(f"  input_feats_list.shape: {batch['input_feats_list'].shape}")
    print(f"  action_obj.shape: {batch['action_obj'].shape}")
    print(f"  action_obj_normalized.shape: {batch['action_obj_normalized'].shape}")
    if "current_obj_pose" in batch:
        cur_pose_batch = batch["current_obj_pose"]
        cur_pose_norm_batch = batch["current_obj_pose_normalized"]
        print(f"  current_obj_pose.shape: {cur_pose_batch.shape}")
        print(f"  current_obj_pose_normalized.shape: {cur_pose_norm_batch.shape}")
        # show first element
        cur0 = cur_pose_batch[0]
        cur0n = cur_pose_norm_batch[0]
        if isinstance(cur0, torch.Tensor):
            cur0 = cur0.cpu().numpy()
        if isinstance(cur0n, torch.Tensor):
            cur0n = cur0n.cpu().numpy()
        print("  current_obj_pose[0]:", np.array2string(cur0, precision=4))
        print("  current_obj_pose_normalized[0]:", np.array2string(cur0n, precision=4))

    if args.visualize:
        cloud_feats_tensor = sample['input_feats_list'][0]
        if isinstance(cloud_feats_tensor, torch.Tensor):
            cloud_feats = cloud_feats_tensor.cpu().numpy()
        else:
            cloud_feats = np.asarray(cloud_feats_tensor)
        pcd = build_point_cloud(cloud_feats)

        action_obj = sample['action_obj'].cpu().numpy()
        traj_geoms = build_traj_geometry(action_obj)

        print("Launching Open3D visualizer...")
        o3d.visualization.draw_geometries([pcd, *traj_geoms])


if __name__ == "__main__":
    main()
