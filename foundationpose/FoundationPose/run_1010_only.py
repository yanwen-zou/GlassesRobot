# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import argparse
import cv2
import imageio
import os

from estimater import *
from zed_datareader import *


if __name__=='__main__':
  code_dir = os.path.dirname(os.path.realpath(__file__))
  default_data_root = os.path.join(os.path.dirname(os.path.dirname(code_dir)), 'data')
  parser = argparse.ArgumentParser()
  parser.add_argument('--demo_name', type=str, default='mustard0')
  parser.add_argument('--data_root', type=str, default=default_data_root)
  # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  # parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  
  set_logging_format()
  set_seed(0)
  
  demo_dir = os.path.join(args.data_root, args.demo_name)
  mesh_file = os.path.join(args.data_root, 'sjtu_lion', 'lion.obj')
  test_scene_dir = demo_dir

  mesh = trimesh.load(mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis')

  ob_in_cam_dir = os.path.join(demo_dir, 'ob_in_cam')
  os.makedirs(ob_in_cam_dir, exist_ok=True)

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator  done")

  reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)

  # 初始化 VideoWriter
  video_path = os.path.join(demo_dir, 'foundationpose_vis.mp4')
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  frame_height, frame_width = reader.get_color(0).shape[:2]
  video_writer = cv2.VideoWriter(video_path, fourcc, 20, (frame_width, frame_height))

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    np.savetxt(os.path.join(ob_in_cam_dir, f'{reader.id_strs[i]}.txt'), pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)
      # 写入视频帧
      video_writer.write(vis[...,::-1])  # BGR

    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

  video_writer.release()
  print(f"🎬 视频已保存到: {video_path}")
