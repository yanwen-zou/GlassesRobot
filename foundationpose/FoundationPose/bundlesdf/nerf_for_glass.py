# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from nerf_runner import *
import glob
import os

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from datareader import *
from bundlesdf.tool import *
import yaml,argparse


def run_neural_object_field(cfg, K, rgbs, depths, masks, cam_in_obs, debug=0, save_dir='/home/bowen/debug/foundationpose_bundlesdf'):
  rgbs = np.asarray(rgbs)
  depths = np.asarray(depths)
  masks = np.asarray(masks)
  cam_in_obs = np.asarray(cam_in_obs)
  glcam_in_obs = cam_in_obs@glcam_in_cvcam

  cfg['save_dir'] = save_dir
  os.makedirs(save_dir, exist_ok=True)

  for i,rgb in enumerate(rgbs):
    imageio.imwrite(f'{save_dir}/rgb_{i:07d}.png', rgb)

  sc_factor,translation,pcd_real_scale, pcd_normalized = compute_scene_bounds(None,glcam_in_obs, K, use_mask=True,base_dir=save_dir,rgbs=rgbs,depths=depths,masks=masks, eps=cfg['dbscan_eps'], min_samples=cfg['dbscan_eps_min_samples'])
  cfg['sc_factor'] = sc_factor
  cfg['translation'] = translation

  o3d.io.write_point_cloud(f'{save_dir}/pcd_normalized.ply', pcd_normalized)

  rgbs_, depths_, masks_, normal_maps,poses = preprocess_data(rgbs, depths, masks,normal_maps=None,poses=glcam_in_obs,sc_factor=cfg['sc_factor'],translation=cfg['translation'])

  nerf = NerfRunner(cfg, rgbs_, depths_, masks_, normal_maps=None, poses=poses, K=K, occ_masks=None, build_octree_pcd=pcd_normalized)
  nerf.train()

  mesh = nerf.extract_mesh(isolevel=0,voxel_size=cfg['mesh_resolution'])
  mesh = nerf.mesh_texture_from_train_images(mesh, rgbs_raw=rgbs, tex_res=1028)
  optimized_cvcam_in_obs,offset = get_optimized_poses_in_real_world(poses,nerf.models['pose_array'],cfg['sc_factor'],cfg['translation'])
  mesh = mesh_to_real_world(mesh, pose_offset=offset, translation=nerf.cfg['translation'], sc_factor=nerf.cfg['sc_factor'])
  return mesh


def _load_frame_filter(base_dir, cfg):
  frame_yaml = cfg.get('frame_yaml', 'select_frames.yml')
  frame_file = os.path.join(base_dir, frame_yaml)
  if not os.path.exists(frame_file):
    return None
  with open(frame_file, 'r') as ff:
    info = yaml.safe_load(ff) or {}
  frames = info.get('frames')
  if not frames:
    return None
  try:
    return {str(int(fid)).zfill(6) for fid in frames}
  except Exception:
    return {str(fid) for fid in frames}


def run_one_ob(base_dir, cfg, use_refined_mask=False):
  save_dir = os.path.join(base_dir, cfg.get('save_subdir', 'nerf'))
  os.system(f'rm -rf {save_dir} && mkdir -p {save_dir}')

  rgb_dir = cfg.get('rgb_dir', 'rgb')
  rgb_ext = cfg.get('rgb_ext', '.png')
  depth_dir = cfg.get('depth_dir', 'depth_enhanced')
  depth_ext = cfg.get('depth_ext', '.png')
  depth_scale = cfg.get('depth_scale', 1.0)
  pose_dir = cfg.get('pose_dir', 'cam_in_ob')
  pose_ext = cfg.get('pose_ext', '.txt')

  if use_refined_mask and cfg.get('mask_refined_dir'):
    mask_dir = cfg['mask_refined_dir']
  else:
    mask_dir = cfg.get('mask_dir', 'mask')
  mask_ext = cfg.get('mask_ext', '.png')

  color_files = sorted(glob.glob(os.path.join(base_dir, rgb_dir, f'*{rgb_ext}')))
  if not color_files:
    raise FileNotFoundError(f'No RGB frames found under {base_dir}/{rgb_dir}')

  K_file = os.path.join(base_dir, cfg.get('K_file', 'K.txt'))
  if not os.path.exists(K_file):
    raise FileNotFoundError(f'Camera intrinsics file not found: {K_file}')
  K = np.loadtxt(K_file)

  frame_filter = _load_frame_filter(base_dir, cfg)

  rgbs = []
  depths = []
  masks = []
  cam_in_obs = []

  for color_file in color_files:
    frame_id = os.path.splitext(os.path.basename(color_file))[0]
    frame_id_key = frame_id.zfill(6)
    if frame_filter and frame_id_key not in frame_filter:
      continue

    depth_file = os.path.join(base_dir, depth_dir, f'{frame_id}{depth_ext}')
    mask_file = os.path.join(base_dir, mask_dir, f'{frame_id}{mask_ext}')
    pose_file = os.path.join(base_dir, pose_dir, f'{frame_id}{pose_ext}')

    if not os.path.exists(depth_file):
      raise FileNotFoundError(f'Depth file missing for frame {frame_id}: {depth_file}')
    if not os.path.exists(mask_file):
      raise FileNotFoundError(f'Mask file missing for frame {frame_id}: {mask_file}')
    if not os.path.exists(pose_file):
      raise FileNotFoundError(f'Pose file missing for frame {frame_id}: {pose_file}')

    rgb = imageio.imread(color_file)
    depth = cv2.imread(depth_file, -1) / depth_scale
    mask = cv2.imread(mask_file, -1)
    cam_in_ob = np.loadtxt(pose_file).reshape(4,4)

    rgbs.append(rgb)
    depths.append(depth)
    masks.append(mask)
    cam_in_obs.append(cam_in_ob)

  if not rgbs:
    raise RuntimeError('No frames selected for reconstruction. Check frame filter configuration.')

  mesh = run_neural_object_field(cfg, K, rgbs, depths, masks, cam_in_obs, save_dir=save_dir, debug=0)
  return mesh


def run_ycbv():
  ob_ids = np.arange(1,22)
  code_dir = os.path.dirname(os.path.realpath(__file__))
  with open(f'{code_dir}/config_ycbv.yml','r') as ff:
    cfg = yaml.safe_load(ff)

  for ob_id in ob_ids:
    base_dir = f'{args.ref_view_dir}/ob_{ob_id:07d}'
    mesh = run_one_ob(base_dir=base_dir, cfg=cfg)
    out_file = f'{base_dir}/model/model.obj'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mesh.export(out_file)


def run_linemod():
  ob_ids = np.setdiff1d(np.arange(1,16), np.array([7,3])).tolist()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  with open(f'{code_dir}/config_linemod.yml','r') as ff:
    cfg = yaml.safe_load(ff)
  for ob_id in ob_ids:
    base_dir = f'{args.ref_view_dir}/ob_{ob_id:07d}'
    mesh = run_one_ob(base_dir=base_dir, cfg=cfg, use_refined_mask=True)
    out_file = f'{base_dir}/model/model.obj'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mesh.export(out_file)
    logging.info(f"saved to {out_file}")


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--ref_view_dir', type=str, default=f'/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16')
  parser.add_argument('--dataset', type=str, default=f'ycbv', help='one of [ycbv/linemod/custom]')
  parser.add_argument('--base_dir', type=str, default=None, help='Base directory for a single sequence (required for dataset=custom)')
  parser.add_argument('--config', type=str, default=None, help='Override path to config yaml')
  parser.add_argument('--use_refined_mask', action='store_true', help='Use refined mask directory if available')
  args = parser.parse_args()

  if args.dataset=='ycbv':
    run_ycbv()
  elif args.dataset=='linemod':
    run_linemod()
  else:
    if args.base_dir is None:
      raise ValueError('For dataset=custom, --base_dir must be provided')
    cfg_path = args.config or f'{code_dir}/config_data_lion.yml'
    with open(cfg_path, 'r') as ff:
      cfg = yaml.safe_load(ff)
    mesh = run_one_ob(base_dir=args.base_dir, cfg=cfg, use_refined_mask=args.use_refined_mask)
    out_file = os.path.join(args.base_dir, cfg.get('output_mesh', 'model/model.obj'))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mesh.export(out_file)
