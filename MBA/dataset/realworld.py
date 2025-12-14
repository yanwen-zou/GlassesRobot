import os
import warnings
import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T
import collections.abc as container_abcs

from PIL import Image
from torch.utils.data import Dataset

from .constants import *
# Use absolute import to allow running when MBA is on PYTHONPATH as a top-level module.
from utils.transformation import xyz_rot_transform

# Default rotation noise (radians) applied to cam_to_base when requested.
CAM_TO_BASE_ROT_NOISE_STD_DEFAULT = np.deg2rad(1.0)

class RealWorldDataset(Dataset):
    """
    Real-world Dataset.
    """
    def __init__(
        self, 
        path, 
        split = 'train', 
        num_obs = 1,
        num_action = 20, 
        voxel_size = 0.005,
        cam_ids = ['104122060902'],
        aug = False,
        aug_trans_min = [-0.2, -0.2, -0.2],
        aug_trans_max = [0.2, 0.2, 0.2],
        aug_rot_min = [-30, -30, -30],
        aug_rot_max = [30, 30, 30],
        aug_jitter = False,
        aug_jitter_params = [0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob = 0.2,
        with_cloud = False,
        with_obj_action = False,
        cam_to_base_rot_noise_std: float = CAM_TO_BASE_ROT_NOISE_STD_DEFAULT,
        head_to_zed_path: str = "glasses_hardware/calib/T_tcp_zed.txt",
    ):
        assert split in ['train', 'eval', 'all']

        self.path = path
        self.split = split
        if split == 'all':
            self.data_path = path
        else:
            self.data_path = os.path.join(path, split)
        self.num_obs = num_obs
        self.num_action = num_action
        self.voxel_size = voxel_size
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob
        self.with_cloud = with_cloud
        self.with_obj_action = with_obj_action
        self.cam_to_base_rot_noise_std = float(cam_to_base_rot_noise_std)

        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"Data path {self.data_path} does not exist.")

        self.all_demos = sorted(
            [
                d for d in os.listdir(self.data_path)
                if os.path.isdir(os.path.join(self.data_path, d))
            ]
        )
        self.num_demos = len(self.all_demos)
        if self.num_demos == 0:
            raise RuntimeError(f"No sequences found under {self.data_path}.")

        self.data_paths = []
        self.obs_frame_ids = []
        self.action_frame_ids = []
        self.seq_ids = []
        if self.with_obj_action:
            self.obj_frame_ids = []
        self.seq_intrinsics = {}
        self.seq_camera_extrinsics = {}
        self.seq_ref_frame = {}
        self.seq_num_frames = {}
        self.seq_last_frame_id = {}
        self.seq_terminal_frame_ids = {}
        self.seq_cam_to_base = {}

        for demo in self.all_demos:
            demo_path = os.path.join(self.data_path, demo)
            rgb_dir = os.path.join(demo_path, "rgb")
            depth_dir = os.path.join(demo_path, "depth")
            if not (os.path.isdir(rgb_dir) and os.path.isdir(depth_dir)):
                continue

            frame_files = sorted(
                [
                    x for x in os.listdir(rgb_dir)
                    if os.path.splitext(x)[1].lower() in ['.png', '.jpg', '.jpeg']
                ]
            )
            frame_ids = [os.path.splitext(x)[0] for x in frame_files]
            if len(frame_ids) == 0:
                continue

            intr_path = os.path.join(demo_path, "cam_K.txt")
            if not os.path.exists(intr_path):
                raise FileNotFoundError(f"Camera intrinsic file missing: {intr_path}")
            cam_K = np.loadtxt(intr_path).astype(np.float32)
            cam_K = cam_K.reshape(3, 3)
            self.seq_intrinsics[demo] = cam_K

            head_pos_dir = os.path.join(demo_path, "head_pos")
            if os.path.isdir(head_pos_dir):
                extr_map = self._load_camera_extrinsics_from_dir(head_pos_dir, head_to_zed_path)
            else:
                extr_map = {}
                warnings.warn(f"[RealWorldDataset] Missing head_pos directory in {demo_path}; using identity extrinsics.")
            self.seq_camera_extrinsics[demo] = extr_map

            cam_to_base_map = self._load_cam_to_base(demo_path, frame_ids, extr_map)
            self.seq_cam_to_base[demo] = cam_to_base_map
            self.seq_ref_frame[demo] = int(frame_ids[0])
            self.seq_num_frames[demo] = len(frame_ids)
            self.seq_last_frame_id[demo] = frame_ids[-1]
            terminal_ids = set(int(fid) for fid in frame_ids[-5:]) if len(frame_ids) >= 5 else set(int(fid) for fid in frame_ids)
            self.seq_terminal_frame_ids[demo] = terminal_ids

            obs_frame_ids_list = []
            action_frame_ids_list = []
            if self.with_obj_action:
                obj_frame_ids_list = []

            for cur_idx in range(len(frame_ids) - 1):
                obs_pad_before = max(0, num_obs - cur_idx - 1)
                action_pad_after = max(0, num_action - (len(frame_ids) - 1 - cur_idx))
                frame_begin = max(0, cur_idx - num_obs + 1)
                frame_end = min(len(frame_ids), cur_idx + num_action + 1)
                obs_frames = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                action_frames = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                obs_frame_ids_list.append(obs_frames)
                action_frame_ids_list.append(action_frames)
                if self.with_obj_action:
                    obj_frame_ids_list.append(action_frames)

            self.data_paths += [demo_path] * len(obs_frame_ids_list)
            self.obs_frame_ids += obs_frame_ids_list
            self.action_frame_ids += action_frame_ids_list
            self.seq_ids += [demo] * len(obs_frame_ids_list)
            if self.with_obj_action:
                self.obj_frame_ids += obj_frame_ids_list

        if len(self.data_paths) == 0:
            raise RuntimeError(f"No valid samples constructed from {self.data_path}.")

    def _get_cam_to_base(self, seq_id, frame_id_str):
        """Fetch T_base_cam for a frame; identity if missing."""
        cam_to_base_map = self.seq_cam_to_base.get(seq_id, {})
        if not cam_to_base_map:
            return np.eye(4, dtype=np.float32)
        if frame_id_str in cam_to_base_map:
            return cam_to_base_map[frame_id_str]
        try:
            frame_int = int(frame_id_str)
            fallback_key = f"{frame_int}"
        except ValueError:
            fallback_key = None
        if fallback_key and fallback_key in cam_to_base_map:
            return cam_to_base_map[fallback_key]
        warnings.warn(f"[dataset] Missing cam_to_base for seq={seq_id} frame={frame_id_str}; using identity.")
        return np.eye(4, dtype=np.float32)
        
    def __len__(self):
        return len(self.obs_frame_ids)

    def _normalize_tcp(self, tcp_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
        return tcp_list
    
    def _normalize_obj(self, obj_list):
        ''' obj_list: [T, 3(trans) + 6(rot)]'''
        obj_list[:, :3] = (obj_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        return obj_list

    def _load_camera_extrinsics_from_dir(self, directory, head_to_zed_path):
        # Load head to zed transformation(from calibration)
        def _load_head_zed(path_str: str) -> np.ndarray:
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Head to zed calibration file missing: {path_str}")
            try:
                if path_str.lower().endswith(".npy"):
                    arr = np.load(path_str).astype(np.float32)
                else:
                    arr = np.loadtxt(path_str).astype(np.float32)
            except Exception as exc:
                raise ValueError(f"Failed to load head_to_zed from {path_str}: {exc}")
            if arr.shape == (3, 4):
                arr = np.vstack([arr, np.array([0, 0, 0, 1], dtype=np.float32)])
            if arr.shape != (4, 4):
                raise ValueError(f"Invalid head to zed transformation shape {arr.shape} in {path_str}")
            return arr

        T_head_zed = _load_head_zed(head_to_zed_path)
        
        extr_map = {}
        files = [f for f in os.listdir(directory) if f.lower().endswith(".txt")]
        if not files:
            warnings.warn(f"[RealWorldDataset] No extrinsic files found in {directory}; using identity.")
            return extr_map

        def sort_key(name):
            stem = os.path.splitext(name)[0]
            try:
                return int(stem)
            except ValueError:
                return stem

        for fname in sorted(files, key=sort_key):
            path = os.path.join(directory, fname)
            values = np.loadtxt(path).astype(np.float32)
            if values.ndim == 1:
                if values.size == 16:
                    mat = values.reshape(4, 4)
                elif values.size == 12:
                    mat = np.vstack([values.reshape(3, 4), np.array([0, 0, 0, 1], dtype=np.float32)])
                elif values.size == 7:
                    x, y, z, qx, qy, qz, qw = values
                    mat = np.eye(4, dtype=np.float32)
                    mat[:3, 3] = np.array([x, y, z], dtype=np.float32)
                    q = np.array([qx, qy, qz, qw], dtype=np.float32)
                    norm = np.linalg.norm(q)
                    if norm < 1e-8:
                        raise ValueError(f"Quaternion norm too small in {path}")
                    q /= norm
                    qx, qy, qz, qw = q
                    mat[:3, :3] = np.array(
                        [
                            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                        ],
                        dtype=np.float32,
                    )
                else:
                    raise ValueError(f"Invalid extrinsic vector length {values.size} in {path}")
            else:
                mat = values
            if mat.shape == (3, 4):
                mat = np.vstack([mat, np.array([0, 0, 0, 1], dtype=np.float32)])
            if mat.shape != (4, 4):
                raise ValueError(f"Invalid extrinsic matrix shape {mat.shape} in {path}")

            key = sort_key(fname)
            extr_map[key] = mat.astype(np.float32) @ T_head_zed # world_zed = world_head * head_zed
        return extr_map

    def _load_cam_to_base(self, demo_path, frame_ids, extr_map):
        """Load T_base_cam using first-frame anchor + head_pos relative motion.

        The first cam_to_base is taken from cam_to_base.[npy|txt] at the
        reference frame, and all other frames are derived by chaining the
        relative head_pos change from that reference.
        """

        def _load_anchor_map():
            cam_to_base_map_local = {}
            txt_path = os.path.join(demo_path, "cam_to_base.txt")
            if not os.path.exists(txt_path):
                raise FileNotFoundError(f"[RealWorldDataset] cam_to_base.txt not found in {demo_path}")
            data = np.loadtxt(txt_path, dtype=np.float32, skiprows=1)
            if data.ndim == 1:
                data = data[None, :]
            for row in data:
                if row.size < 13:
                    continue
                fid = int(round(row[0]))
                rot = row[1:10].reshape(3, 3)
                trans = row[10:13]
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = rot
                T[:3, 3] = trans
                cam_to_base_map_local[f"{fid:06d}"] = T
            if not cam_to_base_map_local:
                raise RuntimeError(f"[RealWorldDataset] cam_to_base.txt in {demo_path} contained no valid rows")
            return cam_to_base_map_local

        file_map = _load_anchor_map()

        if file_map:
            anchor_key = sorted(file_map.keys())[0]
            T_base_cam0 = file_map[anchor_key]

        if not extr_map:
            return {f"{int(fid):06d}": T_base_cam0.copy() for fid in frame_ids}

        anchor_int = int(anchor_key)

        T_world_cam0 = extr_map[anchor_int] # extr_map: world_cam0
        cam_to_base_map = {}
        prev_T = T_base_cam0
        for fid in frame_ids:
            fid_int = int(fid)
            fid_key = f"{fid_int:06d}"
            if fid_int not in extr_map:
                warnings.warn(f"[RealWorldDataset] Missing head_pos for frame {fid_key}; reusing previous cam_to_base.")
                cam_to_base_map[fid_key] = prev_T.copy()
                continue
            T_world_cam = extr_map[fid_int]
            T_cam0_cam = np.linalg.inv(T_world_cam0) @ T_world_cam # head_cam_inv @ world_head0_inv @ world_head @ head_cam
            T_base_cam = T_base_cam0 @ T_cam0_cam
            cam_to_base_map[fid_key] = T_base_cam.astype(np.float32)
            prev_T = T_base_cam
        # Optional rotation noise
        if self.cam_to_base_rot_noise_std > 0.0:
            def _rand_rot(std: float) -> np.ndarray:
                vec = np.random.normal(0.0, std, size=3)
                theta = np.linalg.norm(vec)
                if theta < 1e-12:
                    return np.eye(3, dtype=np.float32)
                k = vec / theta
                kx, ky, kz = k
                K = np.array(
                    [
                        [0, -kz, ky],
                        [kz, 0, -kx],
                        [-ky, kx, 0],
                    ],
                    dtype=np.float32,
                )
                sn, cs = np.sin(theta), np.cos(theta)
                return (np.eye(3, dtype=np.float32) + sn * K + (1 - cs) * (K @ K)).astype(np.float32)
            for key, T in list(cam_to_base_map.items()):
                R_noise = _rand_rot(self.cam_to_base_rot_noise_std)
                T_noisy = T.copy()
                T_noisy[:3, :3] = R_noise @ T[:3, :3]
                cam_to_base_map[key] = T_noisy
        return cam_to_base_map

    def get_camera_extrinsic(self, seq_id, frame_idx, warn_prefix="dataset"):
        extr_map = self.seq_camera_extrinsics.get(seq_id, {})
        if not extr_map:
            warnings.warn(f"[{warn_prefix}] No camera extrinsics found for seq={seq_id}; using identity.")
            return np.eye(4, dtype=np.float32)
        if frame_idx in extr_map:
            return extr_map[frame_idx]
        warnings.warn(f"[{warn_prefix}] Missing camera extrinsic for seq={seq_id} frame={frame_idx}; using identity.")
        return np.eye(4, dtype=np.float32)

    def load_point_cloud(self, colors, depths, cam_intrinsic, depth_scale=1000.0, T_base_cam=None):
        h, w = depths.shape
        fx, fy = cam_intrinsic[0, 0], cam_intrinsic[1, 1]
        cx, cy = cam_intrinsic[0, 2], cam_intrinsic[1, 2]
        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, depth_scale, convert_rgb_to_intensity = False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        points = np.array(cloud.points)
        colors = np.array(cloud.colors)
        points = points.astype(np.float32)
        colors = colors.astype(np.float32)

        if T_base_cam is not None:
            ones = np.ones((points.shape[0], 1), dtype=np.float32)
            homo = np.concatenate([points, ones], axis=1)
            points = (T_base_cam @ homo.T).T[:, :3]

        return points, colors

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        seq_id = self.seq_ids[index]
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]

        # directories
        color_dir = os.path.join(data_path, "rgb")
        depth_dir = os.path.join(data_path, "depth")
        obj_dir = os.path.join(data_path, "ob_in_cam")

        cam_intrinsic = self.seq_intrinsics[seq_id]

        # create color jitter
        if self.split == 'train' and self.aug_jitter:
            jitter = T.ColorJitter(
                brightness = self.aug_jitter_params[0],
                contrast = self.aug_jitter_params[1],
                saturation = self.aug_jitter_params[2],
                hue = self.aug_jitter_params[3]
            )
            jitter = T.RandomApply([jitter], p = self.aug_jitter_prob)

        # load colors and depths
        colors_list = []
        depths_list = []
        for frame_id in obs_frame_ids:
            rgb_path = os.path.join(color_dir, "{}.png".format(frame_id))
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(color_dir, "{}.jpg".format(frame_id))
            colors = Image.open(rgb_path)
            if self.split == 'train' and self.aug_jitter:
                colors = jitter(colors)
            colors_np = np.array(colors, dtype=np.uint8)
            colors_list.append(colors_np)
            depth_path = os.path.join(depth_dir, "{}.png".format(frame_id))
            if not os.path.exists(depth_path):
                depth_path = os.path.join(depth_dir, "{}.jpg".format(frame_id))
            depths_list.append(
                np.array(Image.open(depth_path), dtype = np.float32)
            )
        colors_list = np.stack(colors_list, axis = 0)
        depths_list = np.stack(depths_list, axis = 0)

        # point clouds
        clouds = []
        for i, frame_id in enumerate(obs_frame_ids):
            T_base_cam = self._get_cam_to_base(seq_id, frame_id)
            points_base, colors = self.load_point_cloud(
                colors_list[i], depths_list[i], cam_intrinsic, depth_scale=1000.0, T_base_cam=T_base_cam
            )
            # apply imagenet normalization
            colors = (colors - IMG_MEAN) / IMG_STD
            cloud = np.concatenate([points_base, colors], axis = -1)
            clouds.append(cloud)

        # make voxel input
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            # Upd Note. Make coords contiguous.
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype = np.int32)
            # Upd Note. API change.
            input_coords_list.append(coords)
            input_feats_list.append(cloud.astype(np.float32))

        # convert to torch
        ret_dict = {
            'input_coords_list': input_coords_list,
            'input_feats_list': input_feats_list,
        }
        
        if self.with_cloud:  # warning: this may significantly slow down the training process.
            ret_dict["clouds_list"] = clouds

        if self.with_obj_action:
            def _load_obj_pose(frame_id_str):
                pose_path = os.path.join(obj_dir, "{}.txt".format(frame_id_str))
                if not os.path.exists(pose_path):
                    raise FileNotFoundError(f"Object pose file missing: {pose_path}")
                pose_values = np.loadtxt(pose_path).astype(np.float32)
                if pose_values.ndim == 1:
                    if pose_values.size == 16:
                        pose_mat = pose_values.reshape(4, 4)
                    elif pose_values.size == 12:
                        pose_mat = np.vstack([pose_values.reshape(3, 4), np.array([0, 0, 0, 1], dtype=np.float32)])
                    else:
                        raise ValueError(f"Invalid SE3 vector length {pose_values.size} in {pose_path}")
                else:
                    pose_mat = pose_values
                if pose_mat.shape == (3, 4):
                    pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1], dtype=np.float32)])
                if pose_mat.shape != (4, 4):
                    raise ValueError(f"Invalid SE3 matrix shape {pose_mat.shape} in {pose_path}")
                T_base_cam = self._get_cam_to_base(seq_id, frame_id_str)
                pose_base = T_base_cam @ pose_mat
                return xyz_rot_transform(pose_base, from_rep="matrix", to_rep="rotation_6d")

            action_objs = []
            obj_frame_ids = self.obj_frame_ids[index]
            terminal_frame_ids = self.seq_terminal_frame_ids[seq_id]
            termination_flags = []
            for frame_id in obj_frame_ids:
                action_objs.append(_load_obj_pose(frame_id))
                frame_int = int(frame_id)
                termination_flags.append(1.0 if frame_int in terminal_frame_ids else 0.0)
            action_objs = np.stack(action_objs)
            termination_flags = np.array(termination_flags, dtype=np.float32).reshape(-1, 1)
            action_objs = np.concatenate([action_objs, termination_flags], axis=-1)
            actions_obj_normalized = self._normalize_obj(action_objs.copy())
            ret_dict["action_obj"] = torch.from_numpy(action_objs).float()
            ret_dict["action_obj_normalized"] = torch.from_numpy(actions_obj_normalized).float()

            current_frame_id = obs_frame_ids[-1]
            current_obj = _load_obj_pose(current_frame_id)
            current_obj = current_obj.astype(np.float32)
            current_frame_int = int(current_frame_id)
            current_term_flag = np.array(
                [1.0 if current_frame_int in terminal_frame_ids else 0.0],
                dtype=np.float32
            )
            current_obj = np.concatenate([current_obj, current_term_flag], axis=0)
            current_obj_normalized = self._normalize_obj(current_obj[None, :]).squeeze(0)
            ret_dict["current_obj_pose"] = torch.from_numpy(current_obj).float()
            ret_dict["current_obj_pose_normalized"] = torch.from_numpy(current_obj_normalized).float()

        return ret_dict
        

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        coords_batch = ret_dict['input_coords_list']
        feats_batch = ret_dict['input_feats_list']
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def decode_gripper_width(gripper_width):
    # return gripper_width / 1000. * 0.095
    # robotiq-85: 0.0000 - 0.0085
    #                255 -      0
    return (1. - gripper_width / 255.) * 0.085
