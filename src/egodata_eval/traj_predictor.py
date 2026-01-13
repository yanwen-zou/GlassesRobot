from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import MinkowskiEngine as ME  # type: ignore

from MBA.policy import RISE  # type: ignore
from MBA.utils.constants import TRANS_MIN, TRANS_MAX, IMG_MEAN, IMG_STD  # type: ignore
from MBA.utils.transformation import rotation_transform, mat_to_xyz_rot  # type: ignore
from egodata_eval.eval_utils import _denormalize_obj_traj, _build_pose_mats, _project_points_with_gradient  # type: ignore


class TrajectoryPredictor:
    def __init__(
        self,
        ckpt_path: Path,
        num_action: int = 20,
        obj_pose_mode: str = "delta",
        voxel_size: float = 0.005,
        enable_headpose_head: bool = False,
        headpose_dim: int = 9,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_action = num_action
        self.obj_pose_mode = obj_pose_mode
        self.voxel_size = voxel_size
        self._cached_points_cam: Optional[np.ndarray] = None
        self.last_traj_denorm: Optional[np.ndarray] = None
        self.last_headpose_pred: Optional[np.ndarray] = None
        self.model = RISE(num_action=num_action,
                          input_dim=6,
                          obs_feature_dim=512,
                          action_dim=10,
                          hidden_dim=512,
                          enable_mba=True,
                          obj_dim=10,
                          obj_pose_mode=obj_pose_mode,
                          enable_headpose_head=enable_headpose_head,
                          headpose_dim=headpose_dim).to(self.device).eval()
        if ckpt_path is None:
            raise ValueError("ckpt_path is required; please pass --ckpt to eval.py")
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Trajectory ckpt not found: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location=self.device)
        self.model.load_state_dict(state, strict=False)

    def _make_sparse_input(self, rgb_bgr: np.ndarray, depth_m: np.ndarray, K: np.ndarray, T_base_cam: Optional[np.ndarray] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Backproject depth to xyz and optionally convert to base (ball) frame."""
        h, w = depth_m.shape
        # print(f"[Traj Predictor INFO] depth_m.shape(h,w):{depth_m.shape}")
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        # Subsample grid for speed
        step = max(1, int(max(h, w) / 480)) # for case that h=376, w=672, step = 1
        # print(f"[INFO] step: {step}")
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        zs = depth_m[ys, xs]
        valid = zs > 1e-6
        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        zs = zs[valid].astype(np.float32)
        xs3 = (xs - cx) * zs / fx
        ys3 = (ys - cy) * zs / fy
        xyz_cam = np.stack([xs3, ys3, zs], axis=-1)
        if T_base_cam is not None:
            R = T_base_cam[:3, :3].astype(np.float32)
            t = T_base_cam[:3, 3].astype(np.float32)
            xyz = (R @ xyz_cam.T).T + t
        else:
            xyz = xyz_cam
        # Colors to [0,1]
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        colors = rgb[ys.astype(int), xs.astype(int)].astype(np.float32) / 255.0
        colors = (colors - IMG_MEAN) / IMG_STD
        cloud = np.concatenate([xyz, colors], axis=-1).astype(np.float32)

        # Remove any rows with non-finite values to avoid NaNs in voxelization
        finite_mask = np.isfinite(cloud).all(axis=1)
        cloud = cloud[finite_mask]

        coords = np.ascontiguousarray((cloud[:, :3] / self.voxel_size).astype(np.int32))
        feats = cloud.astype(np.float32)
        # Collate into ME batched format
        coords_me, feats_me = ME.utils.sparse_collate([coords], [feats])

        # ME may already return torch tensors depending on version; handle both
        if isinstance(feats_me, np.ndarray):
            feats_t = torch.from_numpy(feats_me)
        else:
            feats_t = feats_me
        if isinstance(coords_me, np.ndarray):
            coords_t = torch.from_numpy(coords_me)
        else:
            coords_t = coords_me

        return feats_t.to(self.device), coords_t.to(self.device)

    def _current_obj_vec(self, pose_cam_ob: np.ndarray) -> np.ndarray:
        xyz6d = mat_to_xyz_rot(pose_cam_ob, rotation_rep="rotation_6d").astype(np.float32)
        term = np.array([0.0], dtype=np.float32)
        cur = np.concatenate([xyz6d, term], axis=0)
        # normalize like dataset
        norm = cur.copy()
        norm[:3] = (norm[:3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        return norm

    def _absolute_to_delta_np(self, abs_traj_10: np.ndarray, base_pose_cam_ob: np.ndarray) -> np.ndarray:
        """Convert absolute traj (T,10) [xyz(m), rot6d, grip] to delta wrt base_pose.

        Returns: (T,10) with [dxyz, drot6d, grip]
        """
        if abs_traj_10 is None or abs_traj_10.size == 0:
            return abs_traj_10
        base_xyz6d = mat_to_xyz_rot(base_pose_cam_ob, rotation_rep="rotation_6d").astype(np.float32)
        base_xyz = base_xyz6d[:3]
        base_r6 = base_xyz6d[3:9]
        # Translation delta
        delta_xyz = abs_traj_10[:, :3] - base_xyz[None, :]
        # Rotation delta: R_delta = R_abs @ R_base^T
        R_abs = rotation_transform(abs_traj_10[:, 3:9], "rotation_6d", "matrix")
        R_base = rotation_transform(base_r6[None, :], "rotation_6d", "matrix").squeeze(0)
        R_delta = R_abs @ R_base.T
        delta_r6 = rotation_transform(R_delta, "matrix", "rotation_6d")
        # Gripper passthrough if present
        if abs_traj_10.shape[1] > 9:
            grip = abs_traj_10[:, 9:10]
            delta_full = np.concatenate([delta_xyz, delta_r6, grip], axis=1)
        else:
            delta_full = np.concatenate([delta_xyz, delta_r6], axis=1)
        return delta_full.astype(np.float32)

    def predict_and_overlay(
        self,
        image_bgr: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
        pose_cam_ob: np.ndarray,
        T_base_cam: Optional[np.ndarray] = None,
        headpose_cond: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Convert object pose to base frame if provided

        pose_base_ob = T_base_cam @ pose_cam_ob

        feats, coords = self._make_sparse_input(image_bgr, depth_m, K, T_base_cam=T_base_cam)
        st = ME.SparseTensor(feats, coords)
        cur_obj = self._current_obj_vec(pose_base_ob)
        headpose_tensor = None
        if self.model.enable_headpose_head:
            if headpose_cond is None:
                raise ValueError("headpose_cond is required when enable_headpose_head is True.")
            headpose_tensor = torch.from_numpy(headpose_cond[None, :]).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                st,
                actions_obj=None,
                batch_size=1,
                current_obj=torch.from_numpy(cur_obj[None, :]).to(self.device),
                headpose_cond=headpose_tensor,
            )
        obj_traj_norm = outputs["obj_pred"].squeeze(0).detach().cpu().numpy()
        if self.model.enable_headpose_head:
            headpose_pred_norm = outputs["headpose_pred"].squeeze(0).detach().cpu().numpy()

        obj_traj_ref = _denormalize_obj_traj(obj_traj_norm)
        if self.model.enable_headpose_head:
            headpose_pred = _denormalize_obj_traj(headpose_pred_norm) # abs in base frame
            self.last_headpose_pred = headpose_pred
            #print(f"[INFO] Predicted headpose: {headpose_pred[0]}")
        else:
            self.last_headpose_pred = None

        self.last_traj_denorm = obj_traj_ref
        # Deltas are not used in current execution path; keep only absolute trajectory

        pose_mats_ref = _build_pose_mats(obj_traj_ref[:, :3], obj_traj_ref[:, 3:3+6])
        predicted_points = pose_mats_ref[:, :3, 3]  # (N,3)

        T_cam_base = np.linalg.inv(T_base_cam).astype(np.float32)
        R = T_cam_base[:3, :3].astype(np.float32)
        t = T_cam_base[:3, 3].astype(np.float32)
        points_cam = (R @ predicted_points.T).T + t

        self._cached_points_cam = points_cam.copy()
        overlay = _project_points_with_gradient(image_bgr, K, points_cam,
                                                color_start=(255, 0, 0), color_end=(0, 255, 255), radius=4, thickness=-1)
        return overlay

    def overlay_cached(self, image_bgr: np.ndarray, K: np.ndarray) -> np.ndarray:
        if self._cached_points_cam is None:
            return image_bgr
        return _project_points_with_gradient(
            image_bgr, K, self._cached_points_cam,
            color_start=(255, 0, 0), color_end=(0, 255, 255), radius=4, thickness=-1,
        )
