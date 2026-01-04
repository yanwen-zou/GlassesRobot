import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import trimesh

# Make FoundationPose importable
_HERE = Path(__file__).resolve()
_FP_ROOT = _HERE.parents[2] / "foundationpose" / "FoundationPose"
import sys
if str(_FP_ROOT) not in sys.path:
    sys.path.append(str(_FP_ROOT))

from estimater import FoundationPose  # type: ignore


def load_intrinsics(default_path: Optional[Path] = None) -> Tuple[np.ndarray, float]:
    # Reuse FoundationStereo intrinsics file by default
    fs_root = _HERE.parents[1] / "FoundationStereo"
    if default_path is None:
        default_path = fs_root / "assets" / "K_ZED.txt"
    with open(default_path, "r") as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split())), dtype=np.float32).reshape(3, 3)
        baseline = float(lines[1])
    return K, baseline


def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[-1] == 3:
        return image
    raise ValueError("Expected HxWx3 image")


class PoseEstimatorFP:
    """Thin wrapper around FoundationPose for real-time usage."""

    def __init__(self, mesh_path: Path, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load mesh
        mesh = trimesh.load_mesh(str(mesh_path), force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            # If scene, merge to single mesh
            mesh = mesh.dump().sum()
        self.mesh = mesh
        # Prepare model points and normals
        model_pts = np.asarray(mesh.vertices, dtype=np.float32)
        model_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        # Initialize estimator
        # Use a writable debug directory to avoid permission errors
        debug_dir = str((_HERE.parents[2] / "src" / "egodata_eval" / "eval_output" / "fp_debug").resolve())
        os.makedirs(debug_dir, exist_ok=True)
        self.est = FoundationPose(
            model_pts=model_pts,
            model_normals=model_normals,
            mesh=mesh,
            scorer=None,
            refiner=None,
            glctx=None,
            debug=0,
            debug_dir=debug_dir,
        )
        self.est.to_device(self.device)
        self.pose_cam_ob = None  # 4x4

    def initialize(self, rgb_bgr: np.ndarray, depth_m: np.ndarray, mask: np.ndarray, K: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(_ensure_rgb_uint8(rgb_bgr), cv2.COLOR_BGR2RGB)
        mask_bin = (mask > 0).astype(np.uint8)
        
        pose = self.est.register(K=K.astype(np.float32), rgb=rgb, depth=depth_m.astype(np.float32), ob_mask=mask_bin, iteration=5)
        self.pose_cam_ob = pose
        valid = (mask_bin > 0) & np.isfinite(depth_m) & (depth_m > 0)
        if np.any(valid):
            mean_depth = float(depth_m[valid].mean())
            print(f"[INFO] Mask mean depth for book: {mean_depth:.4f}m")
        if pose is not None and np.shape(pose) == (4, 4):
            xyz = pose[:3, 3].astype(np.float32)
            dist = float(np.linalg.norm(xyz))
            print(f"[INFO] Init pose xyz: {xyz}, dist={dist:.4f}m")
        
        return pose


    def track(self, rgb_bgr: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> Optional[np.ndarray]:
        if self.pose_cam_ob is None:
            return None
        rgb = cv2.cvtColor(_ensure_rgb_uint8(rgb_bgr), cv2.COLOR_BGR2RGB)
        try:
            pose = self.est.track_one(rgb=rgb, depth=depth_m.astype(np.float32), K=K.astype(np.float32), iteration=2)
            self.pose_cam_ob = pose
            return pose
        except Exception:
            return None

    @staticmethod
    def _project_points(K: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
        zs = pts_cam[:, 2:3]
        valid = zs[:, 0] > 1e-6
        pts = (K @ pts_cam.T).T
        uv = np.zeros((pts.shape[0], 2), dtype=np.float32)
        uv[valid] = (pts[valid, :2] / zs[valid])
        return uv, valid

    def draw_overlay(self, image_bgr: np.ndarray, K: np.ndarray, pose_cam_ob: Optional[np.ndarray] = None, color=(0, 255, 0)) -> np.ndarray:
        if pose_cam_ob is None:
            pose_cam_ob = self.pose_cam_ob
        if pose_cam_ob is None:
            return image_bgr
        img = image_bgr.copy()
        K = K.astype(np.float32)
        # Draw axes
        axis_len = float(max(self.mesh.extents)) * 0.25
        origin = np.array([[0, 0, 0, 1]], dtype=np.float32)
        axes = np.array([
            [axis_len, 0, 0, 1],
            [0, axis_len, 0, 1],
            [0, 0, axis_len, 1],
        ], dtype=np.float32)
        pts = np.concatenate([origin, axes], axis=0)
        pts_cam = (pose_cam_ob @ pts.T).T[:, :3]
        uv, valid = self._project_points(K, pts_cam)
        o = tuple(np.round(uv[0]).astype(int))
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in range(3):
            if valid[0] and valid[1 + i]:
                p = tuple(np.round(uv[1 + i]).astype(int))
                cv2.arrowedLine(img, o, p, colors[i], 2, tipLength=0.15)

        # Draw wireframe bbox of the mesh
        mn, mx = self.mesh.bounds
        corners = np.array([
            [mn[0], mn[1], mn[2], 1], [mx[0], mn[1], mn[2], 1], [mx[0], mx[1], mn[2], 1], [mn[0], mx[1], mn[2], 1],
            [mn[0], mn[1], mx[2], 1], [mx[0], mn[1], mx[2], 1], [mx[0], mx[1], mx[2], 1], [mn[0], mx[1], mx[2], 1],
        ], dtype=np.float32)
        corners_cam = (pose_cam_ob @ corners.T).T[:, :3]
        uv8, valid8 = self._project_points(K, corners_cam)
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for a,b in edges:
            if valid8[a] and valid8[b]:
                pa = tuple(np.round(uv8[a]).astype(int))
                pb = tuple(np.round(uv8[b]).astype(int))
                cv2.line(img, pa, pb, color, 2)
        return img
