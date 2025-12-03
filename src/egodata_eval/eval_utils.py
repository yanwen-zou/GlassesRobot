from pathlib import Path
from datetime import datetime
from typing import Optional

import sys
import numpy as np
import cv2


from MBA.utils.transformation import rotation_transform  # type: ignore
from MBA.utils.constants import TRANS_MIN, TRANS_MAX  # type: ignore




def save_mask(mask: np.ndarray, ts,out_dir: Optional[Path] = None, prefix: str = "mask") -> Path:
    """Save a binary mask image to eval_output and return the saved path.

    - mask: 2D or 3D numpy array. If 3D, will squeeze singleton channel.
    - out_dir: target directory. Defaults to '<this_dir>/eval_output'.
    - prefix: filename prefix before timestamp.

    Returns the full Path to the saved PNG.
    """
    
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "eval_output" / ts

    out_dir.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"save_mask expects a 2D mask, got shape {arr.shape}")

    mask_u8 = (arr.astype(np.uint8) > 0) * 255

    
    out_path = out_dir / f"mask.png"

    ok = cv2.imwrite(str(out_path), mask_u8)
    if not ok:
        raise IOError(f"Failed to write mask to {out_path}")

    return out_path


def _find_default_ckpt() -> Path:
    root = Path(__file__).resolve().parents[2]
    ckpt_dir = root / "MBA" / "ckpt_deploy"
    if ckpt_dir.is_dir():
        cands = sorted([p for p in ckpt_dir.iterdir() if p.suffix == ".ckpt"], key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            return cands[0]
    return ckpt_dir / "policy_last.ckpt"



def _denormalize_obj_traj(obj_traj: np.ndarray) -> np.ndarray:
    out = obj_traj.copy()
    out[:, :3] = (out[:, :3] + 1) * 0.5 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    return out


def _build_pose_mats(translation: np.ndarray, rotation_6d: np.ndarray) -> np.ndarray:
    if rotation_transform is None:
        raise RuntimeError("MBA not available: rotation_transform is required.")
    mats = np.repeat(np.eye(4)[None, ...], len(translation), axis=0)
    rot_mats = rotation_transform(rotation_6d, "rotation_6d", "matrix")
    mats[:, :3, :3] = rot_mats
    mats[:, :3, 3] = translation
    return mats


def _project_points_with_gradient(image: np.ndarray,
                                  cam_intr: np.ndarray,
                                  points_cam: np.ndarray,
                                  color_start=(255, 0, 0),
                                  color_end=(0, 255, 255),
                                  radius: int = 6,
                                  thickness: int = -1) -> np.ndarray:
    if points_cam.size == 0:
        return image
    overlay = image.copy()
    num_pts = len(points_cam)
    # print(f"[DEBUG] camera_intrinstic: {cam_intr}")
    for idx, pt in enumerate(points_cam):
        # print(f"[DEBUG] point {idx}: {pt}")
        z = float(pt[2])
        if z <= 1e-6:
            continue
        uvw = cam_intr @ pt
        u = int(round(uvw[0] / z))
        v = int(round(uvw[1] / z))
        if not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]):
            continue
        alpha = idx / max(num_pts - 1, 1)
        color = tuple(int(round(cs * (1 - alpha) + ce * alpha)) for cs, ce in zip(color_start, color_end))
        cv2.circle(overlay, (u, v), radius, color, thickness, lineType=cv2.LINE_AA)
    return overlay

def _import_zed_class():
    # Ensure project root is importable, then import the ZED wrapper class
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    sys.path.insert(0, str(project_root))
    from glasses_hardware.hardware.my_device.zed import ZEDCamera
    return ZEDCamera
