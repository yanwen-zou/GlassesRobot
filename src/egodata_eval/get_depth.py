import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

# Make FoundationStereo importable
_HERE = Path(__file__).resolve()
_FS_ROOT = _HERE.parents[1] / "FoundationStereo"
import sys
if str(_FS_ROOT) not in sys.path:
    sys.path.append(str(_FS_ROOT))

from core.foundation_stereo import FoundationStereo  # type: ignore
from core.utils.utils import InputPadder  # type: ignore


def _load_intrinsics(default_path: Optional[Path] = None) -> Tuple[np.ndarray, float]:
    if default_path is None:
        default_path = _FS_ROOT / "assets" / "K_ZED.txt"
    with open(default_path, "r") as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split())), dtype=np.float32).reshape(3, 3)
        baseline = float(lines[1])
    return K, baseline


class DepthEstimator:
    """Encapsulates FoundationStereo model and provides disparity/depth inference."""

    def __init__(self, ckpt_dir: Optional[Path] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if ckpt_dir is None:
            ckpt_dir = _FS_ROOT / "pretrained_models" / "11-33-40" / "model_best_bp2.pth"
        cfg_path = ckpt_dir.parent / "cfg.yaml"

        cfg = OmegaConf.load(str(cfg_path))
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        # Real-time leaning defaults
        cfg["valid_iters"] = int(cfg.get("valid_iters", 24))
        cfg["hiera"] = int(cfg.get("hiera", 0))
        cfg["scale"] = float(cfg.get("scale", 1))
        print(f"[INFO] FoundationStereo config: scale={cfg['scale']}")
        cfg["mixed_precision"] = True

        self.args = OmegaConf.create(cfg)
        self.model = FoundationStereo(self.args)
        ckpt = torch.load(str(ckpt_dir), weights_only=False, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])  # type: ignore
        self.model.to(self.device)
        self.model.eval()

        # Camera intrinsics for depth conversion
        self.K, self.baseline = _load_intrinsics()

    def disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        assert left_bgr is not None and right_bgr is not None
        assert left_bgr.shape[:2] == right_bgr.shape[:2], "Stereo frames must match size"

        # FoundationStereo expects RGB in [0,255], shape BxCxHxW
        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

        H, W = left_rgb.shape[:2]
        img0 = torch.as_tensor(left_rgb, device=self.device).float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb, device=self.device).float()[None].permute(0, 3, 1, 2)

        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)


        with torch.cuda.amp.autocast(self.device == "cuda"):
            if not getattr(self.args, "hiera", 0):
                disp_up = self.model.forward(img0, img1, iters=int(getattr(self.args, "valid_iters", 24)), test_mode=True)
            else:
                disp_up = self.model.run_hierachical(img0, img1, iters=int(getattr(self.args, "valid_iters", 24)), test_mode=True, small_ratio=0.5)

        disp = padder.unpad(disp_up.float())
        disp = disp.data.detach().cpu().numpy().reshape(H, W).astype(np.float32)
        return disp

    def depth(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        disp = self.disparity(left_bgr, right_bgr)
        fx = float(self.K[0, 0])
        depth = np.full_like(disp, np.inf, dtype=np.float32)
        valid = disp > 0
        depth[valid] = fx * float(self.baseline) / disp[valid]
        return depth


def colorize_depth(depth_m: np.ndarray, max_depth: float = 5.0) -> np.ndarray:
    """Colorize depth (m) to BGR. Uses finite-value percentiles for dynamic range.

    - If too few valid depths, falls back to [0, max_depth] scaling.
    - Invalid/infinite depths are mapped to black.
    """
    d = depth_m.astype(np.float32)
    finite = np.isfinite(d) & (d > 0)
    vis = np.zeros((*d.shape, 3), dtype=np.uint8)
    if finite.sum() < 50:
        # fallback static scaling
        d_clip = np.clip(d, 0, max_depth)
        scale_min, scale_max = 0.0, max_depth
    else:
        vals = d[finite]
        p1, p95 = np.percentile(vals, [1, 95]).astype(np.float32)
        if p95 <= p1:
            p1, p95 = 0.0, max_depth
        d_clip = np.clip(d, p1, p95)
        scale_min, scale_max = float(p1), float(p95)
    # Normalize to 0-255
    denom = max(1e-6, (scale_max - scale_min))
    norm = ((d_clip - scale_min) / denom * 255.0).astype(np.uint8)
    cm = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    # Set invalid pixels to black
    cm[~finite] = (0, 0, 0)
    return cm
