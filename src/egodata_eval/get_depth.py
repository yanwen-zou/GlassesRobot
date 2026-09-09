from pathlib import Path
from typing import Optional, Tuple
import sys

import cv2
import numpy as np
import torch


_HERE = Path(__file__).resolve()
_FFS_ROOT = _HERE.parents[1] / "FastFoundationStereo"
_CALIBRATION_ROOT = _HERE.parents[1] / "FoundationStereo" / "assets"

# Fast-FoundationStereo uses top-level imports such as ``core`` and ``Utils``.
if str(_FFS_ROOT) not in sys.path:
    sys.path.insert(0, str(_FFS_ROOT))

from core.utils.utils import InputPadder  # type: ignore  # noqa: E402
from Utils import AMP_DTYPE  # type: ignore  # noqa: E402


def _load_intrinsics(default_path: Optional[Path] = None) -> Tuple[np.ndarray, float]:
    """Load the existing ZED calibration shared by the stereo backends."""
    if default_path is None:
        default_path = _CALIBRATION_ROOT / "K_ZED.txt"
    with default_path.open("r") as f:
        lines = f.readlines()
    K = np.array(list(map(float, lines[0].split())), dtype=np.float32).reshape(3, 3)
    baseline = float(lines[1])
    return K, baseline


class DepthEstimator:
    """Real-time disparity/depth inference using Fast-FoundationStereo.

    The default is NVIDIA's official C-Fast checkpoint with four refinement
    iterations. Override ``ckpt_path``, ``valid_iters`` or ``max_disp`` when a
    different speed/accuracy/near-range trade-off is required.
    """

    def __init__(
        self,
        ckpt_path: Optional[Path] = None,
        device: Optional[str] = None,
        valid_iters: int = 4,
        max_disp: int = 192,
        intrinsics_path: Optional[Path] = None,
        volume_backend: str = "triton",
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if ckpt_path is None:
            ckpt_path = (
                _FFS_ROOT
                / "weights"
                / "c-fast-foundationstereo"
                / "model_best_bp2_serialize.pth"
            )
        self.ckpt_path = Path(ckpt_path)
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(
                "Fast-FoundationStereo checkpoint not found at "
                f"{self.ckpt_path}. Download an official checkpoint into "
                "src/FastFoundationStereo/weights/ (see "
                "src/FastFoundationStereo/readme.md), or pass ckpt_path."
            )

        # Official checkpoints serialize the complete model rather than a
        # state_dict. Importing the vendored package above registers the class
        # names needed by torch.load.
        self.model = torch.load(
            str(self.ckpt_path), map_location="cpu", weights_only=False
        )
        self.model.args.valid_iters = int(valid_iters)
        self.model.args.max_disp = int(max_disp)
        self.model.args.mixed_precision = self.device.type == "cuda"
        self.model.to(self.device).eval()
        self.valid_iters = int(valid_iters)
        if volume_backend not in ("triton", "pytorch1"):
            raise ValueError("volume_backend must be 'triton' or 'pytorch1'")
        self.volume_backend = volume_backend

        if self.device.type == "cuda":
            # The input size is stable in the live pipeline, so autotuning pays
            # off after the model's first-run compilation/warm-up.
            torch.backends.cudnn.benchmark = True

        self.K, self.baseline = _load_intrinsics(intrinsics_path)
        print(
            "[INFO] Fast-FoundationStereo: "
            f"checkpoint={self.ckpt_path.parent.name}, "
            f"valid_iters={self.valid_iters}, max_disp={self.model.args.max_disp}, "
            f"volume_backend={self.volume_backend}"
        )

    def disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        if left_bgr is None or right_bgr is None:
            raise ValueError("Stereo frames must not be None")
        if left_bgr.shape[:2] != right_bgr.shape[:2]:
            raise ValueError("Stereo frames must have matching sizes")

        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        height, width = left_rgb.shape[:2]
        img0 = (
            torch.from_numpy(left_rgb)
            .to(device=self.device, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        img1 = (
            torch.from_numpy(right_rgb)
            .to(device=self.device, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)
        with torch.inference_mode(), torch.amp.autocast(
            self.device.type,
            enabled=self.device.type == "cuda",
            dtype=AMP_DTYPE if self.device.type == "cuda" else torch.bfloat16,
        ):
            disp_up = self.model.forward(
                img0,
                img1,
                iters=self.valid_iters,
                test_mode=True,
                optimize_build_volume=self.volume_backend,
            )

        disp = padder.unpad(disp_up.float())
        return (
            disp.detach()
            .cpu()
            .numpy()
            .reshape(height, width)
            .clip(0, None)
            .astype(np.float32, copy=False)
        )

    def depth(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        disp = self.disparity(left_bgr, right_bgr)
        depth = np.full_like(disp, np.inf, dtype=np.float32)
        valid = disp > 0
        depth[valid] = float(self.K[0, 0]) * self.baseline / disp[valid]
        return depth


def colorize_depth(depth_m: np.ndarray, max_depth: float = 5.0) -> np.ndarray:
    """Colorize metric depth as BGR, mapping invalid pixels to black."""
    d = depth_m.astype(np.float32)
    finite = np.isfinite(d) & (d > 0)
    if finite.sum() < 50:
        d_clip = np.clip(d, 0, max_depth)
        scale_min, scale_max = 0.0, max_depth
    else:
        p1, p95 = np.percentile(d[finite], [1, 95]).astype(np.float32)
        if p95 <= p1:
            p1, p95 = 0.0, max_depth
        d_clip = np.clip(d, p1, p95)
        scale_min, scale_max = float(p1), float(p95)
    denom = max(1e-6, scale_max - scale_min)
    norm = ((d_clip - scale_min) / denom * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    vis[~finite] = (0, 0, 0)
    return vis
