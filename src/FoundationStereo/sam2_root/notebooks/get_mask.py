import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


_PREDICTOR = None
_DEVICE = None


def _init_predictor():
    global _PREDICTOR, _DEVICE
    if _PREDICTOR is not None:
        return _PREDICTOR

    # Notebook-style setup: resolve paths relative to this notebook folder
    # sam_root = .../sam2_root
    sam_root = Path(__file__).resolve().parents[1]

    # Follow notebooks convention: config inside sam_root/sam2/configs, ckpt in sam_root/checkpoints
    model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"

    # Basic sanity checks for clearer errors
    if not (sam_root / model_cfg).exists():
        raise FileNotFoundError(f"Missing SAM2 config: {(sam_root / model_cfg)}")
    if not (sam_root / sam2_checkpoint).exists():
        raise FileNotFoundError(f"Missing SAM2 checkpoint: {(sam_root / sam2_checkpoint)}")
    import sys
    sys.path.append(str(sam_root))

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
    else:
        _DEVICE = torch.device("cpu")

    model_cfg_abs = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_checkpoint_abs = (sam_root / sam2_checkpoint)
    # Ensure Hydra resolves config path relative to the SAM root
    class _WD:
        def __init__(self, path: Path):
            self.path = path
            self.prev = None
        def __enter__(self):
            from os import getcwd, chdir
            self.prev = Path(getcwd())
            chdir(self.path)
        def __exit__(self, exc_type, exc, tb):
            if self.prev is not None:
                from os import chdir
                chdir(self.prev)

    # Match notebook behavior: chdir into sam_root so Hydra finds configs under ./configs
    with _WD(sam_root):
        model = build_sam2(
            config_file=model_cfg_abs,
            ckpt_path=sam2_checkpoint_abs,
            device=_DEVICE,
        )
    _PREDICTOR = SAM2ImagePredictor(model)
    return _PREDICTOR


def click_mask(
    image_rgb: np.ndarray,
    points_xy: List[Tuple[float, float]],
    labels: Optional[List[int]] = None,
    multimask: bool = True,
) -> np.ndarray:
    """Run SAM2 image predictor on a single RGB frame with point prompts.

    - image_bgr: HxWx3 uint8 BGR image (as from OpenCV/ZED)
    - points_xy: list of (x, y) pixel coords wrt the original image size
    - labels: list of 1 (fg) or 0 (bg); defaults to all-ones
    - returns a binary mask HxW (uint8 values 0 or 255)
    """
    assert isinstance(image_rgb, np.ndarray) and image_rgb.ndim == 3, "image must be HxWx3"
    h, w, _ = image_rgb.shape
    predictor = _init_predictor()
    # Convert BGR to RGB
    predictor.set_image(image_rgb)


    if labels is None:
        labels = [1] * len(points_xy)

    pts = np.array(points_xy, dtype=np.float32)
    lbs = np.array(labels, dtype=np.int32)

    masks, ious, _ = predictor.predict(
        point_coords=pts,
        point_labels=lbs,
        multimask_output=multimask,
        normalize_coords=True,
    )

    # Choose the best mask: prefer highest IoU; fallback to largest area
    idx = int(np.argmax(ious)) if ious.size > 0 else 0
    mask_bool = masks[idx].astype(bool)
    if ious.size == 0 and masks.shape[0] > 1:
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        idx = int(np.argmax(areas))
        mask_bool = masks[idx].astype(bool)

    mask_uint8 = (mask_bool.astype(np.uint8)) * 255
    return mask_uint8


def box_mask(
    image_rgb: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    multimask: bool = True,
) -> np.ndarray:
    """Run SAM2 image predictor on a single RGB frame with a box prompt.

    - image_rgb: HxWx3 uint8 RGB image
    - box_xyxy: (x0, y0, x1, y1) in pixel coords wrt original image size
    - returns a binary mask HxW (uint8 values 0 or 255)
    """
    assert isinstance(image_rgb, np.ndarray) and image_rgb.ndim == 3, "image must be HxWx3"
    predictor = _init_predictor()
    predictor.set_image(image_rgb)

    box = np.array(box_xyxy, dtype=np.float32)

    masks, ious, _ = predictor.predict(
        box=box,
        multimask_output=multimask,
        normalize_coords=True,
    )

    # Choose best mask by IoU; fallback to largest area
    idx = int(np.argmax(ious)) if ious.size > 0 else 0
    mask_bool = masks[idx].astype(bool)
    if ious.size == 0 and masks.shape[0] > 1:
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        idx = int(np.argmax(areas))
        mask_bool = masks[idx].astype(bool)

    return (mask_bool.astype(np.uint8)) * 255
