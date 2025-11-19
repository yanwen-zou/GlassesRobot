from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode


def _load_head_pose(pose_path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(pose_path, dtype=np.float32).reshape(-1)
    if values.size < 7:
        raise ValueError(f"Pose file {pose_path} must contain tx ty tz qx qy qz qw.")
    translation = values[:3]
    quat = values[3:7]
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError(f"Quaternion in {pose_path} has zero norm.")
    quat = quat / norm
    return translation, quat


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = float(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))
    angle_rad = 2.0 * np.arccos(dot)
    return np.degrees(angle_rad)


def gather_frame_paths(
    rgb_dir: Path,
    mask_dir: Path,
    hand_mask_dir: Path,
    head_dir: Path,
    trans_thresh: float,
    rot_thresh_deg: float,
) -> tuple[list[str], list[Path], list[Path]]:
    """Collect frames with masks and head poses, selecting keyframes by pose deltas."""
    rgb_paths = sorted(rgb_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")

    candidates: list[tuple[str, Path, Path, np.ndarray, np.ndarray]] = []
    for rgb_path in rgb_paths:
        mask_path = mask_dir / rgb_path.name
        hand_mask_path = hand_mask_dir / rgb_path.name
        pose_path = head_dir / f"{rgb_path.stem}.txt"
        if not mask_path.exists() or not hand_mask_path.exists() or not pose_path.exists():
            continue
        translation, quat = _load_head_pose(pose_path)
        candidates.append((str(rgb_path), mask_path, hand_mask_path, translation, quat))

    if not candidates:
        raise RuntimeError("No frames have masks and head pose data.")

    image_names: list[str] = []
    mask_paths: list[Path] = []
    hand_mask_paths: list[Path] = []
    last_translation = None
    last_quat = None

    for rgb_path, mask_path, hand_mask_path, translation, quat in candidates:
        if last_translation is None:
            select_frame = True
        else:
            trans_delta = np.linalg.norm(translation - last_translation)
            rot_delta = _quat_angle_deg(last_quat, quat)
            select_frame = (trans_delta >= trans_thresh) or (rot_delta >= rot_thresh_deg)

        if select_frame:
            image_names.append(rgb_path)
            mask_paths.append(mask_path)
            hand_mask_paths.append(hand_mask_path)
            last_translation = translation
            last_quat = quat

    if not image_names:
        # Ensure at least the first frame is kept if thresholds are too high.
        first = candidates[0]
        image_names.append(first[0])
        mask_paths.append(first[1])
        hand_mask_paths.append(first[2])
    print(f'selected {len(image_names)} frames from {len(candidates)} candidates based on pose thresholds.')
    return image_names, mask_paths, hand_mask_paths


def build_mask_tensor(mask_paths: list[Path], hand_paths: list[Path], target_hw: tuple[int, int]) -> torch.Tensor:
    """Load mask images, resize them to the network resolution, and combine them."""
    h, w = target_hw
    mask_tensors: list[torch.Tensor] = []
    for mask_path, hand_path in zip(mask_paths, hand_paths):
        mask = TF.to_tensor(Image.open(mask_path).convert("L"))
        hand = TF.to_tensor(Image.open(hand_path).convert("L"))
        combined = torch.clamp(mask + hand, 0.0, 1.0)
        combined = TF.resize(combined, size=[h, w], interpolation=InterpolationMode.NEAREST)
        mask_tensors.append(combined)
    return torch.stack(mask_tensors, dim=0)
