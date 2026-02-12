#!/usr/bin/env python3
"""
Batch-generate hand masks for images under <data-root>/rgb using Grounded SAM 2.

Example:
  python3 scripts_data_processing/grounded_sam_hand_masks.py --data-root data/20251030_113006
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Ensure Grounded-SAM-2 repo is on PYTHONPATH for `sam2` import
_REPO_ROOT = Path(__file__).resolve().parents[1]
_GSAM2_ROOT = _REPO_ROOT / "src" / "FoundationStereo" / "Grounded-SAM-2"
if _GSAM2_ROOT.is_dir():
    sys.path.insert(0, str(_GSAM2_ROOT))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _sorted_images(rgb_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF"}
    images = [p for p in rgb_dir.iterdir() if p.suffix in exts]
    if not images:
        return []
    try:
        images.sort(key=lambda p: int(p.stem))
    except ValueError:
        images.sort(key=lambda p: p.name)
    return images


def parse_args():
    ap = argparse.ArgumentParser(description="Generate hand masks from <data-root>/rgb using Grounded SAM 2")
    ap.add_argument("--data-root", required=True, help="Path to data root containing rgb/")
    ap.add_argument("--text", default="hand and arm.", help="Grounding DINO text prompt (lowercase with trailing dot)")
    ap.add_argument("--output-dir", default="mask_hand", help="Output dir name under data-root")
    ap.add_argument("--box-threshold", type=float, default=0.25, help="Grounding DINO box threshold")
    ap.add_argument("--text-threshold", type=float, default=0.3, help="Grounding DINO text threshold")
    ap.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny", help="Grounding DINO HF model id")
    ap.add_argument(
        "--sam-checkpoint",
        default="src/FoundationStereo/sam2_root/checkpoints/sam2.1_hiera_large.pt",
        help="SAM2 checkpoint path (relative to repo root)",
    )
    ap.add_argument(
        "--sam-config",
        default="src/FoundationStereo/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config path (relative to repo root)",
    )
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for GroundingDINO forward pass")
    return ap.parse_args()


def _discover_episode_dirs(data_root: Path) -> list[Path]:
    if (data_root / "rgb").is_dir():
        return [data_root]
    episodes = []
    for child in sorted(data_root.iterdir()):
        if child.is_dir() and (child / "rgb").is_dir():
            episodes.append(child)
    return episodes


def _iter_batches(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")

    data_root = Path(args.data_root).expanduser().resolve()
    episode_dirs = _discover_episode_dirs(data_root)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode with rgb/ found under: {data_root}")

    # Enable bfloat16 + TF32 on Ampere+ for speed
    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    repo_root = Path(__file__).resolve().parents[1]
    sam_checkpoint = (repo_root / args.sam_checkpoint).resolve()

    # Hydra expects config_name relative to the sam2 package (e.g. "configs/sam2.1/...")
    sam_config_arg = Path(args.sam_config)
    sam_config_name = args.sam_config
    sam_config_path = None

    if sam_config_arg.is_absolute():
        sam_config_path = sam_config_arg
    else:
        sam_config_path = (repo_root / sam_config_arg).resolve()

    if sam_config_path.is_file():
        # Convert absolute path to sam2 package-relative config name if possible
        parts = sam_config_path.parts
        if "sam2" in parts:
            idx = parts.index("sam2")
            rel_parts = parts[idx + 1 :]
            if rel_parts:
                sam_config_name = str(Path(*rel_parts))
        else:
            # Fallback to path relative to repo root
            try:
                sam_config_name = str(sam_config_path.relative_to(repo_root))
            except ValueError:
                sam_config_name = str(sam_config_path)
    else:
        raise FileNotFoundError(f"SAM2 config not found: {sam_config_path}")

    if not sam_checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam_checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam2_image_model = build_sam2(sam_config_name, str(sam_checkpoint), device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    processor = AutoProcessor.from_pretrained(args.model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)
    for episode_dir in episode_dirs:
        rgb_dir = episode_dir / "rgb"
        out_dir = episode_dir / args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        images = _sorted_images(rgb_dir)
        if not images:
            print(f"⚠️ No images found under: {rgb_dir}, skip.")
            continue

        print(f"[INFO] Grounded-SAM hand mask for {episode_dir.name}: {len(images)} frames, batch={args.batch_size}")
        processed = 0
        for batch_paths in _iter_batches(images, args.batch_size):
            pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
            np_images = [np.array(img) for img in pil_images]
            target_sizes = [img.size[::-1] for img in pil_images]

            inputs = processor(
                images=pil_images,
                text=[args.text] * len(pil_images),
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                outputs = grounding_model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=target_sizes,
            )

            for i, img_path in enumerate(batch_paths):
                np_image = np_images[i]
                result = results[i]
                if result["boxes"].shape[0] == 0:
                    mask = np.zeros((np_image.shape[0], np_image.shape[1]), dtype=np.uint8)
                else:
                    image_predictor.set_image(np_image)
                    input_boxes = result["boxes"].cpu().numpy()
                    masks, _, _ = image_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )

                    if masks.ndim == 3:
                        masks = masks[None]
                    elif masks.ndim == 4:
                        masks = masks.squeeze(1)

                    mask = (np.any(masks, axis=0).astype(np.uint8)) * 255
                    mask = np.squeeze(mask)

                out_path = out_dir / f"{img_path.stem}.png"
                Image.fromarray(mask).save(out_path)
                processed += 1
                print(f"[{processed}/{len(images)}] {img_path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
