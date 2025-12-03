#!/usr/bin/env python3
from __future__ import annotations

"""Apply ball masks onto original images and save visualizations.

This script reads `mask_balls` and the corresponding original images
in a given sequence folder, overlays the three ball masks (id1, id2, id3)
onto the images, and writes the results to a `masked_image/` directory.

Image selection priority (under the given data directory):
1. rgb/{frame_id}.png
2. rgb/{frame_id}.jpg
3. jpg/{frame_id}.png
4. jpg/{frame_id}.jpg
5. depth_vis/{frame_id}.png
6. depth_vis/{frame_id}.jpg
7. depth/{frame_id}.png
8. depth/{frame_id}.jpg

Example:
    python scripts_calib_balls/apply_ball_masks_to_images.py \
        --data-dir data/train/20251125_210453
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _find_base_image(data_dir: Path, frame_id: str) -> Path | None:
    """Find the base image for a given frame."""
    candidates = [
        data_dir / "rgb" / f"{frame_id}.png",
        data_dir / "rgb" / f"{frame_id}.jpg",
        data_dir / "jpg" / f"{frame_id}.png",
        data_dir / "jpg" / f"{frame_id}.jpg",
        data_dir / "depth_vis" / f"{frame_id}.png",
        data_dir / "depth_vis" / f"{frame_id}.jpg",
        data_dir / "depth" / f"{frame_id}.png",
        data_dir / "depth" / f"{frame_id}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_mask(mask_path: Path) -> np.ndarray:
    """Load a single-channel mask and return a boolean array."""
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    mask_img = Image.open(mask_path).convert("L")
    mask = np.array(mask_img, dtype=np.uint8)
    return mask > 0


def apply_masks_for_sequence(data_dir: Path) -> None:
    """Apply all ball masks in `mask_balls/` onto images in this sequence."""
    mask_balls_dir = data_dir / "masks_balls"
    if not mask_balls_dir.exists():
        raise FileNotFoundError(f"mask_balls directory not found: {mask_balls_dir}")

    output_dir = data_dir / "masked_image"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all frame IDs that have any ball mask
    frame_ids: set[str] = set()
    for mask_file in sorted(mask_balls_dir.glob("*.png")):
        stem = mask_file.stem  # e.g. "000280_id3"
        if "_id" not in stem:
            continue
        frame_part = stem.split("_id")[0]
        if frame_part:
            frame_ids.add(frame_part)

    if not frame_ids:
        raise RuntimeError(f"No mask files found in {mask_balls_dir}")

    frame_ids_sorted = sorted(frame_ids)
    print(f"[INFO] Found {len(frame_ids_sorted)} frames with ball masks in {mask_balls_dir}")

    # Define colors for three balls (RGBA)
    ball_colors: dict[int, tuple[int, int, int, int]] = {
        1: (255, 0, 0, 120),    # Red, semi-transparent
        2: (0, 255, 0, 120),    # Green, semi-transparent
        3: (0, 0, 255, 120),    # Blue, semi-transparent
    }

    processed = 0
    skipped = 0

    for frame_id in frame_ids_sorted:
        base_path = _find_base_image(data_dir, frame_id)
        if base_path is None:
            print(f"[WARN] No base image found for frame {frame_id}, skipping.")
            skipped += 1
            continue

        try:
            base_img = Image.open(base_path).convert("RGBA")
        except Exception as exc:
            print(f"[WARN] Failed to load base image {base_path} for frame {frame_id}: {exc}")
            skipped += 1
            continue

        w, h = base_img.size

        # Start from original image and overlay each ball mask in turn
        composite = base_img.copy()

        for ball_id in (1, 2, 3):
            mask_path = mask_balls_dir / f"{frame_id}_id{ball_id}.png"
            if not mask_path.exists():
                # It is allowed that some balls are missing in a frame
                continue

            try:
                mask_bool = _load_mask(mask_path)
            except FileNotFoundError:
                continue
            except Exception as exc:
                print(f"[WARN] Failed to load mask {mask_path}: {exc}")
                continue

            if mask_bool.shape != (h, w):
                print(
                    f"[WARN] Mask size {mask_bool.shape[::-1]} does not match image size {(w, h)} "
                    f"for frame {frame_id}, ball {ball_id}, skipping this mask."
                )
                continue

            color_rgba = ball_colors.get(ball_id, (255, 255, 255, 120))

            # Create a solid color image, then use mask as alpha selection
            overlay = Image.new("RGBA", (w, h), color_rgba)

            # Convert boolean mask to uint8 alpha mask (0 or 255)
            alpha = (mask_bool.astype(np.uint8) * 255)
            alpha_img = Image.fromarray(alpha, mode="L")

            # Paste overlay only where mask is True
            composite.paste(overlay, (0, 0), mask=alpha_img)

        out_path = output_dir / f"{frame_id}.png"
        try:
            composite.save(out_path)
            processed += 1
        except Exception as exc:
            print(f"[WARN] Failed to save masked image for frame {frame_id} to {out_path}: {exc}")
            skipped += 1

    print(f"[INFO] Done. Processed {processed} frames, skipped {skipped} frames.")
    print(f"[INFO] Output written to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply mask_balls onto original images and save to masked_image/."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Sequence directory containing mask_balls/ and images (e.g., rgb/, jpg/, depth_vis/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    apply_masks_for_sequence(data_dir)


if __name__ == "__main__":
    main()

