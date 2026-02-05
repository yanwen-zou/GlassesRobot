#!/usr/bin/env python3
from __future__ import annotations

"""Calculate 3D centroids of balls from masks and depth information.

This script processes mask_balls data to extract 3 separate masks (id1, id2, id3) for each frame,
calculates 3D centroids using depth information in the current frame camera coordinate system.

Example:
    python scripts_calib_balls/calculate_ball_centers.py \
        --data-dir data/20251130_150031 \
        --output data/20251130_150031/ball_centers.txt
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


DEPTH_SCALE_DEFAULT = 1000.0  # RealSense depth units -> meters
DEFAULT_MAX_RADIUS_STD_RATIO = 0.3  # max std(r) / mean(r) for boundary radii


def load_intrinsics(path: Path) -> np.ndarray:
    """Load camera intrinsics from a text file (3x3 matrix)."""
    if not path.exists():
        raise FileNotFoundError(f"Intrinsic matrix not found: {path}")
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    mat = np.array(rows, dtype=np.float32)
    if mat.shape != (3, 3):
        raise ValueError(f"Intrinsic matrix must be 3x3, got {mat.shape}")
    return mat


def load_mask(mask_path: Path) -> np.ndarray:
    """Load mask image and return boolean array."""
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    mask = np.array(Image.open(mask_path).convert("L"))
    return mask > 0


def load_depth(depth_path: Path, depth_scale: float) -> np.ndarray:
    """Load depth image and convert to meters."""
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
    depth_m = depth_raw / depth_scale
    return depth_m


def save_invalid_mask_debug(
    mask: np.ndarray,
    boundary: np.ndarray,
    out_dir: Path,
    frame_id: int | None,
    ball_id: int | None,
    reason: str,
) -> Path:
    """Save an RGB visualization of an invalid mask for manual inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask] = (0, 255, 0)  # green mask interior
    rgb[boundary] = (255, 0, 0)  # red boundary pixels
    frame_str = f"{frame_id:06d}" if frame_id is not None else "frame"
    ball_str = f"id{ball_id}" if ball_id is not None else "ball"
    filename = f"{frame_str}_{ball_str}_invalid.png"
    out_path = out_dir / filename
    Image.fromarray(rgb).save(out_path)
    print(f"[WARN] Saved invalid mask debug image ({reason}) to {out_path}")
    return out_path


def calculate_ball_centroid(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    max_radius_std_ratio: float,
    invalid_mask_output_dir: Path | None = None,
    frame_id: int | None = None,
    ball_id: int | None = None,
) -> np.ndarray | None:
    """Calculate 3D centroid of a ball from mask and depth in current frame camera coordinate system.
    
    Args:
        depth_m: Depth image in meters (H, W)
        mask: Boolean mask (H, W)
        intrinsic: Camera intrinsic matrix (3x3)
        max_radius_std_ratio: Maximum allowed std(radius) / mean(radius) for boundary points
        invalid_mask_output_dir: If provided, save debug PNGs for masks failing the shape check
        frame_id: Frame identifier used in debug filenames
        ball_id: Ball identifier used in debug filenames
        
    Returns:
        3D centroid in current frame camera coordinate system, or None if no valid points
    """
    if depth_m.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Depth shape {depth_m.shape} and mask shape {mask.shape} must match.")

    ys_all, xs_all = np.nonzero(mask)
    if ys_all.size == 0 or xs_all.size == 0:
        return None

    # Boundary-based circularity check: std(radius)/mean(radius) should be small
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    ksum = (
        padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
        padded[1:-1, 0:-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:, 0:-2] + padded[2:, 1:-1] + padded[2:, 2:]
    )
    boundary = mask & (ksum < 9)
    bys, bxs = np.nonzero(boundary)
    if bys.size == 0 or bxs.size == 0:
        return None
    cx, cy = bxs.mean(), bys.mean()
    radii = np.sqrt((bxs - cx) ** 2 + (bys - cy) ** 2)
    mean_r = float(radii.mean())
    std_r = float(radii.std())
    if mean_r <= 1e-6:
        return None
    radius_ratio = std_r / mean_r
    if radius_ratio > max_radius_std_ratio:
        reason = (
            f"shape_check_failed ratio={radius_ratio:.3f} "
            f"mean_r={mean_r:.2f} std_r={std_r:.2f} boundary_pixels={bxs.size}"
        )
        if invalid_mask_output_dir is not None:
            save_invalid_mask_debug(
                mask=mask,
                boundary=boundary,
                out_dir=invalid_mask_output_dir,
                frame_id=frame_id,
                ball_id=ball_id,
                reason=reason,
            )
        return None

    valid_mask = mask & np.isfinite(depth_m) & (depth_m > 0)
    ys, xs = np.nonzero(valid_mask)
    if ys.size == 0:
        return None

    z = depth_m[ys, xs].astype(np.float32) / 1# DEBUG: Hardcode scaling for depth unmatching
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    cam_points = np.stack([x, y, z], axis=1)
    
    # Calculate centroid in current frame camera coordinate system
    centroid = np.mean(cam_points, axis=0)
    return centroid.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate 3D centroids of balls from masks and depth information."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Data directory containing mask_balls/, depth/, and cam_K.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (defaults to <data-dir>/ball_centers.txt)",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=DEPTH_SCALE_DEFAULT,
        help="Meters per depth unit (default 1000.0 for RealSense uint16).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip frames if mask/depth is missing instead of failing.",
    )
    parser.add_argument(
        "--max-radius-std-ratio",
        type=float,
        default=DEFAULT_MAX_RADIUS_STD_RATIO,
        help="Maximum std(radius)/mean(radius) for boundary points; higher values are treated as broken masks (default: 0.25).",
    )
    parser.add_argument(
        "--invalid-mask-output-dir",
        type=Path,
        default=None,
        help="If set, save PNG debug images for masks invalidated by the shape check to this directory.",
    )
    parser.add_argument(
        "--first-frame-only",
        action="store_true",
        help="Only process the first depth frame (useful when tracking later frames via head pose).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir: Path = args.data_dir
    
    # Setup paths
    mask_balls_dir = data_dir / "masks_balls"
    depth_dir = data_dir / "depth"
    intrinsic_path = data_dir / "cam_K.txt"
    if not intrinsic_path.exists():
        intrinsic_path = data_dir / "camera_intrinsics.txt"
    output_path = args.output or (data_dir / "ball_centers.txt")

    # Load intrinsics
    print(f"Loading camera intrinsics from {intrinsic_path}")
    intrinsic = load_intrinsics(intrinsic_path)

    # Get all frame IDs from depth directory
    depth_files = sorted(depth_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not depth_files:
        raise FileNotFoundError(f"No depth files found in {depth_dir}")
    
    frame_ids = [int(f.stem) for f in depth_files]
    if args.first_frame_only and frame_ids:
        frame_ids = frame_ids[:1]
    print(f"Found {len(frame_ids)} frames")

    # Process each frame
    results = []
    processed = 0
    skipped = 0
    first_image_loaded = False
    
    for frame_id in frame_ids:
        # Load depth
        depth_path = depth_dir / f"{frame_id:06d}.png"
        try:
            depth_m = load_depth(depth_path, args.depth_scale)
            # Output image dimensions on first load
            if not first_image_loaded:
                h, w = depth_m.shape[:2]
                print(f"[INFO] Depth image dimensions: {w} x {h} (width x height)")
                first_image_loaded = True
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"[WARN] {exc}; skipping frame {frame_id:06d}")
                skipped += 1
                continue
            raise

        # Process each ball (id1, id2, id3)
        for ball_id in [1, 2, 3]:
            mask_path = mask_balls_dir / f"{frame_id:06d}_id{ball_id}.png"
            try:
                mask = load_mask(mask_path)
                # Output mask dimensions on first load
                if not first_image_loaded:
                    h, w = mask.shape[:2]
                    print(f"[INFO] Mask image dimensions: {w} x {h} (width x height)")
                    first_image_loaded = True
            except FileNotFoundError as exc:
                if args.skip_missing:
                    print(f"[WARN] {exc}; skipping ball {ball_id} in frame {frame_id:06d}")
                    continue
                raise

            mask_pixels = int(mask.sum())
            ys_all, xs_all = np.nonzero(mask)
            if ys_all.size and xs_all.size:
                y_min, y_max = ys_all.min(), ys_all.max()
                x_min, x_max = xs_all.min(), xs_all.max()
                bbox_h = y_max - y_min + 1
                bbox_w = x_max - x_min + 1
                bbox_area = bbox_h * bbox_w
            else:
                bbox_h = bbox_w = bbox_area = 0
            print(
                f"[INFO] Frame {frame_id:06d} ball {ball_id} mask pixels: {mask_pixels}, "
                f"bbox: {bbox_w}x{bbox_h} area={bbox_area}"
            )

            centroid = calculate_ball_centroid(
                depth_m,
                mask,
                intrinsic,
                max_radius_std_ratio=args.max_radius_std_ratio,
                invalid_mask_output_dir=args.invalid_mask_output_dir,
                frame_id=frame_id,
                ball_id=ball_id,
            )
            if centroid is not None:
                results.append((frame_id, ball_id, centroid[0], centroid[1], centroid[2]))
            else:
                if not args.skip_missing:
                    print(
                        f"[WARN] No valid points or mask invalid for ball {ball_id} in frame {frame_id:06d}"
                    )
        
        processed += 1
        if processed % 50 == 0:
            print(f"Processed {processed}/{len(frame_ids)} frames...")

    # Save results
    print(f"\nSaving results to {output_path}")
    with open(output_path, "w") as f:
        f.write("frame_id ball_id x y z\n")
        for frame_id, ball_id, x, y, z in results:
            f.write(f"{frame_id} {ball_id} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Saved {len(results)} ball center positions")
    print(f"Processed {processed} frames, skipped {skipped} frames")


if __name__ == "__main__":
    main()
