#!/usr/bin/env python3
"""Calculate 3D centroids of balls from masks and depth information.

This script processes mask_balls data to extract 3 separate masks (id1, id2, id3) for each frame,
calculates 3D centroids using depth information in the current frame camera coordinate system.

Example:
    python scripts_calib_balls/calculate_ball_centers.py \
        --data-dir data/train/20251125_210453 \
        --output ball_centers.txt
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


DEPTH_SCALE_DEFAULT = 1000.0  # RealSense depth units -> meters


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


def calculate_ball_centroid(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray | None:
    """Calculate 3D centroid of a ball from mask and depth in current frame camera coordinate system.
    
    Args:
        depth_m: Depth image in meters (H, W)
        mask: Boolean mask (H, W)
        intrinsic: Camera intrinsic matrix (3x3)
        
    Returns:
        3D centroid in current frame camera coordinate system, or None if no valid points
    """
    if depth_m.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Depth shape {depth_m.shape} and mask shape {mask.shape} must match.")

    valid_mask = mask & np.isfinite(depth_m) & (depth_m > 0)
    ys, xs = np.nonzero(valid_mask)
    if ys.size == 0:
        return None

    z = depth_m[ys, xs].astype(np.float32)
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

            centroid = calculate_ball_centroid(depth_m, mask, intrinsic)
            if centroid is not None:
                results.append((frame_id, ball_id, centroid[0], centroid[1], centroid[2]))
            else:
                if not args.skip_missing:
                    print(f"[WARN] No valid points for ball {ball_id} in frame {frame_id:06d}")
        
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

