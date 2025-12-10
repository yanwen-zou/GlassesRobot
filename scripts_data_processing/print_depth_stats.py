#!/usr/bin/env python3
import argparse
import os
import numpy as np


def load_png_u16(path: str) -> np.ndarray:
    """Load a 16-bit PNG using cv2, Pillow, or imageio (in that order)."""
    # Try OpenCV
    try:
        import cv2  # type: ignore
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is not None and arr.dtype == np.uint16:
            return arr
        if arr is not None:
            raise RuntimeError(f"Loaded with dtype {arr.dtype}, expected uint16")
    except Exception:
        pass

    # Try Pillow
    try:
        from PIL import Image  # type: ignore
        img = Image.open(path)
        arr = np.array(img)
        if arr.dtype == np.uint16:
            return arr
        raise RuntimeError(f"Pillow loaded dtype {arr.dtype}, expected uint16")
    except Exception:
        pass

    # Try imageio
    try:
        import imageio.v3 as iio  # type: ignore
        arr = iio.imread(path)
        if arr.dtype == np.uint16:
            return arr
        raise RuntimeError(f"imageio loaded dtype {arr.dtype}, expected uint16")
    except Exception:
        pass

    raise RuntimeError("Failed to load image as uint16 using cv2/Pillow/imageio.")


def maybe_read_depth_scale(png_path: str):
    """Attempt to read a depth scale (meters/unit) from depth_scale.txt.

    Checks sibling folder of the PNG and its parent directory.
    """
    cand1 = os.path.join(os.path.dirname(png_path), "depth_scale.txt")
    if os.path.isfile(cand1):
        try:
            with open(cand1, "r") as f:
                return float(f.read().strip())
        except Exception:
            return None

    parent = os.path.dirname(os.path.dirname(png_path))
    cand2 = os.path.join(parent, "depth_scale.txt")
    if os.path.isfile(cand2):
        try:
            with open(cand2, "r") as f:
                return float(f.read().strip())
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser(description="Print stats for a uint16 RealSense depth PNG.")
    ap.add_argument("png", help="Path to depth_*.png (uint16)")
    ap.add_argument("--depth-scale", type=float, default=None, help="Meters per unit (e.g., 0.001)")
    ap.add_argument("--no-autoscale", action="store_true", help="Do not read depth_scale.txt automatically")
    ap.add_argument("--patch-size", type=int, default=5, help="Center patch size (odd)")
    args = ap.parse_args()

    arr = load_png_u16(args.png)
    h, w = arr.shape[:2]
    nz = arr[arr > 0]

    print(f"path: {args.png}")
    print(f"shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"min: {int(arr.min())}, max: {int(arr.max())}")
    if nz.size:
        print(f"min>0: {int(nz.min())}, max>0: {int(nz.max())}, mean>0: {float(nz.mean()):.3f}")
    else:
        print("no nonzero pixels")

    # Resolve depth scale
    scale = args.depth_scale
    if scale is None and not args.no_autoscale:
        scale = maybe_read_depth_scale(args.png)
    if scale is not None:
        print(f"depth_scale (m/unit): {scale}")
        if nz.size:
            print(f"mean depth (m) for nonzero: {nz.mean() * scale:.3f}")
        else:
            print("mean depth (m): N/A (no nonzero)")

    # Center patch
    k = max(1, int(args.patch_size))
    if k % 2 == 0:
        k += 1
    y0 = max(0, h // 2 - k // 2)
    x0 = max(0, w // 2 - k // 2)
    patch = arr[y0:y0 + k, x0:x0 + k]
    print(f"center {k}x{k} patch @ ({y0}:{y0+k}, {x0}:{x0+k}):")
    for row in patch:
        print(" ".join(str(int(v)) for v in row))

    # Sample a few coordinates
    sample_coords = [(h // 2, w // 2), (h // 3, w // 3), (2 * h // 3, 2 * w // 3)]
    print("sample pixels (y,x)->value [meters if scale known]:")
    for (yy, xx) in sample_coords:
        val = int(arr[yy, xx])
        if scale is not None:
            print(f"({yy},{xx})->{val} [{val * scale:.3f} m]")
        else:
            print(f"({yy},{xx})->{val}")


if __name__ == "__main__":
    main()

