#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def read_u16(path: Path) -> np.ndarray:
    # Try OpenCV
    try:
        import cv2  # type: ignore
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is not None and arr.dtype == np.uint16:
            return arr
    except Exception:
        pass
    # Try Pillow
    try:
        from PIL import Image  # type: ignore
        arr = np.array(Image.open(str(path)))
        if arr.dtype == np.uint16:
            return arr
    except Exception:
        pass
    # Try imageio
    try:
        import imageio.v3 as iio  # type: ignore
        arr = iio.imread(str(path))
        if arr.dtype == np.uint16:
            return arr
    except Exception:
        pass
    raise RuntimeError(f"Failed to read uint16 PNG: {path}")


def write_u16(path: Path, arr: np.ndarray) -> None:
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    # Try OpenCV
    try:
        import cv2  # type: ignore
        if cv2.imwrite(str(path), arr):
            return
    except Exception:
        pass
    # Try Pillow
    try:
        from PIL import Image  # type: ignore
        Image.fromarray(arr, mode="I;16").save(str(path))
        return
    except Exception:
        pass
    # Try imageio
    try:
        import imageio.v3 as iio  # type: ignore
        iio.imwrite(str(path), arr)
        return
    except Exception:
        pass
    raise RuntimeError(f"Failed to write uint16 PNG: {path}")


def process_dir(depth_dir: Path, threshold: int, dry_run: bool) -> None:
    files = sorted([p for p in depth_dir.iterdir() if p.suffix.lower() == ".png"])
    if not files:
        print(f"No PNG files in {depth_dir}")
        return
    changed = 0
    total = 0
    for p in files:
        total += 1
        arr = read_u16(p)
        if arr.ndim != 2:
            print(f"Skip non-2D image: {p}")
            continue
        mask = arr >= threshold
        if not np.any(mask):
            continue
        new_arr = arr.copy()
        new_arr[mask] = 0
        if dry_run:
            cnt = int(mask.sum())
            print(f"DRY-RUN would zero {cnt} px in {p.name}")
        else:
            write_u16(p, new_arr)
            cnt = int(mask.sum())
            print(f"Zeroed {cnt} px in {p.name}")
            changed += 1
    print(f"Done. Processed {total} files, modified {changed} files.")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_dir = repo_root / "data" / "20251030_113006" / "depth"

    ap = argparse.ArgumentParser(description="Set depth values >= threshold to 0 for all PNGs in a directory.")
    ap.add_argument("--depth-dir", type=str, default=str(default_dir), help="Directory containing depth PNGs")
    ap.add_argument("--threshold", type=int, default=1000, help="Threshold (inclusive). Values >= threshold become 0")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files, only report changes")
    args = ap.parse_args()

    depth_dir = Path(args.depth_dir).expanduser().resolve()
    if not depth_dir.exists() or not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    process_dir(depth_dir, args.threshold, args.dry_run)


if __name__ == "__main__":
    main()

