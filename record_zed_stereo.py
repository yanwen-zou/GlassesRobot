#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import cv2


def find_latest_cam_k(base: Path) -> Path | None:
    """Find latest cam_K.txt under data/train/*/cam_K.txt by folder name/time."""
    if not base.exists():
        return None
    candidates = []
    for p in base.glob("*/cam_K.txt"):
        # sort by parent directory name and mtime as fallback
        candidates.append((p.parent.name, p.stat().st_mtime, p))
    if not candidates:
        return None
    # primary: sort by directory name (timestamp-like), else by mtime
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def ensure_dirs(out_root: Path, stamp: str) -> tuple[Path, Path, Path]:
    ts_root = out_root / stamp
    left_dir = ts_root / "zed_left"
    right_dir = ts_root / "zed_right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    return ts_root, left_dir, right_dir


def copy_cam_k(cam_k_src: Path | None, dest_dirs: list[Path]):
    if cam_k_src is None or not cam_k_src.exists():
        print("[warn] cam_K.txt not found; skip copying.")
        return
    for d in dest_dirs:
        try:
            shutil.copy2(cam_k_src, d / "cam_K.txt")
        except Exception as e:
            print(f"[warn] Failed to copy cam_K.txt to {d}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Record ZED stereo frames to PNGs.")
    parser.add_argument("--resolution", default="720P", help="ZED resolution: 2K/1080P/720P/WVGA")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for capture")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration seconds (0 for until Ctrl+C)")
    parser.add_argument("--cam-k", type=Path, default=None, help="Path to cam_K.txt (defaults to latest under data/train)")
    parser.add_argument("--out-root", type=Path, default=Path("bundle_sdf_data"), help="Output root directory")
    parser.add_argument("--prefix", default="", help="Optional filename prefix for frames")
    args = parser.parse_args()

    # Resolve cam_K.txt
    cam_k_path = args.cam_k
    if cam_k_path is None:
        cam_k_path = find_latest_cam_k(Path("data/train"))
        if cam_k_path is None:
            print("[warn] No cam_K.txt found under data/train/*/cam_K.txt")
        else:
            print(f"[info] Using cam_K.txt: {cam_k_path}")
    else:
        if not cam_k_path.exists():
            print(f"[warn] Provided cam_K path not found: {cam_k_path}")

    # Prepare output directories
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ts_root, out_left, out_right = ensure_dirs(args.out_root, stamp)
    # Copy cam_K.txt into timestamp directory (once)
    copy_cam_k(cam_k_path, [ts_root])
    print(f"[info] Timestamp root: {ts_root}")
    print(f"[info] Output left:    {out_left}")
    print(f"[info] Output right:   {out_right}")

    # Import ZED wrapper
    try:
        from glasses_hardware.hardware.my_device.zed import ZEDCamera
    except Exception as e:
        print("[error] Failed to import ZED wrapper: ", e)
        sys.exit(1)

    # Initialize camera
    try:
        cam = ZEDCamera(resolution=args.resolution, fps=args.fps)
    except Exception as e:
        print("[error] Failed to open ZED camera: ", e)
        sys.exit(2)

    # Capture loop
    t_end = time.time() + args.duration if args.duration and args.duration > 0 else None
    frame_idx = 0
    print("[info] Recording... Press Ctrl+C to stop.")
    try:
        while True:
            if t_end is not None and time.time() >= t_end:
                print("[info] Reached duration limit; stopping.")
                break
            frames = cam.read_stereo()
            if frames is None:
                # brief sleep to avoid tight loop if grabbing fails
                time.sleep(0.001)
                continue
            left, right = frames
            if left is None or right is None:
                continue

            # Ensure BGR uint8
            if left.dtype != 'uint8':
                left = left.astype('uint8')
            if right.dtype != 'uint8':
                right = right.astype('uint8')

            # File names: use epoch ms for uniqueness
            fname_l = f"{frame_idx:06d}.png"
            fname_r = f"{frame_idx:06d}.png"

            # Write PNGs
            cv2.imwrite(str(out_left / fname_l), left)
            cv2.imwrite(str(out_right / fname_r), right)

            frame_idx += 1
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
    finally:
        try:
            cam.close()
        except Exception:
            pass

    print(f"[info] Saved {frame_idx} stereo frames under {ts_root}.")


if __name__ == "__main__":
    main()
