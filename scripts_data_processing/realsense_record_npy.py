#!/usr/bin/env python3
"""
Record RGB + depth streams from an Intel RealSense camera and write:

- rgb/000001.png (8-bit color, optionally resized)
- depth/000001.npy (depth in meters as float32)
- cam_K.txt (3x3 intrinsics matrix for the stored resolution)

python scripts_data_processing/realsense_record_npy.py --out-dir mesh_data/small_book --serial 135122070361

Press 'q' in the preview window (default enabled) or Ctrl+C in the terminal to stop.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


try:
    import pyrealsense2 as rs
except Exception as exc:
    print("ERROR: Failed to import pyrealsense2. Please install librealsense and pyrealsense2.")
    raise

try:
    import cv2
except Exception as exc:
    print("ERROR: Failed to import cv2. Please install opencv-python.")
    raise


def list_devices() -> list[tuple[rs.device, str, str]]:
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "Unknown"
        serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else ""
        devices.append((dev, name, serial))
    return devices


def intrinsics_to_K(intr: rs.intrinsics) -> np.ndarray:
    K = np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K


def scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    scaled = K.copy()
    scaled[0, 0] *= sx
    scaled[1, 1] *= sy
    scaled[0, 2] *= sx
    scaled[1, 2] *= sy
    return scaled


def write_K(out_dir: Path, K: np.ndarray) -> None:
    with (out_dir / "cam_K.txt").open("w") as f:
        for r in range(3):
            f.write("{:.8f} {:.8f} {:.8f}\n".format(K[r, 0], K[r, 1], K[r, 2]))


def ensure_output_dirs(root: Path) -> tuple[Path, Path]:
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    return rgb_dir, depth_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Record RealSense RGB+Depth and save depth as NPY.")
    parser.add_argument("--device-index", type=int, default=0, help="Index of the RealSense device (0=first).")
    parser.add_argument("--serial", type=str, default=None, help="Optional camera serial to select.")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="Base directory for recordings.")
    parser.add_argument("--width", type=int, default=1280, help="Color stream width.")
    parser.add_argument("--height", type=int, default=720, help="Color stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS.")
    parser.add_argument("--target-width", type=int, default=640, help="Output width (resize).")
    parser.add_argument("--target-height", type=int, default=360, help="Output height (resize).")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to record (0 = no limit).")
    args = parser.parse_args()

    devices = list_devices()
    if not devices:
        print("ERROR: No RealSense devices detected.")
        return

    target_serial = args.serial
    selected_serial = None
    if target_serial:
        for _, _, serial in devices:
            if serial == target_serial:
                selected_serial = serial
                break
        if selected_serial is None:
            print(f"ERROR: Serial {target_serial} not found among connected devices.")
            return
    else:
        idx = max(0, min(args.device_index, len(devices) - 1))
        selected_serial = devices[idx][2]

    print(f"[INFO] Using device serial: {selected_serial}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected_serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    align = rs.align(rs.stream.color)

    try:
        profile = pipeline.start(config)
    except Exception as exc:
        print(f"ERROR: Failed to start pipeline: {exc}")
        return

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    base_K = intrinsics_to_K(intr)
    sx = args.target_width / float(args.width)
    sy = args.target_height / float(args.height)
    scaled_K = scale_K(base_K, sx, sy)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.out_dir / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir, depth_dir = ensure_output_dirs(session_dir)
    write_K(session_dir, scaled_K)
    print(f"[INFO] Recording to {session_dir}")

    cv2.namedWindow("RealSense", cv2.WINDOW_NORMAL)

    frame_idx = 0
    last_log = time.time()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_frame.get_units()

            color_resized = cv2.resize(color, (args.target_width, args.target_height), interpolation=cv2.INTER_LINEAR)
            depth_resized = cv2.resize(depth, (args.target_width, args.target_height), interpolation=cv2.INTER_NEAREST)

            frame_idx += 1
            rgb_path = rgb_dir / f"{frame_idx:06d}.png"
            depth_path = depth_dir / f"{frame_idx:06d}.npy"

            cv2.imwrite(str(rgb_path), color_resized)
            np.save(depth_path, depth_resized)

            depth_vis = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            stacked = np.hstack((color_resized, depth_colormap))
            cv2.imshow("RealSense", stacked)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                print(f"[INFO] Reached max frames ({args.max_frames}), stopping.")
                break

            now = time.time()
            if now - last_log > 2.0:
                print(f"[INFO] Captured {frame_idx} frames...")
                last_log = now
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user, stopping.")
    finally:
        pipeline.stop()
        cv2.destroyWindow("RealSense")
        print(f"[INFO] Saved {frame_idx} frames in {session_dir}")


if __name__ == "__main__":
    main()
