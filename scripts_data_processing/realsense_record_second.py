#!/usr/bin/env python3
"""
Record color and depth streams from the second Intel RealSense camera
and save them under data/<timestamp>/ with images resized to 640x360.

Outputs:
- data/<timestamp>/rgb/rgb_000001.png (8-bit color)
- data/<timestamp>/depth/depth_000001.png (16-bit depth)
- data/<timestamp>/cam_K.txt (3x3 intrinsics matrix for resized images)

Controls:
- Press 'q' in the preview window or Ctrl+C in the terminal to stop.

Requirements:
- pyrealsense2, numpy, opencv-python
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    print("ERROR: Failed to import pyrealsense2. Please install librealsense and pyrealsense2.")
    raise

try:
    import cv2
except Exception:
    print("ERROR: Failed to import cv2. Please install opencv-python.")
    raise


def list_devices():
    ctx = rs.context()
    devices = []
    for d in ctx.query_devices():
        name = d.get_info(rs.camera_info.name) if d.supports(rs.camera_info.name) else "Unknown"
        serial = d.get_info(rs.camera_info.serial_number) if d.supports(rs.camera_info.serial_number) else ""
        devices.append((d, name, serial))
    return devices


def ensure_dirs(path):
    os.makedirs(os.path.join(path, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(path, "depth"), exist_ok=True)


def intrinsics_to_K(intr):
    # intr: rs.intrinsics with fx, fy, ppx (cx), ppy (cy)
    K = np.array([[intr.fx, 0.0, intr.ppx],
                  [0.0, intr.fy, intr.ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def scale_K(K, sx, sy):
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[0, 2] *= sx
    K_scaled[1, 2] *= sy
    return K_scaled


def write_K(path, K):
    with open(os.path.join(path, "cam_K.txt"), "w") as f:
        for r in range(3):
            f.write("{:.8f} {:.8f} {:.8f}\n".format(K[r, 0], K[r, 1], K[r, 2]))


def main():
    parser = argparse.ArgumentParser(description="Record from the second RealSense camera and save RGB+Depth frames.")
    parser.add_argument("--device-index", type=int, default=1, help="Index of the RealSense device to use (0-based). Default: 1 (second camera)")
    parser.add_argument("--out-dir", type=str, default="data", help="Base output directory. Timestamped folder will be created inside.")
    parser.add_argument("--width", type=int, default=1280, help="Requested color stream width before resize.")
    parser.add_argument("--height", type=int, default=720, help="Requested color stream height before resize.")
    parser.add_argument("--fps", type=int, default=30, help="Requested FPS.")
    parser.add_argument("--target-width", type=int, default=640, help="Output image width.")
    parser.add_argument("--target-height", type=int, default=360, help="Output image height.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional max frames to record (0 = unlimited).")
    parser.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview window.")
    args = parser.parse_args()

    devices = list_devices()
    if not devices:
        print("No RealSense devices found.")
        sys.exit(1)

    if args.device_index < 0 or args.device_index >= len(devices):
        print(f"Invalid device-index {args.device_index}. Found {len(devices)} device(s):")
        for i, (_, name, serial) in enumerate(devices):
            print(f"  [{i}] {name} (S/N: {serial})")
        sys.exit(1)

    device, name, serial = devices[args.device_index]
    print(f"Using device [{args.device_index}]: {name} (S/N: {serial})")

    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)

    # Enable streams
    # Use the requested resolution for color; depth resolution will be matched if possible.
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    # Common depth mode
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, args.fps)

    profile = pipeline.start(config)

    # Align depth to color for consistent size before resizing
    align = rs.align(rs.stream.color)

    # Get color intrinsics from the active profile
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()

    # Build K at native resolution, then scale to target size
    K_native = intrinsics_to_K(color_intr)
    sx = args.target_width / float(color_intr.width)
    sy = args.target_height / float(color_intr.height)
    K_scaled = scale_K(K_native, sx, sy)

    # Prepare output dirs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.out_dir, ts)
    ensure_dirs(out_root)

    # Write intrinsics once
    write_K(out_root, K_scaled)

    print(f"Saving to: {out_root}")
    print("Press 'q' in the preview window or Ctrl+C to stop.")

    frame_id = 1
    window_name = "RealSense Preview" if not args.no_preview else None
    if window_name:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth = np.asanyarray(depth_frame.get_data())  # uint16
            color = np.asanyarray(color_frame.get_data())  # BGR, uint8

            # Resize
            depth_resized = cv2.resize(depth, (args.target_width, args.target_height), interpolation=cv2.INTER_NEAREST)
            color_resized = cv2.resize(color, (args.target_width, args.target_height), interpolation=cv2.INTER_LINEAR)

            # Save
            rgb_path = os.path.join(out_root, "rgb", f"rgb_{frame_id:06d}.png")
            depth_path = os.path.join(out_root, "depth", f"depth_{frame_id:06d}.png")

            # Color: 8-bit PNG
            cv2.imwrite(rgb_path, color_resized)
            # Depth: 16-bit PNG, keep raw units
            cv2.imwrite(depth_path, depth_resized)

            # Preview
            if window_name:
                # Create a simple depth visualization for preview
                depth_vis = cv2.convertScaleAbs(depth_resized, alpha=0.03)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                preview = np.hstack((color_resized, depth_vis))
                cv2.imshow(window_name, preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.max_frames > 0 and frame_id >= args.max_frames:
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        pipeline.stop()
        if window_name:
            cv2.destroyAllWindows()

    print(f"Done. Saved {frame_id - 1} frame(s) to {out_root}")


if __name__ == "__main__":
    main()

