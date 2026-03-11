#!/usr/bin/env python3
"""
Record an Intel RealSense color stream to an MP4 file.

Example:
python scripts_data_processing/realsense_record_mp4.py --data-dir data/run_001
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:
    print("ERROR: Failed to import cv2. Please install opencv-python.")
    raise

try:
    import pyrealsense2 as rs
except Exception:
    print("ERROR: Failed to import pyrealsense2. Please install librealsense and pyrealsense2.")
    raise


RUNNING = True


def _handle_stop(signum, frame):
    del signum, frame
    global RUNNING
    RUNNING = False


def list_devices() -> list[tuple[str, str]]:
    ctx = rs.context()
    devices: list[tuple[str, str]] = []
    for dev in ctx.query_devices():
        name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "Unknown"
        serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else ""
        devices.append((name, serial))
    return devices


def select_serial(serial: str | None, device_index: int) -> str:
    devices = list_devices()
    if not devices:
        raise RuntimeError("No RealSense devices detected.")

    if serial is not None:
        for _, found_serial in devices:
            if found_serial == serial:
                return found_serial
        raise RuntimeError(f"Serial {serial} not found among connected devices.")

    if device_index < 0 or device_index >= len(devices):
        device_list = ", ".join(f"[{idx}] {name} ({dev_serial})" for idx, (name, dev_serial) in enumerate(devices))
        raise RuntimeError(f"Invalid device-index {device_index}. Found devices: {device_list}")

    return devices[device_index][1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Record RealSense color stream to MP4.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Output directory for the recording.")
    parser.add_argument("--filename", type=str, default="realsense.mp4", help="MP4 filename inside data-dir.")
    parser.add_argument("--device-index", type=int, default=0, help="RealSense device index when serial is not set.")
    parser.add_argument("--serial", type=str, default=None, help="Optional RealSense serial number.")
    parser.add_argument("--width", type=int, default=1280, help="Color stream width.")
    parser.add_argument("--height", type=int, default=720, help="Color stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS.")
    parser.add_argument("--preview", action="store_true", help="Show a preview window while recording.")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        selected_serial = select_serial(args.serial, args.device_index)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    args.data_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.data_dir / args.filename
    meta_path = args.data_dir / "realsense_record_meta.txt"

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected_serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    try:
        profile = pipeline.start(config)
    except Exception as exc:
        print(f"ERROR: Failed to start RealSense pipeline: {exc}")
        return 1

    stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = stream_profile.get_intrinsics()
    actual_width = int(intr.width)
    actual_height = int(intr.height)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (actual_width, actual_height),
    )
    if not writer.isOpened():
        pipeline.stop()
        print(f"ERROR: Failed to open video writer for {video_path}")
        return 1

    if args.preview:
        cv2.namedWindow("RealSense MP4 Recorder", cv2.WINDOW_NORMAL)

    frame_count = 0
    started_at = datetime.now().isoformat(timespec="seconds")
    last_log_time = time.time()
    print(f"[INFO] Recording RealSense to {video_path}")
    print(f"[INFO] Using device serial: {selected_serial}")

    try:
        while RUNNING:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            writer.write(color)
            frame_count += 1

            if args.preview:
                cv2.imshow("RealSense MP4 Recorder", color)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            now = time.time()
            if now - last_log_time >= 5.0:
                print(f"[INFO] Recorded {frame_count} frames...")
                last_log_time = now
    except KeyboardInterrupt:
        pass
    finally:
        writer.release()
        pipeline.stop()
        if args.preview:
            cv2.destroyWindow("RealSense MP4 Recorder")

        ended_at = datetime.now().isoformat(timespec="seconds")
        meta_path.write_text(
            "\n".join(
                [
                    f"video_path={video_path.name}",
                    f"serial={selected_serial}",
                    f"width={actual_width}",
                    f"height={actual_height}",
                    f"fps={args.fps}",
                    f"frames={frame_count}",
                    f"started_at={started_at}",
                    f"ended_at={ended_at}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[INFO] Saved {frame_count} frames to {video_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
