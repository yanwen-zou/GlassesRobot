#!/usr/bin/env python3
import argparse
import os
import sys

import cv2
import pyzed.sl as sl


def parse_args():
    parser = argparse.ArgumentParser(description="Record ZED stereo frames for mesh capture.")
    parser.add_argument("--session", required=True, help="输出文件夹名，例如20240101_120000")
    parser.add_argument("--out-root", default="data", help="输出根目录（默认data）")
    parser.add_argument("--fps", type=int, default=30, help="采样帧率（默认30）")
    parser.add_argument("--serial", type=int, help="可选：指定ZED序列号")
    parser.add_argument("--camera-id", type=int, help="可选：指定USB摄像头ID")
    return parser.parse_args()


def ensure_dirs(out_root: str, session: str):
    base = os.path.join(out_root, session)
    left_dir = os.path.join(base, "left")
    right_dir = os.path.join(base, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    return base, left_dir, right_dir


def print_device_list():
    try:
        devices = sl.Camera.get_device_list()
    except Exception as e:
        print(f"[warn] 无法枚举ZED设备: {e}")
        return
    if not devices:
        print("[warn] 未发现任何ZED设备")
        return
    print("[info] 可用ZED设备：")
    for d in devices:
        state = getattr(d, "camera_state", None)
        print(f"  id={d.id} serial={d.serial_number} model={d.camera_model} state={state}")


def main():
    args = parse_args()
    out_root, left_dir, right_dir = ensure_dirs(args.out_root, args.session)

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.camera_fps = args.fps
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.async_grab_camera_recovery = True
    init_params.sensors_required = False
    if args.camera_id is not None:
        init_params.set_from_camera_id(args.camera_id)
    if args.serial is not None:
        init_params.set_from_serial_number(args.serial)

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open: " + repr(err))
        print_device_list()
        sys.exit(1)

    left_image = sl.Mat()
    right_image = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    frame_idx = 0
    print(f"[info] Recording... Press 'q' to stop. Output: {out_root}")
    try:
        while True:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)

            left = left_image.get_data()
            right = right_image.get_data()

            left_bgr = cv2.cvtColor(left, cv2.COLOR_BGRA2BGR)
            right_bgr = cv2.cvtColor(right, cv2.COLOR_BGRA2BGR)

            cv2.imwrite(os.path.join(left_dir, f"left{frame_idx:06d}.png"), left_bgr)
            cv2.imwrite(os.path.join(right_dir, f"right{frame_idx:06d}.png"), right_bgr)
            frame_idx += 1

            cv2.imshow("Left", left_bgr)
            cv2.imshow("Right", right_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
    finally:
        zed.close()
        cv2.destroyAllWindows()

    print(f"[info] Saved {frame_idx} stereo frames to {left_dir} and {right_dir}")


if __name__ == "__main__":
    main()
