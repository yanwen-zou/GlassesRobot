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
- Optional keyboard robot control (enable with --keyboard-control): u/o/j/l/i/k translate, 1-6 rotate, +/- adjust step.
- Camera pose saving (robot_to_cam.npy) uses recorded TCP poses and eih_camT.npy (tcp->cam SE3).
- Capture uses a fixed key: press 'c' in the preview to save a frame and poses.

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from glasses_hardware.hardware.my_device.robot import FlexivRobot, _pose7_to_mat


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


def quat_multiply(q1, q2):
    """Quaternion multiply, both as [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    return q / n


def axis_angle_to_quat(axis, angle):
    """Axis is 3-array, angle in rad."""
    axis = np.asarray(axis, dtype=np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1e-9 or abs(angle) < 1e-9:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    axis = axis / norm
    half = angle * 0.5
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def apply_tcp_delta(curr_pose7, trans_delta, rot_axis=None, rot_angle=0.0):
    """Return new pose after applying translation and small rotation in world frame."""
    new_pose = np.asarray(curr_pose7, dtype=np.float32).copy()
    new_pose[:3] += np.asarray(trans_delta, dtype=np.float32)
    if rot_axis is not None and abs(rot_angle) > 0:
        curr_quat = new_pose[3:7]
        dq = axis_angle_to_quat(rot_axis, rot_angle)
        new_quat = quat_normalize(quat_multiply(dq, curr_quat))
        new_pose[3:7] = new_quat
    return new_pose


def build_tcp_keymap(trans_step, rot_step):
    """Map keys to (translation delta, rotation axis, angle)."""
    return {
        # Translation (world frame)
        ord("u"): (np.array([ trans_step, 0, 0], dtype=np.float32), None, 0.0),  # +X
        ord("o"): (np.array([-trans_step, 0, 0], dtype=np.float32), None, 0.0),  # -X
        ord("j"): (np.array([0,  trans_step, 0], dtype=np.float32), None, 0.0),  # +Y
        ord("l"): (np.array([0, -trans_step, 0], dtype=np.float32), None, 0.0),  # -Y
        ord("i"): (np.array([0, 0,  trans_step], dtype=np.float32), None, 0.0),  # +Z
        ord("k"): (np.array([0, 0, -trans_step], dtype=np.float32), None, 0.0),  # -Z
        # Rotation about world axes
        ord("1"): (np.zeros(3, dtype=np.float32), np.array([1, 0, 0], dtype=np.float32),  rot_step),  # +roll
        ord("2"): (np.zeros(3, dtype=np.float32), np.array([1, 0, 0], dtype=np.float32), -rot_step),  # -roll
        ord("3"): (np.zeros(3, dtype=np.float32), np.array([0, 1, 0], dtype=np.float32),  rot_step),  # +pitch
        ord("4"): (np.zeros(3, dtype=np.float32), np.array([0, 1, 0], dtype=np.float32), -rot_step),  # -pitch
        ord("5"): (np.zeros(3, dtype=np.float32), np.array([0, 0, 1], dtype=np.float32),  rot_step),  # +yaw
        ord("6"): (np.zeros(3, dtype=np.float32), np.array([0, 0, 1], dtype=np.float32), -rot_step),  # -yaw
    }


def load_tcp_to_cam(path):
    """Load SE3 from TCP to camera (4x4)."""
    T = np.load(path)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix at {path}, got {T.shape}")
    return T.astype(np.float32)


def tcp_pose_to_cam_pose(tcp_pose7, T_tcp_cam):
    """Convert TCP pose7 to camera pose matrix using tcp->cam transform."""
    T_world_tcp = _pose7_to_mat(tcp_pose7)
    return T_world_tcp @ T_tcp_cam


def save_cam_poses(out_root, tcp_poses, eih_cam_path):
    """Save camera poses (world->cam) computed from TCP poses."""
    try:
        T_tcp_cam = load_tcp_to_cam(eih_cam_path)
    except Exception as e:
        print(f"WARNING: Cannot load tcp->cam transform from {eih_cam_path}: {e}")
        return
    cam_poses = []
    for tcp_pose in tcp_poses:
        cam_poses.append(tcp_pose_to_cam_pose(tcp_pose, T_tcp_cam))
    if not cam_poses:
        return
    cam_array = np.stack(cam_poses, axis=0)
    np.save(os.path.join(out_root, "robot_to_cam.npy"), cam_array)

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
    parser.add_argument(
        "--keyboard-control",
        action="store_true",
        help="Enable keyboard control of the robot TCP (u/o/j/l/i/k translate, 1-6 rotate, +/- adjust step). Requires preview window.",
    )
    parser.add_argument(
        "--eih-camT",
        type=str,
        default="glasses_hardware/calib/eih_camT.npy",
        help="Path to tcp->camera transform (eih_camT.npy) used to save robot_to_cam.npy.",
    )
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

    # print(
    #     f"Default color stream: {args.width}x{args.height} @ {args.fps}fps "
    #     f"(resized to {args.target_width}x{args.target_height} for saving)."
    # )
    print(f"Saving to: {out_root}")
    print("Press 'q' in the preview window or Ctrl+C to stop.")

    frame_id = 1
    window_name = "RealSense Preview" if not args.no_preview else None
    if window_name:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    capture_key = "c"
    if window_name is None:
        print("Capture-on-key requires the preview window; running without preview will save every frame.")
        capture_key_code = None
    else:
        capture_key_code = ord(capture_key)
    saved_count = 0

    # Initialize Flexiv robot for keyboard TCP control when requested
    if args.keyboard_control and window_name is None:
        print("Keyboard control requires the preview window; disabling keyboard control because --no-preview was set.")
        args.keyboard_control = False

    robot = None
    current_tcp = np.zeros(7, dtype=float)
    trans_step = 0.005  # meters
    rot_step = 0.02     # radians
    keymap = build_tcp_keymap(trans_step, rot_step)
    if args.keyboard_control:
        robot = FlexivRobot(home=False)
        try:
            current_tcp = robot.get_tcp_pose().astype(float)
        except Exception:
            current_tcp = np.zeros(7, dtype=float)
        print(
            "Keyboard control active: translate with u/o/j/l/i/k, rotate with 1-6, +/- to change step, ESC to quit."
        )
    if capture_key_code is not None:
        print(f"Capture mode: press '{capture_key}' in the preview to save a frame and poses.")

    tcp_poses = []
    printed_frame_shape = False

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
            if not printed_frame_shape:
                print(f"[INFO] Captured color shape: {color.shape}, depth shape: {depth.shape}")
                printed_frame_shape = True

            # Resize once for preview and potential saving
            depth_resized = cv2.resize(depth, (args.target_width, args.target_height), interpolation=cv2.INTER_NEAREST)
            color_resized = cv2.resize(color, (args.target_width, args.target_height), interpolation=cv2.INTER_LINEAR)

            should_save = capture_key_code is None
            if window_name:
                # Create a simple depth visualization for preview
                depth_vis = cv2.convertScaleAbs(depth_resized, alpha=0.03)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                preview = np.hstack((color_resized, depth_vis))
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC to quit
                    break
                elif args.keyboard_control:
                    if key in (ord("+"), ord("=")):
                        trans_step = min(0.05, trans_step + 0.001)
                        rot_step = min(0.2, rot_step + 0.005)
                        keymap = build_tcp_keymap(trans_step, rot_step)
                    elif key in (ord("-"), ord("_")):
                        trans_step = max(0.001, trans_step - 0.001)
                        rot_step = max(0.001, rot_step - 0.005)
                        keymap = build_tcp_keymap(trans_step, rot_step)
                    elif key in keymap and robot is not None:
                        d_trans, axis, angle = keymap[key]
                        current_tcp = apply_tcp_delta(current_tcp, d_trans, axis, angle)
                        robot.send_tcp_pose(current_tcp.tolist())
                        time.sleep(0.02)
                if capture_key_code is not None and key == capture_key_code:
                    should_save = True

            if should_save:
                # Save images
                rgb_path = os.path.join(out_root, "rgb", f"{frame_id:06d}.png")
                depth_path = os.path.join(out_root, "depth", f"{frame_id:06d}.png")

                cv2.imwrite(rgb_path, color_resized)
                cv2.imwrite(depth_path, depth_resized)
                print(f"[SAVE] Frame {frame_id:06d} -> {rgb_path}, {depth_path}")

                # Record current TCP pose for this frame
                if robot is not None:
                    try:
                        tcp_pose = robot.get_tcp_pose()
                    except Exception:
                        tcp_pose = np.zeros(7, dtype=float)
                else:
                    tcp_pose = np.zeros(7, dtype=float)
                tcp_poses.append(np.asarray(tcp_pose, dtype=float))

                saved_count += 1
                frame_id += 1
                if args.max_frames > 0 and saved_count >= args.max_frames:
                    break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        pipeline.stop()
        if window_name:
            cv2.destroyAllWindows()
        try:
            if tcp_poses:
                tcp_array = np.stack(tcp_poses, axis=0)
                np.save(os.path.join(out_root, "tcp_pose.npy"), tcp_array)
                save_cam_poses(out_root, tcp_array, args.eih_camT)
        except Exception as e:
            print(f"WARNING: Failed to save tcp_pose.npy: {e}")

    print(f"Done. Saved {saved_count} frame(s) to {out_root}")


if __name__ == "__main__":
    main()
