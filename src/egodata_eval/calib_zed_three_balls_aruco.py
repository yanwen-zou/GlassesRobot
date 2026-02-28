#!/usr/bin/env python3
"""Calibrate with eval.py three-ball logic, then estimate AprilTag pose online.

Flow:
1) Reuse `calibrate_from_three_balls` from eval_utils (same as eval.py path).
2) Print T_base_cam in terminal.
3) Open left image stream and detect AprilTag continuously, printing T_cam_tag.
"""

from __future__ import annotations

import argparse
import time
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

here = Path(__file__).resolve()
project_root = here.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from egodata_eval.eval_utils import calibrate_from_three_balls, read_zed_intrinsics_baseline, move_i2rt_to_init_angles
from egodata_eval.eval_constant import TASK_CHOICES
from egodata_eval.get_depth import DepthEstimator
from glasses_hardware.hardware.my_device.zed import ZEDCamera
from glasses_hardware.hardware.my_device.i2rt_robo import I2RT, I2RTServer, I2RTClient


def _detect_apriltag_pose(
    frame_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    tag_size_m: float,
    april_dict_name: str,
    target_id: int,
):
    if not hasattr(cv2, "aruco"):
        return None, None, frame_bgr

    dict_id = getattr(cv2.aruco, april_dict_name, cv2.aruco.DICT_APRILTAG_36h11)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params_ctor = getattr(cv2.aruco, "DetectorParameters_create", None)
    params = params_ctor() if params_ctor else cv2.aruco.DetectorParameters()

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detector_cls = getattr(cv2.aruco, "ArucoDetector", None)
    if detector_cls is not None:
        detector = detector_cls(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    vis = frame_bgr.copy()
    if ids is None or len(ids) == 0:
        return None, None, vis

    ids = ids.reshape(-1)
    cv2.aruco.drawDetectedMarkers(vis, corners, ids.reshape(-1, 1))
    try:
        idx = int(np.where(ids == target_id)[0][0]) if target_id >= 0 else 0
    except Exception:
        idx = 0

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, tag_size_m, camera_matrix, dist_coeffs
    )
    rvec = rvecs[idx].reshape(3)
    tvec = tvecs[idx].reshape(3)
    R, _ = cv2.Rodrigues(rvec)
    T_cam_tag = np.eye(4, dtype=np.float32)
    T_cam_tag[:3, :3] = R.astype(np.float32)
    T_cam_tag[:3, 3] = tvec.astype(np.float32)

    for rv, tv in zip(rvecs, tvecs):
        cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rv, tv, tag_size_m * 0.5)
    chosen_id = int(ids[idx])
    return chosen_id, T_cam_tag, vis


def _read_zed_left_dist_coeffs(camera) -> np.ndarray:
    """Read ZED left-camera distortion coefficients in OpenCV format."""
    zed_handle = getattr(camera, "_zed", camera)
    info = zed_handle.get_camera_information()
    config = getattr(info, "camera_configuration", None)
    calib = config.calibration_parameters if config else info.calibration_parameters
    left = calib.left_cam
    dist_raw = np.asarray(getattr(left, "disto", []), dtype=np.float32).reshape(-1)
    if dist_raw.size < 5:
        raise RuntimeError(f"ZED left distortion coefficients unavailable, got size={dist_raw.size}.")
    return dist_raw[:5].reshape(5, 1)


def _init_i2rt_rpc(channel: str) -> tuple[I2RT, I2RTServer, threading.Thread, I2RTClient]:
    """Start local I2RT RPC server/client."""
    robot = I2RT(channel=channel, zero_gravity_mode=False, home=False)
    server = I2RTServer(robot)
    server_thread = threading.Thread(target=server.serve, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    client = I2RTClient()
    base_q = client.current_joint_pos()
    print(f"[INFO] I2RT current_joint_pos(rad): {np.round(base_q, 4)}")
    return robot, server, server_thread, client


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use eval.py three-ball calibration, then detect AprilTag and print transforms."
    )
    parser.add_argument("--resolution", default="WVGA", choices=["2K", "1080P", "720P", "WVGA"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth-scale", type=float, default=0.75)
    parser.add_argument("--task", type=str, default="book", choices=TASK_CHOICES)
    parser.add_argument("--i2rt-channel", type=str, default="can0")
    parser.add_argument("--apriltag-dict", default="DICT_APRILTAG_36h11")
    parser.add_argument("--apriltag-id", type=int, default=2, help="Target AprilTag ID.")
    parser.add_argument("--apriltag-size", type=float, default=0.064, help="AprilTag side length in meters.")
    parser.add_argument("--print-interval", type=float, default=0.5, help="Seconds between T_cam_tag prints.")
    args = parser.parse_args()

    robot = None
    server = None
    server_thread = None
    client = None

    cam = ZEDCamera(resolution=args.resolution, fps=args.fps)
    try:
        print(f"[INFO] Initialize I2RT via RPC for task={args.task} ...")
        robot, server, server_thread, client = _init_i2rt_rpc(args.i2rt_channel)
        # client.send_joint_pos_deg(np.array([0,10,10,0,0,0]))  # Just print current joint pos without moving
        move_i2rt_to_init_angles(client, task_name=args.task)
    except Exception as exc:
        print(f"[WARN] Failed to init/move I2RT via RPC: {exc}")
    time.sleep(1.0)
    try:
        # depth_est = DepthEstimator(scale=float(args.depth_scale), camera=cam)
        # Use ZED Calibration app values for current resolution (left sensor).
        K, _ = read_zed_intrinsics_baseline(cam)
        dist = np.array([-0.0384, 0.0398, 0.0, 0.0, -0.0441], dtype=np.float32).reshape(5, 1)
        print(f"[INFO] Use manual ZED left intrinsics K:\n{K}")
        print(f"[INFO] Use manual ZED left distortion coeffs: {dist.reshape(-1)}")

        # print("[INFO] Left camera intrinsics K:")
        # print(K)
        # print("[INFO] Start three-ball calibration (same logic as eval.py).")
        # T_base_cam = calibrate_from_three_balls(cam, depth_est, move_robot_fn=None, centroid_log_dir=None)
        # if T_base_cam is None:
        #     raise RuntimeError("Failed to calibrate T_base_cam from three balls.")
        # t_base_cam = T_base_cam[:3, 3].astype(np.float32)
        # d_base_cam = float(np.linalg.norm(t_base_cam))
        # print(
        #     "[OK] T_base_cam translation xyz(m)=({:.4f}, {:.4f}, {:.4f}), distance={:.4f} m".format(
        #         t_base_cam[0], t_base_cam[1], t_base_cam[2], d_base_cam
        #     )
        # )

        win = "ZED Left | AprilTag | q:quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        last_print_t = 0.0

        while True:
            frame = cam.read()
            if frame is None:
                continue
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            frame = frame.copy()

            selected_tag_id, T_cam_tag, vis = _detect_apriltag_pose(
                frame,
                K,
                dist,
                float(args.apriltag_size),
                args.apriltag_dict,
                int(args.apriltag_id),
            )

            y = 24
            cv2.putText(
                vis,
                "Three-ball calib done, printing T_cam_tag online",
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 0),
                2,
            )
            y += 28
            if selected_tag_id is not None and T_cam_tag is not None:
                t = T_cam_tag[:3, 3]
                cv2.putText(
                    vis,
                    f"AprilTag id={selected_tag_id} t=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})m",
                    (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 200, 0),
                    2,
                )
                now = time.time()
                if now - last_print_t >= float(args.print_interval):
                    last_print_t = now
                    d_cam_tag = float(np.linalg.norm(t))
                    print(
                        "[INFO] tag id={} xyz(m)=({:.4f}, {:.4f}, {:.4f}), distance={:.4f} m".format(
                            selected_tag_id, t[0], t[1], t[2], d_cam_tag
                        )
                    )
            else:
                cv2.putText(
                    vis,
                    f"AprilTag id={args.apriltag_id}: not detected",
                    (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (80, 80, 255),
                    2,
                )

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

        cv2.destroyAllWindows()
    finally:
        cam.close()
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                print(f"[WARN] Failed to close I2RT RPC client cleanly: {exc}")


if __name__ == "__main__":
    main()
