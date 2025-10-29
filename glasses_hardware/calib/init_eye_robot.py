"""
Camera calibration helper.

Moves the Flexiv arm to expose an ArUco marker, detects the marker pose from the
RealSense color stream, and saves the resulting transform (marker -> camera).
"""

import argparse
import os
import time
import numpy as np
import cv2
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.my_device.robot import FlexivRobot
from hardware.my_device.camera import CameraD400
from hardware.my_device.macros import CAM_SERIAL


def move_robot_for_visibility(robot: FlexivRobot, joint_offset: float, settle_time: float = 2.0):
    """Lift the penultimate joint slightly to expose the marker."""
    joints = robot.get_joint_pos().copy()
    joints[-2] += joint_offset
    print(f"[Robot] Moving penultimate joint by {joint_offset:.3f} rad")
    robot.send_joint_pose(joints)
    time.sleep(settle_time)
    return joints


def build_transform(rvec, tvec):
    """Compose 4x4 transform from marker to camera using OpenCV pose."""
    rot_matrix, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = tvec.reshape(-1)
    return transform


def detect_aruco_pose(
    frame_bgr,
    camera_matrix,
    dist_coeffs,
    aruco_dict,
    marker_length,
    target_id=None,
    detector_params=None,
):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kwargs = {}
    if detector_params is not None:
        kwargs["parameters"] = detector_params
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, **kwargs)
    if ids is None:
        return None
    results = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        marker_length,
        camera_matrix,
        dist_coeffs,
    )
    rvecs, tvecs, _ = results

    for idx, marker_id in enumerate(ids.flatten()):
        if target_id is not None and marker_id != target_id:
            continue
        return corners[idx], marker_id, rvecs[idx], tvecs[idx]

    return None


def main(args):
    robot = FlexivRobot()
    try:
        camera_serial = CAM_SERIAL[args.cam_index]
    except IndexError as exc:
        raise SystemExit(
            f"cam_index {args.cam_index} invalid; available indices: 0..{len(CAM_SERIAL)-1}"
        ) from exc
    camera = CameraD400(serial=camera_serial)

    joint_backup = robot.get_joint_pos().copy()
    try:
        target_joints = move_robot_for_visibility(robot, args.joint_offset)

        intrinsics = camera.mtx
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        aruco_dict = cv2.aruco.getPredefinedDictionary(args.aruco_dict)
        detector_params = getattr(cv2.aruco, "DetectorParameters_create", None)
        if detector_params is not None:
            detector_params = detector_params()
        else:
            detector_params = cv2.aruco.DetectorParameters()

        print("[Calib] Searching for ArUco marker. Press Ctrl+C to abort.")
        transform = None
        while transform is None:
            color_img, _ = camera.get_data()
            detection = detect_aruco_pose(
                color_img,
                intrinsics,
                dist_coeffs,
                aruco_dict,
                args.marker_length,
                args.target_id,
                detector_params,
            )
            if detection is None:
                cv2.imshow("calibration_view", color_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt("User requested exit.")
                continue

            corners, marker_id, rvec, tvec = detection
            transform = build_transform(rvec, tvec)

            cv2.aruco.drawDetectedMarkers(color_img, [corners])
            cv2.drawFrameAxes(color_img, intrinsics, dist_coeffs, rvec, tvec, args.marker_length * 0.5)
            cv2.imshow("calibration_view", color_img)
            cv2.waitKey(1)
            print(f"[Calib] Detected marker id={marker_id}")
            time.sleep(2.0)

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.save(args.output, transform)
        print(f"[Calib] Saved T_cam_aruco to {args.output}")
    finally:
        cv2.destroyAllWindows()
        robot.send_joint_pose(joint_backup)
        time.sleep(2.0)
        robot.stop()
        print("[Calib] Robot returned to safe posture.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate camera pose relative to ArUco marker using Flexiv robot.")
    parser.add_argument(
        "--joint-offset",
        type=float,
        default=0.4,
        help="Radians to lift the penultimate joint forward for better visibility.",
    )
    parser.add_argument(
        "--marker-length",
        type=float,
        default=0.05,
        help="ArUco marker side length in meters.",
    )
    parser.add_argument(
        "--cam-index",
        type=int,
        default=1,
        help="Index into CAM_SERIAL for selecting the camera.",
    )
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default="DICT_6X6_250",
        help="OpenCV ArUco dictionary name, e.g., DICT_6X6_250.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calib/T_cam_aruco.npy",
        help="Path to save the computed transform matrix.",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        default=-1,
        help="ArUco marker ID to detect; use -1 to accept any marker.",
    )

    args = parser.parse_args()
    try:
        args.aruco_dict = getattr(cv2.aruco, args.aruco_dict)
    except AttributeError as exc:
        available = [name for name in dir(cv2.aruco) if name.startswith("DICT")]
        raise SystemExit(
            f"Unknown ArUco dictionary '{args.aruco_dict}'. Available options: {available}"
        ) from exc
    if args.target_id < 0:
        args.target_id = None
    main(args)
