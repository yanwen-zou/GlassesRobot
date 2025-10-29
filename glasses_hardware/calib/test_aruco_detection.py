#!/usr/bin/env python3
"""
Quick ArUco detection sanity check using the CameraD400 feed.
"""

import argparse
import pathlib
import sys

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.my_device.camera import CameraD400
from hardware.my_device.macros import CAM_SERIAL


def main(args):
    try:
        serial = CAM_SERIAL[args.cam_index]
    except IndexError as exc:
        raise SystemExit(
            f"cam_index {args.cam_index} invalid; available indices 0..{len(CAM_SERIAL)-1}"
        ) from exc

    aruco_dict = getattr(cv2.aruco, args.aruco_dict, None)
    if aruco_dict is None:
        available = [name for name in dir(cv2.aruco) if name.startswith("DICT")]
        raise SystemExit(f"Unknown dictionary {args.aruco_dict}. Candidates: {available}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
    detector_params = getattr(cv2.aruco, "DetectorParameters_create", None)
    if detector_params is not None:
        detector_params = detector_params()
    else:
        detector_params = cv2.aruco.DetectorParameters()

    camera = CameraD400(serial=serial)
    intrinsics = camera.mtx
    dist_coeffs = np.zeros((5, 1))

    marker_length = args.marker_length
    print(f"[INFO] Running detection with dictionary={args.aruco_dict}, marker_length={marker_length} m")
    print("[INFO] Press 'q' to exit.")

    try:
        while True:
            color, depth = camera.get_data()
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

            display = color.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_length, intrinsics, dist_coeffs
                )
                for rvec, tvec in zip(rvecs, tvecs):
                    cv2.drawFrameAxes(display, intrinsics, dist_coeffs, rvec, tvec, marker_length * 0.5)
                print(f"[INFO] Detected IDs: {ids.flatten().tolist()}")
            else:
                print("[INFO] No markers detected.")

            cv2.imshow("aruco_detection", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        del camera


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ArUco marker detection using CameraD400.")
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default="DICT_6X6_250",
        help="OpenCV ArUco dictionary name.",
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
        default=0,
        help="Index into CAM_SERIAL for selecting the camera.",
    )
    args = parser.parse_args()
    main(args)
