#!/usr/bin/env python3
"""
Initialize Piper arm to a fixed pose, open ZED camera, detect an ArUco marker,
and cache T_zed_aruco (SE3, 4x4) to T_zed_aruco.npy. If no marker is detected,
raise an error and exit.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cv2


def _add_project_root_to_path():
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _deg_list_to_rad(vals_deg):
    return [np.deg2rad(v) for v in vals_deg]


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Ensure input frame is a valid 3-channel BGR image for OpenCV drawing.
    - Drops alpha if BGRA.
    - Validates non-empty content.
    """
    if image is None:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        # BGRA -> BGR
        image = image[:, :, :3]
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError(f"Invalid image for display/drawing: shape={getattr(image, 'shape', None)}")
    return image


# def load_zed_intrinsics(default_path: Path | None = None) -> np.ndarray:
#     # Reuse FoundationStereo intrinsics file by default
#     root = Path(__file__).resolve().parents[1] / "FoundationStereo" / "assets"
#     intr_path = default_path or (root / "K_ZED.txt")
#     with open(intr_path, "r") as f:
#         lines = f.readlines()
#         K = np.array(list(map(float, lines[0].rstrip().split())), dtype=np.float32).reshape(3, 3)
#     return K


def move_arm_to_pose(pose_deg):
    from glasses_hardware.hardware.my_device.piper import Piper

    arm = Piper(can_port="can0")
    arm.enable_motion(speed_rate=50, is_mit_mode=0)
    targets_rad = _deg_list_to_rad(pose_deg)
    # Send for a short settle duration
    deadline = time.time() + 2
    cmd = arm.to_cmd(targets_rad)
    try:
        while time.time() < deadline:
            arm.iface.JointCtrl(*cmd)
            time.sleep(0.02)
    finally:
        # keep enabled so user can continue if needed
        pass


# def detect_aruco_and_cache(K: np.ndarray, marker_length_m: float, out_path: Path) -> np.ndarray:
#     from glasses_hardware.hardware.my_device.zed import ZEDCamera

#     zed = ZEDCamera(resolution="720P", fps=30)
#     dist_coeffs = np.zeros((5, 1), dtype=np.float32)

#     # Prepare ArUco detector
#     aruco_dict_id = getattr(cv2.aruco, "DICT_6X6_250")
#     aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
#     detector_params = getattr(cv2.aruco, "DetectorParameters_create", None)
#     if detector_params is not None:
#         detector_params = detector_params()
#     else:
#         detector_params = cv2.aruco.DetectorParameters()

#     T = None
#     try:
#         win = "ZED ArUco"
#         cv2.namedWindow(win, cv2.WINDOW_NORMAL)

#         start = time.time()
#         timeout_s = 5.0
#         while time.time() - start < timeout_s:
#             frame = zed.read()
#             if frame is None:
#                 continue
#             try:
#                 frame = _ensure_bgr(frame)
#             except ValueError:
#                 continue
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

#             display = frame.copy()
#             if ids is not None and len(ids) > 0:
#                 cv2.aruco.drawDetectedMarkers(display, corners, ids)
#                 rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length_m, K, dist_coeffs)
#                 # Draw axes for each detection
#                 for rvec, tvec in zip(rvecs, tvecs):
#                     cv2.drawFrameAxes(display, K, dist_coeffs, rvec, tvec, marker_length_m * 0.5)

#                 # Use the first marker to compute and cache T
#                 rvec = rvecs[0].reshape(3)
#                 tvec = tvecs[0].reshape(3)
#                 R, _ = cv2.Rodrigues(rvec)
#                 T = np.eye(4, dtype=float)
#                 T[:3, :3] = R
#                 T[:3, 3] = tvec

#                 # Show the annotated view for ~3 seconds before exiting
#                 show_until = time.time() + 3.0
#                 while time.time() < show_until:
#                     cv2.imshow(win, display)
#                     key = cv2.waitKey(1) & 0xFF
#                     if key == ord('q') or key == 27:
#                         break
#                 np.save(str(out_path), T)
#                 return T

#             # No detection yet; keep showing live feed

#             display = _ensure_bgr(display)

#             cv2.imshow(win, display)
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q') or key == 27:
#                 break
#     finally:
#         try:
#             cv2.destroyAllWindows()
#         except Exception:
#             pass
#         zed.close()

#     raise RuntimeError("未检测到 ArUco 标记，无法缓存 T_zed_aruco")


def preview_zed_3s():
    """Open ZED, display current frame in a window for ~3 seconds, then exit."""
    from glasses_hardware.hardware.my_device.zed import ZEDCamera

    zed = ZEDCamera(resolution="720P", fps=30)
    win = "ZED Preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t0 = time.time()
    try:
        while time.time() - t0 < 3.0:
            frame = zed.read()
            if frame is None:
                continue
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cv2.destroyAllWindows()
        zed.close()


class Calibration:
    """Encapsulated calibration helper for ZED–ArUco detection.

    - Provides loading intrinsics, detecting ArUco, and caching/loading T_zed_aruco.
    - T_zed_aruco is the 4x4 SE3 from ZED camera frame to ArUco frame as returned by
      OpenCV's estimatePoseSingleMarkers: i.e., it maps points from ArUco to ZED (cam<-aruco).
    """

    def __init__(self, marker_length_m: float = 0.045):
        _add_project_root_to_path()
        self.marker_length_m = float(marker_length_m)
        self._K: np.ndarray | None = None

    # @property
    # def K(self) -> np.ndarray:
    #     if self._K is None:
    #         self._K = load_zed_intrinsics()
    #     return self._K

    # def get_T_zed_aruco(self, cache_path: Path | None = None, detect_if_missing: bool = True) -> np.ndarray:
    #     """Load T_zed_aruco from cache or detect once via ZED and save.

    #     Args:
    #         cache_path: where to cache/load the SE3 (default: repo root T_zed_aruco.npy)
    #         detect_if_missing: if True, run detector when cache missing

    #     Returns:
    #         np.ndarray (4,4) SE3 matrix cam<-aruco
    #     """
    #     path = cache_path or Path("T_zed_aruco.npy")
    #     if path.exists():
    #         T = np.load(str(path))
    #         if T.shape == (4, 4):
    #             print(f"[OK] Loaded T_zed_aruco from {path}")
    #             return T.astype(np.float32)
    #     if not detect_if_missing:
    #         raise FileNotFoundError(f"No cached T_zed_aruco at {path}")
    #     return detect_aruco_and_cache(self.K, self.marker_length_m, path)


def main():
    _add_project_root_to_path()

    parser = argparse.ArgumentParser(description="Move Piper, detect ArUco via ZED, and cache T_zed_aruco.")
    parser.add_argument(
        "--marker-length",
        type=float,
        default=0.045,
        help="ArUco marker side length in meters",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("T_zed_aruco.npy"),
        help="Output .npy path for SE3 matrix",
    )

    args = parser.parse_args()

    # 1) Move arm to given pose (degrees)
    target_deg = [-18.00, 16.00, -24.00, -10.00, 20.00, 9.00]
    move_arm_to_pose(target_deg)

    # 2) Detect and cache via Calibration helper
    # calib = Calibration(marker_length_m=args.marker_length)
    # T = calib.get_T_zed_aruco(cache_path=args.out, detect_if_missing=True)
    # print(f"[OK] Cached T_zed_aruco to {args.out} with shape {T.shape}")


if __name__ == "__main__":
    main()
