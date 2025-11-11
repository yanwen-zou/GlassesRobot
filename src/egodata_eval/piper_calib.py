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


def load_zed_intrinsics(default_path: Path | None = None) -> np.ndarray:
    # Reuse FoundationStereo intrinsics file by default
    root = Path(__file__).resolve().parents[1] / "FoundationStereo" / "assets"
    intr_path = default_path or (root / "K_ZED.txt")
    with open(intr_path, "r") as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split())), dtype=np.float32).reshape(3, 3)
    print(f"[INFO] Loaded ZED intrinsics from {intr_path}: \n{K}")
    return K


def move_arm_to_pose(pose_deg):
    from glasses_hardware.hardware.my_device.piper import Piper

    arm = Piper(can_port="can0")
    arm.enable_motion(speed_rate=50, is_mit_mode=0)
    targets_rad = _deg_list_to_rad(pose_deg)
    # Send for a short settle duration
    deadline = time.time() + 2.0
    cmd = arm.to_cmd(targets_rad)
    try:
        while time.time() < deadline:
            arm.iface.JointCtrl(*cmd)
            time.sleep(0.02)
    finally:
        # keep enabled so user can continue if needed
        pass


# NOTE: ArUco detection helpers are encapsulated in ArucoCalibrator class below to avoid duplicate logic.



class ArucoCalibrator:
    """Reusable ArUco calibrator that avoids repeated camera initialization.

    - Opens ZED once in constructor and provides `detect` to obtain T_cam_aruco.
    - Optionally caches result to a .npy file.
    - Call `close()` when done.
    """

    def __init__(self, marker_length_m: float = 0.045, K: np.ndarray | None = None):
        _add_project_root_to_path()
        from glasses_hardware.hardware.my_device.zed import ZEDCamera
        self.marker_length_m = float(marker_length_m)
        K_loaded = load_zed_intrinsics()
        # Stored intrinsics are from a 0.5 downscaled image; rescale by 2x to match the ZED feed we detect on.
        self.K = K_loaded.astype(np.float32).copy()
        self.K[:2, :] *= 2.0
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, "DICT_6X6_250"))
        detector_params_ctor = getattr(cv2.aruco, "DetectorParameters_create", None)
        self.detector_params = detector_params_ctor() if detector_params_ctor else cv2.aruco.DetectorParameters()
        # Open camera once
        self._zed = ZEDCamera(resolution="720P", fps=30)

    def detect(self, timeout_s: float = 5.0, show: bool = True) -> np.ndarray:
        """Detect ArUco once and return T_cam_aruco (cam<-aruco), 4x4."""
        win = None
        if show:
            win = "ZED ArUco"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        start = time.time()
        T = None
        try:
            while time.time() - start < timeout_s:
                frame = self._zed.read()
                if frame is None:
                    continue
                try:
                    frame = _ensure_bgr(frame)
                except ValueError:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.detector_params)
                display = frame.copy()
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(display, corners, ids)
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, self.marker_length_m, self.K, self.dist_coeffs
                    )
                    rvec = rvecs[0].reshape(3)
                    tvec = tvecs[0].reshape(3)
                    R, _ = cv2.Rodrigues(rvec)
                    T = np.eye(4, dtype=float)
                    T[:3, :3] = R
                    T[:3, 3] = tvec
                    if show:
                        for rvec, tvec in zip(rvecs, tvecs):
                            cv2.drawFrameAxes(display, self.K, self.dist_coeffs, rvec, tvec, self.marker_length_m * 0.5)
                        # brief show
                        show_until = time.time() + 1.0
                        while time.time() < show_until:
                            cv2.imshow(win, display)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q') or key == 27:
                                break
                    return T
                if show:
                    cv2.imshow(win, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
        finally:
            if show:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
        raise RuntimeError("未检测到 ArUco 标记，无法获得 T_cam_aruco")

    def detect_and_cache(self, out_path: Path, timeout_s: float = 5.0, show: bool = True) -> np.ndarray:
        """Detect ArUco and cache; on timeout, fallback to repo-root T_zed_aruco.npy if available."""
        try:
            T = self.detect(timeout_s=timeout_s, show=show)
            np.save(str(out_path), T)
            return T
        except Exception:
            # Fallback: repo root cache
            try:
                project_root = Path(__file__).resolve().parents[2]
                fallback = project_root / "T_zed_aruco.npy"
                if fallback.exists():
                    T = np.load(str(fallback)).astype(np.float32)
                    if T.shape == (4, 4) or T.shape == (3, 4):
                        if T.shape == (3, 4):
                            T = np.vstack([T, np.array([0, 0, 0, 1], dtype=np.float32)])
                        print(f"[WARN] 超时未检测到 ArUco，使用根目录缓存: {fallback}")
                        try:
                            np.save(str(out_path), T)
                        except Exception:
                            pass
                        return T
            except Exception:
                pass
            raise

    def close(self):
        try:
            self._zed.close()
        except Exception:
            pass


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

    # 2) Detect and cache via reusable calibrator
    calib = ArucoCalibrator(marker_length_m=args.marker_length)
    try:
        T = calib.detect_and_cache(args.out, timeout_s=5.0, show=True)
    finally:
        calib.close()
    print(f"[OK] Cached T_zed_aruco to {args.out} with shape {T.shape}")


if __name__ == "__main__":
    main()
