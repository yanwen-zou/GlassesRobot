#!/usr/bin/env python3
"""
Move the robot TCP to a point that is +10 cm along the detected ArUco marker's +Z axis,
verifying the chain of transforms using known SE3s:

- eih_camT.npy: SE3 from TCP to camera (T_tcp_cam)
- T_CAM_ARUCO_<id>.npy: SE3 from camera to ArUco (T_cam_aruco)

Target in ArUco frame is a pure translation [0, 0, +0.10] with identity rotation
(i.e., align TCP orientation to ArUco frame). The absolute target in world frame is:

    T_world_target = T_world_tcp_current @ T_tcp_cam @ T_cam_aruco @ T_aruco_target

where T_world_tcp_current is read live from the robot.
"""

import argparse
import numpy as np
import time

from pathlib import Path

from glasses_hardware.hardware.my_device.robot import FlexivRobot, _mat_to_pose7


def load_se3(path: Path) -> np.ndarray:
    T = np.load(str(path))
    if T.shape != (4, 4):
        raise ValueError(f"SE3 at {path} must be 4x4, got {T.shape}")
    return T.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Move TCP to ArUco +Z offset using calibrated SE3s.")
    parser.add_argument(
        "--eih-camT",
        type=Path,
        default=Path("glasses_hardware/calib/eih_camT.npy"),
        help="Path to eih_camT.npy (tcp->cam SE3)",
    )
    parser.add_argument(
        "--cam-aruco",
        type=Path,
        default=Path("glasses_hardware/calib/T_cam_aruco_51.npy"),
        help="Path to cam->aruco SE3 .npy (e.g., T_CAM_ARUCO_51.npy)",
    )
    parser.add_argument(
        "--offset-m",
        type=float,
        default=0.20,
        help="Offset distance along ArUco +Z in meters.",
    )


    args = parser.parse_args()


    # Load transforms
    T_tcp_cam = load_se3(args.eih_camT)
    T_cam_aruco = load_se3(args.cam_aruco)

    # Build desired pose in ArUco frame
    T_aruco_target = np.eye(4, dtype=np.float32)
    T_aruco_target[2, 3] = float(args.offset_m)

    robot = FlexivRobot()
    T_world_tcp = np.eye(4, dtype=np.float32)
    curr_pose7 = robot.get_tcp_pose()
    # Convert to matrix via quaternion
    T_world_tcp[:3, 3] = curr_pose7[:3]
    # Quaternion order is (rw, rx, ry, rz)
    from MBA.utils.transformation import rotation_transform  # type: ignore
    R_world_tcp = rotation_transform(curr_pose7[3:7][None, :], 'quaternion', 'matrix').squeeze(0)
    T_world_tcp[:3, :3] = R_world_tcp

    # Compose relative transform from current TCP to target TCP
    T_tcp_target = T_tcp_cam @ T_cam_aruco @ T_aruco_target

    T_tcp_target = T_tcp_target.copy()
    T_tcp_target[:3, :3] = np.eye(3, dtype=np.float32)

    T_world_target = T_world_tcp @ T_tcp_target
    target_pose7 = _mat_to_pose7(T_world_target)

    print("[INFO] eih_camT (tcp->cam):\n", T_tcp_cam)
    print("[INFO] T_cam_aruco:\n", T_cam_aruco)
    print("[INFO] Current T_world_tcp:\n", T_world_tcp)
    print("[INFO] Relative T_tcp_target:\n", T_tcp_target)
    print("[INFO] Target T_world_target:\n", T_world_target)

    # Save base<-aruco transform (world/base to aruco) for later replay usage
    # Here T_world_target equals T_base_aruco when aligning TCP with ArUco at zero offset.
    # Persist to calib directory as T_base_aruco.npy
    out_path = Path("glasses_hardware/calib/T_base_aruco.npy")
    np.save(str(out_path), T_world_target.astype(np.float32))
    print(f"[OK] Saved T_base_aruco to {out_path}")

    print("[INFO] Sending target TCP pose7 [x y z rw rx ry rz]:\n", np.round(target_pose7, 6))

    robot.send_tcp_pose(target_pose7)
    time.sleep(8)
    print("[DONE] Command sent. Verify robot reaches +Z offset relative to ArUco.")


if __name__ == "__main__":
    main()
