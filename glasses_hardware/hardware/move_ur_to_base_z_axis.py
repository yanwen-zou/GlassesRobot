import argparse
from pathlib import Path

import numpy as np

try:
    from ur5_robot import UR5
except ImportError:
    from glasses_hardware.hardware.ur5_robot import UR5


DEFAULT_BASE_Z_OFFSET = 0.0
DEFAULT_DELTA = 0.02


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -rotation.T @ translation
    return inv


def load_transform(path: Path) -> np.ndarray:
    transform = np.loadtxt(path, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform at {path}, got {transform.shape}.")
    return transform


def move_to_base_z_axis(robot: UR5, t_robot_base: np.ndarray, base_z_offset: float = 0.0) -> np.ndarray:
    current_tcp = robot.get_tcp_pose().copy()
    current_pos_robot = np.append(current_tcp[:3], 1.0)

    # T_robot_base is defined as the pose of the base frame in the robot frame,
    # i.e. it maps points from base frame into robot frame.
    t_base_robot = invert_transform(t_robot_base)
    current_pos_base = t_base_robot @ current_pos_robot

    target_pos_base = current_pos_base.copy()
    target_pos_base[0] = 0.0
    target_pos_base[1] = 0.0
    target_pos_base[2] += base_z_offset

    target_pos_robot = t_robot_base @ target_pos_base

    target_tcp = current_tcp.copy()
    target_tcp[:3] = target_pos_robot[:3]

    print("Current TCP in robot frame:", np.round(current_tcp[:3], 6).tolist())
    print("Current TCP in base frame:", np.round(current_pos_base[:3], 6).tolist())
    print(f"Applied base z offset: {base_z_offset:.6f} m")
    print("Target TCP in base frame:", np.round(target_pos_base[:3], 6).tolist())
    print("Target TCP in robot frame:", np.round(target_tcp[:3], 6).tolist())

    robot.move_tcp_pose(target_tcp, pos_tolerance=0.002, max_steps=300)
    return target_tcp


def move_along_base_axis(
    robot: UR5,
    reference_tcp_robot: np.ndarray,
    t_robot_base: np.ndarray,
    axis_idx: int,
    delta: float,
) -> np.ndarray:
    reference_pos_robot = np.append(reference_tcp_robot[:3], 1.0)
    t_base_robot = invert_transform(t_robot_base)
    reference_pos_base = t_base_robot @ reference_pos_robot

    target_pos_base = reference_pos_base.copy()
    target_pos_base[axis_idx] += delta
    target_pos_robot = t_robot_base @ target_pos_base

    target_tcp_robot = reference_tcp_robot.copy()
    target_tcp_robot[:3] = target_pos_robot[:3]

    axis_name = ["x", "y", "z"][axis_idx]
    print(f"Move +{delta:.6f} m along base {axis_name}-axis")
    print("Reference TCP in base frame:", np.round(reference_pos_base[:3], 6).tolist())
    print("Target TCP in base frame:", np.round(target_pos_base[:3], 6).tolist())
    print("Target TCP in robot frame:", np.round(target_tcp_robot[:3], 6).tolist())

    robot.move_tcp_pose(target_tcp_robot, pos_tolerance=0.002, max_steps=300)
    return target_tcp_robot


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Move UR TCP onto the base frame z-axis.")
    parser.add_argument(
        "--robot-ip",
        default="192.168.2.102",
        help="UR robot IP address.",
    )
    parser.add_argument(
        "--transform",
        default="glasses_hardware/calib/T_robot_base.txt",
        help="Path to T_robot_base transform file.",
    )
    parser.add_argument(
        "--base-z-offset",
        type=float,
        default=DEFAULT_BASE_Z_OFFSET,
        help="Additional offset along the base frame z-axis in meters.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable PyBullet GUI inside UR5 controller.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_DELTA,
        help="Translation delta along each base axis in meters.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    t_robot_base = load_transform(Path(args.transform))

    robot = UR5(robot_ip=args.robot_ip, gui=args.gui, debug=False)
    try:
        reference_tcp = move_to_base_z_axis(robot, t_robot_base, base_z_offset=args.base_z_offset)

        # axis_moves = [
        #     (0, args.delta, "+x"),
        #     (0, -args.delta, "-x"),
        #     (1, args.delta, "+y"),
        #     (1, -args.delta, "-y"),
        #     (2, args.delta, "+z"),
        #     (2, -args.delta, "-z"),
        # ]

        # for axis_idx, delta, label in axis_moves:
        #     print(f"Execute base-axis move {label}")
        #     move_along_base_axis(robot, reference_tcp, t_robot_base, axis_idx=axis_idx, delta=delta)
        #     print("Return to base z-axis reference TCP")
        #     robot.move_tcp_pose(reference_tcp, pos_tolerance=0.002, max_steps=300)
    finally:
        print("Closing Robot, current_tcp=", np.round(robot.get_tcp_pose()[:3], 6).tolist())
        robot.close()


if __name__ == "__main__":
    main()
