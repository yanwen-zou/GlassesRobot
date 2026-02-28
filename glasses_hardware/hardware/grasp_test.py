"""Simple grasp interaction script for Flexiv robot."""

from __future__ import annotations

import sys
from typing import Literal

import numpy as np

from glasses_hardware.hardware.my_device.robot import FlexivGripper, FlexivRobot, compose_relative_delta, compose_global_delta
from MBA.utils.transformation import rotation_transform  # type: ignore
from egodata_eval.eval_constant import TASK_CHOICES


def move_home(robot: FlexivRobot) -> None:
    """Send robot back to its predefined joint-space home."""
    robot.send_joint_pose(robot.home_joint_pos)


def move_relative_x(robot: FlexivRobot, distance_m: float) -> None:
    """Move TCP along +X of the base frame by the provided distance."""
    current_pose = robot.get_tcp_pose()
    delta = np.array([
        distance_m,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ], dtype=np.float32)  # xyz + rotation6d (two column vectors of rotation matrix)
    target_pose = compose_global_delta(current_pose, delta)
    robot.send_tcp_pose(target_pose)


def wait_for_command() -> Literal["p", "q"]:
    """Block on user input until a valid command is received."""
    while True:
        user_input = input("输入'p'闭合夹爪退出，输入'q'回到home退出：").strip().lower()
        if user_input in {"p", "q"}:
            return user_input  # type: ignore[return-value]
        print("仅接受 'p' 或 'q'，请重新输入。")


def _select_delta_xyz(task_name: str) -> tuple[float, float, float]:
    # Placeholder values per task (update later as needed).
    task_to_delta = {
        "teapot": (-0.33, -0.12, 0.07),
        "book": (-0.03, 0.03, 0),
        "sword": (0.08, -0.10, 0.08),
        "cup": (-0.3, -0.1, 0),
        "bread": (0.08, -0.06, -0.02)
    }
    base_dx, base_dy, base_dz = task_to_delta.get(task_name, (0.05, 0.0, 0.05))
    dx = float(np.random.uniform(base_dx - 0.03, base_dx + 0.02))
    dy = float(np.random.uniform(base_dy - 0.07, base_dy + 0.03))
    return dx, dy, base_dz


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Simple grasp interaction script for Flexiv robot.")
    ap.add_argument("--task", type=str, choices=TASK_CHOICES, default="book")
    args = ap.parse_args()

    robot = FlexivRobot(home=True)
    gripper = FlexivGripper(robot)

    move_home(robot)
    center_pose = robot.get_tcp_pose().copy()
    # Base rotation about Z
    z_rad = np.deg2rad(3.0)
    Rz = np.array(
        [
            [np.cos(z_rad), -np.sin(z_rad), 0.0],
            [np.sin(z_rad),  np.cos(z_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    R = Rz
    if args.task == "sword" or args.task == "bread":
        y_rad = np.deg2rad(30.0)
        Ry = np.array(
            [
                [np.cos(y_rad), 0.0, np.sin(y_rad)],
                [0.0, 1.0, 0.0],
                [-np.sin(y_rad), 0.0, np.cos(y_rad)],
            ],
            dtype=np.float32,
        )
        R = Rz @ Ry

    rot6d = rotation_transform(R[None, ...], "matrix", "rotation_6d").squeeze(0)
    delta_rot = np.concatenate([np.zeros(3, dtype=np.float32), rot6d], axis=0)
    delta_x, delta_y, delta_z = _select_delta_xyz(args.task)

    pose_forward = center_pose.copy()
    pose_forward[0] += delta_x
    pose_forward[1] += delta_y
    pose_forward[2] += -delta_z

    pose_forward = compose_global_delta(pose_forward, delta_rot)

    robot.send_tcp_pose(pose_forward)
    gripper.move(gripper.max_width)

    cmd = wait_for_command()

    if cmd == "p":
        gripper.move(0.0)
    elif cmd == "q":
        move_home(robot)
    else:
        print(f"未知指令: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
