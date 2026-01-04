"""Simple grasp interaction script for Flexiv robot."""

from __future__ import annotations

import sys
from typing import Literal

import numpy as np

from glasses_hardware.hardware.my_device.robot import FlexivGripper, FlexivRobot, compose_relative_delta


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
    target_pose = compose_relative_delta(current_pose, delta)
    robot.send_tcp_pose(target_pose)


def wait_for_command() -> Literal["p", "q"]:
    """Block on user input until a valid command is received."""
    while True:
        user_input = input("输入'p'闭合夹爪退出，输入'q'回到home退出：").strip().lower()
        if user_input in {"p", "q"}:
            return user_input  # type: ignore[return-value]
        print("仅接受 'p' 或 'q'，请重新输入。")


def main() -> None:
    robot = FlexivRobot(home=True)
    gripper = FlexivGripper(robot)

    move_home(robot)
    center_pose = robot.get_tcp_pose().copy()

    delta_x = 0.1
    delta_y = 0.15
    delta_z = 0.05

    pose_forward = center_pose.copy()
    pose_forward[0] += delta_x
    pose_forward[1] += delta_y
    pose_forward[2] += -delta_z

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
