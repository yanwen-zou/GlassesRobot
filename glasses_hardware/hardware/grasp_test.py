"""Simple grasp interaction script for Flexiv or UR5 + DH gripper."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pyDHgripper import AG95

here = Path(__file__).resolve()
project_root = here.parents[2]
src_root = project_root / "src"
for path in (project_root, src_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from glasses_hardware.hardware.my_device.robot import FlexivGripper, FlexivRobot, compose_global_delta
from glasses_hardware.hardware.ur5_robot import UR5
from MBA.utils.transformation import rotation_transform  # type: ignore
from egodata_eval.eval_constant import TASK_CHOICES


def select_delta_xyz(task_name: str) -> tuple[float, float, float]:
    task_to_delta = {
        "teapot": (-0.11, -0.14, 0.09),
        # "book": (-0.03, 0.03, 0.0),
        "book": (0.0, 0.0, 0.0),
        "sword": (0.08, -0.10, 0.08),
        "cup": (-0.3, -0.1, 0.0),
        "bread": (0.08, -0.03, -0.02),
    }
    return task_to_delta.get(task_name, (0.05, 0.0, 0.05))


def wait_for_command() -> Literal["p", "q"]:
    while True:
        user_input = input("输入'p'闭合夹爪退出，输入'q'回到home退出：").strip().lower()
        if user_input in {"p", "q"}:
            return user_input  # type: ignore[return-value]
        print("仅接受 'p' 或 'q'，请重新输入。")


def run_flexiv(task_name: str) -> None:
    robot = FlexivRobot(home=True)
    gripper = FlexivGripper(robot)

    robot.send_joint_pose(robot.home_joint_pos)
    center_pose = robot.get_tcp_pose().copy()

    z_rad = np.deg2rad(3.0)
    rz = np.array(
        [
            [np.cos(z_rad), -np.sin(z_rad), 0.0],
            [np.sin(z_rad), np.cos(z_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rotation = rz
    if task_name in {"sword", "bread"}:
        y_rad = np.deg2rad(30.0)
        ry = np.array(
            [
                [np.cos(y_rad), 0.0, np.sin(y_rad)],
                [0.0, 1.0, 0.0],
                [-np.sin(y_rad), 0.0, np.cos(y_rad)],
            ],
            dtype=np.float32,
        )
        rotation = rz @ ry

    rot6d = rotation_transform(rotation[None, ...], "matrix", "rotation_6d").squeeze(0)
    delta_rot = np.concatenate([np.zeros(3, dtype=np.float32), rot6d], axis=0)
    delta_x, delta_y, delta_z = select_delta_xyz(task_name)

    target_pose = center_pose.copy()
    target_pose[0] += delta_x
    target_pose[1] += delta_y
    target_pose[2] -= delta_z
    target_pose = compose_global_delta(target_pose, delta_rot)

    robot.send_tcp_pose(target_pose)
    gripper.move(gripper.max_width)

    cmd = wait_for_command()
    if cmd == "p":
        gripper.move(0.0)
    elif cmd == "q":
        robot.send_joint_pose(robot.home_joint_pos)


def run_ur5(task_name: str, robot_ip: str, gripper_port: str) -> None:
    robot = UR5(robot_ip=robot_ip, gui=False, debug=False)
    gripper = AG95(port=gripper_port)
    gripper.set_vel(80)
    gripper.set_force(50)

    try:
        robot.move_home()
        center_pose = robot.get_tcp_pose().copy()  # [x,y,z,rx,ry,rz]
        delta_x, delta_y, delta_z = select_delta_xyz(task_name)

        target_pose = center_pose.copy()
        target_pose[0] += delta_x
        target_pose[1] += delta_y
        target_pose[2] -= delta_z

        robot.move_tcp_pose(target_pose, pos_tolerance=0.002, max_steps=300)
        gripper.set_pos(1000)
        time.sleep(1.0)

        cmd = wait_for_command()
        if cmd == "p":
            gripper.set_pos(0)
        elif cmd == "q":
            robot.move_home()
    finally:
        try:
            gripper.ser.close()
        except Exception:
            pass
        robot.close()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Simple grasp interaction script for Flexiv or UR5.")
    ap.add_argument("--task", type=str, choices=TASK_CHOICES, default="book")
    ap.add_argument("--arm-hardware", type=str, choices=["flexiv", "ur5"], default="ur5")
    ap.add_argument("--ur5-robot-ip", type=str, default="192.168.2.102")
    ap.add_argument("--dh-gripper-port", type=str, default="/dev/ttyUSB0")
    args = ap.parse_args()

    if args.arm_hardware == "flexiv":
        run_flexiv(args.task)
    elif args.arm_hardware == "ur5":
        run_ur5(args.task, args.ur5_robot_ip, args.dh_gripper_port)
    else:
        print(f"未知机械臂后端: {args.arm_hardware}")
        sys.exit(1)


if __name__ == "__main__":
    main()
