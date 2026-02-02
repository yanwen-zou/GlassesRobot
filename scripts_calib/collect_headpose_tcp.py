#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from glasses_hardware.hardware.my_device.robot import FlexivRobot


class HeadPoseSubscriber(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("headpose_subscriber")
        self.latest_msg: Optional[PoseStamped] = None
        self.create_subscription(PoseStamped, topic, self._on_pose, 10)

    def _on_pose(self, msg: PoseStamped) -> None:
        self.latest_msg = msg

    def get_latest(self) -> Optional[PoseStamped]:
        return self.latest_msg


def _wait_for_headpose(node: HeadPoseSubscriber, timeout_sec: float) -> Optional[PoseStamped]:
    start = time.time()
    while time.time() - start < timeout_sec:
        rclpy.spin_once(node, timeout_sec=0.05)
        msg = node.get_latest()
        if msg is not None:
            return msg
    return node.get_latest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move robot +/- 10cm along XYZ, record TCP + head pose samples."
    )
    parser.add_argument("--pose-topic", type=str, default="/glasses_pose")
    parser.add_argument("--delta-m", type=float, default=0.10, help="Offset per axis (meters).")
    parser.add_argument("--settle-sec", type=float, default=1.0, help="Wait after each move.")
    parser.add_argument("--interp-steps", type=int, default=30, help="Interpolation steps per move.")
    parser.add_argument("--step-sleep", type=float, default=0.05, help="Sleep per interpolation step.")
    parser.add_argument("--wait-headpose-sec", type=float, default=2.0, help="Wait for head pose after move.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calib/headpose_tcp_samples.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--no-return-center", action="store_true", help="Do not return to center between axes.")
    return parser.parse_args()


def _pose_to_list(msg: PoseStamped) -> list[float]:
    return [
        float(msg.pose.position.x),
        float(msg.pose.position.y),
        float(msg.pose.position.z),
        float(msg.pose.orientation.x),
        float(msg.pose.orientation.y),
        float(msg.pose.orientation.z),
        float(msg.pose.orientation.w),
    ]


def main() -> None:
    args = parse_args()
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rclpy.init(args=None)
    node = HeadPoseSubscriber(args.pose_topic)
    robot: Optional[FlexivRobot] = None

    records: list[list[object]] = []
    try:
        robot = FlexivRobot(home=True)
        center_pose = robot.get_tcp_pose().copy()

        # Wait for first head pose.
        first_msg = _wait_for_headpose(node, timeout_sec=5.0)
        if first_msg is None:
            raise RuntimeError("No head pose received; check head pose listener/topic.")

        axis_map = {"x": 0, "y": 1, "z": 2}
        move_idx = 0
        for axis in ("x", "y", "z"):
            idx = axis_map[axis]
            for direction in (1.0, -1.0):
                target_pose = center_pose.copy()
                target_pose[idx] += direction * float(args.delta_m)

                start_pose = robot.get_tcp_pose().astype(float)
                for step in range(int(args.interp_steps)):
                    alpha = (step + 1) / float(args.interp_steps)
                    interp_pose = start_pose.copy()
                    interp_pose[:3] = start_pose[:3] * (1.0 - alpha) + target_pose[:3] * alpha
                    robot.send_tcp_pose(interp_pose)
                    time.sleep(float(args.step_sleep))

                    head_msg = _wait_for_headpose(node, timeout_sec=float(args.wait_headpose_sec))
                    tcp_pose = robot.get_tcp_pose().astype(float)
                    head_pose = _pose_to_list(head_msg) if head_msg is not None else [float("nan")] * 7

                    record = [
                        move_idx,
                        axis,
                        "+" if direction > 0 else "-",
                        step,
                        time.time(),
                        *tcp_pose.tolist(),
                        *head_pose,
                    ]
                    records.append(record)
                    move_idx += 1

            if not args.no_return_center:
                robot.send_tcp_pose(center_pose)
                time.sleep(float(args.settle_sec))

    finally:
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "idx",
                    "axis",
                    "dir",
                    "step",
                    "timestamp_unix",
                    "tcp_x",
                    "tcp_y",
                    "tcp_z",
                    "tcp_rw",
                    "tcp_rx",
                    "tcp_ry",
                    "tcp_rz",
                    "head_x",
                    "head_y",
                    "head_z",
                    "head_qx",
                    "head_qy",
                    "head_qz",
                    "head_qw",
                ]
            )
            writer.writerows(records)

        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
