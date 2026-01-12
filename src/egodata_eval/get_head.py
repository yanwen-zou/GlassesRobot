#!/usr/bin/env python3
"""
ROS2 node: listen to /glasses_pose and compute T_world_cam = T_world_glasses * T_tcp_zed.
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R


def _load_se3(path: str) -> np.ndarray:
    data = np.loadtxt(path).astype(np.float32)
    if data.shape == (4, 4):
        return data
    if data.shape == (3, 4):
        pad = np.array([[0, 0, 0, 1]], dtype=np.float32)
        return np.vstack([data, pad])
    raise ValueError(f"Invalid SE3 matrix shape {data.shape} in {path}")


def _pose_to_mat(msg: PoseStamped) -> np.ndarray:
    q = (
        msg.pose.orientation.w,
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
    )
    rot = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().astype(np.float32)
    trans = np.array(
        [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
        dtype=np.float32,
    )
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


class HeadPoseReader(Node):
    def __init__(self, pose_topic: str, tcp_zed_path: str, T_base_cam0: np.ndarray) -> None:
        super().__init__("egodata_get_head")
        self._T_tcp_zed = _load_se3(tcp_zed_path)
        self._T_base_cam0 = T_base_cam0.astype(np.float32)
        self._T_world_cam0: Optional[np.ndarray] = None
        self._last_base_cam: Optional[np.ndarray] = None
        self.create_subscription(PoseStamped, pose_topic, self._pose_callback, 10)
        self.get_logger().info(f"Listening on {pose_topic}; T_tcp_zed loaded from {tcp_zed_path}")

    def _pose_callback(self, msg: PoseStamped) -> None:
        T_world_glasses = _pose_to_mat(msg)
        T_world_cam = T_world_glasses @ self._T_tcp_zed
        if self._T_world_cam0 is None:
            self._T_world_cam0 = T_world_cam.copy()
        T_cam0_cam = np.linalg.inv(self._T_world_cam0) @ T_world_cam
        T_base_cam = self._T_base_cam0 @ T_cam0_cam
        self._last_base_cam = T_base_cam
        flat = np.array2string(T_base_cam, precision=4, suppress_small=True)
        # self.get_logger().info(f"T_base_cam:\n{flat}")

    def get_headpos(self, timeout_sec: float = 0.0) -> Optional[np.ndarray]:
        rclpy.spin_once(self, timeout_sec=timeout_sec)
        return self._last_base_cam


def parse_cli(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-topic", type=str, default="/glasses_pose", help="Pose topic to subscribe.")
    parser.add_argument(
        "--tcp-zed",
        type=str,
        default="glasses_hardware/calib/T_glasses_zed.txt",
        help="Path to T_tcp_zed (4x4 SE3).",
    )
    return parser.parse_args(argv)


def main(args: Optional[list[str]] = None) -> None:
    cli_args = parse_cli(args)
    rclpy.init(args=args)
    node = HeadPoseReader(cli_args.pose_topic, cli_args.tcp_zed)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
