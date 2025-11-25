#!/usr/bin/env python3
import socket

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class HeadposeListener(Node):
    def __init__(self):
        super().__init__('headpos_listener')

        self.pose_pub = self.create_publisher(PoseStamped, '/glasses_pose', 10)

        self.UDP_IP = '0.0.0.0'
        self.UDP_PORT = 5006
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP, self.UDP_PORT))
        self.sock.setblocking(False)

        self._pose_broadcast_rate_hz = 60.0
        self._latest_pose: PoseStamped | None = None

        self.poll_timer = self.create_timer(0.01, self.poll_udp)
        self.broadcast_timer = self.create_timer(
            1.0 / self._pose_broadcast_rate_hz, self.publish_latest_pose
        )

    def poll_udp(self) -> None:
        try:
            data, _ = self.sock.recvfrom(1024)
        except BlockingIOError:
            return

        message = data.decode().strip()
        parts = message.split(',')
        if not parts:
            return
        if parts[0] != 'pose' or len(parts) < 8:
            self.get_logger().debug(f'Ignoring UDP payload: {message}')
            return

        try:
            x, y, z = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
        except ValueError:
            self.get_logger().warning(f'Failed to parse head pose payload: {message}')
            return

        # 左手系 -> 右手系：位置 y 取反；姿态做同样的镜像变换
        S = np.diag([1.0, -1.0, 1.0])
        R_lh = quaternion_to_matrix(np.array([qw, qx, qy, qz], dtype=np.float64))
        R_rh = S @ R_lh @ S
        qw_r, qx_r, qy_r, qz_r = matrix_to_quaternion(R_rh)
        qx_r, qy_r, qz_r, qw_r = [round(val, 2) for val in (qx_r, qy_r, qz_r, qw_r)]

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = -y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx_r
        pose.pose.orientation.y = qy_r
        pose.pose.orientation.z = qz_r
        pose.pose.orientation.w = qw_r

        self.pose_pub.publish(pose)
        self._latest_pose = pose
        

    def publish_latest_pose(self) -> None:
        if self._latest_pose is None:
            return

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self._latest_pose.pose.position.x
        pose.pose.position.y = self._latest_pose.pose.position.y
        pose.pose.position.z = self._latest_pose.pose.position.z
        pose.pose.orientation.x = self._latest_pose.pose.orientation.x
        pose.pose.orientation.y = self._latest_pose.pose.orientation.y
        pose.pose.orientation.z = self._latest_pose.pose.orientation.z
        pose.pose.orientation.w = self._latest_pose.pose.orientation.w
        self.pose_pub.publish(pose)


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    qw, qx, qy, qz = quaternion
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    x, y, z = qx, qy, qz
    w = qw
    xx, xy, xz = x * x, x * y, x * z
    yy, yz, zz = y * y, y * z, z * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - s * (yy + zz), s * (xy - wz), s * (xz + wy)],
            [s * (xy + wz), 1.0 - s * (xx + zz), s * (yz - wx)],
            [s * (xz - wy), s * (yz + wx), 1.0 - s * (xx + yy)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation matrix -> quaternion (w, x, y, z)."""
    m = np.asarray(R, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


def main(args=None):
    rclpy.init(args=args)
    node = HeadposeListener()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
