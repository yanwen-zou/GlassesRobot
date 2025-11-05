#!/usr/bin/env python3
import socket

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

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

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
