#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import socket
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import time

class UDPListener(Node):
    def __init__(self):
        super().__init__('udp_listener')

        self.cmd_pub = self.create_publisher(String, '/control_cmd', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/glasses_pose', 10)

        self.UDP_IP = "0.0.0.0"
        self.UDP_PORT = 5005
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP, self.UDP_PORT))
        self.sock.setblocking(False)

        self.timer = self.create_timer(0.01, self.poll_udp)

    def poll_udp(self):
        try:
            data, _ = self.sock.recvfrom(1024)
            msg = data.decode().strip().split(",")
            if msg[0] in ["start", "stop"]:
                self.cmd_pub.publish(String(data=msg[0]))
            elif msg[0] == "pose":
                x, y, z = map(float, msg[1:4])
                qx, qy, qz, qw = map(float, msg[4:8])
                pose = PoseStamped()
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = z
                pose.pose.orientation.x = qx
                pose.pose.orientation.y = qy
                pose.pose.orientation.z = qz
                pose.pose.orientation.w = qw
                pose.header.stamp = self.get_clock().now().to_msg()
                self.pose_pub.publish(pose)
        except BlockingIOError:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = UDPListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
