#!/usr/bin/env python3
import socket

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class UDPListener(Node):
    def __init__(self):
        super().__init__('udp_listener')

        self.cmd_pub = self.create_publisher(String, '/control_cmd', 10)

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
