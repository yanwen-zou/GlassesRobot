#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib.pyplot as plt

class TrajectoryRecorder(Node):
    def __init__(self):
        super().__init__('trajectory_recorder')

        self.recording = False
        self.glasses_traj = []
        self.zed_traj = []

        # 保存初始偏移
        self.glasses_origin = None
        self.zed_origin = None

        self.create_subscription(String, '/control_cmd', self.cmd_callback, 10)
        self.create_subscription(PoseStamped, '/glasses_pose', self.glasses_callback, 10)
        self.create_subscription(PoseStamped, '/zed_pose', self.zed_callback, 10)

    def cmd_callback(self, msg):
        if msg.data == "start":
            self.recording = True
            self.glasses_traj.clear()
            self.zed_traj.clear()
            self.glasses_origin = None
            self.zed_origin = None
            self.get_logger().info("🎬 Start recording...")
        elif msg.data == "stop":
            self.recording = False
            self.get_logger().info("🛑 Stop recording, saving files...")
            self.save_and_plot()

    def glasses_callback(self, msg: PoseStamped):
        if self.recording:
            t = self.get_clock().now().nanoseconds * 1e-9

            # 第一次收到数据，记录初始偏移
            if self.glasses_origin is None:
                self.glasses_origin = np.array([
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z
                ])

            p = np.array([msg.pose.position.x,
                          msg.pose.position.y,
                          msg.pose.position.z]) - self.glasses_origin

            self.glasses_traj.append([t, p[0], p[1], p[2],
                                      msg.pose.orientation.x,
                                      msg.pose.orientation.y,
                                      msg.pose.orientation.z,
                                      msg.pose.orientation.w])

    def zed_callback(self, msg: PoseStamped):
        if self.recording:
            t = self.get_clock().now().nanoseconds * 1e-9

            # 第一次收到数据，记录初始偏移（注意z也要反过来）
            if self.zed_origin is None:
                self.zed_origin = np.array([
                    msg.pose.position.y,      # 交换 x 和 y
                    msg.pose.position.x,
                    msg.pose.position.z     
                ])

            # 当前点（交换 xy + 翻转 z）
            p = np.array([
                msg.pose.position.y,
                msg.pose.position.x,
                msg.pose.position.z
            ]) - self.zed_origin

            self.zed_traj.append([t, p[0], p[1], p[2],
                                msg.pose.orientation.x,
                                msg.pose.orientation.y,
                                msg.pose.orientation.z,
                                msg.pose.orientation.w])


    def save_tum(self, filename, traj):
        with open(filename, "w") as f:
            for t in traj:
                f.write(" ".join([f"{x:.6f}" for x in t]) + "\n")
        self.get_logger().info(f"Saved {len(traj)} poses to {filename}")

    def save_and_plot(self):
        if self.glasses_traj:
            self.save_tum("glasses_trajectory.txt", self.glasses_traj)
        if self.zed_traj:
            self.save_tum("zed_trajectory.txt", self.zed_traj)

        if self.glasses_traj or self.zed_traj:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if self.glasses_traj:
                g = np.array(self.glasses_traj)
                ax.plot(g[:,1], g[:,2], g[:,3], 'r-', label="Glasses")
            if self.zed_traj:
                z = np.array(self.zed_traj)
                ax.plot(z[:,1], z[:,2], z[:,3], 'b-', label="ZED")
            ax.legend()
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryRecorder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
