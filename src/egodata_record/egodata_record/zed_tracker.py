#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import pyzed.sl as sl

class ZEDTracker(Node):
    def __init__(self):
        super().__init__('zed_tracker')

        self.pose_pub = self.create_publisher(PoseStamped, '/zed_pose', 10)

        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.coordinate_units = sl.UNIT.METER

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("ZED open failed.")
            exit(1)

        tracking_params = sl.PositionalTrackingParameters()
        if self.zed.enable_positional_tracking(tracking_params) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("Enable tracking failed.")
            self.zed.close()
            exit(1)

        self.runtime_params = sl.RuntimeParameters()
        self.zed_pose = sl.Pose()

        self.timer = self.create_timer(0.03, self.publish_pose)

    def publish_pose(self):
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.get_position(self.zed_pose, sl.REFERENCE_FRAME.WORLD)
            t = self.zed_pose.get_translation(sl.Translation()).get()

            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = t[0]
            pose.pose.position.y = t[1]
            pose.pose.position.z = t[2]
            pose.pose.orientation.w = 1.0

            self.pose_pub.publish(pose)

def main(args=None):
    rclpy.init(args=args)
    node = ZEDTracker()
    rclpy.spin(node)
    node.zed.close()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
