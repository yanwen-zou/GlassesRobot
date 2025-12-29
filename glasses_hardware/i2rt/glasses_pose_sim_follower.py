#!/usr/bin/env python3
"""
ROS2 node: subscribe /glasses_pose and command the YAM arm (simulation or real).

Incoming poses are treated as incremental commands; each delta is applied to the
robot TCP via IK. In sim mode a MuJoCo viewer mirrors the motion.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from glasses_hardware.hardware.my_device.i2rt_robo_sim import I2RTSim
from i2rt.robots.kinematics_mj import Kinematics
from i2rt.robots.utils import YAM_XML_PATH


RDF_TO_ROBOT = np.array(
    [
        [0.0, 0.0, 1.0],   # forward
        [-1.0, 0.0, 0.0],  # left
        [0.0, -1.0, 0.0],  # up
    ],
    dtype=np.float32,
)


def pose_to_vec(msg: PoseStamped) -> np.ndarray:
    return np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32)


def pose_to_rot(msg: PoseStamped) -> np.ndarray:
    q = (
        msg.pose.orientation.w,
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
    )
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


class GlassesPoseFollower(Node):
    """Mirror glasses pose deltas into the robot."""

    def __init__(self, mode: str = "sim", real_channel: str = "can0") -> None:
        super().__init__("glasses_pose_sim_follower")
        self.declare_parameter("pose_topic", "/glasses_pose")
        self.declare_parameter("translation_scale", 1.0)
        self.declare_parameter("deadband_m", 0.002)
        self.declare_parameter("max_step_m", 0.02)
        self.declare_parameter("rotation_scale", 1.0)
        self.declare_parameter("max_rot_rad", 0.1)
        self.declare_parameter("command_duration", 0.2)
        self.declare_parameter("command_steps", 20)

        pose_topic = self.get_parameter("pose_topic").value
        self._scale = float(self.get_parameter("translation_scale").value)
        self._deadband = abs(float(self.get_parameter("deadband_m").value))
        self._max_step = abs(float(self.get_parameter("max_step_m").value))
        self._rot_scale = float(self.get_parameter("rotation_scale").value)
        self._max_rot = abs(float(self.get_parameter("max_rot_rad").value))
        self._cmd_duration = float(self.get_parameter("command_duration").value)
        self._cmd_steps = int(self.get_parameter("command_steps").value)
        self._mode = mode
        self._viewer_stop: Optional[threading.Event] = None
        self._viewer_thread: Optional[threading.Thread] = None

        if self._mode == "sim":
            self._robot = I2RTSim()
            self._viewer_stop = threading.Event()
            self._viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
            self._viewer_thread.start()
        else:
            from glasses_hardware.hardware.my_device.i2rt_robo import I2RT

            self._robot = I2RT(channel=real_channel, zero_gravity_mode=False, home=True)

        self._kin = Kinematics(YAM_XML_PATH, "grasp_site")
        self._arm_dofs = min(6, self._robot.num_dofs())
        self._current_q = self._robot.current_joint_pos()
        self._target_pose = self._kin.fk(self._current_q[: self._arm_dofs])
        self._last_pose_vec: Optional[np.ndarray] = None
        self._last_rot_mat: Optional[np.ndarray] = None

        self.create_subscription(PoseStamped, pose_topic, self._pose_callback, 10)
        self.get_logger().info(
            f"Follower ({self._mode}) listening on {pose_topic} "
            f"(scale={self._scale:.3f}, deadband={self._deadband:.3f} m, max_step={self._max_step:.3f} m)",
        )

    def _pose_callback(self, msg: PoseStamped) -> None:
        vec = RDF_TO_ROBOT @ pose_to_vec(msg)
        rot = pose_to_rot(msg)
        if self._last_pose_vec is None or self._last_rot_mat is None:
            self._last_pose_vec = vec
            self._last_rot_mat = rot
            self.get_logger().info("Received first glasses pose, using as reference.")
            return

        delta = (vec - self._last_pose_vec) * self._scale
        distance = float(np.linalg.norm(delta))
        if distance < self._deadband:
            return

        delta = np.clip(delta, -self._max_step, self._max_step)
        if math.isclose(float(np.linalg.norm(delta)), 0.0, abs_tol=1e-6):
            return

        rot_delta = self._last_rot_mat.T @ rot
        print(
            "rot delta (glasses frame) xyz deg:",
            np.round(R.from_matrix(rot_delta).as_euler("xyz", degrees=True), 2),
        )
        rot_delta_robot = RDF_TO_ROBOT @ rot_delta @ RDF_TO_ROBOT.T
        rotvec = R.from_matrix(rot_delta_robot).as_rotvec() * self._rot_scale
        rotvec[0], rotvec[2] = rotvec[2], rotvec[0] # DEBUG: swap pitch and roll
        rot_norm = float(np.linalg.norm(rotvec))
        if self._max_rot > 0 and rot_norm > self._max_rot:
            rotvec *= self._max_rot / rot_norm
        print(
            "rot delta (robot frame) xyz deg:",
            np.round(R.from_matrix(rot_delta_robot).as_euler("xyz", degrees=True), 2),
        )

        new_pose = self._target_pose.copy()
        new_pose[:3, 3] += delta.astype(np.float32)
        new_pose[:3, :3] = new_pose[:3, :3] @ R.from_rotvec(rotvec).as_matrix().astype(np.float32)

        success, q_sol = self._kin.ik(new_pose, "grasp_site", verbose=False)
        if not success:
            self.get_logger().warning(f"IK failed for relative delta {delta}")
            return

        self._target_pose = new_pose
        self._current_q[: self._arm_dofs] = q_sol[: self._arm_dofs]
        try:
            self._robot.send_joint_pos_rad(self._current_q, duration=self._cmd_duration, steps=self._cmd_steps)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to command robot: {exc}")
            return

        self._last_pose_vec = vec
        self._last_rot_mat = rot

    def _viewer_loop(self) -> None:
        import mujoco
        import mujoco.viewer

        model, _ = self._robot.viewer_handles()
        view_data = mujoco.MjData(model)
        try:
            with mujoco.viewer.launch_passive(model, view_data, show_left_ui=False, show_right_ui=False) as viewer:
                viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
                while viewer.is_running() and self._viewer_stop and not self._viewer_stop.is_set():
                    self._robot.copy_state(view_data)
                    viewer.sync()
                    time.sleep(model.opt.timestep)
                if self._viewer_stop and self._viewer_stop.is_set():
                    viewer.close()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("MuJoCo viewer exited unexpectedly: %s", exc)

    def destroy_node(self) -> bool:
        if self._viewer_stop is not None:
            self._viewer_stop.set()
        if self._viewer_thread is not None and self._viewer_thread.is_alive():
            self._viewer_thread.join(timeout=1.0)
        if hasattr(self._robot, "close"):
            try:
                self._robot.close()
            except Exception:  # noqa: BLE001
                pass
        return super().destroy_node()


def parse_cli(argv: Optional[list[str]]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=("sim", "real"), default="sim", help="Control MuJoCo sim or real arm.")
    parser.add_argument("--channel", type=str, default="can0", help="CAN channel for real arm (mode=real).")
    return parser.parse_known_args(argv)


def main(args=None) -> None:
    cli_args, ros_args = parse_cli(args)
    rclpy.init(args=ros_args)
    node = GlassesPoseFollower(mode=cli_args.mode, real_channel=cli_args.channel)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
