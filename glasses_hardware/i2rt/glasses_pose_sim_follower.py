#!/usr/bin/env python3
"""
ROS2 node: subscribe /glasses_pose and command the MuJoCo YAM arm by relative motion.

The incoming PoseStamped stream is treated as an incremental command source.
Each translation delta is scaled/clipped and applied to the robot TCP in simulation.
"""

from __future__ import annotations

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
    """Extract xyz as numpy vector."""
    return np.array(
        [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
        dtype=np.float32,
    )


def pose_to_rot(msg: PoseStamped) -> np.ndarray:
    q = (
        msg.pose.orientation.w,
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
    )
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


class GlassesPoseSimFollower(Node):
    """Mirror glasses relative translations into a MuJoCo-controlled arm."""

    def __init__(self) -> None:
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

        self._robot = I2RTSim()
        self._kin = Kinematics(YAM_XML_PATH, "grasp_site")
        self._arm_dofs = min(6, self._robot.num_dofs())
        self._current_q = self._robot.current_joint_pos()
        self._target_pose = self._kin.fk(self._current_q[: self._arm_dofs])
        self._last_pose_vec: Optional[np.ndarray] = None
        self._last_rot_mat: Optional[np.ndarray] = None
        self._viewer_stop = threading.Event()
        self._viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self._viewer_thread.start()

        self.create_subscription(PoseStamped, pose_topic, self._pose_callback, 10)
        self.get_logger().info(
            f"Sim follower listening on {pose_topic} (scale={self._scale:.3f}, deadband={self._deadband:.3f} m, max_step={self._max_step:.3f} m)",
        )

    def _pose_callback(self, msg: PoseStamped) -> None:
        self.get_logger().debug(f"Received glasses pose: {msg}")
        vec = RDF_TO_ROBOT @ pose_to_vec(msg)
        rot = RDF_TO_ROBOT @ pose_to_rot(msg) @ RDF_TO_ROBOT
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
        rotvec = R.from_matrix(rot_delta).as_rotvec() * self._rot_scale
        rot_norm = float(np.linalg.norm(rotvec))
        if rot_norm > self._max_rot > 0:
            rotvec = rotvec * (self._max_rot / rot_norm)

        new_pose = self._target_pose.copy()
        new_pose[:3, 3] += delta.astype(np.float32)
        if rot_norm > 1e-6:
            new_pose[:3, :3] = new_pose[:3, :3] @ R.from_rotvec(rotvec).as_matrix().astype(np.float32)
        success, q_sol = self._kin.ik(new_pose, "grasp_site", verbose=False)
        if not success:
            self.get_logger().warning(f"IK failed for relative delta {delta}")
            return

        self._target_pose = new_pose
        self._current_q[: self._arm_dofs] = q_sol[: self._arm_dofs]
        try:
            self._robot.send_joint_pos_rad(
                self._current_q, duration=self._cmd_duration, steps=self._cmd_steps
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to command MuJoCo robot: {exc}")
            return

        self._last_pose_vec = vec
        self._last_rot_mat = rot

    def _viewer_loop(self) -> None:
        import glfw
        import mujoco

        model, data = self._robot.mj_handles()
        if not glfw.init():
            self.get_logger().error("Failed to initialize GLFW for visualization.")
            return
        window = glfw.create_window(1280, 960, "YAM MuJoCo Viewer", None, None)
        if window is None:
            glfw.terminate()
            self.get_logger().error("Failed to create GLFW window for visualization.")
            return
        glfw.make_context_current(window)
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        scene = mujoco.MjvScene(model, maxgeom=2000)
        ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        cam.distance = 2.0
        cam.azimuth = 90
        cam.elevation = -35
        try:
            while not glfw.window_should_close(window) and not self._viewer_stop.is_set():
                viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
                mujoco.mjv_updateScene(
                    model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
                )
                mujoco.mjr_render(viewport, scene, ctx)
                glfw.swap_buffers(window)
                glfw.poll_events()
                time.sleep(model.opt.timestep)
        finally:
            glfw.terminate()

    def destroy_node(self) -> bool:
        self._viewer_stop.set()
        if self._viewer_thread.is_alive():
            self._viewer_thread.join(timeout=1.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlassesPoseSimFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
