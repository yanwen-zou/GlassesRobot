import tempfile
import math
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


class UR5:
    """UR5 hardware wrapper with joint control and PyBullet-based TCP IK."""

    def __init__(
        self,
        robot_ip: str = "192.168.2.102",
        urdf_path: str | None = None,
        gui: bool = False,
        home_joint_pose: np.ndarray | list[float] | None = None,
        move_to_home: bool = True,
        debug: bool = True,
        control_dt: float = 0.05,
        max_joint_speed: float = 0.3,
        servo_lookahead: float = 0.1,
        servo_gain: int = 300,
    ) -> None:
        self.robot_ip = robot_ip
        self.control_dt = float(control_dt)
        self.max_joint_speed = float(max_joint_speed)
        self.servo_lookahead = float(servo_lookahead)
        self.servo_gain = int(servo_gain)
        self.debug = bool(debug)

        hardware_dir = Path(__file__).resolve().parent
        self.urdf_dir = hardware_dir / "ur_description"
        self.urdf_path = Path(urdf_path) if urdf_path is not None else self.urdf_dir / "urdf" / "ur5_robot.urdf"

        self.rtde_c = RTDEControlInterface(self.robot_ip)
        self.rtde_r = RTDEReceiveInterface(self.robot_ip)
        self.home_joint_pose = np.asarray(
            # [-0.19338876, -1.50480682,  1.76545811, -2.13313371, -1.21471769, -0.01526481],
            [-0.07230121, -1.66861946,  2.21617556, -2.91870243, -1.74901325,  0.28110909],
            dtype=np.float64,
        )
        print(f"Home joint pose: {self.home_joint_pose}")

        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot_id = p.loadURDF(self._build_pybullet_urdf(), useFixedBase=True, physicsClientId=self.client)

        self.joint_indices: list[int] = []
        self.joint_lower: list[float] = []
        self.joint_upper: list[float] = []
        self.joint_ranges: list[float] = []
        self.ee_link_index = -1

        for joint_idx in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            joint_info = p.getJointInfo(self.robot_id, joint_idx, physicsClientId=self.client)
            if joint_info[12].decode("utf-8") == "ee_link":
                self.ee_link_index = joint_idx
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(joint_idx)
                self.joint_lower.append(joint_info[8])
                self.joint_upper.append(joint_info[9])
                self.joint_ranges.append(joint_info[9] - joint_info[8])

        if self.ee_link_index < 0:
            raise RuntimeError("Failed to find ee_link in URDF.")

        if move_to_home:
            print("Current joint pose:", self.get_joint_pos())
            self.send_joint_pose(self.home_joint_pose)
        self.sync_pybullet()

    def _build_pybullet_urdf(self) -> str:
        urdf_text = self.urdf_path.read_text(encoding="utf-8")
        urdf_text = urdf_text.replace("package://ur_description/", f"{self.urdf_dir.as_posix()}/")
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".urdf",
            prefix="ur5_robot_pybullet_",
            delete=False,
            encoding="utf-8",
        ) as fp:
            fp.write(urdf_text)
            return fp.name

    @staticmethod
    def _rotvec_to_quat(rotvec: np.ndarray) -> tuple[float, float, float, float]:
        angle = float(np.linalg.norm(rotvec))
        if angle < 1e-12:
            return 0.0, 0.0, 0.0, 1.0
        axis = rotvec / angle
        half = angle * 0.5
        sin_half = math.sin(half)
        return (
            float(axis[0] * sin_half),
            float(axis[1] * sin_half),
            float(axis[2] * sin_half),
            float(math.cos(half)),
        )

    @staticmethod
    def _real_to_pybullet_tcp(tcp: np.ndarray) -> np.ndarray:
        mapped = np.asarray(tcp, dtype=np.float64).copy()
        mapped[0] *= -1.0
        mapped[1] *= -1.0
        mapped[3] *= -1.0
        mapped[4] *= -1.0
        return mapped

    @staticmethod
    def _pybullet_to_real_tcp(tcp: np.ndarray) -> np.ndarray:
        mapped = np.asarray(tcp, dtype=np.float64).copy()
        mapped[0] *= -1.0
        mapped[1] *= -1.0
        mapped[3] *= -1.0
        mapped[4] *= -1.0
        return mapped

    def _reset_pybullet_joints(self, joints: np.ndarray) -> None:
        for joint_idx, joint_pos in zip(self.joint_indices, joints.tolist()):
            p.resetJointState(self.robot_id, joint_idx, float(joint_pos), physicsClientId=self.client)

    def _get_pybullet_tcp_pose(self) -> np.ndarray:
        link_state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        pos = np.asarray(link_state[4], dtype=np.float64)
        quat = np.asarray(link_state[5], dtype=np.float64)
        rotvec = np.asarray(p.getAxisAngleFromQuaternion(quat.tolist())[0], dtype=np.float64)
        angle = float(p.getAxisAngleFromQuaternion(quat.tolist())[1])
        return np.concatenate([pos, rotvec * angle], axis=0)

    def _log(self, message: str) -> None:
        if self.debug:
            print(message)

    def _limit_joint_step(self, target_joints: np.ndarray, actual_joints: np.ndarray) -> np.ndarray:
        max_step = self.max_joint_speed * self.control_dt
        delta = np.clip(target_joints - actual_joints, -max_step, max_step)
        return actual_joints + delta

    def sync_pybullet(self) -> None:
        self._reset_pybullet_joints(self.get_joint_pos())

    def get_joint_pos(self) -> np.ndarray:
        return np.asarray(self.rtde_r.getActualQ(), dtype=np.float64)

    def get_joint_vel(self) -> np.ndarray:
        return np.asarray(self.rtde_r.getActualQd(), dtype=np.float64)

    def get_tcp_pose(self) -> np.ndarray:
        """Return UR TCP pose as [x, y, z, rx, ry, rz] with rotation vector."""
        return np.asarray(self.rtde_r.getActualTCPPose(), dtype=np.float64)

    def send_joint_pose(
        self,
        joints: np.ndarray | list[float],
        speed: float = 0.5,
        acceleration: float = 0.5,
        asynchronous: bool = False,
    ) -> None:
        target = np.asarray(joints, dtype=np.float64)
        if target.shape[0] != len(self.joint_indices):
            raise ValueError(f"Expected {len(self.joint_indices)} joints, got {target.shape[0]}.")
        self.rtde_c.moveJ(target.tolist(), speed, acceleration, asynchronous)
        self._reset_pybullet_joints(target)

    def move_home(self, speed: float = 0.5, acceleration: float = 0.5, asynchronous: bool = False) -> None:
        self.send_joint_pose(self.home_joint_pose, speed=speed, acceleration=acceleration, asynchronous=asynchronous)

    def send_tcp_pose(self, tcp: np.ndarray | list[float]) -> np.ndarray:
        target_tcp = np.asarray(tcp, dtype=np.float64)
        if target_tcp.shape[0] != 6:
            raise ValueError("Expected TCP pose with 6 values: [x, y, z, rx, ry, rz].")
        pb_target_tcp = self._real_to_pybullet_tcp(target_tcp)

        actual_joints = self.get_joint_pos()
        actual_tcp = self.get_tcp_pose()
        self._reset_pybullet_joints(actual_joints)
        pybullet_tcp_before = self._pybullet_to_real_tcp(self._get_pybullet_tcp_pose())

        ik_joint_targets = np.asarray(
            p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link_index,
                targetPosition=pb_target_tcp[:3].tolist(),
                lowerLimits=self.joint_lower,
                upperLimits=self.joint_upper,
                jointRanges=self.joint_ranges,
                restPoses=actual_joints.tolist(),
                maxNumIterations=200,
                residualThreshold=1e-4,
                physicsClientId=self.client,
            )[: len(self.joint_indices)],
            dtype=np.float64,
        )
        joint_targets = self._limit_joint_step(ik_joint_targets, actual_joints)

        self._reset_pybullet_joints(ik_joint_targets)
        pybullet_tcp_ik = self._pybullet_to_real_tcp(self._get_pybullet_tcp_pose())
        self._reset_pybullet_joints(joint_targets)
        pybullet_tcp_limited = self._pybullet_to_real_tcp(self._get_pybullet_tcp_pose())

        # self._log(
        #     "[IK] target_tcp_xyz="
        #     f"{np.round(target_tcp[:3], 5).tolist()} "
        #     "actual_tcp_xyz="
        #     f"{np.round(actual_tcp[:3], 5).tolist()} "
        #     "pb_before_xyz="
        #     f"{np.round(pybullet_tcp_before[:3], 5).tolist()}"
        # )
        # self._log(
        #     "[IK] target_rotvec="
        #     f"{np.round(target_tcp[3:], 5).tolist()} "
        #     "actual_rotvec="
        #     f"{np.round(actual_tcp[3:], 5).tolist()} "
        #     "pb_before_rotvec="
        #     f"{np.round(pybullet_tcp_before[3:], 5).tolist()}"
        # )
        # self._log(
        #     "[IK] ik_joint_delta="
        #     f"{np.round(ik_joint_targets - actual_joints, 5).tolist()} "
        #     "limited_joint_delta="
        #     f"{np.round(joint_targets - actual_joints, 5).tolist()}"
        # )
        # self._log(
        #     "[IK] pb_ik_xyz="
        #     f"{np.round(pybullet_tcp_ik[:3], 5).tolist()} "
        #     "pb_limited_xyz="
        #     f"{np.round(pybullet_tcp_limited[:3], 5).tolist()} "
        #     "target_error_before="
        #     f"{float(np.linalg.norm(target_tcp[:3] - actual_tcp[:3])):.6f} "
        #     "target_error_pb_ik="
        #     f"{float(np.linalg.norm(target_tcp[:3] - pybullet_tcp_ik[:3])):.6f}"
        # )

        self.rtde_c.servoJ(
            joint_targets.tolist(),
            0.5,
            0.5,
            self.control_dt,
            self.servo_lookahead,
            self.servo_gain,
        )
        self._reset_pybullet_joints(joint_targets)
        return joint_targets

    def move_tcp_pose(
        self,
        tcp: np.ndarray | list[float],
        pos_tolerance: float = 0.002,
        max_steps: int = 200,
    ) -> None:
        target_tcp = np.asarray(tcp, dtype=np.float64)
        for _ in range(max_steps):
            current_tcp = self.get_tcp_pose()
            pos_error = float(np.linalg.norm(target_tcp[:3] - current_tcp[:3]))
            self._log(
                "[MOVE_TCP] current_xyz="
                f"{np.round(current_tcp[:3], 5).tolist()} "
                "target_xyz="
                f"{np.round(target_tcp[:3], 5).tolist()} "
                f"pos_error={pos_error:.6f}"
            )
            if pos_error <= pos_tolerance:
                break
            self.send_tcp_pose(target_tcp)
            time.sleep(self.control_dt)

    def stop(self) -> None:
        try:
            self.rtde_c.servoStop()
        except Exception:
            pass

    def close(self) -> None:
        self.stop()
        try:
            self.rtde_c.stopScript()
        except Exception:
            pass
        if self.client is not None and p.isConnected(self.client):
            p.disconnect(self.client)
            self.client = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def move_tcp_linear(robot: UR5, target_tcp: np.ndarray, num_steps: int = 100) -> None:
    start_tcp = robot.get_tcp_pose().copy()
    for alpha in np.linspace(0.0, 1.0, num_steps):
        interp_tcp = (1.0 - alpha) * start_tcp + alpha * target_tcp
        robot.move_tcp_pose(interp_tcp, pos_tolerance=0.002, max_steps=20)


def main() -> None:
    robot = UR5()
    try:
        current_tcp = robot.get_tcp_pose().copy()
        delta = 0.05
        axis_names = ["X", "Y", "Z"]

        print("Current TCP:", current_tcp)
        for axis_idx, axis_name in enumerate(axis_names):
            target_tcp = current_tcp.copy()
            target_tcp[axis_idx] += delta

            print(f"Move +{delta:.3f} m along {axis_name}")
            move_tcp_linear(robot, target_tcp)

            time.sleep(1.0)

            print("Return to start TCP")
            move_tcp_linear(robot, current_tcp)

            time.sleep(1.0)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
