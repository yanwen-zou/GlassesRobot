import time
import numpy as np
from pathlib import Path

# Import Flexiv RDK Python library
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../flexiv_rdk/lib_py"))
import flexivrdk
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from get_ip import get_local_ip
"""
Ensure MBA package path precedes local 'utils' to avoid import shadowing.
MBA/utils/transformation.py imports 'from utils import rotation_utils'. If our local
glasses_hardware/hardware/utils.py shadows it, import fails. Prepend '<repo>/MBA'.
"""
_here = Path(__file__).resolve()
_repo_root = _here.parents[3]
sys.path.insert(0, str(_repo_root / "MBA"))
from MBA.utils.transformation import rotation_transform, xyz_rot_to_mat  # type: ignore


def _pose7_to_mat(pose7: np.ndarray) -> np.ndarray:
    """Convert pose [x,y,z,rw,rx,ry,rz] to 4x4 matrix."""
    pose7 = np.asarray(pose7, dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = pose7[:3]
    # flexiv uses (rw, rx, ry, rz)
    quat = pose7[3:7]
    # convert quaternion to rotation matrix via rotation_transform
    R = rotation_transform(quat[None, :], 'quaternion', 'matrix').squeeze(0)
    T[:3, :3] = R
    return T


def _mat_to_pose7(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 matrix to pose [x,y,z,rw,rx,ry,rz]."""
    T = np.asarray(T, dtype=np.float32)
    xyz = T[:3, 3]
    quat = rotation_transform(T[:3, :3][None, ...], 'matrix', 'quaternion').squeeze(0)
    return np.concatenate([xyz, quat], axis=0)


def compose_relative_delta(curr_pose7: np.ndarray, delta_xyz_rot6d: np.ndarray) -> np.ndarray:
    """Compose current tcp pose with a relative delta (xyz+rot6d) and return target pose7.

    - curr_pose7: [x,y,z,rw,rx,ry,rz]
    - delta_xyz_rot6d: [dx,dy,dz,r6d...]
    """
    T_curr = _pose7_to_mat(curr_pose7)
    # xyz_rot_to_mat expects concatenated xyz + rotation representation
    T_delta = xyz_rot_to_mat(delta_xyz_rot6d.astype(np.float32), rotation_rep='rotation_6d')
    T_target = T_curr @ T_delta
    return _mat_to_pose7(T_target)




class ModeMap:
    idle = "IDLE"
    cart_impedance_online = "NRT_CARTESIAN_MOTION_FORCE"
    joint = "NRT_JOINT_POSITION"


class FlexivRobot:
    """
    Flexiv Robot Control Class.
    """
    logger_name = "FlexivRobot"

    def __init__(self, robot_ip_address='192.168.2.100', pc_ip_address=None, default_pose=[0.6, 0, 0.2, 0, -0.5**0.5, 0.5**0.5, 0], home = True):
        if pc_ip_address is None:
            pc_ip_address = get_local_ip(robot_ip_address)
        
        self.robot_states = flexivrdk.RobotStates()
        self.log = flexivrdk.Log()
        self.mode = flexivrdk.Mode
        self.robot = flexivrdk.Robot(robot_ip_address, pc_ip_address)
        self.default_pose = default_pose
        self.home_pose = self.default_pose
        self.home_joint_pos = [0.218,0.211,-0.075,1.941,0.001,0.414,0.169] # Predefined home joint positions for book picking
        self.init_robot(home)
        self.init_pose = self.get_tcp_pose()
    
    def init_robot(self, home=True):
        log = self.log
        mode = self.mode
        robot = self.robot

        # Clear fault on robot server if any
        if robot.isFault():
            log.warn("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            robot.clearFault()
            time.sleep(2)
            # Check again
            if robot.isFault():
                log.error("Fault cannot be cleared, exiting ...")
                return
            log.info("Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...")
        robot.enable()

        # Wait for the robot to become operational
        while not robot.isOperational():
            time.sleep(1)

        log.info("Robot is now operational")

        # Move robot to home pose
        log.info("Moving to home pose")
        if home:
            self.send_joint_pose(self.home_joint_pos)
        time.sleep(4)
        # self.send_tcp_pose(self.home_pose)
        # time.sleep(4)

        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        # Zero Force-torque Sensor
        # =========================================================================================
        # IMPORTANT: must zero force/torque sensor offset for accurate force/torque measurement
        robot.executePrimitive("ZeroFTSensor()")

        # WARNING: during the process, the robot must not contact anything, otherwise the result
        # will be inaccurate and affect following operations
        log.warn(
            "Zeroing force/torque sensors, make sure nothing is in contact with the robot"
        )

        # Wait for primitive completion
        while robot.isBusy():
            time.sleep(1)
        log.info("Sensor zeroing complete")

    def enable(self, max_time=10):
        """Enable robot after emergency button is released."""
        self.robot.enable()
        tic = time.time()
        while not self.is_operational():
            if time.time() - tic > max_time:
                return "Robot enable failed"
            time.sleep(0.01)
        return

    def _get_robot_status(self):
        self.robot.getRobotStates(self.robot_states)
        return self.robot_states

    def mode_mapper(self, mode):
        assert mode in ModeMap.__dict__.keys(), "unknown mode name: %s" % mode
        return getattr(self.mode, getattr(ModeMap, mode))

    def get_control_mode(self):
        return self.robot.getMode()

    def set_control_mode(self, mode):
        control_mode = self.mode_mapper(mode)
        self.robot.setMode(control_mode)

    def switch_mode(self, mode, sleep_time=0.01):
        """switch to different control modes.

        Args:
            mode: 'idle', 'cart_impedance_online'
            sleep_time: sleep time to control mode switch time

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        if self.get_control_mode() == self.mode_mapper(mode):
            return

        while self.get_control_mode() != self.mode_mapper("idle"):
            self.set_control_mode("idle")
            time.sleep(sleep_time)
        while self.get_control_mode() != self.mode_mapper(mode):
            self.set_control_mode(mode)
            time.sleep(sleep_time)

        print("[Robot] Set mode: {}".format(str(self.get_control_mode())))

    def clear_fault(self):
        self.robot.clearFault()

    def is_fault(self):
        """Check if robot is in FAULT state."""
        return self.robot.isFault()

    def is_stopped(self):
        """Check if robot is stopped."""
        return self.robot.isStopped()

    def is_connected(self):
        """return if connected.

        Returns: True/False
        """
        return self.robot.isConnected()

    def is_operational(self):
        """Check if robot is operational."""
        return self.robot.isOperational()

    def get_tcp_pose(self):
        """get current robot's tool pose in world frame.

        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().tcpPose)

    def get_tcp_vel(self):
        """get current robot's tool velocity in world frame.

        Returns:
            7-dim list consisting of (vx,vy,vz,vrw,vrx,vry,vrz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().tcpVel)

    def get_joint_pos(self):
        """get current joint value.

        Returns:
            7-dim numpy array of 7 joint position

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().q)

    def get_joint_vel(self):
        """get current joint velocity.

        Returns:
            7-dim numpy array of 7 joint velocity

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().dq)

    def stop(self):
        """Stop current motion and switch mode to idle."""
        self.robot.stop()
        while self.get_control_mode() != self.mode_mapper("idle"):
            time.sleep(0.005)

    def set_max_contact_wrench(self, max_wrench):
        self.switch_mode('cart_impedance_online')
        self.robot.setMaxContactWrench(max_wrench)

    def send_impedance_online_pose(self, tcp):
        """make robot move towards target pose in impedance control mode,
        combining with sleep time makes robot move smmothly.

        Args:
            tcp: 7-dim list or numpy array, target pose (x,y,z,rw,rx,ry,rz) in world frame
            wrench: 6-dim list or numpy array, max moving force (fx,fy,fz,wx,wy,wz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode('cart_impedance_online')
        self.robot.sendCartesianMotionForce(np.array(tcp), [0] * 6, 0.1) # 0.1: maximum velocity

    def send_tcp_pose(self, tcp):
        """
        Send tcp pose.
        """
        self.send_impedance_online_pose(tcp)

    def send_joint_pose(self, q):
        """
        Send joint pose.
        """
        self.switch_mode('joint')
        DOF = len(q)
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1] * DOF
        MAX_ACC = [1] * DOF
        self.robot.sendJointPosition(np.array(q), target_vel, target_acc, MAX_VEL, MAX_ACC)

    def get_robot_state(self):
        raw = self._get_robot_status()
        tcpPose = raw.tcpPose
        tcpVel = raw.tcpVel
        jointPose = raw.q
        jointVel = raw.dq
        return tcpPose, jointPose, tcpVel, jointVel
    
class FlexivGripper:
    def __init__(self, r: FlexivRobot, home = True) -> None:
        self.gripper_state = flexivrdk.GripperStates()
        self.gripper = flexivrdk.Gripper(r.robot)
        self.gripper.getGripperStates(self.gripper_state)
        self.max_width = 0.085
        if home:
            self.move(self.max_width)  # Open gripper at start
    def move(self, width):
        # self.gripper.move(self.max_width * width / 1000, 0.1, 20)
        self.gripper.move(width, 0.1, 10)
    def grasp(self, force):
        self.gripper.grasp(force)
    def move_from_sigma(self, width):
        self.gripper.move(self.max_width * width / 1000, 0.1, 5)
    def get_gripper_state(self):
        self.gripper.getGripperStates(self.gripper_state)
        return self.gripper_state.width 
    
if __name__ == "__main__":
    robot = FlexivRobot()
    gripper = FlexivGripper(robot)

    # 获取初始位姿，作为运动中心点
    center_pose = robot.get_tcp_pose().copy()
    print("[INFO] Start repetitive motion around:", center_pose)

    # 设置运动幅度和速度
    delta = 0.05  # 每次偏移 5cm
    sleep_time = 0.01

    # 定义两个往返目标点（在 X 方向前后移动）
    pose_forward = center_pose.copy()
    pose_forward[2] += delta
    pose_backward = center_pose.copy()
    pose_backward[2] -= delta

    try:
        while True:
            # 向前
            print("[INFO] Moving forward...")
            robot.send_tcp_pose(pose_forward)
            gripper.move(0.085)  # 张开夹爪
            time.sleep(1)
            # 打印当前位置，用于观察
            tcp_pose, joint_pos, _, _ = robot.get_robot_state()
            print("[Current TCP Pose]:", np.round(tcp_pose, 4))
            # 向后
            print("[INFO] Moving backward...")
            robot.send_tcp_pose(pose_backward)
            gripper.move(0)  # 夹紧夹爪
            time.sleep(1)

            # 打印当前位置，用于观察
            tcp_pose, joint_pos, _, _ = robot.get_robot_state()
            print("[Current TCP Pose]:", np.round(tcp_pose, 4))

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C). Moving to safe position...")
        robot.send_tcp_pose(center_pose)
        time.sleep(2)
        robot.stop()
        print("[INFO] Robot stopped safely.")
