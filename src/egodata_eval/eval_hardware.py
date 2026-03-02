from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys
import multiprocessing as mp
import time
import select
import termios
import tty
import threading

import numpy as np
from std_msgs.msg import Float32MultiArray

here = Path(__file__).resolve()
project_root = here.parents[2]
src_root = project_root / "src"
mba_root = project_root / "MBA"
for path in (src_root, mba_root, project_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
from scipy.spatial.transform import Rotation as R

from MBA.utils.transformation import rotation_transform  # type: ignore
from egodata_eval.eval_constant import (
    DEFAULT_BASE_TO_ROBOT_TXT,
    GRIP_OPEN_THRESH,
    GRIPPER_OPEN_WIDTH_DEFAULT,
    I2RT_CMD_DURATION,
    I2RT_CMD_STEPS,
    I2RT_MAX_ROT,
    I2RT_SERVER_CHANNEL,
    LOOP_SLEEP_SEC,
    STEPS_TO_EXECUTE,
    ZED_FPS,
    ZED_RESOLUTION,
    DEPTH_EST_SCALE,
    DEFAULT_I2RT_ZED_TXT,
    DEFAULT_GLASSES_ZED_TXT,
    TASK_TCP_TO_OBJECT_SE3,
)
from egodata_eval.eval_utils import (
    _build_pose_mats,
    _import_zed_class,
    _load_calib_mat_safe,
    _run_i2rt_server,
    calibrate_from_three_balls,
    headpose_base_to_i2rt_rel,
    move_i2rt_to_init_angles,
    headpose_to_tcp,
)
from egodata_eval.get_depth import DepthEstimator
from egodata_eval.get_head import HeadPoseReader
from egodata_eval.eval_utils import add_relative
from glasses_hardware.hardware.my_device.robot import FlexivRobot, FlexivGripper  # type: ignore
from glasses_hardware.hardware.my_device.i2rt_robo import (
    I2RTClient,
    DEFAULT_ROBOT_PORT,
)  # type: ignore
from i2rt.robots.kinematics_mj import Kinematics
from i2rt.robots.utils import YAM_XML_PATH, YAM_GLASS_PATH
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException


class I2RTHeadCmdNode(Node):
    def __init__(self, i2rt_port: int, cmd_duration: float, cmd_steps: int) -> None:
        super().__init__("i2rt_head_cmd")
        self._client = I2RTClient(port=i2rt_port)
        self._cmd_duration = cmd_duration
        self._cmd_steps = cmd_steps
        self._sub = self.create_subscription(Float32MultiArray, "head_cmd", self._on_head_cmd, 10)

    def _on_head_cmd(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != 9:
            return
        self._client.send_ee_pos(data.tolist(), duration=self._cmd_duration, steps=self._cmd_steps)

    def num_dofs(self) -> int:
        return self._client.num_dofs()

    def current_joint_pos(self) -> np.ndarray:
        return self._client.current_joint_pos()

    def send_joint_pos_rad(self, target_joint_pos_rad, duration: Optional[float] = None, steps: Optional[int] = None) -> None:
        self._client.send_joint_pos_rad(target_joint_pos_rad, duration=duration, steps=steps)

    def close(self, timeout_s: float = 15.0) -> None:
        self._client.close(timeout_s=timeout_s)


class FlexivArmCmdNode(Node):
    def __init__(self) -> None:
        super().__init__("flexiv_arm_cmd")
        self._robot = FlexivRobot(home=False)
        self._gripper = FlexivGripper(self._robot)
        self._sub = self.create_subscription(Float32MultiArray, "arm_cmd", self._on_arm_cmd, 10)
        self.max_width = getattr(self._gripper, "max_width", GRIPPER_OPEN_WIDTH_DEFAULT)

    def _on_arm_cmd(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != 8:
            return
        pose7 = data[:7]
        grip_val = float(data[7])
        self._robot.send_tcp_pose(pose7.astype(np.float32))
        self._gripper.move(grip_val)

    def get_tcp_pose(self) -> np.ndarray:
        return self._robot.get_tcp_pose().astype(np.float32)

    def get_gripper_state(self) -> float:
        return float(self._gripper.get_gripper_state())

    def get_state(self) -> dict[str, np.ndarray | float]:
        tcp_pose = self.get_tcp_pose().astype(np.float32)
        gripper_width = self.get_gripper_state()
        return {
            "tcp_pose": tcp_pose,
            "gripper_width": float(gripper_width),
        }

    def close(self) -> None:
        try:
            self._robot.close()
        except Exception:
            pass


class EvalHardware:
    """
    Wrapper of I2RT(head) and Flexiv(manipulator), communicate with separate nodes.
    """
    def __init__(
        self,
        base_to_robot_txt: Optional[str] = DEFAULT_BASE_TO_ROBOT_TXT,
        i2rt_max_rot: float = I2RT_MAX_ROT,
        i2rt_cmd_duration: float = I2RT_CMD_DURATION,
        i2rt_cmd_steps: int = I2RT_CMD_STEPS,
        steps_to_execute: int = STEPS_TO_EXECUTE,
        i2rt_channel: str = I2RT_SERVER_CHANNEL,
        i2rt_port: int = DEFAULT_ROBOT_PORT,
        task_name: str = "book",
    ) -> None:
        self.T_robot_base = np.eye(4, dtype=np.float32)
        if base_to_robot_txt:
            loaded = _load_calib_mat_safe(Path(base_to_robot_txt))
            if loaded is not None:
                self.T_robot_base = loaded.astype(np.float32)
                print(f"[INFO] Loaded T_robot_base from {base_to_robot_txt}")
            else:
                print(f"[WARN] Failed to load T_robot_base from {base_to_robot_txt}; using identity.")
        self.T_i2rt_zed = _load_calib_mat_safe(Path(DEFAULT_I2RT_ZED_TXT))

        self.T_zed_glasses = np.linalg.inv(_load_calib_mat_safe(Path(DEFAULT_GLASSES_ZED_TXT))).astype(np.float32)


        self.i2rt_max_rot = i2rt_max_rot
        self.i2rt_cmd_duration = i2rt_cmd_duration
        self.i2rt_cmd_steps = i2rt_cmd_steps
        self.steps_to_execute = steps_to_execute
        self.task_name = task_name
        self.i2rt_server_proc: Optional[mp.Process] = None
        print("[INFO] Initializing Flexiv and gripper...") # First initialize Flexiv, or I2RT comm will go error
        if not rclpy.ok():
            rclpy.init(args=None)
        self.flexiv_robot = FlexivArmCmdNode()
        self._arm_cmd_pub = self.flexiv_robot.create_publisher(Float32MultiArray, "arm_cmd", 10)
        self._flexiv_executor = SingleThreadedExecutor()
        self._flexiv_executor.add_node(self.flexiv_robot)
        self._flexiv_thread = threading.Thread(target=self._spin_executor, args=(self._flexiv_executor,), daemon=True)
        self._flexiv_thread.start()

        print("[INFO] Initializing I2RT (RPC)...")
        self.i2rt_server_proc = mp.Process(
            target=_run_i2rt_server,
            args=(i2rt_channel, False, i2rt_port),
            daemon=True,
        )
        self.i2rt_server_proc.start()
        if not rclpy.ok():
            rclpy.init(args=None)
        self.i2rt_robot = I2RTHeadCmdNode(i2rt_port, i2rt_cmd_duration, i2rt_cmd_steps)
        self._head_cmd_pub = self.i2rt_robot.create_publisher(Float32MultiArray, "head_cmd", 10)
        self._head_executor = SingleThreadedExecutor()
        self._head_executor.add_node(self.i2rt_robot)
        self._head_thread = threading.Thread(target=self._spin_executor, args=(self._head_executor,), daemon=True)
        self._head_thread.start()
        time.sleep(3)

        self.i2rt_kin = Kinematics(YAM_GLASS_PATH, "grasp_site")
        self.i2rt_arm_dofs = min(6, self.i2rt_robot.num_dofs())
        self.i2rt_current_q = self.i2rt_robot.current_joint_pos()
        self.i2rt_target_pose = None
        self.last_headpose_rot: Optional[np.ndarray] = None
        self.last_headpose_xyz: Optional[np.ndarray] = None
        self.camera = self._init_camera()
        self.idx = 0

    def _init_camera(self):
        ZEDCamera = _import_zed_class()
        return ZEDCamera(resolution=ZED_RESOLUTION, fps=ZED_FPS)

    def execute_pred_tcp_rel(self, tcp_rel_seq: np.ndarray) -> np.ndarray: # rel tcp in i2rt frame, [N, xyz+6d rot] DEBUG:Basepose payload
        rel_seq = _build_pose_mats(
            tcp_rel_seq[:, :3],
            tcp_rel_seq[:, 3:3+6],
        ).astype(np.float32)

        self.i2rt_current_q = self.i2rt_robot.current_joint_pos()
        # The starting pose of an action chunk, converted from TCP to glasses frame.
        current_pose = (
            self.i2rt_kin.fk(self.i2rt_current_q[:self.i2rt_arm_dofs]).astype(np.float32)
        )
        # T_tcp_glasses = self.T_i2rt_zed.astype(np.float32) @ self.T_zed_glasses.astype(np.float32)
        # glass_pose = current_pose @ T_tcp_glasses
        # print("[DEBUG] Last rel_seq xyz norm from step 0: {:.4f} m".format(np.linalg.norm(rel_seq[-1, :3, 3])))
        new_pose_seq: list[np.ndarray] = []
        for idx in range(rel_seq.shape[0]):
            # print(f"[DEBUG] Current TCP Pose :\n{current_pose}")
            # print(f"[DEBUG] Rel seq translation:{np.round(rel_seq[idx,:3, 3],4)}")
            # glass_new_pose = add_relative(rel_seq[idx], glass_pose)
            # glass_new_pose = rel_seq[idx] @ glass_pose
            new_pose = rel_seq[idx] @ current_pose
            new_pose_seq.append(new_pose.astype(np.float32))
            print(f"[DEBUG] New TCP Pose :\n{new_pose}")
            print("--------------------------------------------------")
            success, q_sol = self.i2rt_kin.ik(new_pose, "grasp_site", verbose=False)
            if not success:
                print("[WARN] I2RT IK failed for relative tcp pose.")
                continue
            self.i2rt_target_pose = new_pose
            # self.i2rt_current_q[:self.i2rt_arm_dofs] = q_sol[:self.i2rt_arm_dofs]
            target_rot6d = rotation_transform(
                new_pose[:3, :3][None, ...],
                "matrix",
                "rotation_6d",
            ).squeeze(0)
            # target_rot6d = rotation_transform( #DEBUG: no rot version
            #     current_pose[:3, :3][None, ...],
            #     "matrix",
            #     "rotation_6d",
            # ).squeeze(0)
            target_xyz_rot6d = np.concatenate([new_pose[:3, 3], target_rot6d], axis=0).astype(np.float32)
            self._publish_head_cmd(target_xyz_rot6d)
            time.sleep(0.2)
        return np.stack(new_pose_seq, axis=0).astype(np.float32)
    def execute_pred_tcp_abs(self, tcp_i2rt_abs: np.ndarray) -> np.ndarray: # abs headpose in i2rt frame, [N, xyz+6d rot]
        new_pose_seq: list[np.ndarray] = []
        new_pose_base = tcp_i2rt_abs[0].astype(np.float32)
        self.i2rt_current_q = self.i2rt_robot.current_joint_pos()
        current_pose = (
            self.i2rt_kin.fk(self.i2rt_current_q[:self.i2rt_arm_dofs]).astype(np.float32)
        )
        for idx in range(tcp_i2rt_abs.shape[0]):
            new_pose_delta = tcp_i2rt_abs[idx] @ np.linalg.inv(new_pose_base)
            print(f"new_pose_delta:{new_pose_delta[:3,3]}")
            new_pose = new_pose_delta @ current_pose
            new_pose_seq.append(new_pose.copy())
            # print (f"[DEBUG] New TCP Pose :{new_pose[:3,3]}")
            success, q_sol = self.i2rt_kin.ik(new_pose, "grasp_site", verbose=False)
            if not success:
                print("[WARN] I2RT IK failed for absolute tcp pose.")
                continue
            self.i2rt_target_pose = new_pose
            target_rot6d = rotation_transform(
                new_pose[:3, :3][None, ...],
                "matrix",
                "rotation_6d",
            ).squeeze(0)
            target_xyz_rot6d = np.concatenate([new_pose[:3, 3], target_rot6d], axis=0).astype(np.float32)
            self._publish_head_cmd(target_xyz_rot6d)
            time.sleep(0.2)
        return np.stack(new_pose_seq, axis=0).astype(np.float32)

    def execute_robot_traj(
        self,
        traj_denorm: np.ndarray,
        pose_base_ob: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        grip_seq = traj_denorm[:, 9].astype(np.float32)
        pose_robot_ob = self.T_robot_base @ pose_base_ob # [4,4]
        pose_seq_base = _build_pose_mats(
            traj_denorm[:, :3],
            traj_denorm[:, 3:3+6],
        )
        # print(f"[INFO] pose_seq_base: {pose_seq_base}")
        pose_seq_robot = np.einsum(
            'ij,njk->nik',
            self.T_robot_base.astype(np.float32),
            pose_seq_base.astype(np.float32),
        ) # [N,4,4], SE3 in robot frame
        # print(f"[INFO] pose_seq_robot:\n{pose_seq_robot[1,:3,3]}")
        T_tcp_object = TASK_TCP_TO_OBJECT_SE3.get(self.task_name, np.eye(4, dtype=np.float32)).astype(np.float32)
        # print(f"[INFO] Using T_tcp_object for task '{self.task_name}':\n{T_tcp_object}")
        T_object_tcp = np.linalg.inv(T_tcp_object).astype(np.float32)
        # print(f"[INFO] pose_robot_ob:\n{pose_robot_ob}")
        pose_robot_tcp = (pose_robot_ob @ T_object_tcp).astype(np.float32)
        # print(f"[INFO] pose_robot_tcp:\n{pose_robot_tcp}")
        tcp_seq_robot = np.einsum(
            "nij,jk->nik",
            pose_seq_robot.astype(np.float32),
            T_object_tcp,
        ).astype(np.float32)
        # print(f"[INFO] tcp_seq_robot:\n{tcp_seq_robot[:,:3,3]}")
        steps_grip = None
        steps_to_execute = self.steps_to_execute
        if grip_seq is not None:
            steps_grip = grip_seq[1:1+int(steps_to_execute)]
        executed_poses: list[np.ndarray] = []
        tcp_history: list[np.ndarray] = []
        if tcp_seq_robot.size > 0:
            robot_rel_pts = tcp_seq_robot[1:1+int(steps_to_execute), :3, 3] - pose_robot_tcp[:3, 3][None, :]
            # print(f"[INFO] rel_pts:\n{robot_rel_pts}")
            curr_pose7 = self.flexiv_robot.get_tcp_pose().astype(np.float32)
            start_xyz = curr_pose7[:3].astype(np.float32)
            start_quat = curr_pose7[3:7].astype(np.float32)
            start_rot = rotation_transform(start_quat[None, :], "quaternion", "matrix").squeeze(0)
            base_obj_rot = pose_robot_tcp[:3, :3].astype(np.float32)
            open_width = getattr(self.flexiv_robot, "max_width", GRIPPER_OPEN_WIDTH_DEFAULT)
            open_thresh = GRIP_OPEN_THRESH.get(self.task_name, GRIP_OPEN_THRESH["book"])
            for i in range(robot_rel_pts.shape[0]):
                xyz = start_xyz + robot_rel_pts[i]
                step_rot = tcp_seq_robot[1 + i, :3, :3].astype(np.float32)
                rel_rot = step_rot @ base_obj_rot.T 
                target_rot = rel_rot @ start_rot
                target_quat = rotation_transform(target_rot[None, ...], "matrix", "quaternion").squeeze(0)
                pose7 = np.concatenate([xyz, target_quat], axis=0).astype(np.float32)
                width_cmd = 0.0
                if steps_grip is not None and i < len(steps_grip):
                    grip_val = float(steps_grip[i])
                    # print(f"[INFO] grip_val:{grip_val}")
                    if grip_val > open_thresh:
                        if self.task_name == "teapot":
                            print("[INFO] Teapot grip > threshold; stopping arm without opening gripper.")
                            break
                        width_cmd = open_width
                    else:
                        width_cmd = 0.0
                # print(f"[INFO] pose7:\n{pose7}")
                self._publish_arm_cmd(pose7, width_cmd)
                executed_poses.append(pose7.copy())
                tcp_history.append(self.flexiv_robot.get_tcp_pose().astype(np.float32))
                # print(f"[INFO] execute flexiv:{self.idx}")
                self.idx += 1
                time.sleep(LOOP_SLEEP_SEC)
        return pose_robot_ob.astype(np.float32), tcp_seq_robot.astype(np.float32), executed_poses

    def close(self, timeout_s: float = 15.0) -> None:
        """Return I2RT to home pose before closing the clients."""
        if self.i2rt_robot is None:
            return
        try:
            self.i2rt_robot.close(timeout_s=timeout_s)
        except Exception as exc:
            print(f"[WARN] I2RT close 失败: {exc}")
        finally:
            self._head_executor.shutdown()
            self._head_thread.join(timeout=1.0)
            self.i2rt_robot.destroy_node()
            self._flexiv_executor.shutdown()
            self._flexiv_thread.join(timeout=1.0)
            self.flexiv_robot.destroy_node()
            self.flexiv_robot.close()

    @staticmethod
    def _spin_executor(executor: SingleThreadedExecutor) -> None:
        try:
            executor.spin()
        except ExternalShutdownException:
            pass

    def _publish_head_cmd(self, target_xyz_rot6d: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = target_xyz_rot6d.astype(np.float32).ravel().tolist()
        self._head_cmd_pub.publish(msg)

    def _publish_arm_cmd(self, pose7: np.ndarray, grip_val: float) -> None:
        msg = Float32MultiArray()
        payload = np.concatenate([pose7.astype(np.float32), np.array([grip_val], dtype=np.float32)], axis=0)
        msg.data = payload.ravel().tolist()
        self._arm_cmd_pub.publish(msg)


def main() -> None:
    '''
    Test headpose_base -> tcp_i2rt rel traj transformation
    '''
    import argparse
    ap = argparse.ArgumentParser(description="Move I2RT up/down around current headpose.")
    ap.add_argument("--base-to-robot-txt", type=str, default=DEFAULT_BASE_TO_ROBOT_TXT)
    ap.add_argument("--pose-topic", type=str, default="/glasses_pose")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--dwell-sec", type=float, default=1)
    args = ap.parse_args()

    class _StdinCbreak:
        def __init__(self) -> None:
            self._fd = None
            self._old = None

        def __enter__(self):
            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()
                self._old = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
            return self

        def __exit__(self, exc_type, exc, tb):
            if self._fd is not None and self._old is not None:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def _check_quit() -> bool:
        if not sys.stdin.isatty():
            return False
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return False
        ch = sys.stdin.read(1)
        return ch.lower() == "q"

    if not rclpy.ok():
        rclpy.init(args=None)
    hw = EvalHardware(base_to_robot_txt=args.base_to_robot_txt)
    move_i2rt_to_init_angles(hw.i2rt_robot)
    depth_est = DepthEstimator(camera=hw.camera,scale=DEPTH_EST_SCALE)
    T_base_cam = calibrate_from_three_balls(
        hw.camera,
        depth_est,
        move_robot_fn=None,
        centroid_log_dir=None,
    )
    base_q = hw.i2rt_robot.current_joint_pos()
    base_pose = hw.i2rt_kin.fk(base_q[:hw.i2rt_arm_dofs]).astype(np.float32)
    if T_base_cam is None:
        raise RuntimeError("Failed to calibrate T_base_cam from three balls.")
    head_reader = HeadPoseReader(args.pose_topic, DEFAULT_GLASSES_ZED_TXT, T_base_cam)
    try:
        with _StdinCbreak():
            offset = 0
            pitch_rad = np.deg2rad(10.0)
            pitch_rot6d = rotation_transform(
                R.from_euler("y", pitch_rad).as_matrix()[None, ...],
                "matrix",
                "rotation_6d",
            ).squeeze(0).astype(np.float32)
            minus_pitch_rad = np.deg2rad(-10.0)
            minus_pitch_rot6d = rotation_transform(
                R.from_euler("y", minus_pitch_rad).as_matrix()[None, ...],
                "matrix",
                "rotation_6d",
            ).squeeze(0).astype(np.float32)
            base_rel_traj = np.array(
                [
                    np.concatenate([[0, offset, 0], pitch_rot6d], axis=0),
                    np.concatenate([[0, -offset, 0], minus_pitch_rot6d], axis=0),
                ],
                dtype=np.float32,
            )

            while True:
                latest_base_cam = head_reader.get_headpos(timeout_sec=0.0)
                if latest_base_cam is not None:
                    T_base_cam = latest_base_cam.astype(np.float32)
                print(f"base_rel_traj:\n{base_rel_traj}")
                headpose_i2rt_rel = headpose_base_to_i2rt_rel(base_rel_traj, T_base_cam, base_pose)
                print(f"after trans:\n{headpose_i2rt_rel}")
                tcp_i2rt_rel = headpose_to_tcp(headpose_i2rt_rel)
                hw.execute_pred_tcp_rel(tcp_i2rt_rel)
                base_q = hw.i2rt_robot.current_joint_pos()
                base_pose = hw.i2rt_kin.fk(base_q[:hw.i2rt_arm_dofs]).astype(np.float32)
                if _check_quit():
                    print("[INFO] 收到 q，退出控制循环。")
                    hw.close(timeout_s=20.0)
                    break
    finally:
        head_reader.destroy_node()
        rclpy.shutdown()
        if hw.i2rt_server_proc is not None and hw.i2rt_server_proc.is_alive():
            hw.i2rt_server_proc.terminate()
            hw.i2rt_server_proc.join(timeout=2.0)


if __name__ == "__main__":
    main()
