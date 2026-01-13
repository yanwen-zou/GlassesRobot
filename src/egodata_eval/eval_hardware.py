from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys
import multiprocessing as mp
import time
import select
import termios
import tty

import numpy as np

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
from egodata_eval.traj_predictor import TrajectoryPredictor  # type: ignore
from egodata_eval.get_head import HeadPoseReader
from glasses_hardware.hardware.my_device.robot import FlexivRobot, FlexivGripper  # type: ignore
from glasses_hardware.hardware.my_device.i2rt_robo import (
    I2RTClient,
    DEFAULT_ROBOT_PORT,
)  # type: ignore
from i2rt.robots.kinematics_mj import Kinematics
from i2rt.robots.utils import YAM_XML_PATH
import rclpy


class EvalHardware:
    def __init__(
        self,
        base_to_robot_txt: Optional[str] = DEFAULT_BASE_TO_ROBOT_TXT,
        i2rt_max_rot: float = I2RT_MAX_ROT,
        i2rt_cmd_duration: float = I2RT_CMD_DURATION,
        i2rt_cmd_steps: int = I2RT_CMD_STEPS,
        steps_to_execute: int = STEPS_TO_EXECUTE,
        i2rt_channel: str = I2RT_SERVER_CHANNEL,
        i2rt_port: int = DEFAULT_ROBOT_PORT,
    ) -> None:
        self.T_robot_base = np.eye(4, dtype=np.float32)
        if base_to_robot_txt:
            loaded = _load_calib_mat_safe(Path(base_to_robot_txt))
            if loaded is not None:
                self.T_robot_base = loaded.astype(np.float32)
                print(f"[INFO] Loaded T_robot_base from {base_to_robot_txt}")
            else:
                print(f"[WARN] Failed to load T_robot_base from {base_to_robot_txt}; using identity.")

        self.i2rt_max_rot = i2rt_max_rot
        self.i2rt_cmd_duration = i2rt_cmd_duration
        self.i2rt_cmd_steps = i2rt_cmd_steps
        self.steps_to_execute = steps_to_execute
        self.i2rt_server_proc: Optional[mp.Process] = None
        print("[INFO] Initializing Flexiv and gripper...") # First initialize Flexiv, or I2RT comm will go error
        self.flexiv_robot = FlexivRobot(home=False)
        self.flexiv_gripper = FlexivGripper(self.flexiv_robot)

        print("[INFO] Initializing I2RT (RPC)...")
        self.i2rt_server_proc = mp.Process(
            target=_run_i2rt_server,
            args=(i2rt_channel, False, i2rt_port),
            daemon=True,
        )
        self.i2rt_server_proc.start()
        self.i2rt_robot = I2RTClient(port=i2rt_port)
        time.sleep(3)

        self.i2rt_kin = Kinematics(YAM_XML_PATH, "grasp_site")
        self.i2rt_arm_dofs = min(6, self.i2rt_robot.num_dofs())
        self.i2rt_current_q = self.i2rt_robot.current_joint_pos()
        self.i2rt_target_pose = None
        self.last_headpose_rot: Optional[np.ndarray] = None
        self.last_headpose_xyz: Optional[np.ndarray] = None
        self.camera = self._init_camera()

    def _init_camera(self):
        ZEDCamera = _import_zed_class()
        return ZEDCamera(resolution=ZED_RESOLUTION, fps=ZED_FPS)

    def execute_pred_tcp_rel(self, tcp_rel_seq: np.ndarray) -> None: # rel tcp in i2rt frame, [N, xyz+6d rot] DEBUG:Basepose payload
        if tcp_rel_seq is None or tcp_rel_seq.shape[0] == 0:
            return
        rel_seq = _build_pose_mats(
            tcp_rel_seq[:, :3],
            tcp_rel_seq[:, 3:3+6],
        ).astype(np.float32)
        for idx in range(rel_seq.shape[0]):
            self.i2rt_current_q = self.i2rt_robot.current_joint_pos()
            current_pose = self.i2rt_kin.fk(self.i2rt_current_q[:self.i2rt_arm_dofs]).astype(np.float32)
            rel_tcp = np.linalg.inv(current_pose) @ rel_seq[idx] @ current_pose # transform rel traj to tcp frame
            print(f"[DEBUG] Rel seq translation:{np.round(rel_tcp[:3, 3],4)}")
            new_pose = current_pose @ rel_tcp
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
            target_xyz_rot6d = np.concatenate([new_pose[:3, 3], target_rot6d], axis=0).astype(np.float32)
            self.i2rt_robot.send_ee_pos(
                target_xyz_rot6d,
                duration=self.i2rt_cmd_duration,
                steps=self.i2rt_cmd_steps,
            )

    def execute_pred_tcp_rel_open(self, tcp_rel_seq: np.ndarray, base_pose: np.ndarray) -> None:
        """Execute relative TCP sequence using provided base pose as reference."""
        if tcp_rel_seq is None or tcp_rel_seq.shape[0] == 0:
            return
        if base_pose is None or base_pose.shape != (4, 4):
            raise ValueError(f"base_pose must be 4x4, got {None if base_pose is None else base_pose.shape}")
        rel_seq = _build_pose_mats(
            tcp_rel_seq[:, :3],
            tcp_rel_seq[:, 3:3+6],
        ).astype(np.float32)
        base_pose = base_pose.astype(np.float32)
        for idx in range(rel_seq.shape[0]):
            new_pose = rel_seq[idx] @ base_pose
            success, q_sol = self.i2rt_kin.ik(new_pose, "grasp_site", verbose=False)
            if not success:
                print("[WARN] I2RT IK failed for relative tcp pose.")
                continue
            self.i2rt_target_pose = new_pose
            target_rot6d = rotation_transform(
                new_pose[:3, :3][None, ...],
                "matrix",
                "rotation_6d",
            ).squeeze(0)
            target_xyz_rot6d = np.concatenate([new_pose[:3, 3], target_rot6d], axis=0).astype(np.float32)
            self.i2rt_robot.send_ee_pos(
                target_xyz_rot6d,
                duration=self.i2rt_cmd_duration,
                steps=self.i2rt_cmd_steps,
            )

    def execute_robot_traj(
        self,
        traj_pred: TrajectoryPredictor,
        pose_cam_ob: np.ndarray,
        T_base_cam: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        grip_seq = traj_pred.last_traj_denorm[:, 9].astype(np.float32)
        pose_base_ob = T_base_cam @ pose_cam_ob
        pose_robot_ob = self.T_robot_base @ pose_base_ob # [4,4]
        pose_seq_base = _build_pose_mats(
            traj_pred.last_traj_denorm[:, :3],
            traj_pred.last_traj_denorm[:, 3:3+6],
        )
        pose_seq_robot = np.einsum(
            'ij,njk->nik',
            self.T_robot_base.astype(np.float32),
            pose_seq_base.astype(np.float32),
        ) # [N,4,4], SE3 in robot frame

        steps_grip = None
        steps_to_execute = self.steps_to_execute
        if grip_seq is not None:
            steps_grip = grip_seq[1:1+int(steps_to_execute)]
        executed_poses: list[np.ndarray] = []
        tcp_history: list[np.ndarray] = []
        if pose_seq_robot.size > 0:
            robot_rel_pts = pose_seq_robot[1:1+int(steps_to_execute), :3, 3] - pose_robot_ob[:3, 3][None, :]
            curr_pose7 = self.flexiv_robot.get_tcp_pose().astype(np.float32)
            start_xyz = curr_pose7[:3].astype(np.float32)
            start_quat = curr_pose7[3:7].astype(np.float32)
            start_rot = rotation_transform(start_quat[None, :], "quaternion", "matrix").squeeze(0)
            base_obj_rot = pose_robot_ob[:3, :3].astype(np.float32)
            open_width = getattr(self.flexiv_gripper, 'max_width', GRIPPER_OPEN_WIDTH_DEFAULT)
            open_thresh = GRIP_OPEN_THRESH
            for i in range(robot_rel_pts.shape[0]):
                xyz = start_xyz + robot_rel_pts[i]
                step_rot = pose_seq_robot[1 + i, :3, :3].astype(np.float32)
                rel_rot = step_rot @ base_obj_rot.T # TODO: Here suppose object is at TCP, offset should be considered.
                target_rot = rel_rot @ start_rot
                target_quat = rotation_transform(target_rot[None, ...], "matrix", "quaternion").squeeze(0)
                pose7 = np.concatenate([xyz, target_quat], axis=0).astype(np.float32)
                if steps_grip is not None and i < len(steps_grip):
                    grip_val = float(steps_grip[i])
                    width_cmd = open_width if grip_val > open_thresh else 0.0
                    self.flexiv_gripper.move(width_cmd)

                self.flexiv_robot.send_tcp_pose(pose7)
                executed_poses.append(pose7.copy())
                tcp_history.append(self.flexiv_robot.get_tcp_pose().astype(np.float32))
                time.sleep(LOOP_SLEEP_SEC)
        return pose_robot_ob.astype(np.float32), pose_seq_robot.astype(np.float32), executed_poses, tcp_history

    def close_i2rt(self, timeout_s: float = 15.0) -> None:
        """Return I2RT to home pose before closing the client."""
        if self.i2rt_robot is None:
            return
        try:
            self.i2rt_robot.close(timeout_s=timeout_s)
        except Exception as exc:
            print(f"[WARN] I2RT close 失败: {exc}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Move I2RT up/down around current headpose.")
    ap.add_argument("--base-to-robot-txt", type=str, default=DEFAULT_BASE_TO_ROBOT_TXT)
    ap.add_argument("--pose-topic", type=str, default="/glasses_pose")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--dwell-sec", type=float, default=0.5)
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
    try:
        with _StdinCbreak():
            curr_pose = base_pose
            offset = 0.05
            base_rel_traj = np.array(
                [
                    [0, 0, offset, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0, 0, -offset, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0, 0, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
            headpose_i2rt_rel = headpose_base_to_i2rt_rel(base_rel_traj, T_base_cam, base_pose)
            tcp_i2rt_rel = headpose_to_tcp(headpose_i2rt_rel)
            while True:
                # hw.execute_pred_tcp_rel(headpose_i2rt_rel)
                hw.execute_pred_tcp_rel_open(tcp_i2rt_rel, curr_pose)
                last_rel = _build_pose_mats(
                    tcp_i2rt_rel[-1:, :3],
                    tcp_i2rt_rel[-1:, 3:3 + 6],
                ).astype(np.float32)[0]
                curr_pose = last_rel @ curr_pose
                if _check_quit():
                    print("[INFO] 收到 q，退出控制循环。")
                    hw.close_i2rt(timeout_s=20.0)
                    break
    finally:
        if hw.i2rt_server_proc is not None and hw.i2rt_server_proc.is_alive():
            hw.i2rt_server_proc.terminate()
            hw.i2rt_server_proc.join(timeout=2.0)


if __name__ == "__main__":
    main()
