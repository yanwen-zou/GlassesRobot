import numpy as np
import torch
import time
import cv2
import sys
import threading
import importlib
import multiprocessing as mp
from typing import Optional
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R
import pygame
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.transforms import InterpolationMode

from my_device.robot import FlexivRobot, FlexivGripper
from my_device.zed import ZEDCamera
from my_device.keyboard import Keyboard
from my_device.sigma import Sigma7
from my_device.logitechG29_wheel import Controller
from my_device.macros import CAM_SERIAL, INTV, HUMAN, ROBOT
from i2rt.robots.kinematics_mj import Kinematics
from i2rt.robots.utils import YAM_XML_PATH, YAM_GLASS_PATH

# Ensure local packages under repo/{src,MBA} are importable.
here = Path(__file__).resolve()
project_root = here.parents[2]
src_root = project_root / "src"
mba_root = project_root / "MBA"
for p in (project_root, src_root, mba_root):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


# ROS2
import rclpy
from std_msgs.msg import Float32MultiArray
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from MBA.utils.transformation import rotation_transform
from egodata_eval.eval_hardware import I2RTHeadCmdNode
from egodata_eval.eval_utils import _run_i2rt_server, _load_calib_mat_safe, add_relative, move_i2rt_to_init_angles
from egodata_eval.eval_constant import (
    I2RT_MAX_ROT, I2RT_CMD_DURATION, I2RT_CMD_STEPS, STEPS_TO_EXECUTE, I2RT_SERVER_CHANNEL,
    DEFAULT_GLASSES_ZED_TXT, DEFAULT_I2RT_ZED_TXT,
)
from glasses_hardware.hardware.my_device.i2rt_robo import I2RTClient, DEFAULT_ROBOT_PORT

ZED_RESOLUTION = "WVGA"
ZED_FPS = 30


def _headpose_7d_to_se3(hp: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, qx, qy, qz, qw] to a (4, 4) SE3 matrix."""
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 3] = hp[:3]
    mat[:3, :3] = R.from_quat(hp[3:7]).as_matrix().astype(np.float32)
    return mat


class RobotEnv:
    def __init__(self, camera_serial=CAM_SERIAL, img_shape=None, fps=10, is_infer_mode=False, task_name="book", robot_id=None,robot_info_dict=None):
        self.camera_serial = camera_serial
        self.fps = fps
        self.img_shape = img_shape
        self.is_infer_mode = is_infer_mode
        self.i2rt_server_proc = None
        self.i2rt_robot = None
        self.i2rt_kin = None
        self.i2rt_arm_dofs = None
        self.i2rt_current_q = None

        # Initialize hardware components
        if self.is_infer_mode: 
            self.robot = FlexivRobot()
            self._init_i2rt(task_name) # only initialize i2rt in inference mode
        else: 
            self.robot = FlexivRobot()
            self.sigma = Sigma7()
            pygame.init()
            self.controller = Controller(0)
        
        self.gripper = FlexivGripper(self.robot)
        self.zed = self._init_camera() # use ZED camera
        self.keyboard = Keyboard()
        self.home_pose = self.robot.init_pose
        
        # Setup image processors
        BICUBIC = InterpolationMode.BICUBIC
        self.image_processor = Compose([
            Resize((img_shape[1]+8, img_shape[2]+8), interpolation=BICUBIC),
            CenterCrop((img_shape[1], img_shape[2]))
        ])
        
        # Keep track of throttle usage for human intervention
        self.last_throttle = False

    def _init_camera(self):
        return ZEDCamera(resolution=ZED_RESOLUTION, fps=ZED_FPS)
    
    def _init_i2rt(
        self, 
        task_name: str = "book",
        i2rt_max_rot: float = I2RT_MAX_ROT,
        i2rt_cmd_duration: float = I2RT_CMD_DURATION,
        i2rt_cmd_steps: int = I2RT_CMD_STEPS,
        steps_to_execute: int = STEPS_TO_EXECUTE,
        i2rt_channel: str = I2RT_SERVER_CHANNEL,
        i2rt_port: int = DEFAULT_ROBOT_PORT,
    ): 
        self.i2rt_max_rot = i2rt_max_rot
        self.i2rt_cmd_duration = i2rt_cmd_duration
        self.i2rt_cmd_steps = i2rt_cmd_steps
        self.steps_to_execute = steps_to_execute
        self.task_name = task_name
        self.i2rt_server_proc: Optional[mp.Process] = None

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
        
        print("[INFO] Moving I2RT to init angles")
        move_i2rt_to_init_angles(self.i2rt_robot, task_name=task_name)

        # Reference headpose (frame 0) for relative-to-first computation.
        # Set externally via set_headpose_ref(); sourced from the ROS glasses-pose topic.
        self.headpose_ref_xyz: Optional[np.ndarray] = None
        self.headpose_ref_r: Optional[R] = None
        self.headpose_ref_r_inv: Optional[R] = None

    @staticmethod
    def _spin_executor(executor: SingleThreadedExecutor) -> None:
        try:
            executor.spin()
        except ExternalShutdownException:
            pass

    def _move_i2rt_to_zero_slowly(self, duration: float = 4.0, steps: int = 200) -> None:
        """Move I2RT to zero joint pose slowly before shutdown."""
        if self.i2rt_robot is None:
            return
        try:
            q_curr = self.i2rt_robot.current_joint_pos().astype(np.float32)
            q_target = q_curr.copy()
            dofs = int(self.i2rt_arm_dofs) if self.i2rt_arm_dofs is not None else min(6, self.i2rt_robot.num_dofs())
            q_target[:dofs] = 0.0
            print(f"[INFO] Moving I2RT to zero pose (duration={duration}s, steps={steps})")
            self.i2rt_robot.send_joint_pos_rad(q_target, duration=duration, steps=steps)
            self.i2rt_current_q = q_target.copy()
        except Exception as exc:
            print(f"[WARN] Failed to move I2RT to zero pose: {exc}")

    def close(self, timeout_s: float = 15.0) -> None:
        """Close resources. Follows eval_hardware.close order and style."""
        # Close I2RT client and ROS executor/thread/node.
        if self.i2rt_robot is not None:
            try:
                self.i2rt_robot.close(timeout_s=timeout_s)
            except Exception as exc:
                print(f"[WARN] I2RT close failed: {exc}")
            finally:
                self._head_executor.shutdown()
                self._head_thread.join(timeout=1.0)
                self.i2rt_robot.destroy_node()

        # Stop I2RT server process.
        if self.i2rt_server_proc is not None and self.i2rt_server_proc.is_alive():
            self.i2rt_server_proc.terminate()
            self.i2rt_server_proc.join(timeout=2.0)

        # Close remaining devices.
        self.zed.close()
        self.robot.stop()
        if not self.is_infer_mode:
            self.sigma.close()
        pygame.quit()

    def _get_camera_frames(self):
        stereo = self.zed.read_stereo()
        if stereo is None:
            print("Failed to read stereo images from ZED camera.")
            return None, None
        left_img, right_img = stereo
        return left_img, right_img
        
    def reset_robot(self, random_init=False, random_init_pose=None):
        if random_init and random_init_pose is not None:
            self.robot.send_tcp_pose(random_init_pose)
        else:
            self.robot.send_tcp_pose(self.robot.init_pose)
        time.sleep(2)
        self.gripper.move(self.gripper.max_width)
        time.sleep(0.5)
        print("Reset!")
        if self.is_infer_mode: 
            return self.get_robot_state()
        
        # Reset the sigma pose as well
        self.sigma.reset()
        self.last_throttle = False
        if random_init and random_init_pose is not None:
            random_p_drift = random_init_pose[:3] - self.robot.init_pose[:3]
            random_r_drift = R.from_quat(self.robot.init_pose[3:7], scalar_first=True).inv() * R.from_quat(random_init_pose[3:7], scalar_first=True)
            self.sigma.transform_from_robot(random_p_drift, random_r_drift)
        
        return self.get_robot_state()
    
    def get_robot_state(self):
        """Get robot state, images, and joint positions"""
        # Get robot state
        tcp_pose, joint_pos, _, _ = self.robot.get_robot_state()
        
        # Get camera images
        left_img, right_img = self._get_camera_frames()
        if left_img is None or right_img is None:
            return {
                'tcp_pose': tcp_pose,
                'joint_pos': joint_pos,
                'policy_left_img': None,
                'policy_right_img': None,
                'demo_left_img': None,
                'demo_right_img': None,
                'left_img_raw': None,
                'right_img_raw': None
            }
            
        # Process images
        policy_left_img = self.image_processor(torch.from_numpy(left_img.copy()).permute(2, 0, 1))
        policy_right_img = self.image_processor(torch.from_numpy(right_img.copy()).permute(2, 0, 1))
        
        return {
            'tcp_pose': tcp_pose,
            'joint_pos': joint_pos,
            'policy_left_img': policy_left_img,
            'policy_right_img': policy_right_img,
            'demo_left_img': policy_left_img,
            'demo_right_img': policy_right_img,
            'left_img_raw': left_img.copy(),
            'right_img_raw': right_img.copy()
        }
    
    # ------------------------------------------------------------------
    # Headpose helpers (relative-to-first-frame, external-vec based)
    # Headpose is always sourced from the ROS /glasses_pose topic as a
    # 7-dim vector [x, y, z, qx, qy, qz, qw] (scipy xyzw convention).
    # ------------------------------------------------------------------

    def set_headpose_ref(self, headpose_vec: np.ndarray) -> None:
        """Capture frame-0 reference from an external [x,y,z,qx,qy,qz,qw] vector.

        Must be called once at the start of each inference episode before any
        call to compute_headpose_rel or execute_headpose_from_rel.
        """
        hp = np.asarray(headpose_vec, dtype=np.float32)
        self.headpose_ref_xyz = hp[:3].copy()
        q0 = hp[3:7] / np.linalg.norm(hp[3:7])
        r0 = R.from_quat(q0)
        self.headpose_ref_r = r0
        self.headpose_ref_r_inv = r0.inv()

    def compute_headpose_rel(self, headpose_vec: np.ndarray) -> np.ndarray:
        """Compute headpose relative to reference (frame 0) as [x,y,z,qx,qy,qz,qw].

        Matches the formula used in convert_hdf5_to_lerobot.py:
            rel_t = r0_inv.apply(ti - t0)
            rel_q = (r0_inv * R.from_quat(qi)).as_quat()
        """
        if self.headpose_ref_xyz is None or self.headpose_ref_r_inv is None:
            raise RuntimeError("Headpose reference not set. Call set_headpose_ref() first.")
        hp = np.asarray(headpose_vec, dtype=np.float32)
        ti = hp[:3]
        qi = hp[3:7] / np.linalg.norm(hp[3:7])
        rel_t = self.headpose_ref_r_inv.apply(ti - self.headpose_ref_xyz)
        rel_q = (self.headpose_ref_r_inv * R.from_quat(qi)).as_quat()
        return np.concatenate([rel_t, rel_q], axis=0).astype(np.float32)

    def reconstruct_abs_headpose(self, headpose_rel_vec: np.ndarray) -> np.ndarray:
        """Inverse of compute_headpose_rel: relative [x,y,z,qx,qy,qz,qw] -> absolute."""
        if self.headpose_ref_xyz is None or self.headpose_ref_r is None:
            raise RuntimeError("Headpose reference not set. Call set_headpose_ref() first.")
        hp = np.asarray(headpose_rel_vec, dtype=np.float32)
        rel_t = hp[:3]
        rel_q = hp[3:7] / np.linalg.norm(hp[3:7])
        abs_t = self.headpose_ref_r.apply(rel_t) + self.headpose_ref_xyz
        abs_q = (self.headpose_ref_r * R.from_quat(rel_q)).as_quat()
        return np.concatenate([abs_t, abs_q], axis=0).astype(np.float32)

    def execute_headpose_from_rel(
        self,
        headpose_action_rel: np.ndarray,
        current_headpose_abs_7d: np.ndarray,
    ) -> None:
        """Execute a headpose action expressed as relative-to-first-frame.

        Steps:
          1. Reconstruct absolute target headpose from the relative action + frame-0 ref.
          2. Build SE3 matrices for current and target (both in glasses frame).
          3. Compute delta: T_delta = T_current^{-1} @ T_target.
          4. Dispatch to deploy_i2rt_action which converts glasses->TCP and runs IK.

        Args:
            headpose_action_rel:      7-dim [x,y,z,qx,qy,qz,qw] relative-to-first target.
            current_headpose_abs_7d:  7-dim [x,y,z,qx,qy,qz,qw] current absolute headpose
                                      (from the ROS topic at the current step).
        """
        if self.i2rt_robot is None:
            return
        target_abs_7d = self.reconstruct_abs_headpose(headpose_action_rel)
        T_current = _headpose_7d_to_se3(np.asarray(current_headpose_abs_7d, dtype=np.float32))
        T_target = _headpose_7d_to_se3(target_abs_7d)
        print("[DEBUG] Current headpose (abs):", current_headpose_abs_7d)
        print("[DEBUG] Target headpose (abs):", target_abs_7d)
        # Delta in glasses frame: T_delta = T_current^{-1} @ T_target
        T_delta_glasses = np.linalg.inv(T_current).astype(np.float32) @ T_target
        self.deploy_i2rt_action(T_delta_glasses)

    def execute_headpose_from_current_delta(self, headpose_action_delta: np.ndarray) -> None:
        """Execute a headpose action expressed as delta from current headpose.

        Args:
            headpose_action_delta: 7-dim [x,y,z,qx,qy,qz,qw] SE3 delta in glasses frame.
                                   It is interpreted as: T_target = T_current @ T_delta.
        """
        if self.i2rt_robot is None:
            return
        hp = np.asarray(headpose_action_delta, dtype=np.float32).reshape(-1)
        if hp.shape[0] != 7:
            raise ValueError(f"Expected 7-dim headpose delta, got shape {hp.shape}")
        q = hp[3:7]
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            raise ValueError("Headpose delta quaternion norm is too small.")
        hp[3:7] = q / q_norm
        T_delta_glasses = _headpose_7d_to_se3(hp)
        self.deploy_i2rt_action(T_delta_glasses)

    def deploy_action(self, tcp_action, gripper_action):
        self.robot.send_tcp_pose(tcp_action)
        self.gripper.move(gripper_action)
        time.sleep(0.2)

    def update_current_pose(self): 
        if self.i2rt_robot is None:
            return
        self.i2rt_current_q = self.i2rt_robot.current_joint_pos()
        self.i2rt_current_pose = self.i2rt_kin.fk(self.i2rt_current_q[:self.i2rt_arm_dofs]).astype(np.float32)

    def deploy_i2rt_action(self, delta_headpose):
        delta_tcp = self.delta_headpose_to_delta_tcp(delta_headpose)
        # self.i2rt_current_q = self.i2rt_robot.current_joint_pos()
        # current_pose = self.i2rt_kin.fk(self.i2rt_current_q[:self.i2rt_arm_dofs]).astype(np.float32)

        if self.i2rt_current_pose is None:
            raise RuntimeError("Current I2RT pose is not available. Call update_current_pose() first.")

        new_pose = self.i2rt_current_pose @ delta_tcp
        success, q_sol = self.i2rt_kin.ik(new_pose, "grasp_site", verbose=False)
        if not success: 
            print("[WARN] IK failed for delta_headpose.")
            return
        self.i2rt_target_pose = new_pose
        print("[DEBUG] Current TCP pose:\n", self.i2rt_current_pose)
        print("[DEBUG] Target TCP pose:\n", new_pose)
        target_rot6d = rotation_transform(
            new_pose[:3, :3][None, ...],
            "matrix",
            "rotation_6d",
        ).squeeze(0)
        target_xyz_rot6d = np.concatenate([new_pose[:3, 3], target_rot6d], axis=0).astype(np.float32)
        self._publish_head_cmd(target_xyz_rot6d)
        time.sleep(0.2)

    def delta_headpose_to_delta_tcp(
        self,
        delta_headpose: np.ndarray,
        T_glasses_zed: Optional[np.ndarray] = None,
        T_tcp_zed: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert a delta headpose from the glasses/eye frame to the TCP frame.

        First build the fixed transform from TCP frame to glasses frame:
            T_tcp_glasses = T_tcp_zed @ inv(T_glasses_zed)

        Then apply one similarity transform directly:
            delta_tcp = T_tcp_glasses @ delta_glasses @ inv(T_tcp_glasses)

        Args:
            delta_headpose: (4, 4) SE3 delta transform expressed in glasses/eye frame.
            T_glasses_zed:  (4, 4) fixed transform from glasses frame to ZED frame.
                            Loaded from DEFAULT_GLASSES_ZED_TXT when None.
            T_tcp_zed:      (4, 4) fixed transform from TCP frame to ZED frame.
                             Loaded from DEFAULT_I2RT_ZED_TXT when None.

        Returns:
            (4, 4) SE3 delta transform expressed in the TCP frame.
        """
        if T_glasses_zed is None:
            T_glasses_zed = _load_calib_mat_safe(Path(DEFAULT_GLASSES_ZED_TXT))
            # print(f"[INFO] Loaded T_glasses_zed from {DEFAULT_GLASSES_ZED_TXT}:\n{T_glasses_zed}")
            if T_glasses_zed is None:
                raise ValueError(f"Failed to load T_glasses_zed from {DEFAULT_GLASSES_ZED_TXT}")

        if T_tcp_zed is None:
            T_tcp_zed = _load_calib_mat_safe(Path(DEFAULT_I2RT_ZED_TXT))
            # print(f"[INFO] Loaded T_tcp_zed from {DEFAULT_I2RT_ZED_TXT}:\n{T_tcp_zed}")
            if T_tcp_zed is None:
                raise ValueError(f"Failed to load T_tcp_zed from {DEFAULT_I2RT_ZED_TXT}")

        T_glasses_zed = np.asarray(T_glasses_zed, dtype=np.float32)
        T_tcp_zed = np.asarray(T_tcp_zed, dtype=np.float32)
        delta = np.asarray(delta_headpose, dtype=np.float32)

        T_zed_glasses = np.linalg.inv(T_glasses_zed)
        T_tcp_glasses = T_tcp_zed @ T_zed_glasses
        T_glasses_tcp = np.linalg.inv(T_tcp_glasses)
        delta_tcp = T_tcp_glasses @ delta @ T_glasses_tcp

        return delta_tcp

    def _publish_head_cmd(self, target_xyz_rot6d: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = target_xyz_rot6d.astype(np.float32).ravel().tolist()
        self._head_cmd_pub.publish(msg)

    
    def save_scene_images(self, output_dir, episode_idx):
        """Save scene images to output directory"""
        left_img_bgr, right_img_bgr = self._get_camera_frames()
        if left_img_bgr is None or right_img_bgr is None:
            return None, None
        left_img = cv2.cvtColor(left_img_bgr.copy(), cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img_bgr.copy(), cv2.COLOR_BGR2RGB)
        Image.fromarray(left_img).save(f"{output_dir}/left_{episode_idx}.png")
        Image.fromarray(right_img).save(f"{output_dir}/right_{episode_idx}.png")
        return left_img, right_img
    
    def align_with_reference(self, ref_left_img, ref_right_img, raw=False):
        print("=====================================================align_with_reference")
        """Align current scene with reference images"""
        cv2.namedWindow("Left", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Right", cv2.WINDOW_AUTOSIZE)

        while (not input().strip().upper() == 'C'):
            state_data = self.get_robot_state()
        if raw:
            left_img = state_data['left_img_raw']
            right_img = state_data['right_img_raw']
        else:
            left_img = cv2.cvtColor(state_data['demo_left_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
            right_img = cv2.cvtColor(state_data['demo_right_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow("Left", (np.array(left_img) * 0.5 + np.array(ref_left_img) * 0.5).astype(np.uint8))
        cv2.imshow("Right", (np.array(right_img) * 0.5 + np.array(ref_right_img) * 0.5).astype(np.uint8))
        cv2.waitKey(1)
    
    def align_scene_with_file(self, output_dir, episode_idx):
        """Align current scene with reference images from a given file path"""
        ref_left_img = cv2.imread(f"{output_dir}/left_{episode_idx}.png")
        ref_right_img = cv2.imread(f"{output_dir}/right_{episode_idx}.png")
        self.align_with_reference(ref_left_img, ref_right_img, raw=True)
    
    def detach_sigma(self):
        """Detach sigma device and store TCP pose"""
        self.sigma.detach()
        detach_tcp, _, _, _ = self.robot.get_robot_state()
        detach_pos = np.array(detach_tcp[:3])
        detach_rot = R.from_quat(np.array(detach_tcp[3:]), scalar_first=True)
        return detach_pos, detach_rot
    
    def human_teleop_step(self, last_p, last_r):
        """Execute one step of human teleoperation"""
        start_time = time.time()
        
        # Get camera data and robot state
        state_data = self.get_robot_state()
        tcp_pose = state_data['tcp_pose']
        joint_pos = state_data['joint_pos']
        
        # Get teleop controls
        diff_p, diff_r, width = self.sigma.get_control()
        diff_p = self.robot.init_pose[:3] + diff_p
        diff_r = R.from_quat(self.robot.init_pose[3:7], scalar_first=True) * diff_r
        curr_p_action = diff_p - last_p
        curr_r_action = last_r.inv() * diff_r
        last_p = diff_p
        last_r = diff_r
        
        # Check throttle pedal state (for teleop pausing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT :
                self.keyboard.quit = True
        
        throttle = self.controller.get_throttle()
        if throttle < -0.9:
            if not self.last_throttle:
                self.sigma.detach()
                self.last_throttle = True
            return None, last_p, last_r
        
        if self.last_throttle:
            self.last_throttle = False
            self.sigma.resume()
            last_p, last_r, _ = self.sigma.get_control()
            last_p = last_p + self.robot.init_pose[:3]
            last_r = R.from_quat(self.robot.init_pose[3:7], scalar_first=True) * last_r
            return None, last_p, last_r
        
        # Send command to robot
        self.robot.send_tcp_pose(np.concatenate((diff_p, diff_r.as_quat(scalar_first=True)), 0))
        self.gripper.move_from_sigma(width)
        gripper_action = self.gripper.max_width * width / 1000
        
        # Save demo data for return
        processed_data = {
            'policy_right_img': state_data['policy_right_img'],
            'policy_left_img': state_data['policy_left_img'],
            'demo_right_img': state_data['demo_right_img'],
            'demo_left_img': state_data['demo_left_img'],
            'tcp_pose': tcp_pose,
            'joint_pos': joint_pos,
            'action': np.concatenate((curr_p_action, curr_r_action.as_quat(scalar_first=True), [gripper_action])),
            'action_mode': INTV
        }
        
        # Sleep to maintain fps
        time.sleep(max(1 / self.fps - (time.time() - start_time), 0))
        
        return processed_data, diff_p, diff_r
    
    def rewind_robot(self, curr_pos, curr_rot, inverse_action):
        """Rewind the robot by applying inverse actions"""

        p_action = inverse_action[:3]
        r_action = inverse_action[3:7]
        gripper_action = inverse_action[7]
        
        # Apply inverse action
        curr_pos = curr_pos - p_action
        curr_rot = curr_rot * R.from_quat(r_action, scalar_first=True).inv()
        
        # Send command
        self.robot.send_tcp_pose(np.concatenate((curr_pos, curr_rot.as_quat(scalar_first=True)), 0))
        self.gripper.move(gripper_action)
        
        return curr_pos, curr_rot


if __name__ == "__main__": # Test the robustness of Sigma teleoperation# t
    # Example usage of RobotEnv
    robot_env = RobotEnv(camera_serial=CAM_SERIAL, img_shape=(3, 224, 224), fps=10)
    robot_env.reset_robot()
    
    last_p = robot_env.robot.init_pose[:3]
    last_r = R.from_quat(robot_env.robot.init_pose[3:7], scalar_first=True)
    
    while True:
        processed_data, last_p, last_r = robot_env.human_teleop_step(last_p, last_r)
        if processed_data is None:
            continue
        
        if robot_env.keyboard.infer:
            detach_pos, detach_rot = robot_env.detach_sigma()
            random_p = robot_env.robot.init_pose[:3] + (np.random.rand(3) - np.array([0.5, 0.5, 0])) * np.ones((3,)) * 0.3
            random_r = (R.from_quat(robot_env.robot.init_pose[3:7], scalar_first=True) * R.from_euler('xyz', (np.random.rand(3) - 0.5) * np.pi / 6, degrees=False)).as_quat(scalar_first=True)
            robot_env.deploy_action(np.concatenate((random_p, random_r),0), robot_env.gripper.max_width)
            time.sleep(3)

            robot_env.keyboard.infer = False

            while not robot_env.keyboard.ctn:
                cv2.waitKey(1)
            robot_env.keyboard.ctn = False

            robot_env.sigma.resume()
            translate = random_p - detach_pos
            rotation = detach_rot.inv() * R.from_quat(random_r,scalar_first=True)
            robot_env.sigma.transform_from_robot(translate, rotation)
        
        # Break condition for demo
        if robot_env.keyboard.quit:
            break
