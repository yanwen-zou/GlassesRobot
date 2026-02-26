import dataclasses
from datetime import datetime
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rclpy
import tyro
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from openpi_client import image_tools
from openpi_client import websocket_client_policy

here = Path(__file__).resolve()
project_root = here.parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
hardware_root = project_root / "glasses_hardware" / "hardware"
if str(hardware_root) not in sys.path:
    sys.path.insert(0, str(hardware_root))

from my_device.macros import CAM_SERIAL  # type: ignore
from glasses_hardware.hardware.robot_env import RobotEnv  # type: ignore

@dataclasses.dataclass
class Args:
    # Policy server address.
    host: str = "localhost"
    port: int = 8000
    api_key: str | None = None

    # Robot env + headpose.
    headpose_topic: str = "/glasses_pose"
    enable_headpose: bool = True
    # If true, concatenate headpose [x,y,z,qx,qy,qz,qw] to state.
    # NOTE: current `openpi_book_0225` checkpoint uses 8-dim state norm stats,
    # so this should stay False unless your checkpoint was trained with 15-dim state.
    append_headpose_to_state: bool = False

    prompt: str = "Put the book in the shelf"

    # Preprocessing.
    image_size: int = 224
    # Visualize model input image ("observation/image") in a cv2 window.
    visualize_input_image: bool = True

    # Inference loop.
    num_steps: int = 300

    # Save first headpose + policy delta headposes to a timestamped .npz in this directory.
    headpose_log_dir: str = "baseline/openpi/logs"


class HeadPoseSubscriber(Node):
    def __init__(self, pose_topic: str):
        super().__init__("pi05_realworld_headpose_sub")
        self._latest = None
        self.create_subscription(PoseStamped, pose_topic, self._on_pose, 10)

    def _on_pose(self, msg: PoseStamped) -> None:
        self._latest = msg

    def get_headpos(self, timeout_sec: float = 0.0):
        rclpy.spin_once(self, timeout_sec=0.0)
        return self._latest


def _prepare_image_uint8_from_left_bgr(left_bgr: np.ndarray, image_size: int) -> np.ndarray:
    if left_bgr is None:
        raise RuntimeError("ZED returned empty left frame.")
    if left_bgr.ndim != 3 or left_bgr.shape[2] < 3:
        raise ValueError(f"Unexpected left frame shape: {left_bgr.shape}")
    bgr = np.asarray(left_bgr[:, :, :3], dtype=np.uint8)
    rgb = bgr[:, :, ::-1]
    rgb = image_tools.resize_with_pad(rgb, image_size, image_size)
    rgb = image_tools.convert_to_uint8(rgb)
    return np.asarray(rgb, dtype=np.uint8)


def _load_state(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        state = np.load(path)
    else:
        state = np.loadtxt(path, dtype=np.float32, delimiter=",")
    return np.asarray(state, dtype=np.float32).reshape(-1)


def _visualize_model_input_image(image_rgb_uint8: np.ndarray) -> None:
    if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[2] != 3:
        raise ValueError(f"Unexpected model input image shape: {image_rgb_uint8.shape}")
    # image_bgr = image_rgb_uint8[:, :, ::-1]
    cv2.imshow("Left cam", image_rgb_uint8)
    cv2.waitKey(1)


def _headpose_to_vec(msg: PoseStamped | None) -> np.ndarray:
    if msg is None:
        raise RuntimeError("No headpose message received from ROS topic.")
    return np.array(
        [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ],
        dtype=np.float32,
    )


def _build_state_from_robot_state(
    robot_state: dict,
    gripper_state: float,
    headpose_rel: np.ndarray | None,
    append_headpose_to_state: bool,
) -> np.ndarray:
    tcp_pose = np.asarray(robot_state["tcp_pose"], dtype=np.float32).reshape(-1)
    gripper_arr = np.array([float(gripper_state)], dtype=np.float32)
    state = np.concatenate([tcp_pose, gripper_arr], axis=0)
    if append_headpose_to_state and headpose_rel is not None:
        state = np.concatenate([state, headpose_rel.astype(np.float32)], axis=0)
    return state.astype(np.float32, copy=False)


def main(args: Args) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )

    logging.info("Server metadata: %s", client.get_server_metadata())

    robot_env = RobotEnv(
        camera_serial=CAM_SERIAL,
        img_shape=[3, args.image_size, args.image_size],
        fps=10,
        is_infer_mode=True,
    )
    headpose_sub = None
    first_hp_vec: np.ndarray | None = None
    logged_delta_headpose: list[np.ndarray] = []
    logged_step_idx: list[int] = []
    logged_action_idx: list[int] = []
    logged_wall_time: list[float] = []
    if args.enable_headpose:
        if not rclpy.ok():
            rclpy.init(args=None)
        headpose_sub = HeadPoseSubscriber(args.headpose_topic)
        print("[INFO] Subscribed to headpose topic:", args.headpose_topic)
        time.sleep(1.0)  # Wait a moment to ensure we receive the first headpose message.

    try:
        # Capture frame-0 headpose reference before the inference loop.
        if headpose_sub is not None:
            first_hp_msg = headpose_sub.get_headpos(timeout_sec=0.2)
            if first_hp_msg is None:
                raise RuntimeError("No headpose message received before inference loop. "
                                   "Check that the glasses-pose topic is publishing.")
            first_hp_vec = _headpose_to_vec(first_hp_msg)
            logging.info("First headpose (abs) recorded: %s", first_hp_vec)
            if args.append_headpose_to_state:
                # For optional state concat only.
                robot_env.set_headpose_ref(first_hp_vec)
                logging.info("Headpose reference set for state concat: %s", first_hp_vec)

        for step in range(args.num_steps):
            rs = robot_env.get_robot_state()
            left_bgr = rs.get("demo_left_img")
            if left_bgr is None:
                raise RuntimeError("Failed to read left image from RobotEnv/ZED.")
            left_bgr = left_bgr.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            left_cam = left_bgr[..., ::-1]

            if args.visualize_input_image:
                _visualize_model_input_image(left_bgr)

            # Read current headpose (absolute) from the ROS topic.
            headpose_abs_vec = (
                _headpose_to_vec(headpose_sub.get_headpos(timeout_sec=0.2)) if headpose_sub is not None else None
            )

            # Convert to relative-to-first for state, matching the training convention.
            headpose_rel = (
                robot_env.compute_headpose_rel(headpose_abs_vec)
                if (headpose_abs_vec is not None and args.append_headpose_to_state)
                else None
            )

            gripper_state = float(robot_env.gripper.get_gripper_state())
            state = _build_state_from_robot_state(
                rs,
                gripper_state,
                headpose_rel,
                args.append_headpose_to_state,
            )
            logging.info(
                "[step %d] state_dim=%d (append_headpose_to_state=%s)",
                step,
                int(state.shape[-1]),
                args.append_headpose_to_state,
            )

            observation = {
                "observation/image": left_cam,
                "observation/state": state,
                "prompt": args.prompt,
            }

            result = client.infer(observation)
            actions = np.asarray(result["actions"], dtype=np.float32)
            logging.info("[step %d] Observation/State: %s", step, state)
            logging.info("[step %d] state.shape=%s  actions.shape=%s", step, state.shape, actions.shape)

            # Execute actions.
            # The policy may return a single action (shape (15,)) or an action
            # chunk (shape (N, 15)).  Execute all steps in the chunk.
            action_chunk = actions.reshape(-1, actions.shape[-1])
            for action_idx_in_chunk, action in enumerate(action_chunk[:5]):
                print("[INFO] Executing action:", action[:15])
                # Robot TCP pose (7) + gripper (1)
                robot_env.deploy_action(action[:7], float(action[7]))
                # Headpose: action[8:15] is delta from current headpose
                # [x,y,z,qx,qy,qz,qw] in glasses frame.
                if args.enable_headpose and headpose_sub is not None:
                    delta_headpose = np.asarray(action[8:15], dtype=np.float32).copy()
                    logged_delta_headpose.append(delta_headpose)
                    logged_step_idx.append(int(step))
                    logged_action_idx.append(int(action_idx_in_chunk))
                    logged_wall_time.append(float(time.time()))
                    robot_env.execute_headpose_from_current_delta(delta_headpose)
                time.sleep(0.5)
    finally:
        if args.enable_headpose and first_hp_vec is not None:
            log_dir = Path(args.headpose_log_dir)
            if not log_dir.is_absolute():
                log_dir = project_root / log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"policy_headpose_delta_log_{ts}.npz"
            delta_arr = (
                np.stack(logged_delta_headpose, axis=0).astype(np.float32)
                if len(logged_delta_headpose) > 0
                else np.empty((0, 7), dtype=np.float32)
            )
            np.savez(
                log_path,
                first_headpose_abs=first_hp_vec.astype(np.float32),
                policy_delta_headpose=delta_arr,
                step_idx=np.asarray(logged_step_idx, dtype=np.int32),
                action_idx=np.asarray(logged_action_idx, dtype=np.int32),
                wall_time=np.asarray(logged_wall_time, dtype=np.float64),
            )
            logging.info("Saved headpose delta log to: %s (num_deltas=%d)", str(log_path), int(delta_arr.shape[0]))
        logging.info("Shutting down...")
        robot_env.close()
        if args.visualize_input_image:
            cv2.destroyAllWindows()
        if headpose_sub is not None:
            headpose_sub.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
