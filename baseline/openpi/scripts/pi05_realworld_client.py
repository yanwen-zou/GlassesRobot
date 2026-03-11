import dataclasses
from datetime import datetime
import logging
from pathlib import Path
import sys
import time
from typing import Literal

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

    append_headpose_to_state: bool = True

    prompt: str = "Put the book in the shelf"

    # Preprocessing.
    image_size: int = 224
    # Visualize model input image ("observation/image") in a cv2 window.
    visualize_input_image: bool = True

    # Inference loop.
    num_steps: int = 300

    # Save first headpose + policy delta headposes to a timestamped .npz in this directory.
    headpose_log_dir: str = "baseline/openpi/logs"

    task_name: str = "book"


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


def _visualize_model_input_image(image_rgb_uint8: np.ndarray) -> int:
    if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[2] != 3:
        raise ValueError(f"Unexpected model input image shape: {image_rgb_uint8.shape}")
    # image_bgr = image_rgb_uint8[:, :, ::-1]
    cv2.imshow("Left cam", image_rgb_uint8)
    return cv2.waitKey(1) & 0xFF


def _wait_startup_gripper_command() -> Literal["p", "skip"]:
    while True:
        user_input = input("启动前输入'p'闭合夹爪，直接回车跳过：").strip().lower()
        if user_input == "p":
            return "p"
        if user_input == "":
            return "skip"
        print("仅接受 'p' 或直接回车，请重新输入。")


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


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-12, None)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def _absolute_from_chunk_first_action(relative_actions: np.ndarray, first_action_abs: np.ndarray) -> np.ndarray:
    rel = np.asarray(relative_actions, dtype=np.float32)
    base = np.asarray(first_action_abs, dtype=np.float32).reshape(-1)
    if rel.ndim != 2 or rel.shape[0] == 0:
        return rel
    if base.shape[0] < 8:
        raise ValueError(f"first_action_abs must have at least 8 dims, got {base.shape[0]}")
    if rel.shape[1] > base.shape[0]:
        raise ValueError(f"Relative action dim {rel.shape[1]} > base dim {base.shape[0]}")

    out = rel.copy()
    p0 = base[:3]
    q0_wxyz = base[3:7]
    g0 = base[7:8]
    print("Debug: gripper state in first action (base):", g0)
    q0_xyzw = _quat_normalize(_quat_wxyz_to_xyzw(q0_wxyz[None, :]))[0]

    out[:, :3] = _quat_rotate(q0_xyzw[None, :], rel[:, :3]) + p0[None, :]
    out[:, 3:7] = _quat_xyzw_to_wxyz(
        _quat_normalize(_quat_mul(q0_xyzw[None, :], _quat_wxyz_to_xyzw(rel[:, 3:7])))
    )
    out[:, 7:8] = rel[:, 7:8] + g0[None, :]

    if rel.shape[1] >= 15 and base.shape[0] >= 15:
        t0 = base[8:11]
        hq0 = _quat_normalize(base[11:15][None, :])[0]
        out[:, 8:11] = _quat_rotate(hq0[None, :], rel[:, 8:11]) + t0[None, :]
        out[:, 11:15] = _quat_normalize(_quat_mul(hq0[None, :], rel[:, 11:15]))
    return out


def _compose_action_with_state(relative_action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
    rel = np.asarray(relative_action, dtype=np.float32).reshape(-1)
    base = np.asarray(current_state, dtype=np.float32).reshape(-1)
    if rel.shape[0] < 8 or base.shape[0] < 8:
        raise ValueError(f"Need at least 8 dims for rel/base, got rel={rel.shape[0]}, base={base.shape[0]}")
    if rel.shape[0] > base.shape[0]:
        raise ValueError(f"Relative action dim {rel.shape[0]} > state dim {base.shape[0]}")

    out = rel.copy()

    p0 = base[:3]
    q0_xyzw = _quat_normalize(_quat_wxyz_to_xyzw(base[3:7][None, :]))[0]
    out[:3] = _quat_rotate(q0_xyzw[None, :], rel[None, :3])[0] + p0
    out[3:7] = _quat_xyzw_to_wxyz(
        _quat_normalize(_quat_mul(q0_xyzw[None, :], _quat_wxyz_to_xyzw(rel[None, 3:7])))[0]
    )
    out[7] = rel[7] + base[7]

    if rel.shape[0] >= 15 and base.shape[0] >= 15:
        hp_t0 = base[8:11]
        hp_q0 = _quat_normalize(base[11:15][None, :])[0]
        out[8:11] = _quat_rotate(hp_q0[None, :], rel[None, 8:11])[0] + hp_t0
        out[11:15] = _quat_normalize(_quat_mul(hp_q0[None, :], rel[None, 11:15]))[0]
    return out


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
        task_name=args.task_name,
    )
    robot_env.gripper.move(0.085)  # Start with gripper open.
    startup_cmd = _wait_startup_gripper_command()
    if startup_cmd == "p":
        robot_env.gripper.move(0.0)
        logging.info("Startup command received: close gripper.")
    else:
        logging.info("Startup command skipped: keep current gripper state.")
    headpose_sub = None
    first_hp_vec: np.ndarray | None = None
    logged_delta_headpose: list[np.ndarray] = []
    logged_step_idx: list[int] = []
    logged_action_idx: list[int] = []
    logged_wall_time: list[float] = []
    logged_infer_step: list[int] = []
    logged_infer_time: list[float] = []
    logged_robot_state: list[np.ndarray] = []
    logged_headpose_state: list[np.ndarray] = []
    logged_action_abs: list[np.ndarray] = []
    logged_left_cam_raw: list[np.ndarray] = []
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

            key = -1
            if args.visualize_input_image:
                key = _visualize_model_input_image(left_bgr)
                if key == 27:
                    logging.info("ESC pressed, stopping inference loop.")
                    break

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
            actions = result.get("actions")
            action_chunk_abs = _absolute_from_chunk_first_action(actions[:, :15], state)
            action_chunk_rel = actions

            logged_infer_step.append(step)
            logged_infer_time.append(time.time())
            logged_robot_state.append(state[:8].astype(np.float32, copy=True))
            if state.shape[0] >= 15:
                logged_headpose_state.append(state[8:15].astype(np.float32, copy=True))
            else:
                logged_headpose_state.append(np.full((7,), np.nan, dtype=np.float32))
            logged_action_abs.append(np.asarray(action_chunk_abs, dtype=np.float32).copy())
            logged_left_cam_raw.append(left_bgr.copy())

            logging.info("[step %d]", step)

            # Execute actions.
            # Robot action: absolute TCP pose (xyz + quat wxyz) + gripper
            # Headpose: relative-to-first-frame target (xyz + quat xyzw)
            robot_env.update_current_pose()  # Ensure we have the latest current pose before each execution step.
            for action_idx_in_chunk, action_abs in enumerate(action_chunk_abs[:]):
                print("[INFO] Executing robot ABS action:", action_abs[:])
                # Robot TCP pose (7) + gripper (1)
                robot_env.deploy_action(action_abs[:7], float(action_abs[7]))
                # Headpose: action[8:15] is delta from current headpose
                # [x,y,z,qx,qy,qz,qw] in glasses frame.
                if args.enable_headpose and headpose_sub is not None:
                    headpose_action_rel = np.asarray(action_chunk_rel[action_idx_in_chunk, 8:15], dtype=np.float32).copy()
                    print("[DEBUG] Headpose action relative xyz: ", headpose_action_rel[:3])
                    robot_env.execute_headpose_from_current_delta(headpose_action_rel)

    finally:
        log_dir = Path(args.headpose_log_dir)
        if not log_dir.is_absolute():
            log_dir = project_root / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        infer_log_path = log_dir / f"policy_inference_step_log_{ts}.npz"
        max_action_len = max((chunk.shape[0] for chunk in logged_action_abs), default=0)
        if max_action_len > 0:
            action_abs_arr = np.full((len(logged_action_abs), max_action_len, 15), np.nan, dtype=np.float32)
            for i, chunk in enumerate(logged_action_abs):
                n = chunk.shape[0]
                action_abs_arr[i, :n, : min(15, chunk.shape[1])] = chunk[:, :15]
        else:
            action_abs_arr = np.empty((0, 0, 15), dtype=np.float32)

        np.savez(
            infer_log_path,
            infer_step=np.asarray(logged_infer_step, dtype=np.int32),
            infer_time=np.asarray(logged_infer_time, dtype=np.float64),
            robot_state=np.asarray(logged_robot_state, dtype=np.float32),
            headpose_state=np.asarray(logged_headpose_state, dtype=np.float32),
            action_abs=action_abs_arr,
        )
        logging.info("Saved inference step log to: %s (num_steps=%d)", str(infer_log_path), len(logged_infer_step))

        if logged_left_cam_raw:
            video_path = log_dir / f"left_cam_raw_{ts}.mp4"
            height, width = logged_left_cam_raw[0].shape[:2]
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (width, height),
            )
            if writer.isOpened():
                for frame in logged_left_cam_raw:
                    writer.write(np.asarray(frame, dtype=np.uint8))
                writer.release()
                logging.info("Saved left raw camera video to: %s (num_frames=%d)", str(video_path), len(logged_left_cam_raw))
            else:
                logging.warning("Failed to open VideoWriter for: %s", str(video_path))

        if args.enable_headpose and first_hp_vec is not None:
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
