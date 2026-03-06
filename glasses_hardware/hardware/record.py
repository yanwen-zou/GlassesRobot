import time
import os
import h5py
import numpy as np
import argparse
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from robot_env import RobotEnv
from my_device.macros import CAM_SERIAL


def _get_next_episode_id(h5file: h5py.File) -> int:
    if "episodes" not in h5file:
        h5file.create_group("episodes")
        return 0
    episodes = h5file["episodes"]
    if len(episodes) == 0:
        return 0
    ids = []
    for name in episodes.keys():
        if name.startswith("episode_"):
            try:
                ids.append(int(name.split("_", 1)[1]))
            except ValueError:
                continue
    return (max(ids) + 1) if ids else 0


def _write_episode(h5file: h5py.File, episode: dict) -> int:
    episode_id = _get_next_episode_id(h5file)
    grp = h5file["episodes"].create_group(f"episode_{episode_id}")
    for key, value in episode.items():
        grp.create_dataset(key, data=value, compression="gzip", compression_opts=4)
    h5file.flush()
    return episode_id


class HeadPoseSubscriber(Node):
    def __init__(self, pose_topic: str):
        super().__init__("record_headpose_sub")
        self._latest = None
        self.create_subscription(PoseStamped, pose_topic, self._on_pose, 10)

    def _on_pose(self, msg: PoseStamped) -> None:
        self._latest = msg

    def get_latest(self):
        rclpy.spin_once(self, timeout_sec=0.0)
        return self._latest


def record(h5file: h5py.File, robot_env: RobotEnv, headpose_sub: HeadPoseSubscriber | None = None):
    tcp_pose = []
    joint_pos = []
    action = []
    right_cam = []
    left_cam = []
    headpose = []
    robot_state = []

    robot_env.keyboard.start = False
    robot_env.keyboard.discard = False
    robot_env.keyboard.finish = False
    robot_env.keyboard.manual_reset = False
    cnt = 0
    episode_start_time = None

    # robot_env.reset_robot()
    last_p = robot_env.robot.init_pose[:3]
    last_r = R.from_quat(robot_env.robot.init_pose[3:7], scalar_first=True)

    seed = int(time.time())
    np.random.seed(seed)

    while not robot_env.keyboard.quit and not robot_env.keyboard.discard and not robot_env.keyboard.finish:
        if robot_env.keyboard.manual_reset:
            print("Manual reset requested.")
            robot_env.reset_robot()
            last_p = robot_env.robot.init_pose[:3]
            last_r = R.from_quat(robot_env.robot.init_pose[3:7], scalar_first=True)
            robot_env.keyboard.manual_reset = False
            continue

        transition_data, last_p, last_r = robot_env.human_teleop_step(last_p, last_r)
        if not robot_env.keyboard.start or transition_data is None:
            continue

        # Initialize at the beginning of the episode
        if cnt == 0:
            cnt += 1
            episode_start_time = time.time()
            print("Episode start!")
        
        right_cam.append(transition_data['demo_right_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        left_cam.append(transition_data['demo_left_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        tcp_pose.append(transition_data['tcp_pose'])
        joint_pos.append(transition_data['joint_pos'])
        gripper_state = float(robot_env.gripper.get_gripper_state())
        action.append(np.concatenate([last_p, last_r.as_quat(scalar_first=True), [gripper_state]], axis=0))
        robot_state.append(np.concatenate([transition_data['tcp_pose'], [gripper_state]], axis=0))
        if headpose_sub is not None:
            msg = headpose_sub.get_latest()
            if msg is None:
                headpose.append(np.full((7,), np.nan, dtype=np.float32))
                print("No headpose message received! Writing NaN. ")
            else:
                headpose.append(
                    np.array(
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
                )

    if not robot_env.keyboard.start or robot_env.keyboard.quit or robot_env.keyboard.discard:
        print('WARNING: discard the demo!')
        # robot_env.gripper.move(robot_env.gripper.max_width)
        time.sleep(0.5)
        if episode_start_time is not None:
            elapsed_sec = time.time() - episode_start_time
            print(f"Discarded episode elapsed time (including detach): {elapsed_sec:.3f}s")
        return
    
    episode = dict()
    episode['right_cam'] = np.stack(right_cam, axis=0)
    episode['left_cam'] = np.stack(left_cam, axis=0)
    episode['tcp_pose'] = np.stack(tcp_pose, axis=0)
    episode['robot_state'] = np.stack(robot_state, axis=0)
    episode['joint_pos'] = np.stack(joint_pos, axis=0)
    episode['action'] = np.stack(action, axis=0)
    if headpose:
        episode['headpose'] = np.stack(headpose, axis=0)

    # robot_env.gripper.move(robot_env.gripper.max_width)
    time.sleep(0.5)
    if episode_start_time is not None:
        elapsed_sec = time.time() - episode_start_time
    else:
        elapsed_sec = 0.0
    # episode['episode_duration_sec'] = np.array(elapsed_sec, dtype=np.float32)

    episode_id = _write_episode(h5file, episode)
    print('Saved episode ', episode_id)
    print(f"Episode {episode_id} elapsed time (including detach): {elapsed_sec:.3f}s")


def main(args):
    headpose_sub = None
    if args.enable_headpose:
        if not rclpy.ok():
            rclpy.init(args=None)
        headpose_sub = HeadPoseSubscriber(args.headpose_topic)
    if args.resolution is None:
        args.resolution = [224, 224]
    robot_env = RobotEnv(camera_serial=CAM_SERIAL, img_shape=[3] + args.resolution, fps=args.fps)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output, exist_ok=True)
    h5_path = os.path.join(args.output, f"replay_buffer_{timestamp}.hdf5")
    with h5py.File(h5_path, "a") as h5file:
        while not robot_env.keyboard.quit:
            print("start recording...")
            record(h5file, robot_env, headpose_sub=headpose_sub)
            if not robot_env.keyboard.quit:
                print("reset the environment...")
                time.sleep(2)
    if headpose_sub is not None:
        headpose_sub.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-res', '--resolution', nargs='+', type=int)
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--enable-headpose', action='store_true')
    parser.add_argument('--headpose-topic', type=str, default='/glasses_pose')
    args = parser.parse_args()
    main(args)
