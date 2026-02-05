import time
import os
import h5py
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

from hardware.robot_env import RobotEnv
from hardware.my_device.macros import CAM_SERIAL
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


def record(h5file: h5py.File, robot_env:RobotEnv):
    tcp_pose = []
    joint_pos = []
    action = []
    right_cam = []
    left_cam = []

    robot_env.keyboard.start = False
    robot_env.keyboard.discard = False
    robot_env.keyboard.finish = False
    cnt = 0

    robot_env.reset_robot()
    last_p = robot_env.robot.init_pose[:3]
    last_r = R.from_quat(robot_env.robot.init_pose[3:7], scalar_first=True)

    seed = int(time.time())
    np.random.seed(seed)

    while not robot_env.keyboard.quit and not robot_env.keyboard.discard and not robot_env.keyboard.finish:
        transition_data, last_p, last_r = robot_env.human_teleop_step(last_p, last_r)
        if not robot_env.keyboard.start or transition_data is None:
            continue

        # Initialize at the beginning of the episode
        if cnt == 0:
            random_init_pose = robot_env.robot.init_pose + np.random.uniform(-0.1, 0.1, size=7)
            robot_env.reset_robot(random_init=True, random_init_pose=random_init_pose)
            last_p = random_init_pose[:3]
            last_r = R.from_quat(random_init_pose[3:7], scalar_first=True)
            cnt += 1
            print("Episode start!")
            continue
        
        right_cam.append(transition_data['demo_right_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        left_cam.append(transition_data['demo_left_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        tcp_pose.append(transition_data['tcp_pose'])
        joint_pos.append(transition_data['joint_pos'])
        action.append(transition_data['action'])

    if not robot_env.keyboard.start or robot_env.keyboard.quit or robot_env.keyboard.discard:
        print('WARNING: discard the demo!')
        robot_env.gripper.move(robot_env.gripper.max_width)
        time.sleep(0.5)
        return
    
    episode = dict()
    episode['right_cam'] = np.stack(right_cam, axis=0)
    episode['left_cam'] = np.stack(left_cam, axis=0)
    episode['tcp_pose'] = np.stack(tcp_pose, axis=0)
    episode['joint_pos'] = np.stack(joint_pos, axis=0)
    episode['action'] = np.stack(action, axis=0)
    episode_id = _write_episode(h5file, episode)
    print('Saved episode ', episode_id)

    robot_env.gripper.move(robot_env.gripper.max_width)
    time.sleep(0.5)


def main(args):
    robot_env = RobotEnv(camera_serial=CAM_SERIAL, img_shape=[3]+args.resolution, fps=args.fps)
    h5_path = os.path.join(args.output, 'replay_buffer.hdf5')
    with h5py.File(h5_path, "a") as h5file:
        while not robot_env.keyboard.quit:
            print("start recording...")
            record(h5file, robot_env)
            if not robot_env.keyboard.quit:
                print("reset the environment...")
                time.sleep(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-res', '--resolution', nargs='+', type=int)
    parser.add_argument('--fps', type=float, default=10.0)
    args = parser.parse_args()
    main(args)
