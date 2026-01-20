"""
Keyboard teleop for the arm using TCP pose (position/orientation increments).

Controls (world frame):
    q/e : +X / -X (m)
    a/d : +Y / -Y (m)
    w/s : +Z / -Z (m)
    u/o : roll + / - (rad)
    i/k : pitch + / - (rad)
    j/l : yaw + / - (rad)
Press Ctrl+C to quit. Each key applies a small delta and solves IK to move there.
"""
from __future__ import annotations

import argparse
import sys
import termios
import tty
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

# Ensure repo modules import
here = Path(__file__).resolve()
repo_root = here.parents[2]
project_root = repo_root.parent
for path in (project_root, repo_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from glasses_hardware.hardware.my_device.i2rt_robo import I2RT
from i2rt.robots.kinematics_mj import Kinematics
from i2rt.robots.utils import YAM_XML_PATH


def se3_to_pos_quat(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(T[:3, 3], dtype=np.float32)
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    return pos, quat_xyzw


def apply_delta(T: np.ndarray, dpos: np.ndarray, drot_rpy: np.ndarray) -> np.ndarray:
    """Apply position + rpy deltas in world frame."""
    pos, quat_xyzw = se3_to_pos_quat(T)
    pos_new = pos + dpos.astype(np.float32)
    rot = R.from_quat(quat_xyzw) * R.from_euler("xyz", drot_rpy)
    T_new = np.eye(4, dtype=np.float32)
    T_new[:3, :3] = rot.as_matrix().astype(np.float32)
    T_new[:3, 3] = pos_new
    return T_new


def get_key() -> str:
    """Read a single keypress (non-blocking with minimal blocking)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main() -> None:
    ap = argparse.ArgumentParser(description="Keyboard teleop for I2RT.")
    ap.add_argument(
        "--mode",
        choices=("tcp", "joint"),
        default="tcp",
        help="Control mode: TCP pose via IK or direct joint increments.",
    )
    ap.add_argument("--pos-step", type=float, default=0.02, help="TCP position step (m).")
    ap.add_argument("--rot-step", type=float, default=0.05, help="TCP rotation step (rad).")
    ap.add_argument("--joint-step", type=float, default=0.05, help="Joint step (rad).")
    args = ap.parse_args()

    robot = I2RT(channel="can0", zero_gravity_mode=False, home=True)
    model = Kinematics(YAM_XML_PATH, "grasp_site")

    q_start = robot.current_joint_pos()
    target_pose = model.fk(q_start[:6])
    target_q = q_start.copy()

    if args.mode == "tcp":
        print("TCP keyboard teleop started.")
        print("q/e:+/-X  a/d:+/-Y  w/s:+/-Z  u/o:+/-roll  i/k:+/-pitch  j/l:+/-yaw  Ctrl+C to exit.")
    else:
        print("Joint keyboard teleop started.")
        print("q/e:j0  a/d:j1  w/s:j2  u/o:j3  i/k:j4  j/l:j5  Ctrl+C to exit.")

    while True:
        key = get_key()
        if key == "\x03":  # Ctrl+C
            print("Exiting teleop.")
            break

        if args.mode == "tcp":
            dpos = np.zeros(3, dtype=np.float32)
            drot = np.zeros(3, dtype=np.float32)
            if key == "q":
                dpos[0] += args.pos_step
            elif key == "e":
                dpos[0] -= args.pos_step
            elif key == "a":
                dpos[1] += args.pos_step
            elif key == "d":
                dpos[1] -= args.pos_step
            elif key == "w":
                dpos[2] += args.pos_step
            elif key == "s":
                dpos[2] -= args.pos_step
            elif key == "u":
                drot[0] += args.rot_step
            elif key == "o":
                drot[0] -= args.rot_step
            elif key == "i":
                drot[1] += args.rot_step
            elif key == "k":
                drot[1] -= args.rot_step
            elif key == "j":
                drot[2] += args.rot_step
            elif key == "l":
                drot[2] -= args.rot_step
            else:
                continue

            target_pose = apply_delta(target_pose, dpos, drot)
            success, q_sol = model.ik(target_pose, "grasp_site", verbose=False)
            if not success:
                print(f"[WARN] IK failed for key '{key}', skipping.")
                continue
            target_q = q_sol
        else:
            dq = np.zeros_like(target_q)
            if key == "q":
                dq[0] += args.joint_step
            elif key == "e":
                dq[0] -= args.joint_step
            elif key == "a":
                dq[1] += args.joint_step
            elif key == "d":
                dq[1] -= args.joint_step
            elif key == "w":
                dq[2] += args.joint_step
            elif key == "s":
                dq[2] -= args.joint_step
            elif key == "u":
                dq[3] += args.joint_step
            elif key == "o":
                dq[3] -= args.joint_step
            elif key == "i":
                dq[4] += args.joint_step
            elif key == "k":
                dq[4] -= args.joint_step
            elif key == "j":
                dq[5] += args.joint_step
            elif key == "l":
                dq[5] -= args.joint_step
            else:
                continue
            target_q = target_q + dq

        robot.send_joint_pos_rad(target_q, duration=0.1, steps=20)
        print(f"Sent target joint pos: {np.round(target_q, 4)}")
        time.sleep(0.01)


if __name__ == "__main__":
    main()
