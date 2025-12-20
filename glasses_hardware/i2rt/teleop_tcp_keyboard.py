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
import time
import tty
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

def run_sim_teleop(pos_step: float, rot_step: float) -> None:
    """Launch a MuJoCo viewer that mirrors the TCP teleop motions."""
    import glfw
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(YAM_XML_PATH))
    data = mujoco.MjData(model)
    kinematics = Kinematics(YAM_XML_PATH, "grasp_site")

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    window = glfw.create_window(1280, 960, "YAM TCP Teleop (Sim)", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=2000)
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    cam.distance = 2.0
    cam.azimuth = 90
    cam.elevation = -35

    q_target = data.qpos.copy()
    arm_dofs = min(6, q_target.shape[0])
    target_pose = kinematics.fk(q_target[:arm_dofs])

    zero3 = np.zeros(3, dtype=np.float32)

    def key_callback(_win, key, _scancode, action, _mods):
        nonlocal target_pose, q_target
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        dpos = zero3.copy()
        drot = zero3.copy()
        if key == glfw.KEY_Q:
            dpos[0] += pos_step
        elif key == glfw.KEY_E:
            dpos[0] -= pos_step
        elif key == glfw.KEY_A:
            dpos[1] += pos_step
        elif key == glfw.KEY_D:
            dpos[1] -= pos_step
        elif key == glfw.KEY_W:
            dpos[2] += pos_step
        elif key == glfw.KEY_S:
            dpos[2] -= pos_step
        elif key == glfw.KEY_U:
            drot[0] += rot_step
        elif key == glfw.KEY_O:
            drot[0] -= rot_step
        elif key == glfw.KEY_I:
            drot[1] += rot_step
        elif key == glfw.KEY_K:
            drot[1] -= rot_step
        elif key == glfw.KEY_J:
            drot[2] += rot_step
        elif key == glfw.KEY_L:
            drot[2] -= rot_step
        else:
            return
        new_pose = apply_delta(target_pose, dpos, drot)
        success, q_sol = kinematics.ik(new_pose, "grasp_site", verbose=False)
        if success:
            target_pose = new_pose
            q_target[:arm_dofs] = q_sol[:arm_dofs]

    glfw.set_key_callback(window, key_callback)

    try:
        while not glfw.window_should_close(window):
            data.qpos[:] = q_target
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
            mujoco.mjv_updateScene(
                model,
                data,
                opt,
                None,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )
            mujoco.mjr_render(viewport, scene, ctx)

            glfw.swap_buffers(window)
            glfw.poll_events()
            time.sleep(model.opt.timestep)
    finally:
        glfw.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard TCP teleop for YAM robot (real or MuJoCo).")
    parser.add_argument("--mode", choices=("real", "sim"), default="real", help="Run on real hardware or MuJoCo simulation.")
    parser.add_argument("--channel", type=str, default="can0", help="CAN channel for real robot.")
    parser.add_argument("--home", dest="home", action="store_true", help="Home robot at startup (real mode).")
    parser.add_argument("--no-home", dest="home", action="store_false", help="Skip homing at startup.")
    parser.add_argument("--pos-step", type=float, default=0.02, help="Translation step size in meters.")
    parser.add_argument("--rot-step", type=float, default=0.05, help="Rotation step size in radians.")
    parser.set_defaults(home=True)
    args = parser.parse_args()

    if args.mode == "sim":
        run_sim_teleop(args.pos_step, args.rot_step)
        return

    robot = I2RT(channel=args.channel, zero_gravity_mode=False, home=args.home)
    model = Kinematics(YAM_XML_PATH, "grasp_site")

    q_start = robot.current_joint_pos()
    target_pose = model.fk(q_start[:6])

    pos_step = float(args.pos_step)
    rot_step = float(args.rot_step)

    print("TCP keyboard teleop started.")
    print("q/e:+/-X  a/d:+/-Y  w/s:+/-Z  u/o:+/-roll  i/k:+/-pitch  j/l:+/-yaw  Ctrl+C to exit.")

    while True:
        key = get_key()
        if key == "\x03":  # Ctrl+C
            print("Exiting teleop.")
            break

        dpos = np.zeros(3, dtype=np.float32)
        drot = np.zeros(3, dtype=np.float32)
        if key == "q":
            dpos[0] += pos_step
        elif key == "e":
            dpos[0] -= pos_step
        elif key == "a":
            dpos[1] += pos_step
        elif key == "d":
            dpos[1] -= pos_step
        elif key == "w":
            dpos[2] += pos_step
        elif key == "s":
            dpos[2] -= pos_step
        elif key == "u":
            drot[0] += rot_step
        elif key == "o":
            drot[0] -= rot_step
        elif key == "i":
            drot[1] += rot_step
        elif key == "k":
            drot[1] -= rot_step
        elif key == "j":
            drot[2] += rot_step
        elif key == "l":
            drot[2] -= rot_step
        else:
            continue

        target_pose = apply_delta(target_pose, dpos, drot)
        success, q_sol = model.ik(target_pose, "grasp_site", verbose=False)
        if not success:
            print(f"[WARN] IK failed for key '{key}', skipping.")
            continue
        robot.send_joint_pos_rad(q_sol, duration=0.1, steps=20)
        time.sleep(0.01)


if __name__ == "__main__":
    main()
