"""
Thin wrapper around the i2rt MotorChain robot so that other parts of the
glasses hardware stack can drive the arm without depending on the full demo
scripts.

Usage:
    controller = I2RT(channel="can0")
    controller.send_joint_pos_deg([0, 30, ...])
    controller.send_joint_pos_rad(np.zeros(7))
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import sys
import portal

# Make repo modules importable when run as a script.
here = Path(__file__).resolve()
repo_root = here.parents[3]  # unity_comm/glasses_hardware/
project_root = repo_root.parent
sdk_root = repo_root / "i2rt"
for path in (project_root, repo_root, sdk_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
sys.modules.pop("i2rt", None)
sys.modules.pop("i2rt.robots", None)

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics_mj import Kinematics
from i2rt.robots.utils import YAM_XML_PATH, YAM_GLASS_PATH
from MBA.utils.transformation import rotation_transform  # type: ignore


DEFAULT_ROBOT_PORT = 11333


class I2RT:
    """User-facing helper that exposes simple joint-space commands."""

    MAX_JOINT_VEL_RAD_S = 0.2
    MAX_JOINT_ACCEL_RAD_S2 = 6.0

    def __init__(
        self,
        channel: str = "can0",
        zero_gravity_mode: bool = True,
        home: bool = True,
        default_duration: float = 2.0,
        default_steps: int = 50,
    ) -> None:
        self._robot = get_yam_robot(channel=channel, zero_gravity_mode=zero_gravity_mode)
        self._default_duration = default_duration
        self._default_steps = default_steps
        self._home = home
        self._q_cmd = None
        self._dq_cmd = None
        self._kin = Kinematics(YAM_GLASS_PATH, "grasp_site") #TODO: check payload
        self._ee_pos_error = np.zeros(3, dtype=np.float32)

        if self._home:
            zero_pose = np.zeros(self.num_dofs(), dtype=np.float64)
            self.send_joint_pos_rad(zero_pose)

    def num_dofs(self) -> int:
        return self._robot.num_dofs()

    def current_joint_pos(self) -> np.ndarray:
        return self._robot.get_joint_pos()

    def close(self, timeout_s: Optional[float] = None) -> None:
        """Send arm to home pose before closing, with a timeout."""
        duration = self._default_duration
        steps = self._default_steps
        try:
            print("Start go back to 0")
            home_q = np.zeros(self.num_dofs(), dtype=np.float64)
            self.send_joint_pos_rad(home_q, duration=duration, steps=steps)
            if timeout_s is None:
                timeout_s = max(duration + 1.0, 3.0)
            t0 = time.time()
            while (time.time() - t0) < timeout_s:
                try:
                    cur = self.current_joint_pos()
                except Exception:
                    break
                if np.linalg.norm(cur - home_q) < 1e-2:
                    break
                time.sleep(0.1)
        except Exception as exc:
            print(f"[WARN] I2RT 回零失败: {exc}")
        self._robot.close()

    def send_joint_pos_deg(
        self,
        target_joint_pos_deg: Sequence[float],
        duration: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> None:
        """Move to the given joint configuration specified in degrees."""
        target_rad = np.deg2rad(np.asarray(target_joint_pos_deg, dtype=np.float64))
        self._send_joint_pos_rad(target_rad, duration=duration, steps=steps)

    def send_joint_pos_rad(
        self,
        target_joint_pos_rad: Sequence[float],
        duration: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> None:
        """Move to the given joint configuration specified in radians."""
        target = np.asarray(target_joint_pos_rad, dtype=np.float64)
        self._send_joint_pos_rad(target, duration=duration, steps=steps)

    def send_ee_pos(
        self,
        target_xyz_rot6d: Sequence[float],
        duration: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> None:
        """Solve IK for xyz+rot6d and send joint position."""
        target = np.asarray(target_xyz_rot6d, dtype=np.float64)
        if target.shape[0] != 9:
            raise ValueError(f"Expected xyz+rot6d (9,), got {target.shape}")
        xyz = target[:3]
        r6 = target[3:9]

        rot = rotation_transform(r6[None, :], "rotation_6d", "matrix").squeeze(0)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot
        pose[:3, 3] = xyz.astype(np.float32)
        success, q_sol = self._kin.ik(pose, "grasp_site", verbose=False)
        if not success:
            print("[WARN] IK failed for target pose.")
            return
        self.send_joint_pos_rad(q_sol[: self.num_dofs()], duration=duration, steps=steps)

    def _compute_sync_profile(self, q_start: np.ndarray, target: np.ndarray, duration: float):
        dist = np.abs(target - q_start)
        vmax = self.MAX_JOINT_VEL_RAD_S
        amax = self.MAX_JOINT_ACCEL_RAD_S2
        t_min = np.zeros_like(dist)
        tri_limit = (vmax * vmax) / max(amax, 1e-6)
        tri_mask = dist <= tri_limit
        t_min[tri_mask] = 2.0 * np.sqrt(dist[tri_mask] / max(amax, 1e-6))
        t_min[~tri_mask] = 2.0 * vmax / max(amax, 1e-6) + (dist[~tri_mask] - tri_limit) / max(vmax, 1e-6)
        total_time = max(duration, float(np.max(t_min)))
        return dist, t_min, total_time

    def _send_joint_pos_rad(
        self,
        target_joint_pos_rad: Iterable[float],
        duration: Optional[float],
        steps: Optional[int],
    ) -> None:
        if duration is None:
            duration = self._default_duration
        if steps is None:
            steps = self._default_steps

        if steps <= 0:
            raise ValueError("steps must be greater than zero")

        target = np.asarray(target_joint_pos_rad, dtype=np.float64)
        current = self.current_joint_pos()

        # If only 6 joints are provided, append the current last joint value to match DOFs.
        if target.shape[0] == current.shape[0] - 1:
            target = np.concatenate([target, current[-1:]])

        if target.shape != current.shape:
            raise ValueError(
                f"target joint size {target.shape} does not match robot DOFs {current.shape}"
            )
        if steps == 1:
            vel = np.clip((target - current) / max(duration, 1e-6), -self.MAX_JOINT_VEL_RAD_S, self.MAX_JOINT_VEL_RAD_S)
            self._robot.command_joint_state({"pos": target, "vel": vel})
            return

        if self._q_cmd is None or self._q_cmd.shape != current.shape:
            self._q_cmd = current.copy()
        if self._dq_cmd is None or self._dq_cmd.shape != current.shape:
            self._dq_cmd = np.zeros_like(current)

        q_start = self._q_cmd.copy()
        dist, t_min, total_time = self._compute_sync_profile(q_start, target, duration)
        dt = total_time / steps

        for idx in range(steps):
            t = min((idx + 1) * dt, total_time)
            q_ref = q_start.copy()
            dq_ref = np.zeros_like(q_start)
            for j in range(q_start.shape[0]):
                d = dist[j]
                if d < 1e-9:
                    q_ref[j] = target[j]
                    dq_ref[j] = 0.0
                    continue
                t_j = t_min[j]
                if t_j < 1e-9:
                    q_ref[j] = target[j]
                    dq_ref[j] = 0.0
                    continue
                scale = total_time / t_j
                v_lim = self.MAX_JOINT_VEL_RAD_S / scale
                a_lim = self.MAX_JOINT_ACCEL_RAD_S2 / (scale * scale)
                if d <= (v_lim * v_lim) / max(a_lim, 1e-6):
                    t_acc = np.sqrt(d / max(a_lim, 1e-6))
                    t_flat = 0.0
                else:
                    t_acc = v_lim / max(a_lim, 1e-6)
                    t_flat = (d - (v_lim * v_lim) / max(a_lim, 1e-6)) / max(v_lim, 1e-6)
                t_total = 2.0 * t_acc + t_flat
                t_use = min(t, t_total)
                if t_use <= t_acc:
                    pos = 0.5 * a_lim * t_use * t_use
                    vel = a_lim * t_use
                elif t_use <= t_acc + t_flat:
                    pos = 0.5 * a_lim * t_acc * t_acc + v_lim * (t_use - t_acc)
                    vel = v_lim
                else:
                    t_dec = t_use - t_acc - t_flat
                    pos = (
                        0.5 * a_lim * t_acc * t_acc
                        + v_lim * t_flat
                        + v_lim * t_dec
                        - 0.5 * a_lim * t_dec * t_dec
                    )
                    vel = v_lim - a_lim * t_dec
                sgn = 1.0 if (target[j] - q_start[j]) >= 0 else -1.0
                q_ref[j] = q_start[j] + sgn * pos
                dq_ref[j] = sgn * vel
            e = q_ref - self._q_cmd
            dq_ref_track = np.clip(e / max(dt, 1e-6), -self.MAX_JOINT_VEL_RAD_S, self.MAX_JOINT_VEL_RAD_S)
            ddq = np.clip(
                (dq_ref_track - self._dq_cmd) / max(dt, 1e-6),
                -self.MAX_JOINT_ACCEL_RAD_S2,
                self.MAX_JOINT_ACCEL_RAD_S2,
            )
            self._dq_cmd = self._dq_cmd + ddq * dt
            self._q_cmd = self._q_cmd + self._dq_cmd * dt
            # print(f"step {idx + 1}/{steps} joint vel (rad/s): {np.round(self._dq_cmd, 2)}")
            self._robot.command_joint_state({"pos": self._q_cmd, "vel": self._dq_cmd})
            if idx < steps - 1:
                time.sleep(dt)
    def _test_send_joint(
        self,
        target_joint_pos_rad: Iterable[float],
        duration: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> None:
        if duration is None:
            duration = self._default_duration
        if steps is None:
            steps = self._default_steps

        q_test = self.current_joint_pos()
        self._robot.command_joint_state({"pos": q_test, "vel": self._dq_cmd})
        print(f"q_test:{np.round(q_test*100,4)}")

class I2RTServer:
    def __init__(self, robot: I2RT, port: int = DEFAULT_ROBOT_PORT) -> None:
        self._robot = robot
        self._server = portal.Server(str(port))
        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("current_joint_pos", self._robot.current_joint_pos)
        self._server.bind("send_joint_pos_deg", self._robot.send_joint_pos_deg)
        self._server.bind("send_joint_pos_rad", self._robot.send_joint_pos_rad)
        self._server.bind("send_ee_pos", self._robot.send_ee_pos)
        self._server.bind("close", self._robot.close)

    def serve(self) -> None:
        self._server.start()


class I2RTClient:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_ROBOT_PORT) -> None:
        self._client = portal.Client(f"{host}:{port}")

    def num_dofs(self) -> int:
        return self._client.num_dofs().result()

    def current_joint_pos(self) -> np.ndarray:
        return self._client.current_joint_pos().result()

    def send_joint_pos_deg(self, target_joint_pos_deg: Sequence[float], duration: Optional[float] = None, steps: Optional[int] = None) -> None:
        self._client.send_joint_pos_deg(target_joint_pos_deg, duration, steps)

    def send_joint_pos_rad(self, target_joint_pos_rad: Sequence[float], duration: Optional[float] = None, steps: Optional[int] = None) -> None:
        self._client.send_joint_pos_rad(target_joint_pos_rad, duration, steps)

    def send_ee_pos(self, target_xyz_rot6d: Sequence[float], duration: Optional[float] = None, steps: Optional[int] = None) -> None:
        self._client.send_ee_pos(target_xyz_rot6d, duration, steps)

    def close(self, timeout_s: float = 15.0) -> None:
        try:
            self._client.call("close").result(timeout=timeout_s)
        except Exception as exc:
            print(f"[WARN] I2RTClient close RPC failed: {exc!r}")
        finally:
            try:
                self._client.close()
            except Exception as exc:
                print(f"[WARN] I2RTClient socket close failed: {exc!r}")


def main():
    robot = I2RT(channel="can0", zero_gravity_mode=False)
    try:
        current_q = robot.current_joint_pos()
        current_pose = robot._kin.fk(current_q[:6])
        current_rot6d = rotation_transform(
            current_pose[:3, :3][None, ...],
            "matrix",
            "rotation_6d",
        ).squeeze(0)
        target_up = np.concatenate(
            [
                current_pose[:3, 3].astype(np.float32)
                + np.array([0.0, 0.0, 0.2], dtype=np.float32),
                current_rot6d,
            ],
            axis=0,
        ).astype(np.float32)
        robot.send_ee_pos(target_up)
        time.sleep(0.5)
        while True:
            offset = 0.1
            current_q = robot.current_joint_pos()
            current_pose = robot._kin.fk(current_q[:6])
            current_rot6d = rotation_transform(
                current_pose[:3, :3][None, ...],
                "matrix",
                "rotation_6d",
            ).squeeze(0)
            target_pos = current_pose[:3, 3].astype(np.float32)
            target_forward = np.concatenate(
                [target_pos + np.array([offset, 0.0, 0.0], dtype=np.float32), current_rot6d],
                axis=0,
            ).astype(np.float32)
            robot.send_ee_pos(target_forward)

            current_q = robot.current_joint_pos()
            current_pose = robot._kin.fk(current_q[:6])
            current_rot6d = rotation_transform(
                current_pose[:3, :3][None, ...],
                "matrix",
                "rotation_6d",
            ).squeeze(0)
            target_pos = current_pose[:3, 3].astype(np.float32)
            target_back = np.concatenate(
                [target_pos + np.array([-offset, 0.0, 0.0], dtype=np.float32), current_rot6d],
                axis=0,
            ).astype(np.float32)
            robot.send_ee_pos(target_back)
            # robot._test_send_joint(robot.current_joint_pos()) #DEBUG: The arm will fall because of payload
            # time.sleep(0.5)
    finally:
        robot.close()

if __name__ == "__main__":
    main()
