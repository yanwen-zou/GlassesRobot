"""MuJoCo-backed drop-in replacement for the real I2RT wrapper."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

# Keep repo modules importable regardless of launch path.
import sys

here = Path(__file__).resolve()
repo_root = here.parents[3]
project_root = repo_root.parent
sdk_root = repo_root / "i2rt"
for path in (project_root, repo_root, sdk_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import mujoco
import mujoco.viewer

from i2rt.robots.kinematics_mj import Kinematics
from i2rt.robots.utils import YAM_GLASS_PATH, YAM_XML_PATH
from MBA.utils.transformation import rotation_transform  # type: ignore


class I2RTSim:
    """Expose the same API as ``I2RT`` but run purely inside MuJoCo."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        default_duration: float = 2.0,
        default_steps: int = 50,
    ) -> None:
        self._model = mujoco.MjModel.from_xml_path(model_path or str(YAM_XML_PATH))
        self._data = mujoco.MjData(self._model)
        self._default_duration = default_duration
        self._default_steps = default_steps
        self._kin = Kinematics(YAM_GLASS_PATH, "grasp_site")
        self._viewer: Optional[mujoco.viewer.Viewer] = None
        I2RT_TARGET_DEG = [-0.1314, 59.9407, 50.3597, 10.8160, -0.1314, -1.1748]
        # I2RT_TARGET_DEG = [0, 0, 0, 0, 0, 0]
        I2RT_TARGET_RAD = np.deg2rad(I2RT_TARGET_DEG).astype(np.float32)
        self._qpos = I2RT_TARGET_RAD.copy()
        self._set_qpos(self._qpos.copy())

    def num_dofs(self) -> int:
        return self._model.nq

    def current_joint_pos(self) -> np.ndarray:
        return self._qpos.copy()

    def close(self) -> None:
        pass

    def send_joint_pos_deg(
        self,
        target_joint_pos_deg: Sequence[float],
        duration: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> None:
        target_rad = np.deg2rad(np.asarray(target_joint_pos_deg, dtype=np.float64))
        self._send_joint_pos_rad(target_rad, duration=duration, steps=steps)

    def send_joint_pos_rad(
        self,
        target_joint_pos_rad: Sequence[float],
        duration: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> None:
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
        success, q_sol = self._kin.ik(pose, "grasp_site", verbose=True)
        if not success:
            print("[WARN] IK failed for target pose.")
            return
        self.send_joint_pos_rad(q_sol[: self.num_dofs()], duration=duration, steps=steps)

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
        if target.shape[0] == current.shape[0] - 1:
            target = np.concatenate([target, current[-1:]])
        if target.shape != current.shape:
            raise ValueError(
                f"target joint size {target.shape} does not match robot DOFs {current.shape}"
            )

        if steps == 1:
            self._set_qpos(target)
            return

        interval = duration / steps
        alphas = np.linspace(0.0, 1.0, steps)
        for idx, alpha in enumerate(alphas):
            cmd = (1.0 - alpha) * current + alpha * target
            self._set_qpos(cmd)
            if idx < steps - 1:
                time.sleep(interval)

    def _set_qpos(self, qpos: np.ndarray) -> None:
        self._qpos[:] = qpos
        self._data.qpos[:] = qpos
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
        if self._viewer is not None:
            self._viewer.sync()

    def mj_handles(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """Return underlying MuJoCo model/data for visualization."""
        return self._model, self._data


def main() -> None:
    """Test IK back-and-forth motion like i2rt_robo.py."""
    robot = I2RTSim()
    model, data = robot.mj_handles()
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        robot._viewer = viewer
        base_q = robot.current_joint_pos()
        base_pose = robot._kin.fk(base_q[:6])
        base_rot = base_pose[:3, :3].astype(np.float32)
        pitch_rad = np.deg2rad(5.0)
        pitch_rot = np.array(
            [
                [np.cos(pitch_rad), 0.0, np.sin(pitch_rad)],
                [0.0, 1.0, 0.0],
                [-np.sin(pitch_rad), 0.0, np.cos(pitch_rad)],
            ],
            dtype=np.float32,
        )
        yaw_rad = np.deg2rad(5.0)
        yaw_rot = np.array(
            [
                [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
                [np.sin(yaw_rad), np.cos(yaw_rad), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        roll_rad = np.deg2rad(5.0)
        roll_rot = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(roll_rad), -np.sin(roll_rad)],
                [0.0, np.sin(roll_rad), np.cos(roll_rad)],
            ],
            dtype=np.float32,
        )
        offsets = [
            ("pitch+", np.array([0.0, 0.0, 0.0], dtype=np.float32), pitch_rot),
            ("pitch-", np.array([0.0, 0.0, 0.0], dtype=np.float32), pitch_rot.T),
            ("yaw+", np.array([0.0, 0.0, 0.0], dtype=np.float32), yaw_rot),
            ("yaw-", np.array([0.0, 0.0, 0.0], dtype=np.float32), yaw_rot.T),
            ("roll+", np.array([0.0, 0.0, 0.0], dtype=np.float32), roll_rot),
            ("roll-", np.array([0.0, 0.0, 0.0], dtype=np.float32), roll_rot.T),
        ]
        targets_after_trans = [
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.98484576,
                    -0.03372636,
                    -0.17012277,
                    0.03521027,
                    0.9993636,
                    0.00571203,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.98484576,
                    0.03521026,
                    0.16982178,
                    -0.03372635,
                    0.9993636,
                    -0.01161543,
                ],
                dtype=np.float32,
            ),
        ]
        while viewer.is_running():
            # for name, offset, rot_delta in offsets:
            #     print(f"[SIM] Testing offset: {name}")
            #     target_rot = base_rot @ rot_delta
            #     target_rot6d = rotation_transform(
            #         target_rot[None, ...],
            #         "matrix",
            #         "rotation_6d",
            #     ).squeeze(0)
            #     target = np.concatenate(
            #         [
            #             base_pose[:3, 3].astype(np.float32) + offset,
            #             target_rot6d,
            #         ],
            #         axis=0,
            #     ).astype(np.float32)
            #     robot.send_ee_pos(target, duration=1.0, steps=50)
            #     time.sleep(2.0)
            for idx, target in enumerate(targets_after_trans):
                print(f"[SIM] Testing after-trans target: {idx}")
                rel_xyz = target[:3]
                rel_rot = rotation_transform(
                    target[3:9][None, :],
                    "rotation_6d",
                    "matrix",
                ).squeeze(0)
                rel_pose = np.eye(4, dtype=np.float32)
                rel_pose[:3, :3] = rel_rot
                rel_pose[:3, 3] = rel_xyz
                print("Relative pose: ", rel_pose)
                target_pose = np.eye(4, dtype=np.float32)
                target_pose[:3, :3] = rel_pose[:3, :3] @ base_pose[:3, :3]
                target_pose[:3, 3] = rel_pose[:3, 3] + base_pose[:3, 3]
                print("Base pose: ", base_pose)
                print("Target pose: ", target_pose)
                target_rot6d = rotation_transform(
                    target_pose[:3, :3][None, ...],
                    "matrix",
                    "rotation_6d",
                ).squeeze(0)
                print("Target rot6d: ", target_rot6d)
                target_abs = np.concatenate(
                    [target_pose[:3, 3], target_rot6d],
                    axis=0,
                ).astype(np.float32)
                print("Target abs: ", target_abs)
                robot.send_ee_pos(target_abs, duration=1.0, steps=50)
                time.sleep(2.0)


__all__ = ["I2RTSim"]


if __name__ == "__main__":
    main()
