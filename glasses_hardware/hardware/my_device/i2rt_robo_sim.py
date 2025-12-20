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

from i2rt.robots.utils import YAM_XML_PATH


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
        I2RT_TARGET_DEG = [-17, 25, 61, -42, 0, -2]
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

    def mj_handles(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """Return underlying MuJoCo model/data for visualization."""
        return self._model, self._data


__all__ = ["I2RTSim"]
