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
from typing import Iterable, Optional, Sequence

import numpy as np

from ...i2rt.i2rt.robots.get_robot import get_yam_robot


class I2RT:
    """User-facing helper that exposes simple joint-space commands."""

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

        if self._home:
            zero_pose = np.zeros(self.num_dofs(), dtype=np.float64)
            self.send_joint_pos_rad(zero_pose)

    def num_dofs(self) -> int:
        return self._robot.num_dofs()

    def current_joint_pos(self) -> np.ndarray:
        return self._robot.get_joint_pos()

    def close(self) -> None:
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

        if target.shape != current.shape:
            raise ValueError(
                f"target joint size {target.shape} does not match robot DOFs {current.shape}"
            )

        if steps == 1:
            self._robot.command_joint_pos(target)
            return

        interval = duration / steps
        alphas = np.linspace(0.0, 1.0, steps)
        for idx, alpha in enumerate(alphas):
            cmd = (1.0 - alpha) * current + alpha * target
            self._robot.command_joint_pos(cmd)
            if idx < steps - 1:
                time.sleep(interval)
