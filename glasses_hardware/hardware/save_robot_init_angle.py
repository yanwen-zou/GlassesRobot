import os
import sys
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
try:
    from my_device.robot import FlexivRobot  # type: ignore  # noqa: E402
except ImportError:
    from glasses_hardware.hardware.my_device.robot import FlexivRobot  # type: ignore  # noqa: E402


def format_joint_angles(joints: Sequence[float]) -> str:
    """Format joint angles as a single comma-separated line."""
    return ", ".join(f"{angle:.6f}" for angle in joints)


def main() -> None:
    robot = FlexivRobot()
    joint_positions = robot.get_joint_pos()

    output_path = os.path.join(os.path.dirname(__file__), "robot_init_angle.txt")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(format_joint_angles(joint_positions))
        file.write("\n")

    print(f"Saved joint angles to {output_path}")


if __name__ == "__main__":
    main()
