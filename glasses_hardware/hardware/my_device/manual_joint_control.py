import os
import sys
import time
from typing import Dict, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from glasses_hardware.hardware.my_device.robot import FlexivRobot

from glasses_hardware.hardware.my_device.camera import CameraD400
from glasses_hardware.hardware.my_device.macros import CAM_SERIAL



def draw_overlay(frame: np.ndarray, joint_positions: np.ndarray, step: float) -> np.ndarray:
    """Overlay joint info and key bindings on the camera frame."""
    overlay = frame.copy()
    lines = [
        "Manual Joint Control (ESC to exit)",
        f"Step: {step:.3f} rad (+/- to adjust)",
        "q/a: J0  w/s: J1  e/d: J2  r/f: J3",
        "t/g: J4  y/h: J5  u/j: J6",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            overlay,
            line,
            (10, 30 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for joint_idx, joint_val in enumerate(joint_positions):
        cv2.putText(
            overlay,
            f"J{joint_idx}: {joint_val:.3f}",
            (10, frame.shape[0] - 10 - joint_idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def build_keymap(step: float) -> Dict[int, Tuple[int, float]]:
    """Create a key mapping from keyboard input to joint deltas."""
    return {
        ord("q"): (0, step),
        ord("a"): (0, -step),
        ord("w"): (1, step),
        ord("s"): (1, -step),
        ord("e"): (2, step),
        ord("d"): (2, -step),
        ord("r"): (3, step),
        ord("f"): (3, -step),
        ord("t"): (4, step),
        ord("g"): (4, -step),
        ord("y"): (5, step),
        ord("h"): (5, -step),
        ord("u"): (6, step),
        ord("j"): (6, -step),
    }


def main() -> None:
    robot = FlexivRobot(home=False)
    serial = CAM_SERIAL[1] if CAM_SERIAL else None
    camera = CameraD400(serial=serial)

    try:
        current_joints = robot.get_joint_pos().astype(float)
    except Exception:
        current_joints = np.zeros(7, dtype=float)

    step = 0.01  # radians
    keymap = build_keymap(step)

    cv2.namedWindow("cam1", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            color_image, _ = camera.get_data()
            overlay = draw_overlay(color_image, current_joints, step)
            cv2.imshow("cam1", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key in (ord("+"), ord("=")):
                step = min(0.2, step + 0.005)
                keymap = build_keymap(step)
            elif key in (ord("-"), ord("_")):
                step = max(0.001, step - 0.005)
                keymap = build_keymap(step)
            elif key in keymap:
                joint_idx, delta = keymap[key]
                current_joints[joint_idx] += delta
                robot.send_joint_pose(current_joints.tolist())
                time.sleep(0.02)
    finally:
        cv2.destroyAllWindows()
        # Allow camera to release resources explicitly
        del camera
        # Robot stays at the last commanded joint pose by design


if __name__ == "__main__":
    main()
