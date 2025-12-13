from __future__ import annotations

import numpy as np

from glasses_hardware.hardware.my_device.i2rt import I2RT
from glasses_hardware.hardware.my_device.robot import FlexivRobot, FlexivGripper

TARGET_DEG = [-17, 25, 61, -42, 0, -2, 0]
TARGET_RAD = np.deg2rad(TARGET_DEG).astype(np.float32)


def main() -> None:


    print("[INFO] Initializing Flexiv robot...")
    flexiv_robot = FlexivRobot(home=False)
    gripper = FlexivGripper(flexiv_robot)
    print("[INFO] Flexiv initialization complete.")
    input("[PROMPT] Press 'p' then Enter to initialize Flexiv robot...")

    print("[INFO] Initializing I2RT robot...")
    i2rt = I2RT(channel="can0", zero_gravity_mode=True, home=False)
    print(f"[INFO] Moving I2RT to {TARGET_DEG} degrees")
    i2rt.send_joint_pos_rad(TARGET_RAD, duration=2.0, steps=80)

    input("[PROMPT] Press Enter to exit and close robots...")
    gripper = None
    flexiv_robot.close()
    i2rt.close()


if __name__ == "__main__":
    main()
