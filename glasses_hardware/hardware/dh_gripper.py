import time

from pyDHgripper import AG95


def main() -> None:
    gripper = AG95(port="/dev/ttyUSB0")

    # Use a moderate speed/force before moving the gripper.
    gripper.set_vel(80)
    gripper.set_force(50)

    # 0 is fully closed, 1000 is fully open for this driver.
    gripper.set_pos(0)
    time.sleep(2.0)

    gripper.set_pos(1000)
    time.sleep(2.0)


if __name__ == "__main__":
    main()
