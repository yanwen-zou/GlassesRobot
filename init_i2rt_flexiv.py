#!/usr/bin/env python3
"""
Initialize I2RT first, then Flexiv, to test CPU contention.
"""

from __future__ import annotations

import argparse
import time
import multiprocessing as mp

from glasses_hardware.hardware.my_device.i2rt_robo import I2RT, I2RTClient, I2RTServer, DEFAULT_ROBOT_PORT
from glasses_hardware.hardware.my_device.robot import FlexivRobot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Init I2RT then Flexiv.")
    parser.add_argument("--i2rt-channel", type=str, default="can0", help="CAN channel for I2RT.")
    parser.add_argument("--i2rt-home", action="store_true", help="Home I2RT after init.")
    parser.add_argument("--flexiv-home", action="store_true", help="Home Flexiv after init.")
    parser.add_argument("--sleep-sec", type=float, default=2.0, help="Sleep between inits.")
    return parser.parse_args()


def _run_i2rt_server(channel: str, home: bool, port: int) -> None:
    robot = I2RT(channel=channel, zero_gravity_mode=False, home=home)
    server = I2RTServer(robot, port)
    server.serve()


def main() -> None:
    args = parse_args()
    print("[INFO] Initializing I2RT...")
    server_proc = mp.Process(
        target=_run_i2rt_server,
        args=(args.i2rt_channel, args.i2rt_home, DEFAULT_ROBOT_PORT),
        daemon=True,
    )
    server_proc.start()
    client = I2RTClient(port=DEFAULT_ROBOT_PORT)
    time.sleep(args.sleep_sec)

    print("[INFO] Initializing Flexiv...")
    flexiv = FlexivRobot(home=args.flexiv_home)

    print("[OK] Both robots initialized.")
    # Keep objects alive for interactive inspection.
    try:
        while True:
            client.send_joint_pos_deg([0,10,0,0,0,0])
            time.sleep(1.0)
            client.send_joint_pos_deg([0.0]*6)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(client, "close"):
            client.close()
        if server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=2.0)
        if hasattr(flexiv, "close"):
            flexiv.close()


if __name__ == "__main__":
    main()
