import argparse
import time

import numpy as np

try:
    from ur5_robot import UR5
except ImportError:
    from glasses_hardware.hardware.ur5_robot import UR5


DEFAULT_DELTAS = "0.01,0,0;0,0.01,0;0,0,0.01"


def parse_deltas(text: str) -> list[np.ndarray]:
    deltas: list[np.ndarray] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        values = [float(item.strip()) for item in chunk.split(",")]
        if len(values) != 3:
            raise ValueError(f"Each delta must have 3 values, got {values}.")
        deltas.append(np.asarray(values, dtype=np.float64))
    if not deltas:
        raise ValueError("No valid deltas were provided.")
    return deltas


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a sequence of TCP position deltas to UR5.")
    parser.add_argument("--robot-ip", default="192.168.2.102", help="UR robot IP address.")
    parser.add_argument(
        "--deltas",
        default=DEFAULT_DELTAS,
        help="Semicolon-separated xyz deltas in meters, e.g. '0.01,0,0;0,0.01,0;0,0,-0.01'.",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between targets in seconds.")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI inside UR5 controller.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    deltas = parse_deltas(args.deltas)

    robot = UR5(robot_ip=args.robot_ip, gui=args.gui, debug=False)
    try:
        current_tcp = robot.get_tcp_pose().copy()
        print("Start TCP:", np.round(current_tcp, 6).tolist())

        for idx, delta in enumerate(deltas, start=1):
            current_tcp[:3] += delta
            print(f"Step {idx}: delta={np.round(delta, 6).tolist()} target={np.round(current_tcp[:3], 6).tolist()}")
            robot.move_tcp_pose(current_tcp, pos_tolerance=0.002, max_steps=300)
            time.sleep(args.sleep)

        print("Final TCP:", np.round(robot.get_tcp_pose(), 6).tolist())
    finally:
        robot.close()


if __name__ == "__main__":
    main()
