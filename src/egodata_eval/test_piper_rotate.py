import argparse
import time
from typing import Tuple

from piper_sdk import C_PiperInterface_V2


def deg_to_milli_deg(x: float) -> int:
    return int(round(x * 1000.0))


def get_current_xyz(piper: C_PiperInterface_V2) -> Tuple[int, int, int]:
    """Read current end pose and return XYZ in 0.001 mm units (ints)."""
    ep = piper.GetArmEndPoseMsgs().end_pose
    return int(ep.X_axis), int(ep.Y_axis), int(ep.Z_axis)


def main():
    parser = argparse.ArgumentParser(
        description="Set Piper TCP orientation to given rx/ry/rz (deg) while keeping XYZ position."
    )
    parser.add_argument("rx", type=float, help="Target RX angle in degrees (XYZ order)")
    parser.add_argument("ry", type=float, help="Target RY angle in degrees (XYZ order)")
    parser.add_argument("rz", type=float, help="Target RZ angle in degrees (XYZ order)")
    parser.add_argument("--speed", type=int, default=40, help="Move speed percent 0-100 (default: 40)")
    parser.add_argument("--hold_s", type=float, default=2.0, help="Command hold duration in seconds (default: 2.0)")
    parser.add_argument("--dry_run", action="store_true", help="Print command without sending")
    args = parser.parse_args()

    piper = C_PiperInterface_V2(can_name="can0")
    piper.ConnectPort()
    time.sleep(0.2)

    X, Y, Z = get_current_xyz(piper)
    RX = deg_to_milli_deg(args.rx)
    RY = deg_to_milli_deg(args.ry)
    RZ = deg_to_milli_deg(args.rz)

    print(f"Current XYZ (0.001mm): X={X} Y={Y} Z={Z}")
    print(f"Target RPY (0.001deg): RX={RX} RY={RY} RZ={RZ}")

    if not args.dry_run:
        piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        print(f'sending end pose command: X={X} Y={Y} Z={Z} RX={RX} RY={RY} RZ={RZ}')
        time.sleep(0.02)


    else:
        print("[DRY RUN] Not sending commands to Piper.")


if __name__ == "__main__":
    main()
