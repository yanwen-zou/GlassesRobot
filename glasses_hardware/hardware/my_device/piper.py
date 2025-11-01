import math
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from typing import Iterable, List, Optional

from piper_sdk import C_PiperInterface_V2


RAD_TO_MILLI_DEG = 180_000.0 / math.pi
JOINT_COUNT = 6


class Piper:
    def __init__(self, can_port: str = "can0"):
        self.iface = C_PiperInterface_V2(can_name=can_port)
        self.iface.ConnectPort()
        time.sleep(0.2)

    def to_cmd(self, targets_rad: Iterable[float]) -> List[int]:
        return [int(round(v * RAD_TO_MILLI_DEG)) for v in targets_rad]

    def enable_motion(self, speed_rate: int = 60, is_mit_mode: int = 0):
        self.iface.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=speed_rate, is_mit_mode=is_mit_mode)

    def standby(self):
        self.iface.MotionCtrl_2(ctrl_mode=0x00, move_mode=0x01, move_spd_rate_ctrl=0, is_mit_mode=0x00)

    def set_joint_targets(self, targets_rad: Iterable[float]):
        cmd = self.to_cmd(targets_rad)
        self.iface.JointCtrl(*cmd)

    def zero(self):
        self.set_joint_targets([0.0] * JOINT_COUNT)

    def get_status(self):
        return self.iface.GetArmStatus()

    def get_joint_feedback(self):
        return self.iface.GetArmJointMsgs()


@contextmanager
def raw_terminal(fd: int):
    original_attributes = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_attributes)


def read_key(fd: int, timeout: float) -> Optional[str]:
    if not select.select([fd], [], [], timeout)[0]:
        return None
    first = sys.stdin.read(1)
    if first != "\x1b":
        return first
    sequence = first
    for _ in range(2):
        if select.select([fd], [], [], 0.0005)[0]:
            sequence += sys.stdin.read(1)
        else:
            break
    return sequence


def format_joint_feedback(joint_msg) -> str:
    feedback = joint_msg.joint_state
    degrees = [
        feedback.joint_1 * 1e-3,
        feedback.joint_2 * 1e-3,
        feedback.joint_3 * 1e-3,
        feedback.joint_4 * 1e-3,
        feedback.joint_5 * 1e-3,
        feedback.joint_6 * 1e-3,
    ]
    return " | ".join(f"J{i + 1}:{deg:7.2f}°" for i, deg in enumerate(degrees))


def print_help(step_size: float, selected_joint: int):
    help_lines = [
        "",
        "键盘控制已就绪",
        "控制说明:",
        "  1-6         选择关节",
        "  ←/→ 或 a/d  上一个 / 下一个关节",
        "  ↑/↓ 或 w/s  增加 / 减少角度",
        "  z/x         减小 / 增大步长",
        "  space       目标清零",
        "  h           显示帮助",
        "  q 或 ESC    退出",
        "",
        f"当前关节: J{selected_joint + 1}",
        f"步长: {math.degrees(step_size):.2f}° ({step_size:.3f} rad)",
        "",
    ]
    print("\n".join(help_lines))


def main():
    arm = Piper(can_port="can0")
    selected_joint = 0
    step_size = math.radians(2.0)
    joint_targets = [0.0] * JOINT_COUNT
    fd = sys.stdin.fileno()
    print_help(step_size, selected_joint)

    try:
        with raw_terminal(fd):
            while True:
                key = read_key(fd, timeout=0.05)

                if key in ("q", "\x1b"):
                    print("\n退出键盘控制.")
                    break
                elif key in ("h", "H"):
                    print_help(step_size, selected_joint)
                elif key in (" ",):
                    joint_targets = [0.0] * JOINT_COUNT
                elif key in ("z", "Z"):
                    step_size = max(math.radians(0.5), step_size * 0.5)
                elif key in ("x", "X"):
                    step_size = min(math.radians(10.0), step_size * 2.0)
                elif key in ("a", "A", "\x1b[D"):
                    selected_joint = (selected_joint - 1) % JOINT_COUNT
                elif key in ("d", "D", "\x1b[C"):
                    selected_joint = (selected_joint + 1) % JOINT_COUNT
                elif key in ("w", "W", "\x1b[A"):
                    joint_targets[selected_joint] += step_size
                elif key in ("s", "S", "\x1b[B"):
                    joint_targets[selected_joint] -= step_size
                elif key is not None and key.isdigit():
                    index = int(key) - 1
                    if 0 <= index < JOINT_COUNT:
                        selected_joint = index

                arm.enable_motion(speed_rate=60, is_mit_mode=0)
                arm.set_joint_targets(joint_targets)
                status = arm.get_status()
                feedback = arm.get_joint_feedback()
                status_line = format_joint_feedback(feedback)
                sys.stdout.write(
                    f"\r选中 J{selected_joint + 1} | 步长: {math.degrees(step_size):5.2f}° | 目标(°): "
                    f"{', '.join(f'{math.degrees(v):7.2f}' for v in joint_targets)} || 反馈 {status_line}  状态: {status.arm_status.arm_status}"
                )
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C, 正在退出...")
    finally:
        try:
            arm.enable_motion(speed_rate=40, is_mit_mode=0)
            zero_cmd = arm.to_cmd([0.0] * JOINT_COUNT)
            deadline = time.time() + 3.0
            while time.time() < deadline:
                arm.iface.JointCtrl(*zero_cmd)
                time.sleep(0.02)
            arm.standby()
        except Exception:
            pass
        print("控制结束。")


if __name__ == "__main__":
    main()

