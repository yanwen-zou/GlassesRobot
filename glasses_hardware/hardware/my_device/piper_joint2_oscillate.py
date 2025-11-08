#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

用法示例：
  python glasses_hardware/hardware/my_device/piper_joint2_oscillate.py \
      --ifname can0 --speed 60 --hold 1.0 --loops 0

参数：
  --ifname   CAN 口名称，默认 can0
  --speed    关节控制速度比例 [0,100]，默认 60
  --hold     每个目标姿态保持时间（秒），默认 1.0
  --loops    循环次数（0 表示无限）
"""
import argparse
import math
import os
import sys
import time
import cv2


def main():
    parser = argparse.ArgumentParser(description="Piper关节往复：0位 <-> 关节2 +deg 度")
    parser.add_argument("--ifname", default="can0")
    parser.add_argument("--speed", type=int, default=60)
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--loops", type=int, default=0)
    args = parser.parse_args()

    # 允许在源码树中直接运行
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from glasses_hardware.hardware.my_device.piper import Piper, JOINT_COUNT

    arm = Piper(can_port=args.ifname)
    arm.enable_motion(speed_rate=max(0, min(100, int(args.speed))), is_mit_mode=0)

    zero = [0.0] * JOINT_COUNT
    target = [0.0] * JOINT_COUNT
    target[1] = math.radians(10)  # 关节2 +deg 度
    target[2] = math.radians(-10)  # 关节3 -10 度

    def hold_cmd(joints_rad, seconds: float):
        cmd = arm.to_cmd(joints_rad)
        deadline = time.time() + max(0.0, seconds)
        # 以 ~50Hz 发送，给控制器追踪
        while time.time() < deadline:
            arm.iface.JointCtrl(*cmd)
            time.sleep(0.02)

    loops = 0
    img_idx = 0
    try:
        while True:
            # 到零位
            hold_cmd(zero, args.hold)
            time.sleep(1)
            # 到关节2 +deg
            hold_cmd(target, args.hold)
            time.sleep(1)


            if args.loops > 0:
                loops += 1
                if loops >= args.loops:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            # 温和回零并待机
            hold_cmd(zero, 1.5)
            arm.standby()
        except Exception:
            pass
        # 关闭 ZED 相机


if __name__ == "__main__":
    main()
