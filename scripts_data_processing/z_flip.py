#!/usr/bin/env python3
import numpy as np

in_file = "zed_trajectory.txt"
out_file = "zed_trajectory_flipped.txt"

with open(in_file, "r") as f:
    lines = f.readlines()

out_lines = []
for line in lines:
    parts = line.strip().split()
    if len(parts) != 8:
        continue
    # 解析
    t, x, y, z, qx, qy, qz, qw = map(float, parts)
    # 翻转 z
    z = -z
    out_lines.append(f"{t:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

with open(out_file, "w") as f:
    f.writelines(out_lines)

print(f"✅ 已保存翻转后的轨迹到 {out_file}")
