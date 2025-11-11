#!/usr/bin/env python3
"""
Visualize the ArUco marker's coordinate frame relative to the ZED camera base.

This script loads a cached `T_zed_aruco.npy` (4x4 SE3, ArUco -> ZED) and displays both
frames inside Rerun so you can quickly confirm the detected pose. The ZED camera is
used as the visualization root.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_transform(path: Path) -> np.ndarray:
    T = np.load(path)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 SE3 matrix at {path}, got {T.shape}")
    return T.astype(np.float32)


def _log_frame(rr, name: str, T: np.ndarray, axis_len: float) -> None:
    """Log a coordinate frame plus RGB axes."""
    origin = T[:3, 3]
    R = T[:3, :3]
    rr.log(
        f"frames/{name}",
        rr.Transform3D(
            translation=origin,
            mat3x3=R,
        ),
    )
    # Log axes in the frame's local coordinates; Rerun composes the transform.
    origins = np.zeros((3, 3), dtype=np.float32)
    vectors = (np.eye(3, dtype=np.float32) * axis_len).astype(np.float32)
    colors = np.array(
        [
            [255, 0, 0, 255],   # +X red
            [0, 255, 0, 255],   # +Y green
            [0, 0, 255, 255],   # +Z blue
        ],
        dtype=np.uint8,
    )
    rr.log(
        f"frames/{name}/axes",
        rr.Arrows3D(
            origins=origins,
            vectors=vectors,
            colors=colors,
            radii=np.full(3, axis_len * 0.05, dtype=np.float32),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--axis_len", type=float, default=0.2)
    parser.add_argument("--T_zed_aruco", type=Path, default=Path("T_zed_aruco.npy"))
    parser.add_argument("--T_base_aruco", type=Path, default=Path("glasses_hardware/calib/T_base_aruco.npy"))
    args = parser.parse_args()

    # 1️⃣ 加载变换（都是 child→parent 形式）
    T_zed_aruco = _load_transform(args.T_zed_aruco) # aruco → zed
    T_base_aruco = _load_transform(args.T_base_aruco) # aruco → base
    
    # 2️⃣ 打印验证（物理距离必须与实际场景匹配！）
    print(f"\n[验证] ArUco在ZED下的平移: {T_zed_aruco[:3, 3]}")
    print(f"[验证] ArUco在Base下的平移: {T_base_aruco[:3, 3]}")

    # 3️⃣ 计算 Rerun 需要的 parent→child 形式
    T_zed_aruco_vis = T_zed_aruco          # zed → aruco
    T_zed_base_vis = T_zed_aruco @ np.linalg.inv(T_base_aruco)     # zed → base
    
    # 4️⃣ 再次验证（zed→base 的距离）
    print(f"[验证] Base在ZED下的平移: {T_zed_base_vis[:3, 3]}")
    print(f"[验证] 相机到机械臂基座距离: {np.linalg.norm(T_zed_base_vis[:3, 3]):.3f} 米")


    import rerun as rr
    rr.init("ZED/Aruco Frames", spawn=args.spawn)
    rr.log("world", rr.ViewCoordinates.RDF)  # 与ZED/OpenCV一致

    # 5️⃣ 可视化（传入 parent→child 变换）
    _log_frame(rr, "zed_base", np.eye(4, dtype=np.float32), args.axis_len)  # world→zed
    _log_frame(rr, "aruco", T_zed_aruco_vis, args.axis_len)                # zed→aruco
    _log_frame(rr, "robot_base", T_zed_base_vis.astype(np.float32), args.axis_len)  # zed→base

    # 6️⃣ 箭头用正确的平移分量（zed→aruco）
    translation = T_zed_aruco_vis[:3, 3].astype(np.float32)
    rr.log(
        "frames/zed_base/aruco_offset",
        rr.Arrows3D(
            origins=np.zeros((1, 3), dtype=np.float32),
            vectors=translation[None, :],
            colors=np.array([[255, 255, 0, 255]], dtype=np.uint8),
            radii=np.full(1, args.axis_len * 0.03, dtype=np.float32),
        ),
    )

    print("[SUCCESS] 变换方向已修正，请检查机械臂基座位置是否与实际一致")

if __name__ == "__main__":
    main()
