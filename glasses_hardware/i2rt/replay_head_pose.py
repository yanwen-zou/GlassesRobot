#!/usr/bin/env python3
"""
Replay head_pose files under a dataset directory to /glasses_pose as PoseStamped.

Usage:
    python glasses_hardware/i2rt/replay_head_pose.py --data-path DATASET --rate 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


def sort_key(path: Path) -> tuple:
    stem = path.stem
    try:
        numeric = int(stem)
    except ValueError:
        numeric = None
    return (0, numeric) if numeric is not None else (1, stem)


def load_head_pose_files(head_dir: Path) -> List[np.ndarray]:
    poses: List[np.ndarray] = []
    files = sorted([p for p in head_dir.iterdir() if p.suffix.lower() in {".txt", ".npy"}], key=sort_key)
    for file in files:
        try:
            if file.suffix.lower() == ".npy":
                data = np.load(str(file))
            else:
                data = np.loadtxt(str(file))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to load {file}: {exc}", file=sys.stderr)
            continue
        arr = np.asarray(data).reshape(-1)
        if arr.shape[0] < 7:
            print(f"[WARN] Expected >=7 values in {file}, got shape {arr.shape}", file=sys.stderr)
            continue
        poses.append(arr[:7])
    return poses


class HeadPosePublisher(Node):
    def __init__(self, poses: List[np.ndarray], rate_hz: float) -> None:
        super().__init__("head_pose_publisher")
        self._poses = poses
        self._index = 0
        self._publisher = self.create_publisher(PoseStamped, "/glasses_pose", 10)
        self._timer = self.create_timer(1.0 / rate_hz, self._publish_next)
        self.get_logger().info(f"Loaded {len(poses)} head poses, publishing at {rate_hz} Hz")

    def _publish_next(self) -> None:
        if not self._poses:
            return
        pose_arr = self._poses[self._index]
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(pose_arr[0])
        msg.pose.position.y = float(pose_arr[1])
        msg.pose.position.z = float(pose_arr[2])
        msg.pose.orientation.x = float(pose_arr[3])
        msg.pose.orientation.y = float(pose_arr[4])
        msg.pose.orientation.z = float(pose_arr[5])
        msg.pose.orientation.w = float(pose_arr[6])
        self._publisher.publish(msg)
        self._index = (self._index + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay head_pose files to /glasses_pose topic.")
    parser.add_argument("--data-path", type=Path, required=True, help="Dataset directory containing head_pose/")
    parser.add_argument("--rate", type=float, default=60.0, help="Publish rate in Hz.")
    return parser.parse_args()


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args()
    head_dir = args.data_path / "head_pos"
    if not head_dir.is_dir():
        print(f"❌ head_pose directory not found: {head_dir}", file=sys.stderr)
        sys.exit(1)
    poses = load_head_pose_files(head_dir)
    if not poses:
        print(f"❌ No valid head pose files found under {head_dir}", file=sys.stderr)
        sys.exit(1)

    rclpy.init()
    node = HeadPosePublisher(poses, rate_hz=args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
