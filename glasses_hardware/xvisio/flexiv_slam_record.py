#!/usr/bin/env python3
"""
Drive the Flexiv arm through ±X/±Y/±Z translations and log both robot TCP poses
and the corresponding SLAM poses from the XVisio glasses.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

import xvsdk
from glasses_hardware.hardware.my_device.robot import FlexivRobot


def init_glasses() -> None:
    """Initialize the XVisio device for SLAM tracking."""
    xvsdk.init()
    xvsdk.slam_start()
    xvsdk.stereo_start()
    xvsdk.imu_start()


def shutdown_glasses() -> None:
    """Stop SLAM streams and release the device."""
    try:
        xvsdk.slam_stop()
    finally:
        xvsdk.stop()


def wait_for_valid_slam(timeout: float = 5.0) -> bool:
    """Block until SLAM host timestamp/confidence become positive."""
    start = time.time()
    while time.time() - start < timeout:
        (
            _,
            _,
            _,
            _,
            slam_hostTimestamp,
            slam_confidence,
        ) = xvsdk.xv_get_6dof()
        if slam_hostTimestamp.value > 0 and slam_confidence.value > 0:
            return True
        time.sleep(0.05)
    return False


def read_slam_pose() -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """Fetch the latest SLAM pose and return tuple of values."""
    (
        position,
        orientation,
        quaternion,
        slam_edgeTimestamp,
        slam_hostTimestamp,
        slam_confidence,
    ) = xvsdk.xv_get_6dof()
    return (
        slam_hostTimestamp.value,
        slam_edgeTimestamp.value,
        slam_confidence.value,
        position.x,
        position.y,
        position.z,
        orientation.x,
        orientation.y,
        orientation.z,
        quaternion.q0,
        quaternion.q1,
        quaternion.q2,
        quaternion.q3,
    )


def wait_for_target(robot: FlexivRobot, target_pose: np.ndarray, tol: float, timeout: float) -> bool:
    """Wait until robot TCP is within tolerance of target pose."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        curr = robot.get_tcp_pose()
        if np.linalg.norm(curr[:3] - target_pose[:3]) < tol:
            return True
        time.sleep(0.05)
    return False


def interpolate_poses(start: np.ndarray, target: np.ndarray, steps: int) -> List[np.ndarray]:
    """Linearly interpolate translation between start and target (orientation fixed)."""
    steps = max(1, steps)
    poses: List[np.ndarray] = []
    for idx in range(1, steps + 1):
        alpha = idx / steps
        pose = start.copy()
        pose[:3] = (1 - alpha) * start[:3] + alpha * target[:3]
        poses.append(pose)
    return poses


def log_sample(
    writer: csv.writer,
    repeat_idx: int,
    direction: str,
    interp_idx: int,
    flexiv_pose: np.ndarray,
    slam: Tuple[float, ...],
) -> None:
    writer.writerow(
        [
            time.time(),
            repeat_idx,
            direction,
            interp_idx,
            *flexiv_pose.tolist(),
            *slam,
        ]
    )


def sequence_offsets(delta: float) -> List[Tuple[str, np.ndarray]]:
    """Generate label + offset pairs for ±X, ±Y, ±Z translations."""
    return [
        ("+X", np.array([delta, 0.0, 0.0], dtype=np.float32)),
        ("-X", np.array([-delta, 0.0, 0.0], dtype=np.float32)),
        ("+Y", np.array([0.0, delta, 0.0], dtype=np.float32)),
        ("-Y", np.array([0.0, -delta, 0.0], dtype=np.float32)),
        ("+Z", np.array([0.0, 0.0, delta], dtype=np.float32)),
        ("-Z", np.array([0.0, 0.0, -delta], dtype=np.float32)),
    ]


def log_motion(
    robot: FlexivRobot,
    base_pose: np.ndarray,
    path: Sequence[Tuple[str, np.ndarray]],
    repeats: int,
    dwell: float,
    sample_dt: float,
    interp_steps: int,
    csv_writer: csv.writer,
) -> None:
    for idx in range(repeats):
        print(f"[INFO] Starting repeat {idx + 1}/{repeats}")
        for label, delta in path:
            target = base_pose.copy()
            target[:3] += delta
            print(f"[INFO] Move {label}: target {target[:3]}")
            start_pose = robot.get_tcp_pose().copy()
            interp_targets = interpolate_poses(start_pose, target, interp_steps)
            for interp_idx, pose in enumerate(interp_targets, start=1):
                robot.send_tcp_pose(pose)
                reached = wait_for_target(robot, pose, tol=0.002, timeout=5.0)
                if not reached:
                    print(f"[WARN] Interp step {interp_idx}/{interp_steps} for {label} not reached.")
                slam = read_slam_pose()
                if slam[0] <= 0 or slam[2] <= 0:
                    continue
                flexiv_pose = robot.get_tcp_pose()
                log_sample(csv_writer, idx, label, interp_idx, flexiv_pose, slam)
                time.sleep(sample_dt)

            if dwell > 0:
                end_time = time.time() + dwell
                while time.time() < end_time:
                    slam = read_slam_pose()
                    if slam[0] <= 0 or slam[2] <= 0:
                        time.sleep(sample_dt)
                        continue
                    flexiv_pose = robot.get_tcp_pose()
                    log_sample(csv_writer, idx, label, interp_steps, flexiv_pose, slam)
                    time.sleep(sample_dt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate Flexiv TCP along ±XYZ while logging Flexiv and SLAM poses."
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Translation magnitude in meters for each direction (default: 0.05m).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat the ±XYZ motion sequence.",
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=2.0,
        help="Time in seconds to dwell/log at each waypoint.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=20.0,
        help="Logging rate in Hz while dwelling at a waypoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name(
            f"flexiv_slam_log_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        ),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--interp-steps",
        type=int,
        default=10,
        help="Number of interpolation steps between current pose and target.",
    )
    parser.add_argument(
        "--home",
        action="store_true",
        help="Move robot to its predefined home pose before starting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    robot = FlexivRobot(home=FlexivRobot)
    base_pose = robot.get_tcp_pose().copy()
    print(f"[INFO] Using base pose: {base_pose}")

    init_glasses()
    if not wait_for_valid_slam():
        print("[WARN] SLAM pose not stable; proceeding anyway.")

    with args.output.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "wall_time",
                "repeat_idx",
                "direction",
                "interp_idx",
                "flexiv_x",
                "flexiv_y",
                "flexiv_z",
                "flexiv_rw",
                "flexiv_rx",
                "flexiv_ry",
                "flexiv_rz",
                "slam_host_ts",
                "slam_edge_ts",
                "slam_confidence",
                "slam_x",
                "slam_y",
                "slam_z",
                "slam_roll",
                "slam_pitch",
                "slam_yaw",
                "slam_qw",
                "slam_qx",
                "slam_qy",
                "slam_qz",
            ]
        )
        try:
            log_motion(
                robot=robot,
                base_pose=base_pose,
                path=sequence_offsets(args.delta),
                repeats=args.repeats,
                dwell=args.dwell,
                sample_dt=1.0 / max(args.sample_rate, 1e-3),
                interp_steps=args.interp_steps,
                csv_writer=writer,
            )
        finally:
            shutdown_glasses()
            print(f"[INFO] Log saved to {args.output}")


if __name__ == "__main__":
    main()
