#!/usr/bin/env python3
"""Drive Piper arm and glasses head tracker to measure 6DoF consistency."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from glasses_hardware.hardware.utils import quaternion_from_matrix, quaternion_to_matrix
from piper_sdk import C_PiperInterface_V2

COUNT_TO_M = 1e-6  # Piper end pose counts are 0.001 mm.
COUNT_TO_DEG = 1e-3
DEFAULT_SEQUENCE = (
    (57.0, 0.0, 215.0, 0.0, 85.0, 0.0, 0.0),
    (57.0, 0.0, 260.0, 0.0, 85.0, 0.0, 0.0),
)


@dataclass(frozen=True)
class EndPoseTarget:
    """Desired TCP pose command expressed in mm / deg."""

    x_mm: float
    y_mm: float
    z_mm: float
    rx_deg: float
    ry_deg: float
    rz_deg: float
    gripper: float = 0.0

    @property
    def as_list(self) -> List[float]:
        return [
            self.x_mm,
            self.y_mm,
            self.z_mm,
            self.rx_deg,
            self.ry_deg,
            self.rz_deg,
            self.gripper,
        ]

    def to_counts(self) -> Tuple[int, int, int, int, int, int, int]:
        mm_to_counts = 1000.0
        return (
            int(round(self.x_mm * mm_to_counts)),
            int(round(self.y_mm * mm_to_counts)),
            int(round(self.z_mm * mm_to_counts)),
            int(round(self.rx_deg * mm_to_counts)),
            int(round(self.ry_deg * mm_to_counts)),
            int(round(self.rz_deg * mm_to_counts)),
            int(round(self.gripper * mm_to_counts)),
        )


@dataclass
class HeadPoseSample:
    stamp: float
    position_m: np.ndarray
    quat_xyzw: np.ndarray


@dataclass
class StepRecord:
    step_idx: int
    target_index: int
    robot_stamp: float
    head_stamp: float
    robot_transform: np.ndarray
    head_transform: np.ndarray


def rotation_matrix_from_euler_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Create rotation matrix from XYZ intrinsic Euler angles (degrees)."""
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    r_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    r_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return r_z @ r_y @ r_x


def matrix_to_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion in xyzw ordering."""
    qw, qx, qy, qz = quaternion_from_matrix(R)
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def xyzw_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """Convert quaternion in xyzw ordering to rotation matrix."""
    qx, qy, qz, qw = quat
    return quaternion_to_matrix(np.array([qw, qx, qy, qz], dtype=np.float64))


def invert_transform(T: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    delta = R_pred.T @ R_gt
    trace = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(trace))


class GlassesHeadposeAccuracy(Node):
    """ROS2 node coordinating Piper motion with glasses head pose logging."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("glasses_headpose_accuracy")
        self.args = args
        self._head_lock = Lock()
        self._latest_head: Optional[HeadPoseSample] = None
        self.records: List[StepRecord] = []
        self.step_idx = 0
        self.target_ptr = 0
        self.stop_requested = False
        self.targets = [EndPoseTarget(*vals) for vals in DEFAULT_SEQUENCE]
        self.head_scale = float(args.head_scale)
        self.log_root = Path(args.log_root).expanduser().resolve()
        self.robot_dir = self.log_root / "robot_tcp"
        self.head_dir = self.log_root / "headpos"
        self._prepare_output_dirs(force=args.force)

        self.create_subscription(PoseStamped, args.head_topic, self._headpose_callback, 10)

        self.get_logger().info(f"Logging to {self.log_root}")
        self._connect_robot(can_port=args.can_port)

    def _prepare_output_dirs(self, force: bool) -> None:
        self.log_root.mkdir(parents=True, exist_ok=True)
        for path in (self.robot_dir, self.head_dir):
            if path.exists():
                entries = list(path.iterdir())
                if entries and not force:
                    raise RuntimeError(f"{path} is not empty, pass --force to overwrite existing logs.")
                if force:
                    for item in entries:
                        if item.is_file():
                            item.unlink()
            path.mkdir(parents=True, exist_ok=True)
        if force:
            for fname in ("T_tcp_head.npy", "T_head_base.npy", "T_robot_base.npy", "headpose_errors.csv"):
                fpath = self.log_root / fname
                if fpath.exists():
                    fpath.unlink()

    def _connect_robot(self, can_port: str) -> None:
        self.get_logger().info(f"Connecting to Piper arm on {can_port} ...")
        self.piper = C_PiperInterface_V2(can_name=can_port)
        self.piper.ConnectPort()
        time.sleep(0.2)
        start = time.time()
        while not self.piper.EnablePiper():
            if time.time() - start > 5.0:
                raise RuntimeError("Failed to enable Piper arm within 5s.")
            time.sleep(0.05)
        self.get_logger().info("Piper arm enabled.")

    def _headpose_callback(self, msg: PoseStamped) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        position = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float64,
        ) * self.head_scale
        orientation = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        sample = HeadPoseSample(stamp=stamp, position_m=position, quat_xyzw=orientation)
        with self._head_lock:
            self._latest_head = sample

    def wait_for_head_pose(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while rclpy.ok():
            with self._head_lock:
                if self._latest_head is not None:
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._pump_ros_once(min(0.1, remaining))
        return False

    def _pump_ros_once(self, timeout: float) -> None:
        try:
            rclpy.spin_once(self, timeout_sec=max(timeout, 1e-3))
        except rclpy.executors.ExternalShutdownException:
            self.stop_requested = True

    def _send_end_pose(self, target: EndPoseTarget, speed: int) -> None:
        X, Y, Z, RX, RY, RZ, grip = target.to_counts()
        self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x00, move_spd_rate_ctrl=speed, is_mit_mode=0x00)
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        self.piper.GripperCtrl(abs(grip), 1000, 0x01, 0)

    def _capture_robot_pose(
        self, target_index: int, target: EndPoseTarget, command_stamp: float
    ) -> Tuple[np.ndarray, dict, float]:
        measurement_stamp = time.time()
        msg = self.piper.GetArmEndPoseMsgs()
        ep = msg.end_pose
        counts = {
            "X_axis": int(ep.X_axis),
            "Y_axis": int(ep.Y_axis),
            "Z_axis": int(ep.Z_axis),
            "RX_axis": int(ep.RX_axis),
            "RY_axis": int(ep.RY_axis),
            "RZ_axis": int(ep.RZ_axis),
        }
        pos_m = np.array(
            [counts["X_axis"] * COUNT_TO_M, counts["Y_axis"] * COUNT_TO_M, counts["Z_axis"] * COUNT_TO_M],
            dtype=np.float64,
        )
        rx_deg = counts["RX_axis"] * COUNT_TO_DEG
        ry_deg = counts["RY_axis"] * COUNT_TO_DEG
        rz_deg = counts["RZ_axis"] * COUNT_TO_DEG
        R = rotation_matrix_from_euler_xyz(rx_deg, ry_deg, rz_deg)
        quat_xyzw = matrix_to_xyzw(R)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = pos_m
        robot_stamp = measurement_stamp
        data = {
            "stamp": command_stamp,
            "position_m": pos_m.tolist(),
            "orientation_xyzw": quat_xyzw.tolist(),
            "raw_end_pose_counts": counts,
            "measurement_stamp": robot_stamp,
            "stamp_head_aligned": command_stamp,
            "capture_latency_s": robot_stamp - command_stamp,
            "target_index": target_index,
            "end_pose_target": target.as_list,
        }
        return T, data, robot_stamp

    def _capture_head_pose(self) -> Optional[HeadPoseSample]:
        with self._head_lock:
            sample = self._latest_head
        if sample is None:
            return None
        if self.args.headpose_max_age_s > 0.0:
            if time.time() - sample.stamp > self.args.headpose_max_age_s:
                return None
        return sample

    def _record_step(
        self,
        robot_T: np.ndarray,
        robot_json: dict,
        robot_stamp: float,
        head_sample: HeadPoseSample,
    ) -> None:
        head_T = np.eye(4, dtype=np.float64)
        head_T[:3, :3] = xyzw_to_matrix(head_sample.quat_xyzw)
        head_T[:3, 3] = head_sample.position_m
        step = StepRecord(
            step_idx=self.step_idx,
            target_index=robot_json["target_index"],
            robot_stamp=robot_stamp,
            head_stamp=head_sample.stamp,
            robot_transform=robot_T,
            head_transform=head_T,
        )
        self.records.append(step)
        robot_json_path = self.robot_dir / f"step_{self.step_idx:04d}.json"
        robot_mat_path = self.robot_dir / f"T_robot_{self.step_idx:04d}.npy"
        head_json_path = self.head_dir / f"step_{self.step_idx:04d}.json"
        head_mat_path = self.head_dir / f"T_head_{self.step_idx:04d}.npy"

        with robot_json_path.open("w", encoding="utf-8") as fp:
            json.dump(robot_json, fp, indent=2)
        np.save(robot_mat_path, robot_T)

        head_json = {
            "stamp": head_sample.stamp,
            "position_m": head_sample.position_m.tolist(),
            "orientation_xyzw": head_sample.quat_xyzw.tolist(),
        }
        with head_json_path.open("w", encoding="utf-8") as fp:
            json.dump(head_json, fp, indent=2)
        np.save(head_mat_path, head_T)

        self.get_logger().info(
            f"Logged step {self.step_idx:04d} | target {step.target_index} | "
            f"robot_ts={robot_stamp:.3f} head_ts={head_sample.stamp:.3f}"
        )
        self.step_idx += 1

    def perform_step(self) -> None:
        target = self.targets[self.target_ptr]
        target_index = self.target_ptr
        self.target_ptr = (self.target_ptr + 1) % len(self.targets)
        command_stamp = time.time()
        self.get_logger().info(f"Commanding target {target_index} ({target.as_list})")
        command_deadline = command_stamp + self.args.hold_s
        while time.time() < command_deadline and rclpy.ok() and not self.stop_requested:
            self._send_end_pose(target, speed=self.args.move_speed)
            self._pump_ros_once(self.args.command_period_s)

        if not rclpy.ok() or self.stop_requested:
            return

        self._pump_ros_once(self.args.settle_s)

        head_sample = self._capture_head_pose()
        if head_sample is None:
            self.get_logger().warning("Skipping step because no recent head pose was received.")
            return

        robot_T, robot_json, robot_stamp = self._capture_robot_pose(target_index, target, command_stamp)
        self._record_step(robot_T, robot_json, robot_stamp, head_sample)

        if self.args.max_steps and self.step_idx >= self.args.max_steps:
            self.stop_requested = True

    def finalize(self) -> None:
        if not self.records:
            self.get_logger().warning("No samples recorded; nothing to export.")
            return
        T_robot_base = self.records[0].robot_transform
        T_head_base = self.records[0].head_transform
        T_tcp_head = invert_transform(T_robot_base) @ T_head_base
        np.save(self.log_root / "T_robot_base.npy", T_robot_base)
        np.save(self.log_root / "T_head_base.npy", T_head_base)
        np.save(self.log_root / "T_tcp_head.npy", T_tcp_head)

        base_head = self.records[0].head_transform
        csv_path = self.log_root / "headpose_errors.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "step",
                    "target_index",
                    "timestamp",
                    "rel_translation_pred_m",
                    "rel_translation_head_m",
                    "rel_translation_mag_error_m",
                    "rel_rotation_error_deg",
                ]
            )
            for rec in self.records[1:]:
                T_pred_abs = rec.robot_transform @ T_tcp_head
                delta_pred = invert_transform(base_head) @ T_pred_abs
                delta_head = invert_transform(base_head) @ rec.head_transform

                pred_trans_mag = float(np.linalg.norm(delta_pred[:3, 3]))
                head_trans_mag = float(np.linalg.norm(delta_head[:3, 3]))
                trans_mag_err = abs(pred_trans_mag - head_trans_mag)
                rel_rot_err = rotation_error_deg(delta_pred[:3, :3], delta_head[:3, :3])
                writer.writerow(
                    [
                        rec.step_idx,
                        rec.target_index,
                        rec.head_stamp,
                        pred_trans_mag,
                        head_trans_mag,
                        trans_mag_err,
                        rel_rot_err,
                    ]
                )
        self.get_logger().info(
            f"Saved calibration (T_tcp_head/T_robot_base/T_head_base) and {len(self.records) - 1} error rows to {csv_path}"
        )

    def run(self) -> None:
        if not self.wait_for_head_pose(self.args.headpose_timeout_s):
            raise RuntimeError(
                f"No head pose received on {self.args.head_topic} within {self.args.headpose_timeout_s} seconds."
            )
        self.get_logger().info("Head pose stream detected. Press Ctrl+C to stop measurement.")
        try:
            while rclpy.ok() and not self.stop_requested:
                self.perform_step()
        except KeyboardInterrupt:
            self.get_logger().info("Interrupted by user.")
            self.stop_requested = True
        finally:
            self.finalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log Piper TCP and glasses head pose for 6DoF accuracy checks.")
    parser.add_argument("--log-root", type=str, default="glasses_hardware/calib/glasses_test", help="Directory to store logs.")
    parser.add_argument("--head-topic", type=str, default="/glasses_pose", help="Head pose ROS topic (PoseStamped).")
    parser.add_argument("--can-port", type=str, default="can0", help="CAN interface used by Piper arm.")
    parser.add_argument("--hold-s", type=float, default=2.0, help="Duration to stream the target pose before logging.")
    parser.add_argument("--settle-s", type=float, default=0.3, help="Extra wait after commanding before sampling.")
    parser.add_argument("--command-period-s", type=float, default=0.02, help="Delay between repeated EndPoseCtrl commands.")
    parser.add_argument(
        "--head-scale",
        type=float,
        default=1e-3,
        help="Scale applied to incoming head pose translation (use 1.0 if already in meters).",
    )
    parser.add_argument("--headpose-timeout-s", type=float, default=5.0, help="Timeout when waiting for initial head pose.")
    parser.add_argument("--headpose-max-age-s", type=float, default=0.2, help="Max allowed age of head pose samples.")
    parser.add_argument("--move-speed", type=int, default=60, help="Motion speed percentage passed to MotionCtrl_2.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional number of steps before auto-stop (0 disables).")
    parser.add_argument("--force", action="store_true", help="Clear existing logs under log_root before recording.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = GlassesHeadposeAccuracy(args)
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
