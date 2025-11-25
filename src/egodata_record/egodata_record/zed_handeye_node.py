#!/usr/bin/env python3
"""
ROS2 节点：订阅 /glasses_pose (PoseStamped)，用棋盘格做 ZED -> TCP 手眼标定。

操作与 glasses_hardware/calib/zed_cam_calib.py 一致：
- 棋盘默认 12x9，单元 0.024 m，可用参数调整。
- 按空格/回车采样一帧（需检测到棋盘且收到最新 TCP 位姿）。
- 按 s 计算标定，保存 T_zed_to_tcp 和其逆。
- 按 q 退出。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyzed.sl as sl
import rclpy
from rclpy.utilities import remove_ros_args
import os
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

# Ensure we can import from the workspace (glasses_hardware, etc.) when running from install space.
def _find_workspace_root(start: Path) -> Path:
    for candidate in (start,) + tuple(start.parents):
        if (candidate / "src").is_dir() and (candidate / "install").is_dir():
            return candidate
    return start.parent


REPO_ROOT = _find_workspace_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glasses_hardware.hardware.utils import quaternion_to_matrix  # type: ignore

os.environ["PATH"] = "/home/yanwen/miniconda3/envs/zed/bin:" + os.environ["PATH"]
os.environ["PYTHONPATH"] = "/home/yanwen/miniconda3/envs/zed/lib/python3.10/site-packages:" + os.environ.get("PYTHONPATH", "")


def load_intrinsics(path: Path, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """加载相机内参/畸变，支持 txt/npy/npz，K_ZED.txt 需 --intrinsic-scale 设为 2.0。"""
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: np.ndarray = np.zeros((5, 1), dtype=np.float64)

    if path.suffix.lower() == ".txt":
        nums: List[float] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                tokens = [t for t in line.strip().split() if t]
                nums.extend([float(t) for t in tokens])
        if len(nums) < 9:
            raise ValueError(f"{path} 中数字不足 9 个，无法组成 3x3 内参矩阵。")
        camera_matrix = np.array(nums[:9], dtype=np.float64).reshape(3, 3)
        rest = nums[9:]
        if rest:
            dist_coeffs = np.array(rest, dtype=np.float64).reshape(-1, 1)
    else:
        data = np.load(str(path), allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "camera_matrix" in data:
                camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
            elif "K" in data:
                camera_matrix = np.array(data["K"], dtype=np.float64)
            if "dist" in data:
                dist_coeffs = np.array(data["dist"], dtype=np.float64)
            elif "dist_coeffs" in data:
                dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
        else:
            arr = np.array(data, dtype=np.float64)
            if arr.ndim == 2 and arr.shape == (3, 3):
                camera_matrix = arr
            elif arr.ndim == 0 and isinstance(arr.item(), dict):
                payload = arr.item()
                if "camera_matrix" in payload:
                    camera_matrix = np.array(payload["camera_matrix"], dtype=np.float64)
                elif "K" in payload:
                    camera_matrix = np.array(payload["K"], dtype=np.float64)
                if "dist" in payload:
                    dist_coeffs = np.array(payload["dist"], dtype=np.float64)
                elif "dist_coeffs" in payload:
                    dist_coeffs = np.array(payload["dist_coeffs"], dtype=np.float64)

    if camera_matrix is None:
        raise ValueError(f"无法从 {path} 读取相机内参。")

    if scale != 1.0:
        camera_matrix = camera_matrix.copy()
        camera_matrix[0, 0] *= scale
        camera_matrix[1, 1] *= scale
        camera_matrix[0, 2] *= scale
        camera_matrix[1, 2] *= scale
    return camera_matrix, dist_coeffs


def build_board_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float64)
    objp[:, :2] = grid * square_size_m
    return objp


@dataclass
class PoseSample:
    stamp: float
    position: np.ndarray
    quat_xyzw: np.ndarray

    def to_matrix(self) -> np.ndarray:
        qw, qx, qy, qz = (
            float(self.quat_xyzw[3]),
            float(self.quat_xyzw[0]),
            float(self.quat_xyzw[1]),
            float(self.quat_xyzw[2]),
        )
        R = quaternion_to_matrix(np.array([qw, qx, qy, qz], dtype=np.float64))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = self.position.astype(np.float64)
        return T


class ZedHandeyeNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("zed_handeye_calib")
        self.args = args
        self.latest_pose: Optional[PoseSample] = None

        self.create_subscription(PoseStamped, args.pose_topic, self._pose_cb, 10)

        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = int(args.frame_rate)
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED open failed.")

        self.runtime_params = sl.RuntimeParameters()
        self.left = sl.Mat()

        self.camera_matrix, self.dist_coeffs = load_intrinsics(args.intrinsics, args.intrinsic_scale)
        self.obj_points = build_board_points(args.pattern_cols, args.pattern_rows, args.square_size_m)
        self.pattern_size = (args.pattern_cols, args.pattern_rows)

        method_map = {
            "tsai": cv2.CALIB_HAND_EYE_TSAI,
            "park": cv2.CALIB_HAND_EYE_PARK,
            "horaud": cv2.CALIB_HAND_EYE_HORAUD,
            "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
            "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        self.calib_method = method_map[args.method]

        self.R_gripper2base: List[np.ndarray] = []
        self.t_gripper2base: List[np.ndarray] = []
        self.R_target2cam: List[np.ndarray] = []
        self.t_target2cam: List[np.ndarray] = []
        self.frame_id = 0

        cv2.namedWindow("zed_calib", cv2.WINDOW_NORMAL)
        self.get_logger().info(
            f"按空格/回车采样，s 求解，q 退出；订阅 {args.pose_topic}，最少样本 {args.min_samples}"
        )

    def _pose_cb(self, msg: PoseStamped) -> None:
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
        quat = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.latest_pose = PoseSample(stamp=stamp, position=pos, quat_xyzw=quat)

    def _solve_board_pose(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not ret:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        success, rvec, tvec = cv2.solvePnP(self.obj_points, corners_sub, self.camera_matrix, self.dist_coeffs)
        if not success:
            return None
        return corners_sub, rvec, tvec

    def _calibrate(self) -> None:
        T_zed_to_tcp = calibrate_handeye(
            self.R_gripper2base,
            self.t_gripper2base,
            self.R_target2cam,
            self.t_target2cam,
            self.calib_method,
        )
        T_tcp_to_zed = np.linalg.inv(T_zed_to_tcp)
        rot_err_deg, trans_err = compute_pairwise_error(
            self.R_gripper2base, self.t_gripper2base, self.R_target2cam, self.t_target2cam, T_zed_to_tcp
        )

        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        inverse_path = (
            self.args.inverse_output
            if self.args.inverse_output is not None
            else self.args.output.with_name(self.args.output.stem + "_tcp_to_zed.npy")
        )
        np.save(str(self.args.output), T_zed_to_tcp.astype(np.float32))
        np.save(str(inverse_path), T_tcp_to_zed.astype(np.float32))
        self.get_logger().info(
            f"标定完成，pairwise 平均误差：rot={rot_err_deg:.3f} deg, trans={trans_err:.4f} m；已保存 {self.args.output} 和 {inverse_path}"
        )
        self.get_logger().info(f"T_zed_to_tcp:\n{T_zed_to_tcp}")
        self.get_logger().info(f"T_tcp_to_zed:\n{T_tcp_to_zed}")

    def run(self) -> None:
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.0)
                if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
                    continue
                self.zed.retrieve_image(self.left, sl.VIEW.LEFT)
                frame = self.left.get_data()
                if frame is None:
                    continue

                detection = self._solve_board_pose(frame)
                display = frame.copy()
                if detection is not None:
                    corners, _, _ = detection
                    cv2.drawChessboardCorners(display, self.pattern_size, corners, True)
                cv2.putText(
                    display,
                    f"samples: {len(self.R_gripper2base)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("zed_calib", display)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                if key in (ord(" "), ord("\r"), ord("\n")):
                    if detection is None:
                        self.get_logger().warning("未检测到棋盘，跳过采样。")
                        continue
                    if self.latest_pose is None:
                        self.get_logger().warning("未收到眼镜 TCP 位姿，跳过采样。")
                        continue
                    _, rvec, tvec = detection
                    T_base_tcp = self.latest_pose.to_matrix()

                    self.R_gripper2base.append(T_base_tcp[:3, :3].astype(np.float64))
                    self.t_gripper2base.append(T_base_tcp[:3, 3].reshape(3, 1).astype(np.float64))
                    R_cam, _ = cv2.Rodrigues(rvec)
                    self.R_target2cam.append(R_cam.astype(np.float64))
                    self.t_target2cam.append(tvec.reshape(3, 1).astype(np.float64))
                    self.frame_id += 1
                    self.get_logger().info(f"采样成功，累计 {len(self.R_gripper2base)} 帧。")

                if key == ord("s"):
                    if len(self.R_gripper2base) < self.args.min_samples:
                        self.get_logger().warning(
                            f"样本不足（{len(self.R_gripper2base)}/{self.args.min_samples}），继续采集。"
                        )
                        continue
                    self._calibrate()
        finally:
            cv2.destroyAllWindows()
            if self.zed.is_opened():
                self.zed.close()


def calibrate_handeye(
    R_gripper2base: List[np.ndarray],
    t_gripper2base: List[np.ndarray],
    R_target2cam: List[np.ndarray],
    t_target2cam: List[np.ndarray],
    method: int,
) -> np.ndarray:
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=method,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam2gripper
    T[:3, 3] = t_cam2gripper.reshape(3)
    return T


def compute_pairwise_error(
    Rg_list: List[np.ndarray],
    tg_list: List[np.ndarray],
    Rt_list: List[np.ndarray],
    tt_list: List[np.ndarray],
    T_cam_to_tcp: np.ndarray,
) -> tuple[float, float]:
    """AX=XB 残差：逐对样本计算 (A X) 与 (X B) 的旋转/平移差异平均值。"""
    if len(Rg_list) < 2:
        return 0.0, 0.0
    rot_errs = []
    trans_errs = []
    T_cam_to_tcp = np.asarray(T_cam_to_tcp, dtype=np.float64)
    for i in range(len(Rg_list) - 1):
        Rg1, tg1 = Rg_list[i], tg_list[i].reshape(3, 1)
        Rg2, tg2 = Rg_list[i + 1], tg_list[i + 1].reshape(3, 1)
        Rt1, tt1 = Rt_list[i], tt_list[i].reshape(3, 1)
        Rt2, tt2 = Rt_list[i + 1], tt_list[i + 1].reshape(3, 1)

        Ra = Rg2 @ Rg1.T
        ta = tg2 - Ra @ tg1
        Rb = Rt2 @ Rt1.T
        tb = tt2 - Rb @ tt1

        Rlhs = Ra @ T_cam_to_tcp[:3, :3]
        tlhs = Ra @ T_cam_to_tcp[:3, 3].reshape(3, 1) + ta
        Rrhs = T_cam_to_tcp[:3, :3] @ Rb
        trhs = T_cam_to_tcp[:3, :3] @ tb + T_cam_to_tcp[:3, 3].reshape(3, 1)

        Rdelta = Rlhs @ Rrhs.T
        trace = np.clip((np.trace(Rdelta) - 1.0) * 0.5, -1.0, 1.0)
        rot_errs.append(float(np.degrees(np.arccos(trace))))
        trans_errs.append(float(np.linalg.norm(tlhs - trhs)))

    return float(np.mean(rot_errs)), float(np.mean(trans_errs))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS2 节点：ZED -> 眼镜 TCP 手眼标定（棋盘格）。")
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=Path("src/FoundationStereo/assets/K_ZED.txt"),
        help="ZED 左相机内参路径（npy/npz/txt），默认使用 FoundationStereo 的 K_ZED.txt。",
    )
    parser.add_argument(
        "--intrinsic-scale",
        type=float,
        default=2.0,
        help="若内参是下采样结果，按此倍率放大 fx,fy,cx,cy（K_ZED.txt 需 *2 还原）。",
    )
    parser.add_argument("--output", type=Path, default=Path("glasses_hardware/calib/T_zed_tcp.npy"))
    parser.add_argument("--inverse-output", type=Path, default=None, help="可选：保存 TCP->ZED 的路径。")
    parser.add_argument("--pose-topic", type=str, default="/glasses_pose", help="PoseStamped 话题名。")
    parser.add_argument("--frame-rate", type=float, default=30.0, help="ZED 采集帧率。")
    parser.add_argument("--pattern-cols", type=int, default=11, help="棋盘格列数（内角点数）。")
    parser.add_argument("--pattern-rows", type=int, default=8, help="棋盘格行数（内角点数）。")
    parser.add_argument("--square-size-m", type=float, default=0.024, help="棋盘格单元边长（米）。")
    parser.add_argument(
        "--method",
        type=str,
        default="tsai",
        choices=["tsai", "park", "horaud", "andreff", "daniilidis"],
        help="手眼标定算法。",
    )
    parser.add_argument("--min-samples", type=int, default=10, help="最少采样次数。")
    return parser.parse_args(argv)


def main(args=None) -> None:
    # Strip ROS-specific launch args before argparse parsing.
    cli_args = parse_args(remove_ros_args(sys.argv)[1:])
    rclpy.init(args=args)
    node = ZedHandeyeNode(cli_args)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
