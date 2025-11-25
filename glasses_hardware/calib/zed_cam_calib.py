#!/usr/bin/env python3
"""
手眼标定：求解 ZED 相机到眼镜 TCP 的变换矩阵。

使用步骤（默认 12x9 棋盘格）：
1. 运行眼镜端姿态推流（UDP，格式：pose,x,y,z,qx,qy,qz,qw，默认端口 5006）。
2. 放置棋盘格保持静止，移动眼镜（带 ZED）到多种姿态，确保棋盘被完整观测。
3. 运行本脚本，窗口中按空格/回车采样一帧（需检测到棋盘且收到最新 TCP 位姿）。
4. 累积若干帧后按 s 计算手眼结果，保存到 --output。
   默认保存：
     - `--output`：T_zed_to_tcp，作用：p_tcp = T_zed_to_tcp @ p_zed
     - 同目录追加 `_tcp_to_zed.npy`：其逆变换。
按 q 退出。
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glasses_hardware.hardware.my_device.zed import ZEDCamera  # type: ignore
from glasses_hardware.hardware.utils import quaternion_to_matrix  # type: ignore


def load_intrinsics(path: Path, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """加载相机内参和畸变系数，支持 npy/npz/txt，键名 camera_matrix/K、dist/dist_coeffs。"""
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
        raise ValueError(f"无法从 {path} 读取相机内参（camera_matrix/K）。")

    if scale != 1.0:
        camera_matrix = camera_matrix.copy()
        camera_matrix[0, 0] *= scale
        camera_matrix[1, 1] *= scale
        camera_matrix[0, 2] *= scale
        camera_matrix[1, 2] *= scale
    return camera_matrix, dist_coeffs


def build_board_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    """生成棋盘格三维角点（位于 z=0 平面，X 沿列，Y 沿行）。"""
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float64)
    objp[:, :2] = grid * square_size_m
    return objp


@dataclass
class PoseSample:
    stamp: float
    position: np.ndarray  # (3,)
    quat_xyzw: np.ndarray  # (4,)

    def to_matrix(self) -> np.ndarray:
        """返回 base->tcp 4x4 变换（假设四元数为 xyzw）。"""
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


class UDPPoseReceiver:
    """接收眼镜 TCP 位姿，消息格式：pose,x,y,z,qx,qy,qz,qw（同 headpos_listener.py）。"""

    def __init__(self, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", port))
        self._sock.setblocking(False)
        self.latest: Optional[PoseSample] = None

    def poll(self) -> None:
        try:
            data, _ = self._sock.recvfrom(1024)
        except BlockingIOError:
            return
        parts = data.decode(errors="ignore").strip().split(",")
        if len(parts) < 8 or parts[0].lower() != "pose":
            return
        try:
            pos = np.array([float(x) for x in parts[1:4]], dtype=np.float64)
            quat = np.array([float(x) for x in parts[4:8]], dtype=np.float64)
        except ValueError:
            return
        self.latest = PoseSample(time.time(), pos, quat)


def solve_board_pose(
    frame_bgr: np.ndarray,
    pattern_size: Tuple[int, int],
    obj_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """检测棋盘并求解棋盘到相机的位姿。"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # print("pattern size: ", pattern_size)
    # print("obj points: ", obj_points)
    ret, corners = cv2.findChessboardCorners(
        gray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ret:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_sub = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    success, rvec, tvec = cv2.solvePnP(obj_points, corners_sub, camera_matrix, dist_coeffs)
    if not success:
        return None
    return corners_sub, rvec, tvec


def calibrate_handeye(
    R_gripper2base: List[np.ndarray],
    t_gripper2base: List[np.ndarray],
    R_target2cam: List[np.ndarray],
    t_target2cam: List[np.ndarray],
    method: int,
) -> np.ndarray:
    """运行 OpenCV 手眼标定，返回 T_zed_to_tcp（相机->TCP）。"""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="使用棋盘格的手眼标定（ZED -> 眼镜 TCP）")
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
    parser.add_argument("--pose-port", type=int, default=5006, help="UDP 端口（pose,x,y,z,qx,qy,qz,qw）。")
    parser.add_argument("--pattern-cols", type=int, default=12, help="棋盘格列数（内角点数）。")
    parser.add_argument("--pattern-rows", type=int, default=9, help="棋盘格行数（内角点数）。")
    parser.add_argument("--square-size-m", type=float, default=0.024, help="棋盘格单元边长（米）。")
    parser.add_argument(
        "--method",
        type=str,
        default="tsai",
        choices=["tsai", "park", "horaud", "andreff", "daniilidis"],
        help="手眼标定算法。",
    )
    parser.add_argument("--min-samples", type=int, default=10, help="最少采样次数。")
    parser.add_argument("--log-dir", type=Path, default=None, help="可选：保存每帧图像与位姿。")
    args = parser.parse_args()

    camera_matrix, dist_coeffs = load_intrinsics(args.intrinsics, args.intrinsic_scale)
    obj_points = build_board_points(args.pattern_cols, args.pattern_rows, args.square_size_m)
    pattern_size = (args.pattern_cols, args.pattern_rows)

    method_map = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    calib_method = method_map[args.method]

    receiver = UDPPoseReceiver(args.pose_port)
    zed = ZEDCamera()

    R_gripper2base: List[np.ndarray] = []
    t_gripper2base: List[np.ndarray] = []
    R_target2cam: List[np.ndarray] = []
    t_target2cam: List[np.ndarray] = []

    save_dir = args.log_dir
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[提示] q 退出 | 空格/回车 采样 | s 计算标定\n"
        f"[提示] 监听 UDP 端口 {args.pose_port}，等待眼镜 TCP 位姿..."
    )
    cv2.namedWindow("zed_calib", cv2.WINDOW_NORMAL)

    frame_id = 0
    try:
        # print("dist_coeffs: ", dist_coeffs)
        while True:
            receiver.poll()
            frame = zed.read()
            if frame is None:
                continue

            detection = solve_board_pose(frame, pattern_size, obj_points, camera_matrix, dist_coeffs)
            display = frame.copy()
            if detection is not None:
                corners, rvec, tvec = detection
                cv2.drawChessboardCorners(display, pattern_size, corners, True)
            cv2.putText(
                display,
                f"samples: {len(R_gripper2base)}",
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
                    print("[WARN] 未检测到棋盘，跳过采样。")
                    continue
                if receiver.latest is None:
                    print("[WARN] 未收到眼镜 TCP 位姿，跳过采样。")
                    continue
                corners, rvec, tvec = detection
                pose = receiver.latest
                T_base_tcp = pose.to_matrix()

                R_gripper2base.append(T_base_tcp[:3, :3].astype(np.float64))
                t_gripper2base.append(T_base_tcp[:3, 3].reshape(3, 1).astype(np.float64))
                R_cam, _ = cv2.Rodrigues(rvec)
                R_target2cam.append(R_cam.astype(np.float64))
                t_target2cam.append(tvec.reshape(3, 1).astype(np.float64))

                if save_dir is not None:
                    cv2.imwrite(str(save_dir / f"frame_{frame_id:03d}.png"), frame)
                    np.savez(
                        save_dir / f"sample_{frame_id:03d}.npz",
                        T_base_tcp=T_base_tcp,
                        rvec=rvec,
                        tvec=tvec,
                        timestamp=pose.stamp,
                    )
                frame_id += 1
                print(f"[INFO] 采样成功，累计 {len(R_gripper2base)} 帧。")

            if key == ord("s"):
                if len(R_gripper2base) < args.min_samples:
                    print(f"[WARN] 样本不足（{len(R_gripper2base)}/{args.min_samples}），继续采集。")
                    continue
                T_zed_to_tcp = calibrate_handeye(
                    R_gripper2base,
                    t_gripper2base,
                    R_target2cam,
                    t_target2cam,
                    calib_method,
                )
                T_tcp_to_zed = np.linalg.inv(T_zed_to_tcp)

                args.output.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(args.output), T_zed_to_tcp.astype(np.float32))
                inverse_path = (
                    args.inverse_output
                    if args.inverse_output is not None
                    else args.output.with_name(args.output.stem + "_tcp_to_zed.npy")
                )
                np.save(str(inverse_path), T_tcp_to_zed.astype(np.float32))
                print("[OK] 标定完成")
                print("T_zed_to_tcp:\n", T_zed_to_tcp)
                print("T_tcp_to_zed:\n", T_tcp_to_zed)
                print(f"已保存到 {args.output} 和 {inverse_path}")
    finally:
        cv2.destroyAllWindows()
        try:
            zed.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
