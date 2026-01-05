#!/usr/bin/env python3
"""
手眼标定：求解 ZED 相机到 I2RT TCP 的变换矩阵。

使用步骤（默认 12x9 棋盘格）：
1. 放置棋盘格保持静止，移动 I2RT（带 ZED）到多种姿态，确保棋盘被完整观测。
2. 运行本脚本，窗口中按空格/回车采样一帧（需检测到棋盘且读取到当前 TCP 位姿）。
3. 累积若干帧后按 s 计算手眼结果，保存到 --output（默认 T_i2rt_zed.txt）。
按 q 退出。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
I2RT_SDK_ROOT = REPO_ROOT / "glasses_hardware" / "i2rt"
if str(I2RT_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(I2RT_SDK_ROOT))

from glasses_hardware.hardware.my_device.zed import ZEDCamera  # type: ignore
from glasses_hardware.hardware.my_device.i2rt_robo import I2RTClient  # type: ignore
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from egodata_eval.eval_utils import move_i2rt_to_init_angles, _run_i2rt_server  # type: ignore
from i2rt.robots.kinematics_mj import Kinematics  # type: ignore
from i2rt.robots.utils import YAM_XML_PATH  # type: ignore


def read_zed_intrinsics(zed: ZEDCamera) -> Tuple[np.ndarray, np.ndarray]:
    """从 ZED SDK 读取左目内参和畸变系数。"""
    cam_info = zed._zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    left = calib.left_cam
    camera_matrix = np.array(
        [
            [float(left.fx), 0.0, float(left.cx)],
            [0.0, float(left.fy), float(left.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.array(getattr(left, "disto", []), dtype=np.float64).reshape(-1, 1)
    if dist.size == 0:
        dist = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist


def build_board_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    """生成棋盘格三维角点（位于 z=0 平面，X 沿列，Y 沿行）。"""
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float64)
    objp[:, :2] = grid * square_size_m
    return objp


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


def apply_tcp_delta(T: np.ndarray, dpos: np.ndarray, drot_rpy: np.ndarray) -> np.ndarray:
    """Apply position + rpy deltas in world frame."""
    pos = np.asarray(T[:3, 3], dtype=np.float32)
    rot = R.from_matrix(T[:3, :3]) * R.from_euler("xyz", drot_rpy)
    T_new = np.eye(4, dtype=np.float32)
    T_new[:3, :3] = rot.as_matrix().astype(np.float32)
    T_new[:3, 3] = (pos + dpos.astype(np.float32)).astype(np.float32)
    return T_new


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
    parser = argparse.ArgumentParser(description="使用棋盘格的手眼标定（ZED -> I2RT TCP）")
    parser.add_argument("--output", type=Path, default=Path("glasses_hardware/calib/T_i2rt_zed.txt"))
    parser.add_argument("--inverse-output", type=Path, default=None, help="可选：保存 ZED->TCP 的路径。")
    parser.add_argument("--i2rt-channel", type=str, default="can0", help="I2RT CAN 通道（仅用于启动 server）。")
    parser.add_argument("--i2rt-port", type=int, default=11333, help="I2RT RPC 端口。")
    parser.add_argument(
        "--start-i2rt-server",
        action="store_true",
        default=True,
        help="若未启动 I2RT server，则自动启动（默认开启）。",
    )
    parser.add_argument(
        "--no-start-i2rt-server",
        action="store_false",
        dest="start_i2rt_server",
        help="不自动启动 I2RT server。",
    )
    parser.add_argument("--i2rt-connect-timeout", type=float, default=15.0, help="等待 I2RT server 连接的超时秒数。")
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
    parser.add_argument("--log-dir", type=Path, default=None, help="可选：保存每帧图像与位姿。")
    parser.add_argument("--pos-step", type=float, default=0.01, help="键盘平移步长（米）。")
    parser.add_argument("--rot-step", type=float, default=0.03, help="键盘旋转步长（弧度）。")
    parser.add_argument("--move-duration", type=float, default=0.2, help="键盘控制关节移动时长。")
    parser.add_argument("--move-steps", type=int, default=20, help="键盘控制关节移动插值步数。")
    args = parser.parse_args()

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

    i2rt_server_proc: Optional[mp.Process] = None
    if args.start_i2rt_server:
        i2rt_server_proc = mp.Process(
            target=_run_i2rt_server,
            args=(args.i2rt_channel, False, args.i2rt_port),
            daemon=True,
        )
        i2rt_server_proc.start()
        time.sleep(0.5)

    robot = I2RTClient(host="127.0.0.1", port=args.i2rt_port)
    start_time = time.time()
    while True:
        try:
            _ = robot.num_dofs()
            break
        except Exception:
            if time.time() - start_time > args.i2rt_connect_timeout:
                raise RuntimeError("Failed to connect to I2RT server within timeout.")
            time.sleep(0.5)
    kin = Kinematics(YAM_XML_PATH, "grasp_site")
    move_i2rt_to_init_angles(robot)
    current_q = robot.current_joint_pos()
    target_pose = kin.fk(current_q[:6]).astype(np.float32)
    zed = ZEDCamera()
    camera_matrix, dist_coeffs = read_zed_intrinsics(zed)

    R_gripper2base: List[np.ndarray] = []
    t_gripper2base: List[np.ndarray] = []
    R_target2cam: List[np.ndarray] = []
    t_target2cam: List[np.ndarray] = []

    save_dir = args.log_dir
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[提示] q 退出 | 空格/回车 采样 | s 计算标定\n"
        "[提示] 键盘移动: t/g(+/-X) f/h(+/-Y) r/v(+/-Z)\n"
        "[提示] 旋转: u/o(+/-roll) i/k(+/-pitch) j/l(+/-yaw)"
    )
    cv2.namedWindow("zed_calib", cv2.WINDOW_NORMAL)

    frame_id = 0
    try:
        while True:
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
            if key in (ord("t"), ord("g"), ord("f"), ord("h"), ord("r"), ord("v"), ord("u"), ord("o"), ord("i"), ord("k"), ord("j"), ord("l")):
                dpos = np.zeros(3, dtype=np.float32)
                drot = np.zeros(3, dtype=np.float32)
                if key == ord("t"):
                    dpos[0] += args.pos_step
                elif key == ord("g"):
                    dpos[0] -= args.pos_step
                elif key == ord("f"):
                    dpos[1] += args.pos_step
                elif key == ord("h"):
                    dpos[1] -= args.pos_step
                elif key == ord("r"):
                    dpos[2] += args.pos_step
                elif key == ord("v"):
                    dpos[2] -= args.pos_step
                elif key == ord("u"):
                    drot[0] += args.rot_step
                elif key == ord("o"):
                    drot[0] -= args.rot_step
                elif key == ord("i"):
                    drot[1] += args.rot_step
                elif key == ord("k"):
                    drot[1] -= args.rot_step
                elif key == ord("j"):
                    drot[2] += args.rot_step
                elif key == ord("l"):
                    drot[2] -= args.rot_step
                next_pose = apply_tcp_delta(target_pose, dpos, drot)
                success, q_sol = kin.ik(next_pose, "grasp_site", verbose=False)
                if success:
                    target_pose = next_pose
                    current_q[:6] = q_sol[:6]
                    robot.send_joint_pos_rad(
                        current_q,
                        duration=args.move_duration,
                        steps=args.move_steps,
                    )
                else:
                    print("[WARN] IK failed for key input, skipping.")
            if key in (ord(" "), ord("\r"), ord("\n")):
                if detection is None:
                    print("[WARN] 未检测到棋盘，跳过采样。")
                    continue
                corners, rvec, tvec = detection
                current_q = robot.current_joint_pos()
                T_base_tcp = kin.fk(current_q[:6]).astype(np.float64)

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
                        timestamp=time.time(),
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
                np.savetxt(str(args.output), T_tcp_to_zed.astype(np.float64), fmt="%.8f")
                inverse_path = args.inverse_output
                if inverse_path is not None:
                    np.savetxt(str(inverse_path), T_zed_to_tcp.astype(np.float64), fmt="%.8f")
                print("[OK] 标定完成")
                print("T_tcp_to_zed (i2rt->zed):\n", T_tcp_to_zed)
                if inverse_path is not None:
                    print("T_zed_to_tcp:\n", T_zed_to_tcp)
                    print(f"已保存到 {args.output} 和 {inverse_path}")
                else:
                    print(f"已保存到 {args.output}")
    finally:
        cv2.destroyAllWindows()
        try:
            zed.close()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        if i2rt_server_proc is not None:
            i2rt_server_proc.terminate()


if __name__ == "__main__":
    main()
