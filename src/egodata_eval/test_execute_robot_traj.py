from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R

here = Path(__file__).resolve()
project_root = here.parents[2]
src_root = project_root / "src"
mba_root = project_root / "MBA"
for path in (src_root, mba_root, project_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from MBA.utils.transformation import rotation_transform  # type: ignore
from egodata_eval.eval_constant import DEFAULT_BASE_TO_ROBOT_TXT, DEPTH_EST_SCALE, TASK_TCP_TO_OBJECT_SE3
from egodata_eval.eval_hardware import EvalHardware
from egodata_eval.eval_utils import calibrate_from_three_balls, click_mask
from egodata_eval.get_depth import DepthEstimator
from egodata_eval.get_pose import PoseEstimatorFP


def _build_abs_base_traj_x_single_direction(
    pose_base_ob: np.ndarray,
    start_angle_deg: float,
    target_angle_deg: float,
    grip_value: float,
) -> np.ndarray:
    xyz = pose_base_ob[:3, 3].astype(np.float32)
    base_rot = pose_base_ob[:3, :3].astype(np.float32)
    # IMPORTANT: execute_robot_traj treats row-0 as the reference pose.
    # So we always build exactly 2 rows: [origin, target].
    angles_rad = np.deg2rad(np.asarray([start_angle_deg, target_angle_deg], dtype=np.float32))

    traj = np.zeros((len(angles_rad), 10), dtype=np.float32)
    traj[:, :3] = xyz[None, :]
    traj[:, 9] = np.float32(grip_value)

    for i, ang in enumerate(angles_rad):
        rot_x = R.from_euler("x", ang).as_matrix().astype(np.float32)
        rot_i = rot_x @ base_rot
        rot6d = rotation_transform(rot_i[None, ...], "matrix", "rotation_6d").squeeze(0).astype(np.float32)
        traj[i, 3:9] = rot6d
    return traj


def _pose7_to_se3(pose7: np.ndarray) -> np.ndarray:
    xyz = pose7[:3].astype(np.float32)
    quat = pose7[3:7].astype(np.float32)
    rot = rotation_transform(quat[None, :], "quaternion", "matrix").squeeze(0).astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3, 3] = xyz
    return T


def _traj_row_to_pose_base(traj_row: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = np.asarray(traj_row[:3], dtype=np.float32)
    T[:3, :3] = rotation_transform(
        np.asarray(traj_row[3:9], dtype=np.float32)[None, :],
        "rotation_6d",
        "matrix",
    ).squeeze(0).astype(np.float32)
    return T


def _collect_object_mask(frame_bgr: np.ndarray) -> np.ndarray:
    win = "Click Object (Enter confirm, Esc quit)"
    clicks: list[tuple[float, float]] = []
    preview = frame_bgr.copy()

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((float(x), float(y)))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _on_mouse)
    try:
        while True:
            preview[:] = frame_bgr
            for pt in clicks:
                cv2.circle(preview, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
            cv2.imshow(win, preview)
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 10):  # Enter
                break
            if k == 27:  # Esc
                raise RuntimeError("User cancelled object click.")
        if not clicks:
            raise RuntimeError("No click provided for object mask.")
        return click_mask(frame_bgr[..., ::-1].copy(), clicks, labels=[1] * len(clicks), multimask=True)
    finally:
        cv2.destroyWindow(win)


def _quat_to_rot(quat_xyzw: np.ndarray) -> np.ndarray:
    return rotation_transform(quat_xyzw[None, :], "quaternion", "matrix").squeeze(0).astype(np.float32)


def _persist_task_tcp_to_object_se3(task: str, T_tcp_object: np.ndarray) -> None:
    const_path = Path(__file__).resolve().parent / "eval_constant.py"
    text = const_path.read_text()

    # Update in-memory dict first, then persist full dict block.
    TASK_TCP_TO_OBJECT_SE3[task] = T_tcp_object.astype(np.float32)

    keys = ["teapot", "book", "sword", "cup", "bread"]
    lines = ["TASK_TCP_TO_OBJECT_SE3 = {"]
    for k in keys:
        mat = TASK_TCP_TO_OBJECT_SE3.get(k, np.eye(4, dtype=np.float32)).astype(np.float32)
        lines.append(f'    "{k}": np.array(')
        lines.append("        [")
        for r in range(4):
            row = ", ".join(f"{float(v):.8f}" for v in mat[r])
            lines.append(f"            [{row}],")
        lines.append("        ],")
        lines.append("        dtype=np.float32,")
        lines.append("    ),")
    lines.append("}")
    new_block = "\n".join(lines)

    start = text.find("TASK_TCP_TO_OBJECT_SE3 = {")
    end = text.find("\n\n# Task-specific calibration init pose", start)
    if start < 0 or end < 0:
        raise RuntimeError("Failed to locate TASK_TCP_TO_OBJECT_SE3 block in eval_constant.py")

    updated = text[:start] + new_block + text[end:]
    const_path.write_text(updated)
    print(f"[INFO] Persisted TASK_TCP_TO_OBJECT_SE3[{task}] to {const_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test execute_robot_traj with absolute base-frame trajectory: fixed translation + x-axis rotation oscillation."
    )
    ap.add_argument("--task", type=str, default="book", help="Task name used to pick TASK_TCP_TO_OBJECT_SE3.")
    ap.add_argument("--base-to-robot-txt", type=str, default=DEFAULT_BASE_TO_ROBOT_TXT)
    ap.add_argument("--angle-deg", type=float, default=15.0, help="Oscillation amplitude in degrees.")
    ap.add_argument("--grip", type=float, default=0.0, help="Constant gripper signal in traj_denorm[:, 9].")
    ap.add_argument("--obj-x", type=float, default=0.5, help="Object base-frame x for pose_base_ob.")
    ap.add_argument("--obj-y", type=float, default=-0.20, help="Object base-frame y for pose_base_ob.")
    ap.add_argument("--obj-z", type=float, default=0.25, help="Object base-frame z for pose_base_ob.")
    ap.add_argument("--grip-close-width", type=float, default=0.0, help="Closed gripper width command.")
    ap.add_argument("--settle-sec", type=float, default=1.0, help="Sleep after execution for observation.")
    args = ap.parse_args()

    if not rclpy.ok():
        rclpy.init(args=None)

    hw = EvalHardware(
        base_to_robot_txt=args.base_to_robot_txt,
        task_name=args.task,
    )

    try:
        print("[INFO] Running ball calibration to get T_base_cam...")
        depth_est = DepthEstimator(scale=DEPTH_EST_SCALE, camera=hw.camera)
        T_base_cam = calibrate_from_three_balls(
            hw.camera,
            depth_est,
            move_robot_fn=None,
            centroid_log_dir=None,
        ).astype(np.float32)
        print(f"[INFO] T_base_cam:\n{T_base_cam}")

        print("[INFO] Capture one frame and click object for FoundationPose init...")
        stereo = hw.camera.read_stereo()
        if stereo is None:
            raise RuntimeError("Failed to read stereo frame from camera.")
        frame, frame_right = stereo
        depth_m = depth_est.depth(frame, frame_right)
        if depth_m is None:
            raise RuntimeError("Depth estimation failed.")
        obj_mask = _collect_object_mask(frame)

        mesh_path = Path(__file__).resolve().parents[2] / "data" / args.task / "mesh.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
        pose_est = PoseEstimatorFP(mesh_path)
        pose_cam_ob = pose_est.initialize(frame, depth_m, obj_mask, depth_est.K.astype(np.float32))
        if pose_cam_ob is None:
            raise RuntimeError("FoundationPose initialize failed.")
        pose_base_ob = (T_base_cam @ pose_cam_ob.astype(np.float32)).astype(np.float32)
        print(f"[INFO] first-frame pose_base_ob:\n{pose_base_ob}")

        # Update task transform using the requested formula:
        # R_object_tcp = inv(R_tcp_current) @ R_object_robot
        R_object_robot = (hw.T_robot_base[:3, :3].astype(np.float32) @ pose_base_ob[:3, :3].astype(np.float32)).astype(np.float32)
        curr_tcp_pose7 = hw.flexiv_robot.get_tcp_pose().astype(np.float32)
        R_tcp_current = _quat_to_rot(curr_tcp_pose7[3:7].astype(np.float32))
        R_object_tcp = (np.linalg.inv(R_tcp_current) @ R_object_robot).astype(np.float32)

        T_tcp_object = TASK_TCP_TO_OBJECT_SE3.get(args.task, np.eye(4, dtype=np.float32)).astype(np.float32)
        T_tcp_object_new = T_tcp_object.copy()
        # Only update rotation; keep existing translation unchanged.
        T_tcp_object_new[:3, :3] = np.linalg.inv(R_object_tcp).astype(np.float32)
        TASK_TCP_TO_OBJECT_SE3[args.task] = T_tcp_object_new
        _persist_task_tcp_to_object_se3(args.task, T_tcp_object_new)
        T_object_tcp = np.linalg.inv(T_tcp_object_new).astype(np.float32)
        print(f"[INFO] updated T_object_tcp rotation for task={args.task}:\n{T_object_tcp}")

        gripper_closed = False
        current_angle_deg = 0.0
        target_sign = 1
        while True:
            key = input("[INPUT] 输入 p 后闭合夹爪并开始执行轨迹 (Ctrl+C 退出): ").strip().lower()
            if key != "p":
                continue

            target_angle_deg = float(args.angle_deg) * (1.0 if target_sign > 0 else -1.0)
            traj_denorm = _build_abs_base_traj_x_single_direction(
                pose_base_ob=pose_base_ob,
                start_angle_deg=current_angle_deg,
                target_angle_deg=target_angle_deg,
                grip_value=args.grip,
            )
            sign_str = "+" if target_sign > 0 else "-"
            # print(
            #     f"[INFO] one-way traj: {current_angle_deg:.2f}deg -> {target_angle_deg:.2f}deg ({sign_str}x), "
            #     f"shape={traj_denorm.shape}"
            # )
            # print(f"[INFO] one-way traj direction: {sign_str}x, shape={traj_denorm.shape}")
            # print(f"[INFO] traj first row: {traj_denorm[0]}")
            # print(f"[INFO] traj last row: {traj_denorm[-1]}")

            if not gripper_closed:
                print("[INFO] close gripper before trajectory execution")
                curr_pose7 = hw.flexiv_robot.get_tcp_pose().astype(np.float32)
                hw._publish_arm_cmd(curr_pose7, float(args.grip_close_width))
                time.sleep(0.5)
                gripper_closed = True
            # print(f"[INFO] traj_denorm:\n{traj_denorm}")
            pose_robot_ob, tcp_seq_robot, executed = hw.execute_robot_traj(
                traj_denorm=traj_denorm,
                pose_base_ob=_traj_row_to_pose_base(traj_denorm[0]),
            )
            # print(f"[INFO] pose_robot_ob:\n{pose_robot_ob}")
            # print(f"[INFO] tcp_seq_robot shape: {tcp_seq_robot.shape}")
            # print(f"[INFO] executed steps: {len(executed)}")
            curr_tcp_pose7 = hw.flexiv_robot.get_tcp_pose().astype(np.float32)
            curr_tcp_se3 = _pose7_to_se3(curr_tcp_pose7)
            # print(f"[INFO] current tcp se3:\n{curr_tcp_se3}")
            time.sleep(float(args.settle_sec))
            current_angle_deg = target_angle_deg
            target_sign *= -1
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received, exiting.")
    finally:
        hw.close(timeout_s=5.0)
        if rclpy.ok():
            rclpy.shutdown()
        if hw.i2rt_server_proc is not None and hw.i2rt_server_proc.is_alive():
            hw.i2rt_server_proc.terminate()
            hw.i2rt_server_proc.join(timeout=2.0)


if __name__ == "__main__":
    main()
