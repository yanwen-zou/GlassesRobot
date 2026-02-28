#!/usr/bin/env python3
"""Calibrate task-specific T_tcp_object from live robot/camera observation."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
import tempfile

import cv2
import numpy as np
import rclpy

here = Path(__file__).resolve()
project_root = here.parents[2]
src_root = project_root / "src"
mba_root = project_root / "MBA"
for path in (src_root, mba_root, project_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from MBA.utils.transformation import rotation_transform  # type: ignore
from egodata_eval.eval_constant import (
    CALIB_DIR_REL,
    DEFAULT_BASE_TO_ROBOT_TXT,
    DEPTH_EST_SCALE,
    TASK_CHOICES,
    TASK_TCP_TO_OBJECT_SE3,
)
from egodata_eval.eval_hardware import EvalHardware
from egodata_eval.eval_utils import _load_calib_mat_safe, calibrate_from_three_balls, click_mask, move_i2rt_to_init_angles
from egodata_eval.get_depth import DepthEstimator
from egodata_eval.get_pose import PoseEstimatorFP


def _pose7_to_se3_robot(pose7: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = pose7[:3].astype(np.float32)
    T[:3, :3] = rotation_transform(
        np.asarray(pose7[3:7], dtype=np.float32)[None, :],
        "quaternion",
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
            if k in (13, 10):
                break
            if k == 27:
                raise RuntimeError("User cancelled object click.")
        if not clicks:
            raise RuntimeError("No click provided for object mask.")
        return click_mask(frame_bgr[..., ::-1].copy(), clicks, labels=[1] * len(clicks), multimask=True)
    finally:
        cv2.destroyWindow(win)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute live T_tcp_object from Flexiv TCP + FoundationPose object pose.")
    ap.add_argument("--task", type=str, required=True, choices=TASK_CHOICES, help="Task name / mesh folder name.")
    ap.add_argument("--base-to-robot-txt", type=str, default=DEFAULT_BASE_TO_ROBOT_TXT, help="Path of T_robot_base.")
    ap.add_argument(
        "--out-npz",
        type=str,
        default=None,
        help="Output npz path. Default: glasses_hardware/calib/task_tcp_object_<task>_<ts>.npz",
    )
    ap.add_argument("--skip-ball-calib", action="store_true", help="Load latest T_base_cam from --base-cam-txt instead.")
    ap.add_argument(
        "--base-cam-txt",
        type=str,
        default=None,
        help="When --skip-ball-calib is set, load T_base_cam from this txt (4x4 or 3x4).",
    )
    args = ap.parse_args()

    calib_dir = project_root / CALIB_DIR_REL
    calib_dir.mkdir(parents=True, exist_ok=True)
    temp_centroid_dir = Path(tempfile.mkdtemp(prefix="ball_centroids_", dir="/tmp"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_npz = Path(args.out_npz) if args.out_npz else (calib_dir / f"task_tcp_object_{args.task}_{ts}.npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    T_robot_base = _load_calib_mat_safe(Path(args.base_to_robot_txt))
    if T_robot_base is None:
        raise FileNotFoundError(f"Failed to load T_robot_base from: {args.base_to_robot_txt}")
    T_robot_base = T_robot_base.astype(np.float32)

    if not rclpy.ok():
        rclpy.init(args=None)
    exec_ctx = EvalHardware(
        base_to_robot_txt=args.base_to_robot_txt,
        task_name=args.task,
    )
    camera = exec_ctx.camera
    try:
        depth_est = DepthEstimator(scale=DEPTH_EST_SCALE, camera=camera)

        if args.skip_ball_calib:
            if not args.base_cam_txt:
                raise ValueError("--skip-ball-calib requires --base-cam-txt.")
            T_base_cam = _load_calib_mat_safe(Path(args.base_cam_txt))
            if T_base_cam is None:
                raise FileNotFoundError(f"Failed to load T_base_cam from: {args.base_cam_txt}")
            T_base_cam = T_base_cam.astype(np.float32)
        else:
            print("[INFO] Running three-ball calibration...")
            T_base_cam = calibrate_from_three_balls(
                camera,
                depth_est,
                move_robot_fn=lambda: move_i2rt_to_init_angles(exec_ctx.i2rt_robot, task_name=args.task),
                centroid_log_dir=temp_centroid_dir,
            ).astype(np.float32)

        stereo = camera.read_stereo()
        if stereo is None:
            raise RuntimeError("Failed to read stereo frame from camera.")
        frame, frame_right = stereo
        depth_m = depth_est.depth(frame, frame_right)
        if depth_m is None:
            raise RuntimeError("Depth estimation failed.")

        print("[INFO] Click object pixels for SAM mask...")
        obj_mask = _collect_object_mask(frame)

        mesh_path = project_root / "data" / args.task / "mesh.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
        pose_est = PoseEstimatorFP(mesh_path)
        T_cam_obj = pose_est.initialize(frame, depth_m, obj_mask, depth_est.K.astype(np.float32))
        if T_cam_obj is None:
            raise RuntimeError("FoundationPose initialize failed.")
        T_cam_obj = T_cam_obj.astype(np.float32)

        T_base_obj = (T_base_cam @ T_cam_obj).astype(np.float32)
        T_robot_obj = (T_robot_base @ T_base_obj).astype(np.float32)

        tcp_pose7 = exec_ctx.flexiv_robot.get_tcp_pose().astype(np.float32)
        T_robot_tcp = _pose7_to_se3_robot(tcp_pose7)
        T_tcp_obj_calib = (np.linalg.inv(T_robot_tcp) @ T_robot_obj).astype(np.float32)
        T_tcp_obj_const = np.asarray(
            TASK_TCP_TO_OBJECT_SE3.get(args.task, np.eye(4, dtype=np.float32)),
            dtype=np.float32,
        )
        # Use calibrated rotation + constant translation.
        T_tcp_obj = T_tcp_obj_calib.copy()
        T_tcp_obj[:3, 3] = T_tcp_obj_const[:3, 3]

        np.savez_compressed(
            str(out_npz),
            task=np.array(args.task),
            T_robot_base=T_robot_base,
            T_base_cam=T_base_cam,
            T_cam_obj=T_cam_obj,
            T_base_obj=T_base_obj,
            tcp_pose7=tcp_pose7,
            T_robot_tcp=T_robot_tcp,
            T_robot_obj=T_robot_obj,
            T_tcp_obj_calib=T_tcp_obj_calib,
            T_tcp_obj_const=T_tcp_obj_const,
            T_tcp_obj=T_tcp_obj,
        )
        print(f"[OK] Saved calibration result to: {out_npz}")
        print("[INFO] T_robot_tcp:")
        print(T_robot_tcp)
        print("[INFO] T_robot_obj:")
        print(T_robot_obj)
        print("[INFO] T_tcp_obj_calib (raw):")
        print(T_tcp_obj_calib)
        print("[INFO] T_tcp_obj_const (from eval_constant):")
        print(T_tcp_obj_const)
        print("[INFO] T_tcp_obj (rotation=calib, translation=constant):")
        print(T_tcp_obj)
    finally:
        try:
            exec_ctx.close(timeout_s=5.0)
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        try:
            if exec_ctx.i2rt_server_proc is not None and exec_ctx.i2rt_server_proc.is_alive():
                exec_ctx.i2rt_server_proc.terminate()
                exec_ctx.i2rt_server_proc.join(timeout=2.0)
        except Exception:
            pass
        try:
            shutil.rmtree(temp_centroid_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
