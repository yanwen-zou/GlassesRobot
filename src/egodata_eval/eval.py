import time
from datetime import datetime
from pathlib import Path
import sys

import cv2
import signal
import numpy as np
import torch
import rclpy

here = Path(__file__).resolve()
project_root = here.parents[2] # 指向仓库根目录
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from egodata_eval.eval_utils import (
    calibrate_from_three_balls,
    click_mask,
    headpose_base_to_tcp_abs,
    move_i2rt_to_init_angles,
    save_mask,
)  # type: ignore
from egodata_eval.eval_hardware import EvalHardware

from egodata_eval.get_depth import DepthEstimator  # type: ignore
from egodata_eval.get_pose import PoseEstimatorFP  # type: ignore
from egodata_eval.get_head import HeadPoseReader

# ========== MBA Trajectory Prediction (RISE) ==========

from MBA.utils.constants import TRANS_MIN, TRANS_MAX, IMG_MEAN, IMG_STD  # type: ignore
from MBA.utils.transformation import rotation_transform, mat_to_xyz_rot, xyz_rot_transform  # type: ignore
from egodata_eval.traj_predictor import TrajectoryPredictor  # type: ignore
from egodata_eval.eval_constant import *

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Online evaluation with manual ckpt path")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to RISE policy checkpoint (.ckpt)")
    ap.add_argument("--num_action", type=int, default=10)
    ap.add_argument("--mesh-name", type=str, default=DEFAULT_MESH_NAME, help="Name of mesh folder under data/ containing mesh.obj.")
    ap.add_argument("--enable-headpose-head", action="store_true", help="Enable headpose diffusion head in RISE model.")
    ap.add_argument("--tcp-zed", type=str, default=DEFAULT_TCP_ZED_TXT, help="Path to T_tcp_zed (4x4 SE3).")
    args = ap.parse_args()
    # Shared interval controlling how often heavy ops run
    update_interval = UPDATE_INTERVAL
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Prepare video output
    out_dir = Path(__file__).resolve().parent / "eval_output" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    headpose_topic = DEFAULT_POSE_TOPIC
    video_path = out_dir / "stream.mp4"
    # Initialize depth estimator and load model at start

    print("[INFO] Loading FoundationStereo depth model...")
    depth_est = DepthEstimator(scale=DEPTH_EST_SCALE) # no need to modify intrinsics;
    try:
        zed_handle = getattr(cam, "_zed", cam)
        info = zed_handle.get_camera_information()
        config = getattr(info, "camera_configuration", None)
        calibration = config.calibration_parameters if config else info.calibration_parameters
        left_cam = calibration.left_cam
        fx = float(getattr(left_cam, "fx"))
        fy = float(getattr(left_cam, "fy"))
        cx = float(getattr(left_cam, "cx"))
        cy = float(getattr(left_cam, "cy"))
        depth_est.K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        print(f"[INFO] Loaded intrinsics from ZED: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    except Exception as exc:
        print(f"[WARN] Failed to load intrinsics from ZED; using file defaults. Reason: {exc}")
    # One-time calibration to compute T_base_cam
    project_root = Path(__file__).resolve().parents[2]
    calib_dir = project_root / CALIB_DIR_REL
    calib_dir.mkdir(parents=True, exist_ok=True)
    T_base_cam0 = None

    exec_ctx = EvalHardware()
    cam = exec_ctx.camera

    T_base_cam0 = calibrate_from_three_balls(
        cam,
        depth_est,
        move_robot_fn=lambda: move_i2rt_to_init_angles(exec_ctx.i2rt_robot),
        centroid_log_dir=out_dir,
    )
    cam_size = cam.size

    disp_w = int(cam_size[0])
    disp_h = int(cam_size[1])
    print(f"[INFO] Display resolution set from ZEDCamera: {disp_w}x{disp_h}")

    win = WIN_STREAM
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # Video writer (writes displayed frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (disp_w, disp_h))

    # Handle interrupt signal to ensure video is saved
    interrupted = {"flag": False}
    def _on_sigint(signum, frame):
        interrupted["flag"] = True
    signal.signal(signal.SIGINT, _on_sigint)

    last_size = (0, 0)
    click_state = {"pending": False, "pt": (0, 0)}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and last_frame_full is not None:
            orig_w, orig_h = last_size
            sx = orig_w / disp_w
            sy = orig_h / disp_h
            x_orig = float(x) * sx
            y_orig = float(y) * sy
            click_state["pending"] = True
            click_state["pt"] = (x_orig, y_orig)

    cv2.setMouseCallback(win, on_mouse)

    depth_enabled = False
    pose_est = None
    pose_ready = False
    last_mask = None
    last_depth_m = None  # cache last computed depth
    traj_pred = None
    headpose_reader = None

    # Try initialize trajectory predictor (optional)

    traj_pred = TrajectoryPredictor(ckpt_path = Path(args.ckpt), 
                                    num_action = args.num_action, 
                                    enable_headpose_head = args.enable_headpose_head) # current traj pred is under base frame
    print(f"[INFO] Loaded RISE trajectory predictor from {args.ckpt}")
    if args.enable_headpose_head:
        rclpy.init(args=None)
        headpose_reader = HeadPoseReader(headpose_topic, args.tcp_zed, T_base_cam0)
    pose_records: list[dict[str, object]] = []
    executed_poses: list[np.ndarray] = []
    tcp_history: list[np.ndarray] = []
    pose_records: list[dict[str, object]] = []
    T_base_cam_runtime: list[np.ndarray] = []

    frame_idx = 0
    try:
        while True:
            stereo = cam.read_stereo()
            if stereo is None:
                continue
            frame, frame_right = stereo

            K_rs = depth_est.K.astype(np.float32)

            # Prepare an RGB copy if needed downstream (already resized)
            image_bgr = frame
            image_rgb = image_bgr[..., ::-1].copy()

            last_frame_full = image_rgb
            last_size = (frame.shape[1], frame.shape[0])
            disp = frame

            if click_state["pending"]:
                print("[INFO] Enter click_state")
                click_state["pending"] = False
                pt = click_state["pt"]
                try:
                    mask = click_mask(image_rgb, [pt], labels=[1], multimask=True)
                    print("[INFO] Mask calculated")
                except Exception as e:
                    cv2.displayStatusBar(win, f"SAM error: {e}", 3000)
                    mask = None

                if mask is not None:
                    # Save mask into eval_output directory
                    save_mask(mask,ts)
                    last_mask = mask
                    depth_enabled = True
                    

            if depth_enabled:
                do_update = (frame_idx % update_interval == 0)
                # print(f"[INFO] Frame {frame_idx}: depth/pred update={do_update}")
                if do_update:
                    with torch.no_grad():
                        # Depth on resized stereo
                        last_depth_m = depth_est.depth(frame, frame_right)
                depth_m = last_depth_m
                if depth_m is None:
                    continue
                # Initialize FoundationPose once we have a mask and depth
                if (not pose_ready) and (last_mask is not None) and (depth_m is not None):
                    mesh_path = Path(__file__).resolve().parents[2] / "data" / args.mesh_name / "mesh.obj" #TODO: Hardcode
                    if not mesh_path.exists():
                        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
                    if pose_est is None:
                        pose_est = PoseEstimatorFP(mesh_path)
                    pose = pose_est.initialize(frame, depth_m, last_mask, K_rs)
                    pose_ready = pose is not None

                # Track every 10 frames; overlay every frame using last pose
                if pose_ready and pose_est is not None:
                    # Use the same `update_interval` for pose tracking
                    if do_update:
                        pose_est.track(frame, depth_m, K_rs)
                    frame = pose_est.draw_overlay(frame, K_rs)

                    # Overlay trajectory prediction; then execute a few steps on robot
                    if traj_pred is not None and pose_est.pose_cam_ob is not None:
                        if do_update:
                            # print("[INFO] Running trajectory prediction...")
                            headpose_norm = None
                            if args.enable_headpose_head:
                                T_base_cam = headpose_reader.get_headpos(timeout_sec=0.0) if headpose_reader else None
                                if T_base_cam is None:
                                    raise RuntimeError("[eval] No headpose received yet from topic.")
                                headpose_raw = xyz_rot_transform(T_base_cam, from_rep="matrix", to_rep="rotation_6d").astype(np.float32)
                                headpose_norm = headpose_raw.copy() # [x,y,z,r6d] 9-dim
                                headpose_norm[:3] = (headpose_norm[:3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
                            else:
                                T_base_cam = T_base_cam0 # fixed head
                            if T_base_cam is not None:
                                T_base_cam_runtime.append(T_base_cam.astype(np.float32))
                            frame = traj_pred.predict_and_overlay(
                                frame,
                                depth_m,
                                K_rs,
                                pose_est.pose_cam_ob.astype(np.float32),
                                T_base_cam=T_base_cam,
                                headpose_cond=headpose_norm,
                            )
                            if args.enable_headpose_head and traj_pred.last_headpose_pred is not None: # execute headpose
                                headpose_base = traj_pred.last_headpose_pred.astype(np.float32)
                                headpose_tcp_abs = headpose_base_to_tcp_abs(headpose_base, T_base_cam)

                                end_idx = min(headpose_tcp_abs.shape[0], 1 + STEPS_TO_EXECUTE)
                                for step_idx in range(1, end_idx):
                                    exec_ctx.execute_headpose_delta(headpose_tcp_abs[step_idx:step_idx+1])

                            # Saving execution info
                            pose_cam_ob = pose_est.pose_cam_ob.astype(np.float32)
                            pose_robot_ob, pose_seq_robot, step_poses, step_tcp = exec_ctx.execute_robot_traj(
                                traj_pred,
                                pose_cam_ob,
                                T_base_cam,
                            )
                            pose_records.append(
                                {
                                    "timestamp": float(time.time()),
                                    "frame_idx": int(frame_idx),
                                    "object_pose_robot": pose_robot_ob,
                                    "pred_seq_robot": pose_seq_robot,
                                }
                            )
                            if step_poses:
                                executed_poses.extend(step_poses)
                            if step_tcp:
                                tcp_history.extend(step_tcp)
                        else:
                            frame = traj_pred.overlay_cached(frame, K_rs)
                frame_idx += 1 # increment frame index only when depth is enabled

            # Refresh display from possibly overlaid frame
            disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            cv2.imshow(win, disp)
            # Write current display frame to video
            if writer is not None and writer.isOpened():
                writer.write(disp)
            key = cv2.waitKey(1) & 0xFF
            if interrupted["flag"] or key == ord('q') or key == 27:
                break
            gripper_width = float(exec_ctx.flexiv_gripper.get_gripper_state())
            open_width = getattr(exec_ctx.flexiv_gripper, "max_width", GRIPPER_OPEN_WIDTH_DEFAULT)
            if gripper_width >= 0.8 * open_width:
                print(f"[INFO] Gripper open ({gripper_width:.4f}m); stopping.")
                break
            del frame, frame_right
            torch.cuda.empty_cache()
    finally:
        # Release resources and save video
        if headpose_reader is not None:
            headpose_reader.destroy_node()
        if args.enable_headpose_head:
            rclpy.shutdown()

        if writer is not None:
            writer.release()
            print(f"[INFO] Saved video to: {video_path}")

        if pose_records:
            pose_log_path = out_dir / "robot_pose_records.npy"
            np.save(pose_log_path, np.array(pose_records, dtype=object))
            print(f"[INFO] Saved robot-frame pose log to: {pose_log_path}")

        if executed_poses:
            executed_path = out_dir / "robot_executed_poses.npy"
            np.save(executed_path, np.stack(executed_poses, axis=0))
            print(f"[INFO] Saved executed robot poses to: {executed_path}")

        if tcp_history:
            tcp_path = out_dir / "robot_tcp_history.npy"
            np.save(tcp_path, np.stack(tcp_history, axis=0))
            print(f"[INFO] Saved robot TCP history to: {tcp_path}")

        if T_base_cam_runtime:
            t_base_cam_path = out_dir / "T_base_cam_runtime.npy"
            np.save(t_base_cam_path, np.stack(T_base_cam_runtime, axis=0))
            print(f"[INFO] Saved T_base_cam runtime log to: {t_base_cam_path}")

        cv2.destroyAllWindows()

        cam.close()

        if exec_ctx.i2rt_robot is not None:
            exec_ctx.i2rt_robot.close()
        if exec_ctx.i2rt_server_proc is not None and exec_ctx.i2rt_server_proc.is_alive():
            exec_ctx.i2rt_server_proc.terminate()
            exec_ctx.i2rt_server_proc.join(timeout=2.0)


if __name__ == "__main__":
    main()
