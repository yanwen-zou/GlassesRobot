import time
from datetime import datetime
from pathlib import Path
import sys

import cv2
import signal
import numpy as np
import torch
import rclpy
import os

here = Path(__file__).resolve()
project_root = here.parents[2] # 指向仓库根目录
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from egodata_eval.eval_utils import (
    calibrate_from_three_balls,
    click_mask,
    headpose_base_to_i2rt_rel,
    headpose_base_seq_to_rel,
    headpose_to_tcp,
    move_i2rt_to_init_angles,
    save_mask,
    _build_pose_mats,
    init_robot_mask_tracker,
    update_robot_mask_tracker,
    cleanup_robot_mask_tracker,
)  # type: ignore
from egodata_eval.eval_hardware import EvalHardware

from egodata_eval.get_depth import DepthEstimator  # type: ignore
from egodata_eval.get_pose import PoseEstimatorFP
from egodata_eval.get_head import HeadPoseReader
from egodata_eval.traj_predictor import TrajectoryPredictor


from MBA.utils.constants import TRANS_MIN, TRANS_MAX, IMG_MEAN, IMG_STD  # type: ignore
from MBA.utils.transformation import rotation_transform, mat_to_xyz_rot, xyz_rot_transform  # type: ignore
from egodata_eval.eval_constant import *

last_overlay = None


def _safe_save_mask_png(path: Path, mask: np.ndarray, win: str | None = None) -> bool:
    """Save a binary/uint8 mask as a valid PNG; emits debug info on failure."""
    arr = np.asarray(mask)
    # Common SAM2 output is (1, H, W); squeeze batch/channel dims.
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        msg = f"[robot_mask] invalid shape {arr.shape} dtype={arr.dtype} for {path}"
        print(msg)
        if win is not None:
            cv2.displayStatusBar(win, msg, 5000)
        return False
    if arr.dtype != np.uint8:
        # Normalize to {0,255} uint8.
        arr = (arr > 0).astype(np.uint8) * 255
    # Make sure the array is contiguous for OpenCV/libpng.
    arr = np.ascontiguousarray(arr)
    # Use a temp filename that still ends with ".png" so OpenCV picks the correct encoder.
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    ok = cv2.imwrite(str(tmp), arr)
    if not ok:
        msg = f"[robot_mask] cv2.imwrite failed for {path} shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()}"
        print(msg)
        if win is not None:
            cv2.displayStatusBar(win, msg, 5000)
        # Dump a numpy copy for post-mortem.
        try:
            np.save(str(path.with_suffix(".npy")), arr)
        except Exception:
            pass
        return False
    os.replace(tmp, path)
    return True


def _depth_to_pointcloud_np(
    depth_m: np.ndarray,
    K: np.ndarray,
    *,
    rgb: np.ndarray | None = None,
    stride: int = 2,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Backproject depth to a point cloud in camera frame, suitable for np.savez_compressed."""
    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be (H,W), got {depth.shape}")
    if stride < 1:
        raise ValueError(f"stride must be >=1, got {stride}")

    K = np.asarray(K, dtype=np.float32)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    depth_s = depth[::stride, ::stride].astype(np.float32, copy=False)
    Hs, Ws = depth_s.shape
    us = (np.arange(Ws, dtype=np.float32) * stride)[None, :]
    vs = (np.arange(Hs, dtype=np.float32) * stride)[:, None]
    z = depth_s

    if valid_mask is not None:
        vm = np.asarray(valid_mask)
        if vm.shape != depth.shape:
            raise ValueError(f"valid_mask shape {vm.shape} != depth shape {depth.shape}")
        vm_s = vm[::stride, ::stride]
        keep = (z > 0) & (vm_s > 0)
    else:
        keep = z > 0

    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    keep_f = keep.reshape(-1)
    xyz = xyz[keep_f]

    out: dict[str, np.ndarray] = {"xyz_cam": xyz.astype(np.float32, copy=False)}
    if rgb is not None:
        rgb_arr = np.asarray(rgb)
        if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3 or rgb_arr.shape[:2] != depth.shape:
            raise ValueError(f"rgb must be (H,W,3) matching depth, got {rgb_arr.shape}")
        rgb_s = rgb_arr[::stride, ::stride].reshape(-1, 3)
        out["rgb"] = rgb_s[keep_f].astype(np.uint8, copy=False)

    out["K"] = K
    out["stride"] = np.array([stride], dtype=np.int32)
    out["hw"] = np.array(depth.shape, dtype=np.int32)
    return out


def _run_depth_traj(
    state: dict,
    ctx: dict,
    depth_est: DepthEstimator,
    traj_pred: TrajectoryPredictor,
    *,
    cloud_out_path: Path | None = None,
) -> tuple[np.ndarray, dict | None]:
    frame = state["frame"]
    frame_right = state["frame_right"]
    if state.get("mask") is not None:
        ctx["last_mask"] = state["mask"]
    with torch.no_grad():
        ctx["last_depth_m"] = depth_est.depth(frame, frame_right)
    depth_m = ctx["last_depth_m"]
    if depth_m is None:
        return frame, None
    # if ctx.get("robot_mask") is not None:
    #     depth_m = depth_m.copy()
    #     depth_m[ctx["robot_mask"] > 0] = 0.0
    #     ctx["last_depth_m"] = depth_m

    if (not ctx["pose_ready"]) and (ctx["last_mask"] is not None):
        print("[INFO] Initialize Pose Estimator")
        if ctx["pose_est"] is None:
            ctx["pose_est"] = PoseEstimatorFP(ctx["mesh_path"])
        pose = ctx["pose_est"].initialize(frame, depth_m, ctx["last_mask"], ctx["K"])
        ctx["pose_ready"] = pose is not None

    frame_overlay = frame
    pred_state = None
    if ctx["pose_ready"] and ctx["pose_est"] is not None:
        ctx["pose_est"].track(frame, depth_m, ctx["K"])

        if ctx.get("robot_mask") is not None:
            depth_m = depth_m.copy()
            depth_m[ctx["robot_mask"] > 0] = 0.0
            ctx["last_depth_m"] = depth_m

        if traj_pred is not None and ctx["pose_est"].pose_cam_ob is not None:
            T_base_cam = state["T_base_cam"]
            headpose_norm = state.get("headpose_norm")
            frame_overlay, frame_cloud = traj_pred.predict_and_overlay(
                frame,
                depth_m,
                ctx["K"],
                ctx["pose_est"].pose_cam_ob.astype(np.float32),
                T_base_cam=T_base_cam,
                headpose_cond=headpose_norm,
            ) # overlay traj
            if cloud_out_path is not None:
                try:
                    np.savez_compressed(
                        str(cloud_out_path),
                        cloud=np.asarray(frame_cloud, dtype=np.float32),
                        frame_idx=np.array([int(state.get("frame_idx", -1))], dtype=np.int32),
                    )
                except Exception as e:
                    print(f"[WARN] Failed to save frame_cloud npz: {cloud_out_path} ({e})")
            frame_overlay = ctx["pose_est"].draw_overlay(frame_overlay, ctx["K"]) # overlay pose
            pred_state = {
                "pose_cam_ob": ctx["pose_est"].pose_cam_ob.astype(np.float32),
                "traj_denorm": traj_pred.last_traj_denorm.astype(np.float32),
                "headpose_pred": None
                if traj_pred.last_headpose_pred is None
                else traj_pred.last_headpose_pred.astype(np.float32),
                "T_base_cam": T_base_cam.astype(np.float32),
            }
    else:
        print("[DEBUG] Pose not ready...")

    return frame_overlay, pred_state


def main():
    import argparse
    global last_overlay
    ap = argparse.ArgumentParser(description="Online evaluation with manual ckpt path")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to RISE policy checkpoint (.ckpt)")
    ap.add_argument("--num_action", type=int, default=10)
    ap.add_argument("--task", type=str, choices=TASK_CHOICES, default="book", help="Task name (also used as mesh-name).")
    ap.add_argument("--enable-headpose-head", action="store_true", help="Enable headpose diffusion head in RISE model.")
    ap.add_argument("--glass-zed", type=str, default=DEFAULT_GLASSES_ZED_TXT, help="Path to T_tcp_zed (4x4 SE3).")
    args = ap.parse_args()
    # Shared interval controlling how often heavy ops run
    update_interval = UPDATE_INTERVAL
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Prepare video output
    out_dir = Path(__file__).resolve().parent / "eval_output" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    headpose_topic = DEFAULT_POSE_TOPIC
    video_path = out_dir / "stream.mp4"
    infer_left_dir = out_dir / "zed_left"
    infer_right_dir = out_dir / "zed_right"
    infer_left_dir.mkdir(parents=True, exist_ok=True)
    infer_right_dir.mkdir(parents=True, exist_ok=True)
    robot_mask_dir = out_dir / "mask_hand"
    robot_mask_dir.mkdir(parents=True, exist_ok=True)
    # Save pointclouds returned by predict_and_overlay here.
    cloud_dir = out_dir / "pointcloud"
    cloud_dir.mkdir(parents=True, exist_ok=True)
    # Initialize depth estimator and load model at start

    print("[INFO] Loading FoundationStereo depth model...")

    # One-time calibration to compute T_base_cam
    project_root = Path(__file__).resolve().parents[2]
    calib_dir = project_root / CALIB_DIR_REL
    calib_dir.mkdir(parents=True, exist_ok=True)
    T_base_cam0 = None

    exec_ctx = EvalHardware(task_name=args.task)
    cam = exec_ctx.camera
    depth_est = DepthEstimator(scale=DEPTH_EST_SCALE, camera=cam)
    K = depth_est.K.astype(np.float32)

    T_base_cam0 = calibrate_from_three_balls(
        cam,
        depth_est,
        move_robot_fn=lambda: move_i2rt_to_init_angles(exec_ctx.i2rt_robot, task_name=args.task),
        centroid_log_dir=out_dir,
    )
    exec_ctx.i2rt_current_q = exec_ctx.i2rt_robot.current_joint_pos()
    curr_headpose = exec_ctx.i2rt_kin.fk(exec_ctx.i2rt_current_q[:exec_ctx.i2rt_arm_dofs])
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
    mask = None
    robot_clicks: list[tuple[float, float]] = []
    robot_prompt_active = True
    robot_mask = None
    robot_tracker = None
    # Keep robot mask update cadence consistent with inference updates.
    # Use frame_idx-based filenames so masks align with inference frames.
    robot_mask_frame_idx = 0  # legacy counter (kept for backward compatibility if needed)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and last_frame_full is not None:
            orig_w, orig_h = last_size
            sx = orig_w / disp_w
            sy = orig_h / disp_h
            x_orig = float(x) * sx
            y_orig = float(y) * sy
            if robot_prompt_active:
                robot_clicks.append((x_orig, y_orig))
                print(f"[INFO] Robot click point added: ({x_orig:.1f}, {y_orig:.1f})")
            else:
                click_state["pending"] = True
                click_state["pt"] = (x_orig, y_orig)

    cv2.setMouseCallback(win, on_mouse)

    depth_enabled = False
    headpose_reader = None
    pose_est = None
    print("[INFO] Robot mask: click foreground points, press Enter to finish (Esc to skip).")

    traj_pred = TrajectoryPredictor(ckpt_path = Path(args.ckpt), 
                                    num_action = args.num_action, 
                                    enable_headpose_head = args.enable_headpose_head) # current traj pred is under base frame
    print(f"[INFO] Loaded RISE trajectory predictor from {args.ckpt}")

    if args.enable_headpose_head:
        if not rclpy.ok():
            rclpy.init(args=None)
        headpose_reader = HeadPoseReader(headpose_topic, args.glass_zed, T_base_cam0)
    mesh_path = Path(__file__).resolve().parents[2] / "data" / args.task / "mesh.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    infer_ctx = {
        "pose_est": None,
        "pose_ready": False,
        "last_mask": None,
        "last_depth_m": None,
        "K": K,
        "mesh_path": mesh_path,
        "robot_mask": None,
    }
    pose_records: list[dict[str, object]] = []
    executed_poses: list[np.ndarray] = []
    # tcp_history: list[np.ndarray] = []
    pose_records: list[dict[str, object]] = []
    headpose_pred_records: list[np.ndarray] = []
    T_base_cam_runtime: list[np.ndarray] = []

    frame_idx = 0
    try:
        while True:

            stereo = cam.read_stereo()
            if stereo is None:
                continue
            frame, frame_right = stereo

            # Prepare an RGB copy if needed downstream (already resized)
            image_bgr = frame
            image_rgb = image_bgr[..., ::-1].copy()

            last_frame_full = image_rgb
            last_size = (frame.shape[1], frame.shape[0])
            disp = frame

            # NOTE: robot mask tracking update is moved below to match inference cadence (do_update).

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
                    depth_enabled = True
                    print("[INFO] Saved Mask")

            if depth_enabled:
                do_update = (frame_idx % update_interval == 0)
                headpose_norm = None
                T_base_cam = T_base_cam0

                if do_update:
                    # Update robot mask at the same cadence as inference/update.
                    if (not robot_prompt_active) and (robot_tracker is not None):
                        tracked_mask = update_robot_mask_tracker(robot_tracker, image_rgb)
                        if tracked_mask is not None:
                            robot_mask = tracked_mask
                            infer_ctx["robot_mask"] = robot_mask
                            out_path = robot_mask_dir / f"{frame_idx:06d}.png"
                            ok = _safe_save_mask_png(out_path, robot_mask, win=win)
                            if not ok:
                                print(f"[ERROR] Failed to save robot mask: {out_path}")

                    if args.enable_headpose_head:
                        T_base_cam = headpose_reader.get_headpos(timeout_sec=0.2)
                        if T_base_cam is None:
                            raise RuntimeError("[eval] No headpose received yet from topic.")
                        headpose_raw = xyz_rot_transform(T_base_cam, from_rep="matrix", to_rep="rotation_6d").astype(np.float32)
                        headpose_norm = headpose_raw.copy()
                        headpose_norm[:3] = (headpose_norm[:3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
                    if T_base_cam is not None:
                        T_base_cam_runtime.append(T_base_cam.astype(np.float32))

                if do_update:
                    # Record the exact stereo pair used for inference/update.
                    frame_name = f"{frame_idx:06d}.png"
                    cv2.imwrite(str(infer_left_dir / frame_name), frame)
                    cv2.imwrite(str(infer_right_dir / frame_name), frame_right)
                    state = {
                        "frame": frame,
                        "frame_right": frame_right,
                        "T_base_cam": T_base_cam,
                        "headpose_norm": headpose_norm,
                        "mask": mask,
                    }
                    cloud_out_path = cloud_dir / f"{frame_idx:06d}.npz"
                    last_overlay, pred_state = _run_depth_traj(
                        state,
                        infer_ctx,
                        depth_est,
                        traj_pred,
                        cloud_out_path=cloud_out_path,
                    )
                    if pred_state is not None:
                        pose_cam_ob = pred_state["pose_cam_ob"]
                        traj_denorm = pred_state["traj_denorm"]
                        headpose_pred = pred_state["headpose_pred"]
                        T_base_cam_used = pred_state["T_base_cam"]
                        # print(f"[INFO] pred_state:{pred_state}")
                        if args.enable_headpose_head and headpose_pred is not None:
                            headpose_base_seq = headpose_pred.astype(np.float32)
                            headpose_pred_records.append(headpose_base_seq.copy())
                            T_i2rt_tcp = exec_ctx.i2rt_kin.fk(exec_ctx.i2rt_current_q[:exec_ctx.i2rt_arm_dofs])
                            headpose_rel_seq = headpose_base_seq_to_rel(
                                headpose_base_seq,
                                T_base_cam_used,
                            ) # relative to the frame that starts inference
                            # print(f"[DEBUG] headpose_rel_seq: {np.round(headpose_rel_seq[0]*100,3)}")
                            headpose_i2rt_rel = headpose_base_to_i2rt_rel(
                                headpose_rel_seq,
                                T_base_cam_used,
                                T_i2rt_tcp,
                            )
                            # print(f"[DEBUG] headpose_i2rt_rel: {np.round(headpose_i2rt_rel[0]*100,3)}")
                            pred_tcp_i2rt_rel = headpose_to_tcp(headpose_i2rt_rel)
                            # print(f"[DEBUG] pred_tcp_i2rt_rel: {np.round(pred_tcp_i2rt_rel[0]*100,3)}")
                            end_idx = min(pred_tcp_i2rt_rel.shape[0], STEPS_TO_EXECUTE)
                            exec_ctx.execute_pred_tcp_rel(pred_tcp_i2rt_rel[0:end_idx])
                            time.sleep(1.0)  # wait for motion to finish

                        pose_robot_ob, pose_seq_robot, step_poses = exec_ctx.execute_robot_traj(
                            traj_denorm,
                            pose_cam_ob.astype(np.float32),
                            T_base_cam_used.astype(np.float32),
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

                frame_idx += 1 # increment frame index only when depth is enabled

            # Refresh display from possibly overlaid frame
            disp_src = last_overlay if last_overlay is not None else frame
            # print(f'[INFO] Frame updated')
            disp = cv2.resize(disp_src, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            cv2.imshow(win, disp)

            key = cv2.waitKey(1) & 0xFF
            if robot_prompt_active and key in (13, 10):  # Enter
                if robot_clicks:
                    try:
                        robot_mask = click_mask(image_rgb, robot_clicks, labels=[1] * len(robot_clicks), multimask=True)
                        print("[INFO] Get Robot Mask from clicks")
                        robot_tracker = init_robot_mask_tracker(image_rgb, robot_mask)
                        print("[INFO] Init Robot Tracker")
                        infer_ctx["robot_mask"] = robot_mask
                        out_path = robot_mask_dir / f"{robot_mask_frame_idx:06d}.png"
                        ok = _safe_save_mask_png(out_path, robot_mask, win=win)
                        if not ok:
                            print(f"[ERROR] Failed to save robot mask: {out_path}")
                        robot_mask_frame_idx += 1
                        print("[INFO] Robot mask initialized.")
                    except Exception as e:
                        print("[ERROR] Robot mask exception:", e)
                        # robot_mask = None
                        raise e
                else:
                    print("[INFO] No robot clicks provided; skipping robot mask.")
                robot_prompt_active = False
            if robot_prompt_active and key == 27:
                print("[INFO] Robot mask skipped.")
                robot_prompt_active = False
                key = 0
            if interrupted["flag"] or key == ord('q') or key == 27:
                break

            if depth_enabled:
                # Write current display frame to video
                if writer is not None and writer.isOpened():
                    writer.write(disp)
                gripper_width = float(exec_ctx.flexiv_robot.get_gripper_state())
                open_width = getattr(exec_ctx.flexiv_robot, "max_width", GRIPPER_OPEN_WIDTH_DEFAULT)
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
        if robot_tracker is not None:
            cleanup_robot_mask_tracker(robot_tracker)

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

        if headpose_pred_records:
            headpose_pred_path = out_dir / "headpose_pred.npy"
            np.save(headpose_pred_path, np.stack(headpose_pred_records, axis=0))
            print(f"[INFO] Saved headpose predictions to: {headpose_pred_path}")

        # if tcp_history:
        #     tcp_path = out_dir / "robot_tcp_history.npy"
        #     np.save(tcp_path, np.stack(tcp_history, axis=0))
        #     print(f"[INFO] Saved robot TCP history to: {tcp_path}")

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
