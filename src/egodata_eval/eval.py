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
    _normalize_obj_pose,
    init_robot_mask_tracker,
    update_robot_mask_tracker,
    cleanup_robot_mask_tracker,
    headpose_i2rt_to_base_abs,
    _load_calib_mat_safe,
    add_relative,
)  # type: ignore
from egodata_eval.eval_hardware import EvalHardware

from egodata_eval.get_depth import DepthEstimator  # type: ignore
from egodata_eval.get_pose import PoseEstimatorFP
from egodata_eval.get_head import HeadPoseReader
from egodata_eval.traj_predictor import TrajectoryPredictor

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


def _mul_pose_seq_fixed(pose_seq: np.ndarray, T_fixed: np.ndarray, side: str = "right") -> np.ndarray:
    """Multiply an Nx4x4 pose sequence with one fixed 4x4 transform."""
    if pose_seq.ndim != 3 or pose_seq.shape[1:] != (4, 4):
        raise ValueError(f"pose_seq must be Nx4x4, got {pose_seq.shape}")
    if T_fixed.shape != (4, 4):
        raise ValueError(f"T_fixed must be 4x4, got {T_fixed.shape}")

    pose_seq = pose_seq.astype(np.float32)
    T_fixed = T_fixed.astype(np.float32)
    if side == "right":
        return np.einsum("nij,jk->nik", pose_seq, T_fixed).astype(np.float32)
    if side == "left":
        return np.einsum("ij,njk->nik", T_fixed, pose_seq).astype(np.float32)
    raise ValueError(f"Unsupported side={side}, expected 'left' or 'right'")


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
    if state["mask"] is not None:
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

        if ctx["robot_mask"] is not None:
            depth_m = depth_m.copy()
            depth_m[ctx["robot_mask"] > 0] = 0.0
            ctx["last_depth_m"] = depth_m

        if traj_pred is not None and ctx["pose_est"].pose_cam_ob is not None:
            T_base_cam = state["T_base_cam"]
            headpose_norm = state["headpose_norm"]
            frame_overlay, frame_cloud = traj_pred.predict_and_overlay(
                frame,
                depth_m,
                ctx["K"],
                ctx["pose_est"].pose_cam_ob.astype(np.float32),
                T_base_cam=T_base_cam,
                headpose_cond=headpose_norm,
            ) # overlay traj
            if cloud_out_path is not None:
                np.savez_compressed(
                    str(cloud_out_path),
                    cloud=np.asarray(frame_cloud, dtype=np.float32),
                    frame_idx=np.array([int(state["frame_idx"])], dtype=np.int32),
                )
            frame_overlay = ctx["pose_est"].draw_overlay(frame_overlay, ctx["K"]) # overlay pose
            pred_state = {
                "pose_cam_ob": ctx["pose_est"].pose_cam_ob.astype(np.float32),
                "traj_denorm": traj_pred.last_traj_pred.astype(np.float32), # abs both in delta/abs option
                "headpose_pred": None
                if traj_pred.last_headpose_pred is None
                else traj_pred.last_headpose_pred.astype(np.float32),
                "T_base_cam": T_base_cam.astype(np.float32),
                "pose_mode": traj_pred.obj_pose_mode,
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
    ap.add_argument("--obj-pose-mode", type=str, choices=["abs", "delta"], default="delta", help="Model output pose mode: abs or delta.")
    ap.add_argument('--add_curr_cond', action = 'store_true', help = 'add curr obj pose as extra cond for diffusion head')
    ap.add_argument("--glass-zed", type=str, default=DEFAULT_GLASSES_ZED_TXT, help="Path to T_tcp_zed (4x4 SE3).")
    ap.add_argument(
        "--calib-init-pose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After ball calibration, whether to compute T_i2rt_base_ball_base and move I2RT to task init pose.",
    )
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
    T_cam0_base = np.linalg.inv(T_base_cam0).astype(np.float32)
    T_tcp_cam = _load_calib_mat_safe(Path(DEFAULT_I2RT_ZED_TXT))
    if args.calib_init_pose:
        T_base_cam1 = calib_init_pose[args.task]
        if T_base_cam1 is None:
            raise RuntimeError(f"[eval] Missing calib_init_pose for task={args.task}.")
        exec_ctx.i2rt_current_q = exec_ctx.i2rt_robot.current_joint_pos()
        curr_headpose = exec_ctx.i2rt_kin.fk(exec_ctx.i2rt_current_q[:exec_ctx.i2rt_arm_dofs]).astype(np.float32)
        # Calib-time transform chain:
        # i2rt_base -> tcp (FK), tcp -> cam (fixed extrinsic), cam -> ball_base (from 3-ball calibration).
        
        if T_tcp_cam is None:
            raise RuntimeError(f"Failed to load fixed tcp->cam transform from {DEFAULT_I2RT_ZED_TXT}")
        T_i2rt_base_ball_base = (
            curr_headpose.astype(np.float32)
            @ T_tcp_cam.astype(np.float32)
            @ T_cam0_base.astype(np.float32)
        ).astype(np.float32)
        T_ball_base_i2rt_base = np.linalg.inv(T_i2rt_base_ball_base).astype(np.float32)
        t_i2rt_ball_path = out_dir / "T_i2rt_base_ball_base.txt"
        t_ball_i2rt_path = out_dir / "T_ball_base_i2rt_base.txt"
        np.savetxt(t_i2rt_ball_path, T_i2rt_base_ball_base, fmt="%.8f")
        np.savetxt(t_ball_i2rt_path, T_ball_base_i2rt_base, fmt="%.8f")
        print(f"[INFO] Saved T_i2rt_base_ball_base to: {t_i2rt_ball_path}")
        print(f"[INFO] Saved T_ball_base_i2rt_base to: {t_ball_i2rt_path}")

        T_cam_tcp = np.linalg.inv(T_tcp_cam).astype(np.float32)
        T_i2rt_tcp = (
            T_i2rt_base_ball_base.astype(np.float32)
            @ T_base_cam1.astype(np.float32)
            @ T_cam_tcp.astype(np.float32)
        ).astype(np.float32)
        print(f"[INFO] Computed T_i2rt_tcp for init pose:\n{T_i2rt_tcp}")
        success, q_sol = exec_ctx.i2rt_kin.ik(T_i2rt_tcp, "grasp_site", verbose=False)
        if not success:
            exec_ctx.i2rt_robot.close()
            exec_ctx.i2rt_robot.destroy_node()
            exec_ctx.i2rt_server_proc.terminate()
            exec_ctx.i2rt_server_proc.join(timeout=2.0)
            raise RuntimeError(f"[eval] I2RT IK failed for init pose (task={args.task}).")
        q_target = exec_ctx.i2rt_robot.current_joint_pos().astype(np.float32)
        q_target[:exec_ctx.i2rt_arm_dofs] = q_sol[:exec_ctx.i2rt_arm_dofs].astype(np.float32)
        exec_ctx.i2rt_robot.send_joint_pos_rad(
            q_target,
            duration=I2RT_INIT_DURATION,
            steps=I2RT_INIT_STEPS,
        )
        exec_ctx.i2rt_current_q = q_target.copy()
        print(f"[INFO] Reached task init pose via IK for task={args.task}.")
    else:
        print("[INFO] Skip post-calib init-pose step (--no-calib-init-pose).")
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

    traj_pred = TrajectoryPredictor(
        ckpt_path=Path(args.ckpt),
        num_action=args.num_action,
        obj_pose_mode=args.obj_pose_mode,
        enable_headpose_head=args.enable_headpose_head,
        add_curr_cond=args.add_curr_cond,
    ) # current traj pred is under base frame
    print(f"[INFO] Loaded RISE trajectory predictor from {args.ckpt}")

    if args.enable_headpose_head:
        if not rclpy.ok():
            rclpy.init(args=None)
        headpose_reader = HeadPoseReader(headpose_topic, args.glass_zed, T_base_cam0 if not args.calib_init_pose else T_base_cam1)
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
    headpose_abs_seq_records: list[np.ndarray] = []
    T_base_cam_runtime: list[np.ndarray] = []
    tcp_obj_ready = False

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
                T_base_cam = T_base_cam0 if not args.calib_init_pose else T_base_cam1

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
                        headpose_norm = _normalize_obj_pose(
                            headpose_raw,
                            obj_pose_mode="abs",
                        )
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
                        "frame_idx": frame_idx,
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
                        # print(f"headpose_pred:{headpose_pred[:5]}")
                        T_base_cam_used = pred_state["T_base_cam"]
                        pose_mode = pred_state["pose_mode"]
                        # print(f"[INFO] pred_state:{pred_state}")
                        pred_tcp_after_trans = np.zeros((0, 4, 4), dtype=np.float32)
                        headpose_abs_seq_rec = np.zeros((0, 4, 4), dtype=np.float32)
                        tcp_i2rt_abs_rec = np.zeros((0, 4, 4), dtype=np.float32)
                        headpose_i2rt_abs_rec = np.zeros((0, 4, 4), dtype=np.float32)
                        if args.enable_headpose_head and headpose_pred is not None:
                            T_i2rt_tcp = exec_ctx.i2rt_kin.fk(exec_ctx.i2rt_current_q[:exec_ctx.i2rt_arm_dofs])
                            # if pose_mode == "abs": 
                            #     headpose_base_seq = headpose_pred.astype(np.float32) # headpose base abs
                            #     headpose_pred_records.append(headpose_base_seq.copy())
                            #     headpose_rel_seq = headpose_base_seq_to_rel(
                            #         headpose_base_seq,
                            #         T_base_cam_used,
                            #     ) # relative to the frame that starts inference
                            # elif pose_mode == "delta":
                            #     headpose_rel_seq = headpose_pred.astype(np.float32) # headpose base relative
                            #     # record headpose (under base frame)
                            #     headpose_base_seq = headpose_i2rt_to_base_abs(
                            #         headpose_rel_seq,
                            #         T_base_cam_used,
                            #         T_i2rt_tcp,
                            #     ) 
                            #     headpose_pred_records.append(headpose_base_seq.copy())

                            # only use delta to control headpose robot
                            headpose_rel_seq = headpose_pred.astype(np.float32) # headpose base relative
                            # record headpose (under base frame)
                            # TODO: need fix
                            # headpose_base_seq = headpose_i2rt_to_base_abs(
                            #     headpose_rel_seq,
                            #     T_base_cam_used,
                            #     T_i2rt_tcp,
                            # ) 
                            
                            # print(f"T_base_cam_used:\n{T_base_cam_used}")
                            headpose_rel_pose_seq = _build_pose_mats(headpose_rel_seq[:, :3],headpose_rel_seq[:, 3:3 + 6]).astype(np.float32)
                            # print(f"headpose_rel_pose_seq: {headpose_rel_pose_seq[:5,:3,3]}")
                            headpose_abs_seq = add_relative(headpose_rel_pose_seq,T_base_cam_used.astype(np.float32)) # abs under base
                            headpose_abs_seq_rec = headpose_abs_seq.astype(np.float32).copy()
                            headpose_abs_seq_records.append(headpose_abs_seq_rec.copy())  # abs under base
                            # print(f"headpose_abs_seq: {headpose_abs_seq[:5,:3,3]}")
                            T_i2rt_base = T_i2rt_tcp @ T_tcp_cam @ np.linalg.inv(T_base_cam_used)
                            headpose_i2rt_abs = _mul_pose_seq_fixed(
                                headpose_abs_seq,
                                T_i2rt_base,
                                side="left",
                            ) # abs under i2rt
                            headpose_i2rt_abs_rec = headpose_i2rt_abs.astype(np.float32).copy()
                            tcp_i2rt_abs = _mul_pose_seq_fixed(
                                headpose_i2rt_abs,
                                np.linalg.inv(T_tcp_cam),
                                side="right",
                            ) # headpose traj -> tcp traj
                            tcp_i2rt_abs_rec = tcp_i2rt_abs.astype(np.float32).copy()
                            print(f"tcp_i2rt_abs: {tcp_i2rt_abs[:5,:3,3]}")
                            
                            end_idx = min(tcp_i2rt_abs.shape[0], STEPS_HEAD_TO_EXECUTE)
                            pred_tcp_after_trans_i2rt = exec_ctx.execute_pred_tcp_abs(tcp_i2rt_abs[0:end_idx])
                            T_base_i2rt = np.linalg.inv(T_i2rt_base).astype(np.float32)
                            pred_tcp_after_trans = np.einsum(
                                "ij,njk->nik",
                                T_base_i2rt,
                                pred_tcp_after_trans_i2rt.astype(np.float32),
                            ).astype(np.float32)


                            # pred_tcp_i2rt_rel = headpose_to_tcp(headpose_i2rt_rel)
                            # print(f"pred_tcp_i2rt_rel: {pred_tcp_i2rt_rel[:3]}")
                            '''
                            end_idx = min(headpose_rel_seq.shape[0], STEPS_HEAD_TO_EXECUTE)
                            
                            headpose_i2rt_rel = headpose_base_to_i2rt_rel(
                                headpose_rel_seq,
                                T_base_cam_used,
                                T_i2rt_tcp,
                            )
                            pred_tcp_after_trans = exec_ctx.execute_pred_tcp_rel(headpose_i2rt_rel[0:end_idx])
                            '''
                            time.sleep(1.0)  # wait for motion to finish

                        if not tcp_obj_ready:
                            pose_base_ob_calib = (T_base_cam_used.astype(np.float32) @ pose_cam_ob.astype(np.float32)).astype(np.float32)
                            T_robot_obj = (exec_ctx.T_robot_base.astype(np.float32) @ pose_base_ob_calib).astype(np.float32)
                            curr_tcp_pose7 = exec_ctx.flexiv_robot.get_tcp_pose().astype(np.float32)
                            T_robot_tcp = np.eye(4, dtype=np.float32)
                            T_robot_tcp[:3, 3] = curr_tcp_pose7[:3].astype(np.float32)
                            T_robot_tcp[:3, :3] = rotation_transform(
                                curr_tcp_pose7[3:7][None, :],
                                "quaternion",
                                "matrix",
                            ).squeeze(0).astype(np.float32)
                            T_tcp_obj_calib = (np.linalg.inv(T_robot_tcp) @ T_robot_obj).astype(np.float32)
                            T_tcp_obj_const = TASK_TCP_TO_OBJECT_SE3[args.task].astype(np.float32)
                            T_tcp_obj_new = T_tcp_obj_calib.copy()
                            T_tcp_obj_new[:3, 3] = T_tcp_obj_const[:3, 3].astype(np.float32)
                            TASK_TCP_TO_OBJECT_SE3[args.task] = T_tcp_obj_new
                            tcp_obj_ready = True
                            # print(f"[INFO] T_tcp_obj calibrated for task={args.task} (rot=calib, trans=const):\n{T_tcp_obj_new}")

                        if args.add_curr_cond:
                            pose_base_ob = T_base_cam_used.astype(np.float32) @ pose_cam_ob.astype(np.float32)
                        else:
                            pose_base_ob = _build_pose_mats(
                                traj_denorm[:1, :3],
                                traj_denorm[:1, 3:3+6],
                            )[0].astype(np.float32)
                        pred_obj_seq_base = _build_pose_mats(
                            traj_denorm[:, :3],
                            traj_denorm[:, 3:3+6],
                        ).astype(np.float32)
                        pred_obj_seq_robot = np.einsum(
                            "ij,njk->nik",
                            exec_ctx.T_robot_base.astype(np.float32),
                            pred_obj_seq_base.astype(np.float32),
                        ).astype(np.float32)
                        pose_robot_ob, tcp_seq_robot, step_poses = exec_ctx.execute_robot_traj(
                            traj_denorm,
                            pose_base_ob,
                        )
                        # Stop policy if predicted gripper signal exceeds threshold within steps_to_execute.
                        grip_seq = traj_denorm[:, 9].astype(np.float32)
                        steps_to_execute = int(exec_ctx.steps_to_execute)
                        grip_window = grip_seq[1:1 + steps_to_execute]
                        open_thresh = GRIP_OPEN_THRESH[args.task]
                        if grip_window.size > 0 and np.any(grip_window > open_thresh):
                            print("[INFO] Predicted gripper open in steps_to_execute; stopping policy.")
                            break
                        pose_records.append(
                            {
                                "timestamp": float(time.time()),
                                "frame_idx": int(frame_idx),
                                "object_pose_robot": pose_robot_ob,
                                "pred_obj_seq_robot": pred_obj_seq_robot,
                                # Keep pred_seq_robot for visualizer compatibility; set it to object trajectory.
                                "pred_seq_robot": pred_obj_seq_robot,
                                # This sequence is TCP poses in robot frame.
                                "pred_tcp_seq_robot": tcp_seq_robot,
                                # Transformed TCP sequence returned by execute_pred_tcp_rel (Nx4x4).
                                "pred_tcp_after_trans": pred_tcp_after_trans,
                                # TCP absolute sequence in i2rt frame (Nx4x4).
                                "tcp_i2rt_abs": tcp_i2rt_abs_rec,
                                # Headpose absolute sequence under base frame (Nx4x4).
                                "headpose_abs_seq": headpose_abs_seq_rec,
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

        if headpose_abs_seq_records:
            headpose_abs_seq_path = out_dir / "headpose_abs_seq.npy"
            np.save(headpose_abs_seq_path, np.stack(headpose_abs_seq_records, axis=0))
            print(f"[INFO] Saved headpose abs sequences to: {headpose_abs_seq_path}")

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
