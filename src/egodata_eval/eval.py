import time
from datetime import datetime
from pathlib import Path
import sys
import os
from typing import Optional

import cv2
import signal
import numpy as np
import torch

from pathlib import Path
import sys
here = Path(__file__).resolve()
project_root = here.parents[2] # 指向仓库根目录
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure '<project>/src' is importable, then import click_mask as package

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from FoundationStereo.sam2_root.notebooks.get_mask import click_mask  # type: ignore
from egodata_eval.eval_utils import save_mask,  \
_denormalize_obj_traj, _build_pose_mats, _project_points_with_gradient, \
_import_zed_class   # type: ignore

from egodata_eval.get_depth import DepthEstimator, colorize_depth  # type: ignore
from egodata_eval.get_pose import PoseEstimatorFP  # type: ignore
from glasses_hardware.hardware.my_device.robot import FlexivRobot, FlexivGripper  # type: ignore
from glasses_hardware.hardware.my_device.i2rt_robo import I2RT  # type: ignore
from egodata_eval.eval_utils import _build_pose_mats  # type: ignore
from egodata_eval.eval_utils import _import_zed_class  # already imported below; keep for clarity

# ========== MBA Trajectory Prediction (RISE) ==========

import MinkowskiEngine as ME  # type: ignore
from MBA.policy import RISE  # type: ignore
from MBA.utils.constants import TRANS_MIN, TRANS_MAX, IMG_MEAN, IMG_STD  # type: ignore
from MBA.utils.transformation import rotation_transform, mat_to_xyz_rot  # type: ignore
from scripts_calib_balls.calculate_ball_centers import (
    calculate_ball_centroid,
    DEFAULT_MAX_RADIUS_STD_RATIO,
)
from scripts_calib_balls.compute_base_from_ball_centers import compute_base_from_three_points

class TrajectoryPredictor:
    def __init__(
        self,
        ckpt_path: Path,
        num_action: int = 20,
        obj_pose_mode: str = "delta",
        voxel_size: float = 0.005,
        enable_headpose_head: bool = False,
        headpose_dim: int = 9,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_action = num_action
        self.obj_pose_mode = obj_pose_mode
        self.voxel_size = voxel_size
        self._cached_points_cam: Optional[np.ndarray] = None
        self.last_traj_denorm: Optional[np.ndarray] = None
        self.model = RISE(num_action=num_action,
                          input_dim=6,
                          obs_feature_dim=512,
                          action_dim=10,
                          hidden_dim=512,
                          enable_mba=True,
                          obj_dim=10,
                          obj_pose_mode=obj_pose_mode,
                          enable_headpose_head=enable_headpose_head,
                          headpose_dim=headpose_dim).to(self.device).eval()
        if ckpt_path is None:
            raise ValueError("ckpt_path is required; please pass --ckpt to eval.py")
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Trajectory ckpt not found: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location=self.device)
        self.model.load_state_dict(state, strict=False)

    def _make_sparse_input(self, rgb_bgr: np.ndarray, depth_m: np.ndarray, K: np.ndarray, T_base_cam: Optional[np.ndarray] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Backproject depth to xyz and optionally convert to base (ball) frame."""
        h, w = depth_m.shape
        print(f"[Traj Predictor INFO] depth_m.shape(h,w):{depth_m.shape}")
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        # Subsample grid for speed
        step = max(1, int(max(h, w) / 480)) # for case that h=376, w=672, step = 1
        print(f"[INFO] step: {step}")
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        zs = depth_m[ys, xs]
        valid = zs > 1e-6
        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        zs = zs[valid].astype(np.float32)
        xs3 = (xs - cx) * zs / fx
        ys3 = (ys - cy) * zs / fy
        xyz_cam = np.stack([xs3, ys3, zs], axis=-1)
        if T_base_cam is not None:
            R = T_base_cam[:3, :3].astype(np.float32)
            t = T_base_cam[:3, 3].astype(np.float32)
            xyz = (R @ xyz_cam.T).T + t
        else:
            xyz = xyz_cam
        # Colors to [0,1]
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        colors = rgb[ys.astype(int), xs.astype(int)].astype(np.float32) / 255.0
        colors = (colors - IMG_MEAN) / IMG_STD
        cloud = np.concatenate([xyz, colors], axis=-1).astype(np.float32)

        # Remove any rows with non-finite values to avoid NaNs in voxelization
        finite_mask = np.isfinite(cloud).all(axis=1)
        cloud = cloud[finite_mask]

        coords = np.ascontiguousarray((cloud[:, :3] / self.voxel_size).astype(np.int32))
        feats = cloud.astype(np.float32)
        # Collate into ME batched format
        coords_me, feats_me = ME.utils.sparse_collate([coords], [feats])

        # ME may already return torch tensors depending on version; handle both
        if isinstance(feats_me, np.ndarray):
            feats_t = torch.from_numpy(feats_me)
        else:
            feats_t = feats_me
        if isinstance(coords_me, np.ndarray):
            coords_t = torch.from_numpy(coords_me)
        else:
            coords_t = coords_me

        return feats_t.to(self.device), coords_t.to(self.device)

    def _current_obj_vec(self, pose_cam_ob: np.ndarray) -> np.ndarray:
        xyz6d = mat_to_xyz_rot(pose_cam_ob, rotation_rep="rotation_6d").astype(np.float32)
        term = np.array([0.0], dtype=np.float32)
        cur = np.concatenate([xyz6d, term], axis=0)
        # normalize like dataset
        norm = cur.copy()
        norm[:3] = (norm[:3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        return norm

    def _absolute_to_delta_np(self, abs_traj_10: np.ndarray, base_pose_cam_ob: np.ndarray) -> np.ndarray:
        """Convert absolute traj (T,10) [xyz(m), rot6d, grip] to delta wrt base_pose.

        Returns: (T,10) with [dxyz, drot6d, grip]
        """
        if abs_traj_10 is None or abs_traj_10.size == 0:
            return abs_traj_10
        base_xyz6d = mat_to_xyz_rot(base_pose_cam_ob, rotation_rep="rotation_6d").astype(np.float32)
        base_xyz = base_xyz6d[:3]
        base_r6 = base_xyz6d[3:9]
        # Translation delta
        delta_xyz = abs_traj_10[:, :3] - base_xyz[None, :]
        # Rotation delta: R_delta = R_abs @ R_base^T
        R_abs = rotation_transform(abs_traj_10[:, 3:9], "rotation_6d", "matrix")
        R_base = rotation_transform(base_r6[None, :], "rotation_6d", "matrix").squeeze(0)
        R_delta = R_abs @ R_base.T
        delta_r6 = rotation_transform(R_delta, "matrix", "rotation_6d")
        # Gripper passthrough if present
        if abs_traj_10.shape[1] > 9:
            grip = abs_traj_10[:, 9:10]
            delta_full = np.concatenate([delta_xyz, delta_r6, grip], axis=1)
        else:
            delta_full = np.concatenate([delta_xyz, delta_r6], axis=1)
        return delta_full.astype(np.float32)

    def predict_and_overlay(self, image_bgr: np.ndarray, depth_m: np.ndarray, K: np.ndarray, pose_cam_ob: np.ndarray, T_base_cam: Optional[np.ndarray] = None) -> np.ndarray:
        # Convert object pose to base frame if provided
        if T_base_cam is not None:
            pose_base_ob = T_base_cam @ pose_cam_ob
        else:
            pose_base_ob = pose_cam_ob

        feats, coords = self._make_sparse_input(image_bgr, depth_m, K, T_base_cam=T_base_cam)
        st = ME.SparseTensor(feats, coords)
        cur_obj = self._current_obj_vec(pose_base_ob)
        with torch.no_grad():
            outputs = self.model(st, actions_obj = None ,batch_size=1, current_obj=torch.from_numpy(cur_obj[None, :]).to(self.device))
        if "obj_pred" not in outputs:
            self.last_traj_denorm = None
            return image_bgr
        obj_traj_norm = outputs["obj_pred"].squeeze(0).detach().cpu().numpy()
        # In delta mode, model already returns absolute poses relative to current pose; just denormalize translation.
        obj_traj_ref = _denormalize_obj_traj(obj_traj_norm)
        self.last_traj_denorm = obj_traj_ref
        # Deltas are not used in current execution path; keep only absolute trajectory

        # Debug prints to compare current FP pose and first predicted absolute pose
        # fp_xyz6d = mat_to_xyz_rot(pose_cam_ob, rotation_rep="rotation_6d").astype(np.float32)
        fp_xyz6d = mat_to_xyz_rot(pose_base_ob, rotation_rep="rotation_6d").astype(np.float32)
        traj_first_xyz6d = obj_traj_ref[0, :9].astype(np.float32)
        np.set_printoptions(precision=4, suppress=True)
        print("[DEBUG] FP xyz6d:", fp_xyz6d)
        print("[DEBUG] Traj first xyz6d:", traj_first_xyz6d)

        pose_mats_ref = _build_pose_mats(obj_traj_ref[:, :3], obj_traj_ref[:, 3:3+6])
        predicted_points = pose_mats_ref[:, :3, 3]  # (N,3)
        if T_base_cam is not None:
            T_cam_base = np.linalg.inv(T_base_cam).astype(np.float32)
            R = T_cam_base[:3, :3].astype(np.float32)
            t = T_cam_base[:3, 3].astype(np.float32)
            points_cam = (R @ predicted_points.T).T + t
        else:
            points_cam = predicted_points
        self._cached_points_cam = points_cam.copy()
        overlay = _project_points_with_gradient(image_bgr, K, points_cam,
                                                color_start=(255, 0, 0), color_end=(0, 255, 255), radius=4, thickness=-1)
        return overlay

    def overlay_cached(self, image_bgr: np.ndarray, K: np.ndarray) -> np.ndarray:
        if self._cached_points_cam is None:
            return image_bgr
        return _project_points_with_gradient(
            image_bgr, K, self._cached_points_cam,
            color_start=(255, 0, 0), color_end=(0, 255, 255), radius=4, thickness=-1,
        )


def _load_calib_mat_safe(path: Path) -> Optional[np.ndarray]:
    try:
        arr = np.load(str(path)).astype(np.float32)
        if arr.shape == (4, 4):
            return arr
        if arr.shape == (3, 4):
            arr = np.vstack([arr, np.array([0, 0, 0, 1], dtype=np.float32)])
            return arr
    except Exception:
        return None
    return None


I2RT_TARGET_DEG = [-17, 25, 61, -42, 0, -2,0]
I2RT_TARGET_RAD = np.deg2rad(I2RT_TARGET_DEG).astype(np.float32)


def move_i2rt_to_init_angles(robot: Optional["I2RT"], target_rad: np.ndarray = I2RT_TARGET_RAD, duration: float = 2.0, steps: int = 80) -> None:
    """Move I2RT arm to the evaluation target joint configuration."""
    if robot is None:
        print("[WARN] I2RT arm not initialized; cannot move to init pose.")
        return
    try:
        robot.send_joint_pos_rad(target_rad, duration=duration, steps=steps)
        print(f"[INFO] Moved I2RT joints to deg {I2RT_TARGET_DEG}")
    except Exception as exc:
        print(f"[WARN] I2RT init move failed: {exc}")


def calibrate_from_three_balls(
    cam_handle,
    depth_est: DepthEstimator,
    move_robot_fn=None,
    centroid_log_dir: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Perform ball-based calibration to compute T_base_cam."""
    if move_robot_fn is not None:
        move_robot_fn()
    print("[INFO] Click three ball centers (id1, id2, id3) on the first frame to calibrate base.")
    first = cam_handle.read_stereo()
    if first is None:
        print("[WARN] Could not grab frame for ball calibration.")
        return None
    frame, frame_right = first
    K_rs = depth_est.K.astype(np.float32)
    pts: list[tuple[float, float]] = []

    click_state = {"done": False}

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 3:
            pts.append((float(x), float(y)))
            print(f"[INFO] Clicked point {len(pts)}: ({x}, {y})")
            if len(pts) == 3:
                click_state["done"] = True

    win_calib = "Ball Calibration"
    cv2.namedWindow(win_calib, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_calib, _on_mouse)
    disp = frame.copy()
    while True:
        cv2.imshow(win_calib, disp)
        k = cv2.waitKey(10) & 0xFF
        if click_state["done"] or k in (27, ord('q')):
            break
    cv2.destroyWindow(win_calib)

    # Depth for clicked points + mask-based centroid refinement
    depth_m = depth_est.depth(frame, frame_right)
    fx, fy = K_rs[0, 0], K_rs[1, 1]
    cx, cy = K_rs[0, 2], K_rs[1, 2]
    print(f"[INFO] intrinsics fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    frame_rgb = frame[..., ::-1].copy()

    def _show_mask(mask_img: np.ndarray, window: str) -> None:
        overlay = frame.copy()
        mask_bool = mask_img.astype(bool)
        overlay[mask_bool] = (
            0.4 * overlay[mask_bool].astype(np.float32) + 0.6 * np.array([0, 0, 255], dtype=np.float32)
        )
        overlay = overlay.astype(np.uint8)
        cv2.imshow(window, overlay)
        cv2.waitKey(10)

    cam_pts = []
    for idx, (u, v) in enumerate(pts, 1):
        u = int(round(u))
        v = int(round(v))
        mask = None
        try:
            mask = click_mask(frame_rgb, [(u, v)], labels=[1], multimask=True)
            if isinstance(mask, list):
                mask = mask[0]
        except Exception as exc:
            print(f"[WARN] SAM mask failed for point {idx}: {exc}")

        centroid = None
        if mask is not None:
            mask_arr = np.asarray(mask)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr.squeeze(axis=2)
            centroid = calculate_ball_centroid(
                depth_m=depth_m,
                mask=mask_arr.astype(bool),
                intrinsic=K_rs,
                max_radius_std_ratio=DEFAULT_MAX_RADIUS_STD_RATIO,
                frame_id=0,
                ball_id=idx,
            )
            # _show_mask(mask_arr.astype(np.uint8), f"Ball Mask {idx}")
        if centroid is not None:
            cam_pts.append(centroid)
            print(f"[INFO] Mask centroid for p{idx}: {centroid}")
            continue

    if len(cam_pts) != 3:
        print("[WARN] Failed to compute all three ball centroids; aborting calibration.")
        return None

    p1, p2, p3 = cam_pts

    # Persist individual ball centroids for debugging/reuse
    if centroid_log_dir is None:
        centroid_log_dir = Path(__file__).resolve().parents[2] / "glasses_hardware" / "calib"
    centroid_log_dir.mkdir(parents=True, exist_ok=True)
    centroid_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    centroid_log_path = centroid_log_dir / f"ball_centroids_{centroid_ts}.txt"
    with open(centroid_log_path, "w", encoding="utf-8") as fh:
        fh.write("ball_id x y z\n")
        for idx, pt in enumerate((p1, p2, p3), start=1):
            fh.write(f"ball_{idx} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
    print(f"[INFO] Saved per-ball centroids to: {centroid_log_path}")

    R_base_cam, t_base_cam = compute_base_from_three_points(p1, p2, p3)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_base_cam
    T[:3, 3] = t_base_cam
    print("[OK] Ball calibration produced T_base_cam:")
    print(T)
    return T


def run():
    import argparse
    ap = argparse.ArgumentParser(description="Online evaluation with manual ckpt path")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to RISE policy checkpoint (.ckpt)")
    ap.add_argument("--base-to-robot-npy", type=str, default='glasses_hardware/calib/T_robot_base.npy', help="Path to T_robot_base.npy (maps base->robot). Default: identity.")
    ap.add_argument("--num_action", type=int, default=20)
    ap.add_argument("--mesh-name", type=str, default="book", help="Name of mesh folder under data/ containing mesh.obj.")
    args = ap.parse_args()
    # Shared interval controlling how often heavy ops run
    update_interval = 10
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Prepare video output
    out_dir = Path(__file__).resolve().parent / "eval_output" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "stream.mp4"
    ZEDCamera = _import_zed_class()
    cam = ZEDCamera(resolution="WVGA", fps=30)
    # Initialize depth estimator and load model at start

    print("[INFO] Loading FoundationStereo depth model...")
    depth_est = DepthEstimator(scale=0.75) # no need to modify intrinsics;
    # One-time calibration to compute T_base_cam
    project_root = Path(__file__).resolve().parents[2]
    calib_dir = project_root / 'glasses_hardware' / 'calib'
    calib_dir.mkdir(parents=True, exist_ok=True)
    T_base_cam = None
    # Optional base->robot transform (default identity)
    T_robot_base = np.eye(4, dtype=np.float32)
    if args.base_to_robot_npy:
        loaded = _load_calib_mat_safe(Path(args.base_to_robot_npy))
        if loaded is not None:
            T_robot_base = loaded.astype(np.float32)
            print(f"[INFO] Loaded T_robot_base from {args.base_to_robot_npy}")
        else:
            print(f"[WARN] Failed to load T_robot_base from {args.base_to_robot_npy}; using identity.")

    # Initialize robot and gripper
    print("[INFO] Initializing Flexiv and gripper...") # First initialize Flexiv, or I2RT comm will go error
    robot = FlexivRobot(home=False)
    gripper = FlexivGripper(robot)

    print("[INFO] Initializing I2RT...")
    i2rt_robot = I2RT(channel="can0", zero_gravity_mode=True, home=False)
    time.sleep(3)

    T_base_cam = calibrate_from_three_balls(
        cam,
        depth_est,
        move_robot_fn=lambda: move_i2rt_to_init_angles(i2rt_robot),
        centroid_log_dir=out_dir,
    )
    if T_base_cam is not None:
        runtime_cam_path = out_dir / "T_base_cam_runtime.npy"
        np.save(runtime_cam_path, T_base_cam.astype(np.float32))
        print(f"[INFO] Saved runtime T_base_cam to: {runtime_cam_path}")

    cam_size = cam.size

    disp_w = int(cam_size[0])
    disp_h = int(cam_size[1])
    print(f"[INFO] Display resolution set from ZEDCamera: {disp_w}x{disp_h}")

    win = "ZED Stream (click to segment)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # Video writer (writes displayed frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (disp_w, disp_h))

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

    # Try initialize trajectory predictor (optional)

    traj_pred = TrajectoryPredictor(ckpt_path=Path(args.ckpt), num_action= args.num_action) # current traj pred is under base frame
    print(f"[INFO] Loaded RISE trajectory predictor from {args.ckpt}")
    pose_records: list[dict[str, object]] = []
    executed_poses: list[np.ndarray] = []
    tcp_history: list[np.ndarray] = []
    pose_records: list[dict[str, object]] = []

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
                    mesh_path = Path(__file__).resolve().parents[2] / "data" / args.mesh_name / "mesh.obj"
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
                            print("[INFO] Running trajectory prediction...")
                            frame = traj_pred.predict_and_overlay(
                                frame,
                                depth_m,
                                K_rs,
                                pose_est.pose_cam_ob.astype(np.float32),
                                T_base_cam=T_base_cam,
                            )
                            # Execute first N steps relative to current TCP using robot_replay logic
                            steps_to_execute = 3  # how many relative steps to send each update
                            # Gripper signal per step if available (10th channel)
                            grip_seq = traj_pred.last_traj_denorm[:, 9].astype(np.float32)
                            # Record current pose + predicted sequence in robot frame
                            # First convert current object pose from FP to robot frame
                            pose_cam_ob = pose_est.pose_cam_ob.astype(np.float32)
                            print(f"[INFO] OB In Cam: {pose_cam_ob}")

                            pose_base_ob = T_base_cam @ pose_cam_ob
                            pose_robot_ob = T_robot_base @ pose_base_ob # [4,4]
                            pose_seq_robot = None

                            pose_seq_base = _build_pose_mats(
                                traj_pred.last_traj_denorm[:, :3],
                                traj_pred.last_traj_denorm[:, 3:3+6],
                            )
                            pose_seq_robot = np.einsum(
                                'ij,njk->nik',
                                T_robot_base.astype(np.float32),
                                pose_seq_base.astype(np.float32),
                            ) # [N,4,4], SE3 in robot frame

                            pose_records.append(
                                {
                                    "timestamp": float(time.time()),
                                    "frame_idx": int(frame_idx),
                                    "object_pose_robot": pose_robot_ob.astype(np.float32),
                                    "pred_seq_robot": pose_seq_robot.astype(np.float32),
                                }
                            )

                            # Take the first `steps_to_execute` non-zero steps starting from index 1
                            steps_grip = None
                            if grip_seq is not None:
                                steps_grip = grip_seq[1:1+int(steps_to_execute)]
                            if pose_seq_robot.size > 0:
                                # convert pose to pts
                                robot_rel_pts = pose_seq_robot[1:1+int(steps_to_execute), :3, 3] - pose_robot_ob[:3, 3][None, :]
                                # Send absolute targets: start_xyz + p_rel_base, keep start quaternion
                                curr_pose7 = robot.get_tcp_pose().astype(np.float32)
                                start_xyz = curr_pose7[:3].astype(np.float32)
                                start_quat = curr_pose7[3:7].astype(np.float32)
                                open_width = getattr(gripper, 'max_width', 0.085)
                                open_thresh = 0.8
                                for i in range(robot_rel_pts.shape[0]):
                                    xyz = start_xyz + robot_rel_pts[i]
                                    pose7 = np.concatenate([xyz, start_quat], axis=0).astype(np.float32)
                                    # Gripper control if grip available
                                    if steps_grip is not None and i < len(steps_grip):
                                        grip_val = float(steps_grip[i])
                                        width_cmd = open_width if grip_val > open_thresh else 0.0
                                        print(f"[EVAL] step {i+1}/{robot_rel_pts.shape[0]} grip={grip_val:.3f} -> width={width_cmd:.3f}")
                                        gripper.move(width_cmd)

                                    print(f"[EVAL] send step {i+1}/{robot_rel_pts.shape[0]} pose7=", np.round(pose7, 6))
                                    # robot.send_tcp_pose(pose7)
                                    executed_poses.append(pose7.copy())

                                    tcp_history.append(robot.get_tcp_pose().astype(np.float32))

                                    time.sleep(0.05)
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
            
            del frame, frame_right
            torch.cuda.empty_cache()
    finally:
        # Release resources and save video

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

        cv2.destroyAllWindows()

        cam.close()

        if i2rt_robot is not None:
            i2rt_robot.close()



if __name__ == "__main__":
    run()
