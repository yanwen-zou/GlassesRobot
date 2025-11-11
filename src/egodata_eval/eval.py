import time
from datetime import datetime
from pathlib import Path
import sys
import os

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
from egodata_eval.eval_utils import _build_pose_mats  # type: ignore
from egodata_eval.eval_utils import _import_zed_class  # already imported below; keep for clarity

# For live ArUco detection to build camera->base mapping
from egodata_eval import piper_calib  # type: ignore

# ========== MBA Trajectory Prediction (RISE) ==========

import MinkowskiEngine as ME  # type: ignore
from MBA.policy import RISE  # type: ignore
from MBA.utils.constants import TRANS_MIN, TRANS_MAX, IMG_MEAN, IMG_STD  # type: ignore
from MBA.utils.transformation import rotation_transform, mat_to_xyz_rot  # type: ignore

class TrajectoryPredictor:
    def __init__(self, ckpt_path: Path, num_action: int = 20, obj_pose_mode: str = "delta", voxel_size: float = 0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_action = num_action
        self.obj_pose_mode = obj_pose_mode
        self.voxel_size = voxel_size
        self._cached_points_cam: np.ndarray | None = None
        self.last_traj_denorm: np.ndarray | None = None
        self.model = RISE(num_action=num_action,
                          input_dim=6,
                          obs_feature_dim=512,
                          action_dim=10,
                          hidden_dim=512,
                          enable_mba=True,
                          obj_dim=10,
                          obj_pose_mode=obj_pose_mode).to(self.device).eval()
        if ckpt_path is None:
            raise ValueError("ckpt_path is required; please pass --ckpt to eval.py")
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Trajectory ckpt not found: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location=self.device)
        self.model.load_state_dict(state, strict=False)

    def _make_sparse_input(self, rgb_bgr: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        # Backproject depth to camera xyz
        h, w = depth_m.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        # Subsample grid for speed
        step = max(1, int(max(h, w) / 480))
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        zs = depth_m[ys, xs]
        valid = zs > 1e-6
        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        zs = zs[valid].astype(np.float32)
        xs3 = (xs - cx) * zs / fx
        ys3 = (ys - cy) * zs / fy
        xyz = np.stack([xs3, ys3, zs], axis=-1)
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

    def predict_and_overlay(self, image_bgr: np.ndarray, depth_m: np.ndarray, K: np.ndarray, pose_cam_ob: np.ndarray) -> np.ndarray:
        feats, coords = self._make_sparse_input(image_bgr, depth_m, K)
        st = ME.SparseTensor(feats, coords)
        cur_obj = self._current_obj_vec(pose_cam_ob)
        with torch.no_grad():
            outputs = self.model(st, actions=None, batch_size=1, current_obj=torch.from_numpy(cur_obj[None, :]).to(self.device))
        if "obj_pred" not in outputs:
            self.last_traj_denorm = None
            return image_bgr
        obj_traj_norm = outputs["obj_pred"].squeeze(0).detach().cpu().numpy()
        # In delta mode, model already returns absolute poses relative to current pose; just denormalize translation.
        obj_traj_ref = _denormalize_obj_traj(obj_traj_norm)
        self.last_traj_denorm = obj_traj_ref
        # Deltas are not used in current execution path; keep only absolute trajectory

        # Debug prints to compare current FP pose and first predicted absolute pose
        fp_xyz6d = mat_to_xyz_rot(pose_cam_ob, rotation_rep="rotation_6d").astype(np.float32)
        traj_first_xyz6d = obj_traj_ref[0, :9].astype(np.float32)
        np.set_printoptions(precision=4, suppress=True)
        print("[DEBUG] FP xyz6d:", fp_xyz6d)
        print("[DEBUG] Traj first xyz6d:", traj_first_xyz6d)

        pose_mats_ref = _build_pose_mats(obj_traj_ref[:, :3], obj_traj_ref[:, 3:3+6])
        points_cam = pose_mats_ref[:, :3, 3]
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


def _load_calib_mat_safe(path: Path) -> np.ndarray | None:
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


def run():
    import argparse
    ap = argparse.ArgumentParser(description="Online evaluation with manual ckpt path")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to RISE policy checkpoint (.ckpt)")
    args = ap.parse_args()
    # Shared interval controlling how often heavy ops run
    update_interval = 10
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Prepare video output
    out_dir = Path(__file__).resolve().parent / "eval_output" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "stream.mp4"
    ZEDCamera = _import_zed_class()
    # Initialize depth estimator and load model at start
    print("[INFO] Loading FoundationStereo depth model...")
    depth_est = DepthEstimator()
    
    # One-time ArUco calibration to compute T_base_cam
    project_root = Path(__file__).resolve().parents[2]
    calib_dir = project_root / 'glasses_hardware' / 'calib'
    calib_dir.mkdir(parents=True, exist_ok=True)
    T_base_cam = None

    T_base_aruco = _load_calib_mat_safe(calib_dir / 'T_base_aruco.npy')

    print("[INFO] Detecting ArUco once for T_cam_aruco...")
    calibrator = piper_calib.ArucoCalibrator(marker_length_m=0.045, K=depth_est.K.astype(np.float32))

    T_cam_aruco = calibrator.detect_and_cache(calib_dir / 'T_zed_aruco.npy', timeout_s=5.0, show=True)
    T_base_cam = T_base_aruco @ np.linalg.inv(T_cam_aruco)
    print("[OK] Computed T_base_cam from T_base_aruco and T_cam_aruco")

    calibrator.close()


    # Open ZED for main loop
    cam = ZEDCamera(resolution="WVGA", fps=30)

    # Initialize robot and gripper
    print("[INFO] Initializing robot and gripper...")
    robot = FlexivRobot(home=False)
    gripper = FlexivGripper(robot,home=False)

    disp_w, disp_h = 640, 360 # target working/display size
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

    traj_pred = TrajectoryPredictor(ckpt_path=Path(args.ckpt))
    print(f"[INFO] Loaded RISE trajectory predictor from {args.ckpt}")

    frame_idx = 0
    try:
        while True:
            stereo = cam.read_stereo()
            if stereo is None:
                continue
            frame, frame_right = stereo

            # Immediately downscale stereo to 640x360 for both display and model input
            h0, w0 = frame.shape[:2]
            if (w0, h0) != (disp_w, disp_h):
                frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                frame_right = cv2.resize(frame_right, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            # Use pre-downscaled intrinsics directly (K already matches 640x360)
            K_rs = depth_est.K.astype(np.float32)

            # Prepare an RGB copy if needed downstream (already resized)
            image_bgr = frame
            image_rgb = image_bgr[..., ::-1].copy()

            last_frame_full = image_rgb
            last_size = (frame.shape[1], frame.shape[0])  # (640,360)

            disp = frame  # already 640x360

            if click_state["pending"]:
                click_state["pending"] = False
                pt = click_state["pt"]
                # visual click feedback
                cx = int(pt[0] * (disp_w / max(1, last_size[0])))
                cy = int(pt[1] * (disp_h / max(1, last_size[1])))
                cv2.circle(disp, (cx, cy), 6, (0, 0, 255), -1)
                try:
                    mask = click_mask(image_rgb, [pt], labels=[1], multimask=True)
                except Exception as e:
                    cv2.displayStatusBar(win, f"SAM error: {e}", 3000)
                    mask = None

                if mask is not None:
                    # Save mask into eval_output directory
                    save_mask(mask,ts)
                    last_mask = mask
                    depth_enabled = True
                    

            if depth_enabled:
                # Run depth only every `update_interval` frames; reuse cached otherwise
                if frame_idx % update_interval == 0:
                    with torch.no_grad():
                        # Depth on resized stereo
                        last_depth_m = depth_est.depth(frame, frame_right)
                depth_m = last_depth_m
                # Initialize FoundationPose once we have a mask and depth
                if (not pose_ready) and (last_mask is not None) and (depth_m is not None):
                    # try:
                    mesh_path = Path(__file__).resolve().parents[2] / "data" / "book" / "mesh.obj"
                    if pose_est is None:
                        pose_est = PoseEstimatorFP(mesh_path)
                    pose = pose_est.initialize(frame, depth_m, last_mask, K_rs)
                    pose_ready = pose is not None
                    print(f'pose_ready: {pose_ready}')
                    # except Exception as e:
                    #     print(f"FoundationPose init error: {e}")
                    #     pose_ready = False

                # Track every 10 frames; overlay every frame using last pose
                if pose_ready and pose_est is not None:
                    # Use the same `update_interval` for pose tracking
                    if (frame_idx % update_interval == 0) and (depth_m is not None):
                        pose_est.track(frame, depth_m, K_rs)
                    frame = pose_est.draw_overlay(frame, K_rs)

                    # Overlay trajectory prediction; then execute a few steps on robot
                    if traj_pred is not None and depth_m is not None and pose_est.pose_cam_ob is not None:
                        if (frame_idx % update_interval == 0):
                            print("[INFO] Running trajectory prediction...")
                            frame = traj_pred.predict_and_overlay(
                                frame, depth_m, K_rs,
                                pose_est.pose_cam_ob.astype(np.float32)
                            )

                            # Execute first N steps relative to current TCP using robot_replay logic
                            if traj_pred.last_traj_denorm is not None:
                                if T_base_cam is None:
                                    print("[WARN] T_base_cam unavailable; skipping execution.")
                                else:
                                    try:
                                        steps_to_execute = 5  # how many relative steps to send each update
                                        # Absolute predicted points in camera (ZED) frame
                                        xyz_abs_cam = traj_pred.last_traj_denorm[:, :3].astype(np.float32)
                                        # Gripper signal per step if available (10th channel)
                                        grip_seq = None
                                        if traj_pred.last_traj_denorm.shape[1] > 9:
                                            grip_seq = traj_pred.last_traj_denorm[:, 9].astype(np.float32)
                                        if xyz_abs_cam.shape[0] >= 2:
                                            # Base<-cam rotation
                                            R_base_cam = T_base_cam[:3, :3].astype(np.float32)
                                            p0_cam = xyz_abs_cam[0]
                                            # Relative-to-first in base frame
                                            base_rel_pts = (R_base_cam @ (xyz_abs_cam - p0_cam).T).T  # (N,3)
                                            # Take the first `steps_to_execute` non-zero steps starting from index 1
                                            steps_pts = base_rel_pts[1:1+int(steps_to_execute), :]
                                            steps_grip = None
                                            if grip_seq is not None:
                                                steps_grip = grip_seq[1:1+int(steps_to_execute)]
                                            if steps_pts.size > 0:
                                                # Send absolute targets: start_xyz + p_rel_base, keep start quaternion
                                                curr_pose7 = robot.get_tcp_pose().astype(np.float32)
                                                start_xyz = curr_pose7[:3].astype(np.float32)
                                                start_quat = curr_pose7[3:7].astype(np.float32)
                                                open_width = getattr(gripper, 'max_width', 0.085)
                                                open_thresh = 0.8
                                                for i in range(steps_pts.shape[0]):
                                                    xyz = start_xyz + steps_pts[i]
                                                    pose7 = np.concatenate([xyz, start_quat], axis=0).astype(np.float32)
                                                    # Gripper control if grip available
                                                    if steps_grip is not None and i < len(steps_grip):
                                                        grip_val = float(steps_grip[i])
                                                        width_cmd = open_width if grip_val > open_thresh else 0.0
                                                        print(f"[EVAL] step {i+1}/{steps_pts.shape[0]} grip={grip_val:.3f} -> width={width_cmd:.3f}")
                                                        try:
                                                            gripper.move(width_cmd)
                                                        except Exception:
                                                            pass
                                                    print(f"[EVAL] send step {i+1}/{steps_pts.shape[0]} pose7=", np.round(pose7, 6))
                                                    robot.send_tcp_pose(pose7)
                                                    time.sleep(0.05)
                                        else:
                                            print("[INFO] Predicted traj has <2 points; skip execution.")
                                    except Exception as e:
                                        print(f"[WARN] Execution error: {e}")


                        else:
                            # Persist last predicted trajectory between updates
                            frame = traj_pred.overlay_cached(frame, K_rs)

                # Visualize depth
                # depth_vis = colorize_depth(depth_m, max_depth=5.0)
                # depth_disp = cv2.resize(depth_vis, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                # cv2.imshow(win_depth, depth_disp)

            # Refresh display from possibly overlaid frame
            disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            cv2.imshow(win, disp)
            # Write current display frame to video
            try:
                if writer is not None and writer.isOpened():
                    writer.write(disp)
            except Exception:
                pass

            key = cv2.waitKey(1) & 0xFF
            if interrupted["flag"] or key == ord('q') or key == 27:
                break
            frame_idx += 1
            del frame, frame_right
            torch.cuda.empty_cache()
    finally:
        # Release resources and save video

        if writer is not None:
            writer.release()
            print(f"[INFO] Saved video to: {video_path}")

        cv2.destroyAllWindows()

        cam.close()



if __name__ == "__main__":
    run()
