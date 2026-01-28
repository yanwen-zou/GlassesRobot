from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import sys

import numpy as np
import cv2
import os

from MBA.utils.transformation import rotation_transform  # type: ignore
from MBA.utils.constants import TRANS_MIN, TRANS_MAX  # type: ignore
from egodata_eval.eval_constant import (
    CALIB_DIR_REL,
    DEFAULT_GLASSES_ZED_TXT,
    DEFAULT_I2RT_ZED_TXT,
    I2RT_INIT_DURATION,
    I2RT_INIT_STEPS,
    TASK_I2RT_TARGET_RAD,
    WIN_CALIB,
)
from egodata_eval.get_depth import DepthEstimator  # type: ignore
from glasses_hardware.hardware.my_device.i2rt_robo import I2RT, I2RTServer  # type: ignore

RDF_TO_ROBOT = np.array(
    [
        [0.0, 0.0, 1.0],   # forward
        [-1.0, 0.0, 0.0],  # left
        [0.0, -1.0, 0.0],  # up
    ],
    dtype=np.float32,
)

from scripts_calib_balls.calculate_ball_centers import (
    calculate_ball_centroid,
    DEFAULT_MAX_RADIUS_STD_RATIO,
)
from scripts_calib_balls.compute_base_from_ball_centers import compute_base_from_three_points

here = Path(__file__).resolve()
project_root = here.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
_SAM2_VIDEO_PREDICTOR = None
_SAM2_IMAGE_PREDICTOR = None


def _get_sam2_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    # Use Grounded-SAM-2's video predictor which supports real-time streaming APIs.
    candidate = root / "src" / "FoundationStereo" / "Grounded-SAM-2"
    if not candidate.exists():
        raise FileNotFoundError(f"未找到 Grounded-SAM-2 目录: {candidate}")
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    return candidate


def _get_sam2_video_predictor():
    global _SAM2_VIDEO_PREDICTOR
    if _SAM2_VIDEO_PREDICTOR is not None:
        return _SAM2_VIDEO_PREDICTOR
    gs_root = _get_sam2_root()
    config_name = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # Grounded-SAM-2 keeps checkpoints under src/FoundationStereo/sam2_root/checkpoints.
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = repo_root / "src" / "FoundationStereo" / "sam2_root" / "checkpoints" / "sam2.1_hiera_large.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 SAM2 checkpoint: {checkpoint_path}")

    class _WD:
        def __init__(self, path: Path):
            self.path = path
            self.prev = None
        def __enter__(self):
            self.prev = Path.cwd()
            os.chdir(self.path)
        def __exit__(self, exc_type, exc, tb):
            if self.prev is not None:
                os.chdir(self.prev)

    # Ensure Hydra finds configs via pkg://sam2 (Grounded-SAM-2 provides the sam2 package).
    with _WD(gs_root):
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore
        _SAM2_VIDEO_PREDICTOR = build_sam2_video_predictor(config_name, str(checkpoint_path))
    return _SAM2_VIDEO_PREDICTOR


def _get_sam2_image_predictor():
    """Lazy-load SAM2 image predictor from Grounded-SAM-2 to avoid importing sam2_root (offline-only)."""
    global _SAM2_IMAGE_PREDICTOR
    if _SAM2_IMAGE_PREDICTOR is not None:
        return _SAM2_IMAGE_PREDICTOR

    gs_root = _get_sam2_root()
    config_name = "configs/sam2.1/sam2.1_hiera_l.yaml"
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = repo_root / "src" / "FoundationStereo" / "sam2_root" / "checkpoints" / "sam2.1_hiera_large.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 SAM2 checkpoint: {checkpoint_path}")

    class _WD:
        def __init__(self, path: Path):
            self.path = path
            self.prev = None
        def __enter__(self):
            self.prev = Path.cwd()
            os.chdir(self.path)
        def __exit__(self, exc_type, exc, tb):
            if self.prev is not None:
                os.chdir(self.prev)

    with _WD(gs_root):
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        model = build_sam2(config_name, str(checkpoint_path))
        _SAM2_IMAGE_PREDICTOR = SAM2ImagePredictor(model)
    return _SAM2_IMAGE_PREDICTOR


def click_mask(
    image_rgb: np.ndarray,
    points_xy: list[tuple[float, float]],
    labels: Optional[list[int]] = None,
    multimask: bool = True,
) -> np.ndarray:
    """SAM2 image click mask using Grounded-SAM-2 package (supports online video tracking coexistence)."""
    assert isinstance(image_rgb, np.ndarray) and image_rgb.ndim == 3, "image must be HxWx3"
    predictor = _get_sam2_image_predictor()
    predictor.set_image(image_rgb)

    if labels is None:
        labels = [1] * len(points_xy)
    pts = np.array(points_xy, dtype=np.float32)
    lbs = np.array(labels, dtype=np.int32)

    masks, ious, _ = predictor.predict(
        point_coords=pts,
        point_labels=lbs,
        multimask_output=multimask,
        normalize_coords=True,
    )
    idx = int(np.argmax(ious)) if ious is not None and np.size(ious) > 0 else 0
    mask_bool = masks[idx].astype(bool)
    if (ious is None or np.size(ious) == 0) and masks.shape[0] > 1:
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        idx = int(np.argmax(areas))
        mask_bool = masks[idx].astype(bool)
    return (mask_bool.astype(np.uint8)) * 255


def init_robot_mask_tracker(init_frame_rgb: np.ndarray, init_mask: np.ndarray) -> Dict[str, Any]:
    predictor = _get_sam2_video_predictor()
    inference_state = predictor.init_state(video_path=None)
    # Grounded-SAM-2 streaming mode leaves these unset; required by mask consolidation code.
    h, w = int(init_frame_rgb.shape[0]), int(init_frame_rgb.shape[1])
    inference_state["video_height"] = h
    inference_state["video_width"] = w
    # Add first frame and set initial mask as prompt.
    frame_idx = predictor.add_new_frame(inference_state, init_frame_rgb)
    predictor.reset_state(inference_state)
    predictor.add_new_mask(inference_state, frame_idx=frame_idx, obj_id=1, mask=(init_mask > 0))
    # Run inference on the first frame to warm up tracking state.
    _, _, video_res_masks = predictor.infer_single_frame(inference_state, frame_idx=frame_idx)
    mask0 = (video_res_masks[0] > 0.0).detach().cpu().numpy()
    # video_res_masks[0] is usually shaped (1, H, W); squeeze to 2D for saving/usage.
    if mask0.ndim == 3 and mask0.shape[0] == 1:
        mask0 = mask0[0]
    mask0 = mask0.astype(np.uint8) * 255
    return {
        "predictor": predictor,
        "state": inference_state,
        "obj_id": 1,
        "last_mask": mask0,
    }


def update_robot_mask_tracker(tracker: Dict[str, Any], frame_rgb: np.ndarray) -> np.ndarray | None:
    if not tracker:
        return None
    predictor = tracker["predictor"]
    inference_state = tracker["state"]
    frame_idx = predictor.add_new_frame(inference_state, frame_rgb)
    _, _, video_res_masks = predictor.infer_single_frame(inference_state, frame_idx=frame_idx)
    mask = (video_res_masks[0] > 0.0).detach().cpu().numpy()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    mask = mask.astype(np.uint8) * 255
    tracker["last_mask"] = mask
    return mask


def cleanup_robot_mask_tracker(tracker: Dict[str, Any]) -> None:
    # No persistent resources besides GPU memory.
    return




def save_mask(mask: np.ndarray, ts,out_dir: Optional[Path] = None, prefix: str = "mask") -> Path:
    """Save a binary mask image to eval_output and return the saved path.

    - mask: 2D or 3D numpy array. If 3D, will squeeze singleton channel.
    - out_dir: target directory. Defaults to '<this_dir>/eval_output'.
    - prefix: filename prefix before timestamp.

    Returns the full Path to the saved PNG.
    """
    
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "eval_output" / ts

    out_dir.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"save_mask expects a 2D mask, got shape {arr.shape}")

    mask_u8 = (arr.astype(np.uint8) > 0) * 255

    
    out_path = out_dir / f"mask.png"

    ok = cv2.imwrite(str(out_path), mask_u8)
    if not ok:
        raise IOError(f"Failed to write mask to {out_path}")

    return out_path


def _find_default_ckpt() -> Path:
    root = Path(__file__).resolve().parents[2]
    ckpt_dir = root / "MBA" / "ckpt_deploy"
    if ckpt_dir.is_dir():
        cands = sorted([p for p in ckpt_dir.iterdir() if p.suffix == ".ckpt"], key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            return cands[0]
    return ckpt_dir / "policy_last.ckpt"



def _denormalize_obj_traj(obj_traj: np.ndarray) -> np.ndarray:
    out = obj_traj.copy()
    out[:, :3] = (out[:, :3] + 1) * 0.5 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    return out


def _build_pose_mats(translation: np.ndarray, rotation_6d: np.ndarray) -> np.ndarray:
    if rotation_transform is None:
        raise RuntimeError("MBA not available: rotation_transform is required.")
    translation = np.asarray(translation, dtype=np.float32)
    rotation_6d = np.asarray(rotation_6d, dtype=np.float32)
    mats = np.repeat(np.eye(4, dtype=np.float32)[None, ...], len(translation), axis=0)
    rot_mats = rotation_transform(rotation_6d, "rotation_6d", "matrix")
    mats[:, :3, :3] = rot_mats.astype(np.float32)
    mats[:, :3, 3] = translation
    return mats


def _project_points_with_gradient(image: np.ndarray,
                                  cam_intr: np.ndarray,
                                  points_cam: np.ndarray,
                                  color_start=(255, 0, 0),
                                  color_end=(0, 255, 255),
                                  radius: int = 6,
                                  thickness: int = -1) -> np.ndarray:
    if points_cam.size == 0:
        return image
    overlay = image.copy()
    num_pts = len(points_cam)
    # print(f"[DEBUG] camera_intrinstic: {cam_intr}")
    for idx, pt in enumerate(points_cam):
        # print(f"[DEBUG] point {idx}: {pt}")
        z = float(pt[2])
        if z <= 1e-6:
            continue
        uvw = cam_intr @ pt
        u = int(round(uvw[0] / z))
        v = int(round(uvw[1] / z))
        if not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]):
            continue
        alpha = idx / max(num_pts - 1, 1)
        color = tuple(int(round(cs * (1 - alpha) + ce * alpha)) for cs, ce in zip(color_start, color_end))
        cv2.circle(overlay, (u, v), radius, color, thickness, lineType=cv2.LINE_AA)
    return overlay

def _import_zed_class():
    # Ensure project root is importable, then import the ZED wrapper class
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    sys.path.insert(0, str(project_root))
    from glasses_hardware.hardware.my_device.zed import ZEDCamera
    return ZEDCamera


def read_zed_intrinsics_baseline(camera) -> Tuple[np.ndarray, float]:
    """Read ZED left intrinsics and baseline from a camera handle."""
    zed_handle = getattr(camera, "_zed", camera)
    info = zed_handle.get_camera_information()
    config = getattr(info, "camera_configuration", None)
    calib = config.calibration_parameters if config else info.calibration_parameters
    left = calib.left_cam
    K = np.array(
        [
            [float(left.fx), 0.0, float(left.cx)],
            [0.0, float(left.fy), float(left.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    try:
        raw = zed_handle.get_camera_information().camera_configuration.calibration_parameters_raw
        baseline = float(raw.get_camera_baseline()) / 1000
    except Exception as exc:
        raise RuntimeError("ZED baseline not available from SDK.") from exc
    return K, baseline


def headpose_base_to_i2rt_rel(
    headpose_base: np.ndarray,
    T_base_cam: np.ndarray,
    T_i2rt_tcp: np.ndarray,
    T_tcp_zed: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    将base坐标系下的相对轨迹变换到i2rt坐标系下
    
    Args:
        headpose_base: [N, 9] 相对轨迹，每行包含 [dx, dy, dz, rotation_6d...]
        T_base_cam: 从相机到base坐标系的变换矩阵
        T_i2rt_tcp: 从TCP到i2rt坐标系的变换矩阵
        T_tcp_zed: 从ZED相机到TCP的变换矩阵（可选）
    
    Returns:
        [N, 9] 在i2rt坐标系下的相对轨迹
    """
    # 提取平移和旋转部分
    trans_base = headpose_base[:, :3]  # [N, 3] 相对平移
    rot6d_base = headpose_base[:, 3:9]  # [N, 6] 旋转6D
    
    # 构建相对旋转矩阵
    rot_mat_base = rotation_transform(
        rot6d_base,
        "rotation_6d",
        "matrix",
    )  # [N, 3, 3]
    
    # 计算从base到i2rt的旋转矩阵
    if T_tcp_zed is None:
        T_tcp_zed = _load_calib_mat_safe(Path(DEFAULT_I2RT_ZED_TXT))
        if T_tcp_zed is None:
            raise ValueError(f"Failed to load T_tcp_zed from {DEFAULT_I2RT_ZED_TXT}")
    
    # 计算从base到i2rt的变换链
    # T_i2rt_base = T_i2rt_tcp @ T_tcp_zed @ T_cam_base
    T_cam_base = np.linalg.inv(T_base_cam.astype(np.float32))
    T_i2rt_base = (
        T_i2rt_tcp.astype(np.float32)
        @ T_tcp_zed.astype(np.float32)
        @ T_cam_base.astype(np.float32)
    )
    
    # 提取旋转部分
    R_i2rt_base = T_i2rt_base[:3, :3]  # [3, 3]
    
    # 对于相对平移向量，只应用旋转变换，不改变量级
    # 注意：相对平移是向量，坐标系变换时只受旋转影响
    trans_i2rt = np.einsum("ij,nj->ni", R_i2rt_base, trans_base)
    
    # 对于相对旋转，进行相似变换：R_i2rt = R_i2rt_base @ R_base @ R_i2rt_base.T
    rot_mat_i2rt = np.einsum(
        "ij,njk,kl->nil",
        R_i2rt_base,
        rot_mat_base,
        R_i2rt_base.T,
    )
    
    # 转换回6D表示
    rot6d_i2rt = rotation_transform(
        rot_mat_i2rt,
        "matrix",
        "rotation_6d",
    )
    
    # 组合结果
    return np.concatenate([trans_i2rt, rot6d_i2rt], axis=1).astype(np.float32)


def headpose_base_seq_to_rel(
    headpose_base_seq: np.ndarray,
    T_base_cam: np.ndarray,
) -> np.ndarray:
    """Convert absolute base-frame headpose sequence to relative poses wrt base pose."""
    base_pose = T_base_cam.astype(np.float32)
    base_xyz = base_pose[:3, 3]
    base_rot = base_pose[:3, :3]

    pose_seq_base = _build_pose_mats(
        headpose_base_seq[:, :3],
        headpose_base_seq[:, 3:3 + 6],
    ).astype(np.float32)
    xyz_rel = pose_seq_base[:, :3, 3] - base_xyz[None, :]
    rel_rot = np.einsum(
        "nij,jk->nik",
        pose_seq_base[:, :3, :3],
        base_rot.T,
    )
    r6_rel = rotation_transform(
        rel_rot,
        "matrix",
        "rotation_6d",
    )
    return np.concatenate([xyz_rel, r6_rel], axis=1).astype(np.float32)


def headpose_to_tcp(
    headpose_i2rt_rel: np.ndarray,
    T_glasses_zed: Optional[np.ndarray] = None,
    T_zed_tcp: Optional[np.ndarray] = None,
) -> np.ndarray:
    pose_seq_i2rt = _build_pose_mats(
        headpose_i2rt_rel[:, :3],
        headpose_i2rt_rel[:, 3:3+6],
    )
    if T_glasses_zed is None:
        T_glasses_zed = _load_calib_mat_safe(Path(DEFAULT_GLASSES_ZED_TXT))
        if T_glasses_zed is None:
            raise ValueError(f"Failed to load T_glasses_zed from {DEFAULT_GLASSES_ZED_TXT}")
    if T_zed_tcp is None:
        T_zed_tcp = np.linalg.inv(T_glasses_zed.astype(np.float32))
    T_glasses_tcp = T_zed_tcp.astype(np.float32) @ T_glasses_zed.astype(np.float32)
    T_tcp_glasses = np.linalg.inv(T_glasses_tcp)
    pose_seq_tcp = np.einsum(
        "ij,njk,kl->nil",
        T_glasses_tcp,
        pose_seq_i2rt.astype(np.float32),
        T_tcp_glasses,
    )
    xyz_tcp = pose_seq_tcp[:, :3, 3]
    r6_tcp = rotation_transform(
        pose_seq_tcp[:, :3, :3],
        "matrix",
        "rotation_6d",
    )
    return np.concatenate([xyz_tcp, r6_tcp], axis=1).astype(np.float32)


def _load_calib_mat_safe(path: Path) -> Optional[np.ndarray]:
    try:
        arr = np.loadtxt(str(path), dtype=np.float32)
        if arr.shape == (4, 4):
            return arr
        if arr.shape == (3, 4):
            arr = np.vstack([arr, np.array([0, 0, 0, 1], dtype=np.float32)])
            return arr
    except Exception:
        return None
    return None


def move_i2rt_to_init_angles(
    robot: Optional["I2RT"],
    task_name: Optional[str] = None,
    target_rad: np.ndarray = TASK_I2RT_TARGET_RAD["book"],
    duration: float = I2RT_INIT_DURATION,
    steps: int = I2RT_INIT_STEPS,
) -> None:
    """Move I2RT arm to the evaluation target joint configuration."""
    if robot is None:
        print("[WARN] I2RT arm not initialized; cannot move to init pose.")
        return
    try:
        if task_name is not None:
            if task_name in TASK_I2RT_TARGET_RAD:
                target_rad = TASK_I2RT_TARGET_RAD[task_name]
        robot.send_joint_pos_rad(target_rad, duration=duration, steps=steps)
        print(f"[INFO] Moved I2RT joints to rad {TASK_I2RT_TARGET_RAD.get(task_name, TASK_I2RT_TARGET_RAD['book'])}")
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
    K_rs = depth_est.K.astype(np.float32)
    pts: list[tuple[float, float]] = []
    last_frame = None
    last_frame_right = None

    click_state = {"done": False}

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 3:
            pts.append((float(x), float(y)))
            print(f"[INFO] Clicked point {len(pts)}: ({x}, {y})")
            if len(pts) == 3:
                click_state["done"] = True

    win_calib = WIN_CALIB
    cv2.namedWindow(win_calib, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_calib, _on_mouse)
    while True:
        stereo = cam_handle.read_stereo()
        if stereo is None:
            continue
        frame, frame_right = stereo
        last_frame = frame.copy()
        last_frame_right = frame_right
        disp = frame.copy()
        for pt in pts:
            cv2.circle(disp, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)
        cv2.imshow(win_calib, disp)
        k = cv2.waitKey(10) & 0xFF
        if click_state["done"] or k in (27, ord('q')):
            break
    cv2.destroyWindow(win_calib)

    if last_frame is None or last_frame_right is None:
        print("[WARN] Could not grab frame for ball calibration.")
        return None
    frame = last_frame
    frame_right = last_frame_right
    depth_m = depth_est.depth(frame, frame_right)
    fx, fy = K_rs[0, 0], K_rs[1, 1]
    cx, cy = K_rs[0, 2], K_rs[1, 2]
    print(f"[INFO] intrinsics fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    frame_rgb = frame[..., ::-1].copy()

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
            centroid = calculate_ball_centroid( # in cam coordinate
                depth_m=depth_m,
                mask=mask_arr.astype(bool),
                intrinsic=K_rs,
                max_radius_std_ratio=DEFAULT_MAX_RADIUS_STD_RATIO,
                frame_id=0,
                ball_id=idx,
            )
        if centroid is not None:
            cam_pts.append(centroid)
            print(f"[INFO] Mask centroid for p{idx}: {centroid}")
            continue

    if len(cam_pts) != 3:
        print("[WARN] Failed to compute all three ball centroids; aborting calibration.")
        return None

    p1, p2, p3 = cam_pts

    if centroid_log_dir is None:
        centroid_log_dir = Path(__file__).resolve().parents[2] / CALIB_DIR_REL
    centroid_log_dir.mkdir(parents=True, exist_ok=True)
    centroid_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    centroid_log_path = centroid_log_dir / f"ball_centroids_{centroid_ts}.txt"
    with open(centroid_log_path, "w", encoding="utf-8") as fh:
        fh.write("ball_id x y z\n")
        for idx, pt in enumerate((p1, p2, p3), start=1):
            fh.write(f"ball_{idx} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
    print(f"[INFO] Saved per-ball centroids to: {centroid_log_path}") # in cam coordinate

    R_base_cam, t_base_cam = compute_base_from_three_points(p1, p2, p3)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_base_cam
    T[:3, 3] = t_base_cam
    print("[OK] Ball calibration produced T_base_cam:")
    print(T)
    return T


def _run_i2rt_server(channel: str, home: bool, port: int) -> None:
    robot = I2RT(channel=channel, zero_gravity_mode=True, home=home)
    server = I2RTServer(robot, port)
    try:
        server.serve()
    except KeyboardInterrupt:
        try:
            server._server.close(internal=True)  # type: ignore[attr-defined]
        except Exception:
            pass
    finally:
        try:
            robot.close()
        except Exception:
            pass


# VERY IMPORTANT 
def add_relative(rel_mat: np.ndarray, base_mat: np.ndarray) -> np.ndarray: 
    res_mat = np.eye(4, dtype=np.float32)
    res_mat[:3, :3] = rel_mat[:3, :3] @ base_mat[:3, :3]
    res_mat[:3, 3] = rel_mat[:3, 3] + base_mat[:3, 3]
    return res_mat
