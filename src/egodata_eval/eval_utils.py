from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import sys

import numpy as np
import cv2

from MBA.utils.transformation import rotation_transform  # type: ignore
from MBA.utils.constants import TRANS_MIN, TRANS_MAX  # type: ignore
from egodata_eval.eval_constant import (
    CALIB_DIR_REL,
    DEFAULT_GLASSES_ZED_TXT,
    DEFAULT_I2RT_ZED_TXT,
    I2RT_INIT_DURATION,
    I2RT_INIT_STEPS,
    I2RT_TARGET_DEG,
    I2RT_TARGET_RAD,
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
from FoundationStereo.sam2_root.notebooks.get_mask import click_mask  # type: ignore




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


def headpose_base_to_i2rt_rel( # headpose_base: rel traj in base frame
    headpose_base: np.ndarray,
    T_base_cam: np.ndarray,
    T_i2rt_tcp: np.ndarray,
    T_tcp_zed: Optional[np.ndarray] = None,
) -> np.ndarray:
    pose_seq_base = _build_pose_mats(
        headpose_base[:, :3],
        headpose_base[:, 3:3+6],
    )
    T_cam_base = np.linalg.inv(T_base_cam.astype(np.float32))
    if T_tcp_zed is None:
        T_tcp_zed = _load_calib_mat_safe(Path(DEFAULT_I2RT_ZED_TXT))
        if T_tcp_zed is None:
            raise ValueError(f"Failed to load T_tcp_zed from {DEFAULT_I2RT_ZED_TXT}")
    T_i2rt_base = (
        T_i2rt_tcp.astype(np.float32)
        @ T_tcp_zed.astype(np.float32)
        @ T_cam_base.astype(np.float32)
    )
    # print(f"[INFO] T_i2rt_base:\n{T_i2rt_base}")
    T_base_i2rt = np.linalg.inv(T_i2rt_base)
    pose_seq_i2rt = np.einsum(
        "ij,njk,kl->nil",
        T_i2rt_base,
        pose_seq_base.astype(np.float32),
        T_base_i2rt,
    )
    xyz_i2rt = pose_seq_i2rt[:, :3, 3]
    r6_i2rt = rotation_transform(
        pose_seq_i2rt[:, :3, :3],
        "matrix",
        "rotation_6d",
    )
    return np.concatenate([xyz_i2rt, r6_i2rt], axis=1).astype(np.float32)


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
    target_rad: np.ndarray = I2RT_TARGET_RAD,
    duration: float = I2RT_INIT_DURATION,
    steps: int = I2RT_INIT_STEPS,
) -> None:
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
