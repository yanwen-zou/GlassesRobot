import argparse
import os
import warnings
from typing import Dict, List, Tuple

import cv2
import MinkowskiEngine as ME
import numpy as np
import torch

from dataset.realworld import RealWorldDataset
from policy import RISE
from utils.constants import TRANS_MAX, TRANS_MIN
from utils.transformation import rotation_transform


def denormalize_obj_traj(obj_traj: np.ndarray) -> np.ndarray:
    obj_out = obj_traj.copy()
    obj_out[:, :3] = (obj_out[:, :3] + 1) * 0.5 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    return obj_out


def build_pose_mats(translation: np.ndarray, rotation_6d: np.ndarray) -> np.ndarray:
    mats = np.repeat(np.eye(4)[None, ...], len(translation), axis=0)
    rot_mats = rotation_transform(rotation_6d, "rotation_6d", "matrix")
    mats[:, :3, :3] = rot_mats
    mats[:, :3, 3] = translation
    return mats


def delta_to_absolute_traj(delta_traj: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    if delta_traj.ndim != 2 or base_pose.ndim != 1:
        raise ValueError("delta_traj must be 2D array and base_pose must be 1D array.")
    if delta_traj.shape[1] < 3 + 6 or base_pose.shape[0] < 3 + 6:
        raise ValueError("delta trajectory and base pose must include translation and 6D rotation.")
    abs_traj = delta_traj.copy()
    abs_traj[:, :3] = delta_traj[:, :3] + base_pose[:3]

    delta_rot_mats = rotation_transform(delta_traj[:, 3:3 + 6], "rotation_6d", "matrix")
    base_rot_mat = rotation_transform(base_pose[None, 3:3 + 6], "rotation_6d", "matrix")[0]
    abs_rot_mats = delta_rot_mats @ base_rot_mat
    abs_rot_6d = rotation_transform(abs_rot_mats, "matrix", "rotation_6d")
    abs_traj[:, 3:3 + 6] = abs_rot_6d
    return abs_traj


def interpolate_color(color_start: Tuple[int, int, int],
                      color_end: Tuple[int, int, int],
                      alpha: float) -> Tuple[int, int, int]:
    return tuple(int(round(cs * (1 - alpha) + ce * alpha))
                 for cs, ce in zip(color_start, color_end))


def project_points_with_gradient(image: np.ndarray,
                                 cam_intr: np.ndarray,
                                 points_cam: np.ndarray,
                                 color_start: Tuple[int, int, int] = (0, 0, 255),
                                 color_end: Tuple[int, int, int] = (0, 255, 255),
                                 radius: int = 6,
                                 thickness: int = -1) -> np.ndarray:
    if points_cam.size == 0:
        return image
    overlay = image.copy()
    num_pts = len(points_cam)
    for idx, pt in enumerate(points_cam):
        z = pt[2]
        if z <= 1e-6:
            continue
        uvw = cam_intr @ pt
        u = int(round(uvw[0] / z))
        v = int(round(uvw[1] / z))
        if not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]):
            continue
        alpha = idx / max(num_pts - 1, 1)
        color = interpolate_color(color_start, color_end, alpha)
        cv2.circle(overlay, (u, v), radius, color, thickness, lineType=cv2.LINE_AA)
    return overlay


def draw_coordinate_frame(image: np.ndarray,
                          cam_intr: np.ndarray,
                          pose_cam: np.ndarray,
                          axis_length: float = 0.05,
                          thickness: int = 2,
                          label: str = "",
                          label_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    overlay = image.copy()
    origin = pose_cam[:3, 3]
    if origin[2] <= 1e-6:
        return image
    axes = pose_cam[:3, :3]
    endpoints = [origin + axis_length * axes[:, i] for i in range(3)]
    points = [origin] + endpoints
    pixels = []
    for pt in points:
        z = pt[2]
        if z <= 1e-6:
            return image
        uvw = cam_intr @ pt
        u = int(round(uvw[0] / z))
        v = int(round(uvw[1] / z))
        pixels.append((u, v))
    h, w = overlay.shape[:2]
    origin_px = pixels[0]
    if not (0 <= origin_px[0] < w and 0 <= origin_px[1] < h):
        return image
    axis_colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
    ]
    for idx in range(3):
        end_px = pixels[idx + 1]
        cv2.line(overlay, origin_px, end_px, axis_colors[idx], thickness, lineType=cv2.LINE_AA)
    if label:
        pos = (origin_px[0] + 5, origin_px[1] - 5)
        cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, lineType=cv2.LINE_AA)
    return overlay


def write_video(frames: List[np.ndarray], output_path: str, fps: int = 5) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def convert_base_pose_to_cam(pose_base: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """Convert a pose in ball/base frame to camera frame using T_base_cam."""
    T_cam_base = np.linalg.inv(T_base_cam)
    return T_cam_base @ pose_base


def render_item_predictions(ds: RealWorldDataset,
                            idx: int,
                            model: RISE,
                            obj_pose_mode: str = "abs",
                            add_legend: bool = False,
                            compare_mode: str = "traj") -> Tuple[List[Tuple[str, np.ndarray]], np.ndarray]:
    item = ds[idx]
    coords, feats = ME.utils.sparse_collate(item["input_coords_list"],
                                            item["input_feats_list"])
    st = ME.SparseTensor(feats.cuda(), coords.cuda())

    current_obj = item.get("current_obj_pose_normalized")
    current_obj_np = None
    if current_obj is not None:
        current_obj_np = current_obj.numpy()
        current_obj = current_obj.unsqueeze(0).cuda()
    with torch.no_grad():
        outputs = model(st, actions=None, batch_size=1, current_obj=current_obj)
    if "obj_pred" not in outputs:
        raise RuntimeError("Model did not return object predictions.")
    obj_traj_norm = outputs["obj_pred"].squeeze(0).cpu().numpy()
    if obj_pose_mode == "delta":
        if current_obj_np is None:
            raise RuntimeError("Current object pose is required to convert delta predictions.")
        if getattr(model.action_decoder, "returns_absolute_pose", True):
            pass
        else:
            print("warning: Converting delta object trajectory to absolute poses (base frame).")
            obj_traj_norm = delta_to_absolute_traj(obj_traj_norm, current_obj_np)

    # Predicted poses are in ball/base frame. Denormalize and convert to camera frame of the current image.
    obj_traj_base = denormalize_obj_traj(obj_traj_norm)
    pose_mats_base = build_pose_mats(obj_traj_base[:, :3], obj_traj_base[:, 3:3 + 6])

    gt_pose_mats_base = None
    if "action_obj" in item:
        gt_traj_base = item["action_obj"].numpy()
        gt_pose_mats_base = build_pose_mats(gt_traj_base[:, :3], gt_traj_base[:, 3:3 + 6])

    seq_id = ds.seq_ids[idx]
    obs_frame_ids = ds.obs_frame_ids[idx]
    cur_frame_id = obs_frame_ids[-1]

    demo_path = ds.data_paths[idx]
    rgb_dir = os.path.join(demo_path, "rgb")
    rgb_path_png = os.path.join(rgb_dir, f"{cur_frame_id}.png")
    rgb_path_jpg = os.path.join(rgb_dir, f"{cur_frame_id}.jpg")
    if os.path.exists(rgb_path_png):
        base_rgb = cv2.imread(rgb_path_png)
    else:
        base_rgb = cv2.imread(rgb_path_jpg)
    if base_rgb is None:
        warnings.warn(f"[vis_prediction_ball2cam] Missing RGB image for frame {cur_frame_id} in {rgb_dir}")
        return [], np.empty((0,), dtype=np.float32)

    cam_intr = ds.seq_intrinsics[seq_id].copy()
    T_base_cam = ds._get_cam_to_base(seq_id, cur_frame_id)  # noqa: SLF001

    overlay = base_rgb

    if compare_mode == "traj":
        pred_points_cam = []
        for pose_base in pose_mats_base:
            pose_cam = convert_base_pose_to_cam(pose_base, T_base_cam)
            pred_points_cam.append(pose_cam[:3, 3])
        pred_points_cam = np.asarray(pred_points_cam)

        overlay = project_points_with_gradient(
            overlay,
            cam_intr,
            pred_points_cam,
            color_start=(255, 0, 0),
            color_end=(0, 255, 255),
            radius=6,
            thickness=-1
        )

        if gt_pose_mats_base is not None:
            gt_points_cam = []
            for pose_base in gt_pose_mats_base:
                pose_cam = convert_base_pose_to_cam(pose_base, T_base_cam)
                gt_points_cam.append(pose_cam[:3, 3])
            gt_points_cam = np.asarray(gt_points_cam)
            overlay = project_points_with_gradient(
                overlay,
                cam_intr,
                gt_points_cam,
                color_start=(0, 255, 0),
                color_end=(255, 0, 255),
                radius=4,
                thickness=-1
            )
    elif compare_mode == "pose":
        if gt_pose_mats_base is None:
            warnings.warn("GT pose not available; pose comparison disabled.")
        else:
            pred_last = pose_mats_base[-1]
            gt_last = gt_pose_mats_base[-1]
            for mat, label, label_color in [
                (pred_last, "Pred", (255, 255, 0)),
                (gt_last, "GT", (0, 255, 255)),
            ]:
                pose_cam = convert_base_pose_to_cam(mat, T_base_cam)
                overlay = draw_coordinate_frame(
                    overlay,
                    cam_intr,
                    pose_cam,
                    axis_length=0.06,
                    thickness=2,
                    label=label,
                    label_color=label_color
                )
    else:
        raise ValueError(f"Unsupported compare_mode {compare_mode}.")

    if add_legend:
        if compare_mode == "traj":
            cv2.putText(overlay, "Pred traj: blue→yellow", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if gt_pose_mats_base is not None:
                cv2.putText(overlay, "GT traj: green→magenta", (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(overlay, "Axes colors: X-red, Y-green, Z-blue", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, "Pred label yellow, GT label cyan", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    first_pose_cam = convert_base_pose_to_cam(pose_mats_base[0], T_base_cam) if len(pose_mats_base) else None
    first_pred_point_cam = first_pose_cam[:3, 3].copy() if first_pose_cam is not None else np.empty((0,), dtype=np.float32)
    return [(str(cur_frame_id), overlay)], first_pred_point_cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num_action", type=int, default=20)
    ap.add_argument("--demo_index", type=int, default=0, help="Index of episode (demo folder) to visualize")
    ap.add_argument("--demo_step", type=int, default=0,
                    help="When not using --full_episode, pick the N-th window inside the chosen episode")
    ap.add_argument("--output_video", type=str, default=None, help="Optional mp4 path")
    ap.add_argument("--split", type=str, default="train", choices=["train", "eval", "all"],
                    help="Dataset split to load")
    ap.add_argument("--full_episode", action="store_true", help="Use entire episode length for prediction")
    ap.add_argument("--fps", type=int, default=20, help="Frames per second for the rendered video")
    ap.add_argument("--compare_mode", type=str, default="traj",
                    choices=["traj", "pose"],
                    help="Comparison mode: full trajectory versus single-pose")
    ap.add_argument("--obj_pose_mode", type=str, default="delta", choices=["abs", "delta"],
                    help="Object pose prediction target type used by the checkpoint.")
    args = ap.parse_args()

    ds = RealWorldDataset(args.data_path, split=args.split, num_obs=1,
                          num_action=args.num_action, with_obj_action=True)
    available_seq_ids = getattr(ds, "all_demos", sorted(set(ds.seq_ids)))
    if args.demo_index < 0 or args.demo_index >= len(available_seq_ids):
        raise IndexError(f"demo_index {args.demo_index} out of range for {len(available_seq_ids)} episodes.")
    seq_id = available_seq_ids[args.demo_index]
    seq_indices_all = [i for i, sid in enumerate(ds.seq_ids) if sid == seq_id]
    if not seq_indices_all:
        raise RuntimeError(f"No samples found for episode id {seq_id}.")

    if args.full_episode:
        seq_indices = sorted(seq_indices_all)
    else:
        demo_step = max(0, min(args.demo_step, len(seq_indices_all) - 1))
        seq_indices = [seq_indices_all[demo_step]]

    model = RISE(num_action=args.num_action,
                 input_dim=6,
                 obs_feature_dim=512,
                 action_dim=10,
                 hidden_dim=512,
                 enable_mba=True,
                 obj_dim=10,
                 obj_pose_mode=args.obj_pose_mode).cuda().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"), strict=False)

    frame_map: Dict[str, np.ndarray] = {}
    frame_sequence: List[str] = []
    traj_records: List[Tuple[str, np.ndarray]] = []
    legend_used = False

    for seq_idx in seq_indices:
        overlays, first_pred_point = render_item_predictions(
            ds,
            seq_idx,
            model,
            obj_pose_mode=args.obj_pose_mode,
            add_legend=not legend_used,
            compare_mode=args.compare_mode
        )
        if overlays and not legend_used:
            legend_used = True
        for frame_id, overlay in overlays:
            frame_map[frame_id] = overlay
            if frame_id not in frame_sequence:
                frame_sequence.append(frame_id)
        if first_pred_point.size:
            obs_frames = ds.obs_frame_ids[seq_idx]
            traj_frame_id = str(obs_frames[-1]) if len(obs_frames) else str(seq_idx)
            traj_records.append((traj_frame_id, first_pred_point))

    rendered_frames = [frame_map[fid] for fid in frame_sequence]

    traj_output_path = None
    if args.output_video:
        output_dir = os.path.dirname(args.output_video)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        write_video(rendered_frames, args.output_video, fps=args.fps)
        print(f"Saved visualization to {args.output_video}")

    if args.output_video is None:
        print(f"Rendered {len(rendered_frames)} frames.")
    else:
        print(f"Rendered {len(rendered_frames)} frames for video output.")

    if traj_records:
        if args.output_video:
            traj_base = os.path.splitext(args.output_video)[0]
        else:
            default_traj_dir = os.path.join("outputs", "trajectories")
            os.makedirs(default_traj_dir, exist_ok=True)
            traj_base = os.path.join(default_traj_dir, seq_id)
        traj_output_path = f"{traj_base}_traj_cam.txt"
        traj_dir = os.path.dirname(traj_output_path)
        if traj_dir:
            os.makedirs(traj_dir, exist_ok=True)
        with open(traj_output_path, "w", encoding="utf-8") as f:
            f.write("# frame_id x y z (camera frame)")
            for frame_id, point in traj_records:
                f.write(f"\n{frame_id} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")
        print(f"Saved predicted first-step trajectory points (cam frame) to {traj_output_path}")
    else:
        print("No trajectory points recorded (empty predictions).")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
