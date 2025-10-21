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
    """
    Convert a delta trajectory (normalized translation + 6D rotation) into absolute pose
    coordinates relative to the provided base pose.
    """
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


def get_cam_extrinsics(ds: RealWorldDataset, seq_id: str, frame_ids: List[str]) -> (np.ndarray, List[np.ndarray]):
    ref_frame = ds.seq_ref_frame[seq_id]
    ref_extr = ds.get_camera_extrinsic(seq_id, ref_frame, warn_prefix="vis_prediction(ref)")
    out = []
    for fid in frame_ids:
        fid_int = int(fid)
        out.append(ds.get_camera_extrinsic(seq_id, fid_int, warn_prefix="vis_prediction"))
    return ref_extr, out


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
        (0, 0, 255),   # X axis in red
        (0, 255, 0),   # Y axis in green
        (255, 0, 0),   # Z axis in blue
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


def render_item_predictions(ds: RealWorldDataset,
                            idx: int,
                            model: RISE,
                            obj_pose_mode: str = "abs",
                            add_legend: bool = False,
                            compare_mode: str = "trajectory"):
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
        if getattr(model.action_decoder, "returns_absolute_pose", True): #model has done the conversion already
            pass
        else:
            print("warning: Converting delta object trajectory to absolute poses.")
            obj_traj_norm = delta_to_absolute_traj(obj_traj_norm, current_obj_np)
    term_signal = obj_traj_norm[:, -1]
    term_reached = term_signal > 0.5
    # print(f"[termination] seq={ds.seq_ids[idx]} flags={np.round(term_signal.astype(float), 4)} >0.5?={term_reached}")
    obj_traj_ref = denormalize_obj_traj(obj_traj_norm)
    pose_mats_ref = build_pose_mats(obj_traj_ref[:, :3], obj_traj_ref[:, 3:3 + 6])

    gt_pose_mats_ref = None
    if "action_obj" in item:
        gt_traj_ref = item["action_obj"].numpy()
        gt_pose_mats_ref = build_pose_mats(gt_traj_ref[:, :3], gt_traj_ref[:, 3:3 + 6])

    seq_id = ds.seq_ids[idx]
    frame_ids = ds.action_frame_ids[idx]
    ref_extr, _ = get_cam_extrinsics(ds, seq_id, frame_ids)
    cam_intr = ds.seq_intrinsics[seq_id].copy()
    #cam_intr[:2] *= 0.5 # delete if not downscaled
    ref_extr = np.asarray(ref_extr)

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
        warnings.warn(f"[vis_prediction] Missing RGB image for frame {cur_frame_id} in {rgb_dir}")
        return []

    cam_extr_cur = ds.get_camera_extrinsic(seq_id, int(cur_frame_id), warn_prefix="vis_prediction(cur)")
    cam_extr_cur = np.asarray(cam_extr_cur)
    cam_world_cur = np.linalg.inv(cam_extr_cur)

    overlay = base_rgb

    if compare_mode == "trajectory":
        pred_points_cam = []
        for pose_ref in pose_mats_ref:
            pose_world = ref_extr @ pose_ref
            pose_cam = cam_world_cur @ pose_world
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

        if gt_pose_mats_ref is not None:
            gt_points_cam = []
            for pose_ref in gt_pose_mats_ref:
                pose_world = ref_extr @ pose_ref
                pose_cam = cam_world_cur @ pose_world
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
        if gt_pose_mats_ref is None:
            warnings.warn("GT pose not available; pose comparison disabled.")
        else:
            pred_last = pose_mats_ref[-1]
            gt_last = gt_pose_mats_ref[-1]
            for mat, label, label_color in [
                (pred_last, "Pred", (255, 255, 0)),
                (gt_last, "GT", (0, 255, 255)),
            ]:
                pose_world = ref_extr @ mat
                pose_cam = cam_world_cur @ pose_world
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
        if compare_mode == "trajectory":
            cv2.putText(overlay, "Pred traj: blue→yellow", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if gt_pose_mats_ref is not None:
                cv2.putText(overlay, "GT traj: green→magenta", (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(overlay, "Axes colors: X-red, Y-green, Z-blue", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, "Pred label yellow, GT label cyan", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return [(cur_frame_id, overlay)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data_lion")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num_action", type=int, default=20)
    ap.add_argument("--demo_index", type=int, default=0, help="Index of sequence to visualize")
    ap.add_argument("--output_video", type=str, default=None, help="Optional mp4 path")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "all"],
                    help="Dataset split to load")
    ap.add_argument("--full_episode", action="store_true", help="Use entire episode length for prediction")
    ap.add_argument("--fps", type=int, default=5, help="Frames per second for the rendered video")
    ap.add_argument("--compare_mode", type=str, default="trajectory",
                    choices=["trajectory", "pose"],
                    help="Comparison mode: full trajectory versus single-pose")
    ap.add_argument("--obj_pose_mode", type=str, default="abs", choices=["abs", "delta"],
                    help="Object pose prediction target type used by the checkpoint.")
    args = ap.parse_args()

    ds = RealWorldDataset(args.data_path, split=args.split, num_obs=1,
                          num_action=args.num_action, with_obj_action=True)
    if args.demo_index >= len(ds):
        raise IndexError(f"demo_index {args.demo_index} out of range for dataset of size {len(ds)}.")
    seq_id = ds.seq_ids[args.demo_index]

    if args.full_episode:
        seq_indices = sorted([i for i, sid in enumerate(ds.seq_ids) if sid == seq_id])
    else:
        seq_indices = [args.demo_index]

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
    legend_used = False

    for seq_idx in seq_indices:
        overlays = render_item_predictions(
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

    rendered_frames = [frame_map[fid] for fid in frame_sequence]

    if args.output_video:
        output_dir = os.path.dirname(args.output_video)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        write_video(rendered_frames, args.output_video, fps=args.fps)
        print(f"Saved visualization to {args.output_video}")

    if args.output_video is None:
        print(f"Rendered {len(rendered_frames)} frames.")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
