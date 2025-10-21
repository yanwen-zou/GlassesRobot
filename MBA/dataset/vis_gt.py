import argparse
import os
import warnings
from typing import Dict, List

import cv2
import numpy as np

from dataset.constants import IMG_MEAN, IMG_STD
from dataset.realworld import RealWorldDataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, _REPO_ROOT)
from data_back.constant import POSE_DRAW_TRANSFORM, BBOX


def to_pcd(cloud):
    pts, cols = cloud[:, :3], np.clip(cloud[:, 3:] * IMG_STD + IMG_MEAN, 0, 1)
    import open3d as o3d  # lazy import keeps Open3D optional
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def to_homo(pts: np.ndarray) -> np.ndarray:
    if pts.ndim != 2:
        raise ValueError(f"Expected 2D array for pts, got shape {pts.shape}")
    return np.concatenate((pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)), axis=-1)


def project_3d_to_2d(pt: np.ndarray, K: np.ndarray, ob_in_cam: np.ndarray) -> np.ndarray:
    pt = pt.reshape(4, 1)
    projected = K @ ((ob_in_cam @ pt)[:3, :])
    projected = projected.reshape(-1)
    projected = projected / projected[2]
    return projected.reshape(-1)[:2].round().astype(int)


def draw_xyz_axis(color: np.ndarray,
                  ob_in_cam: np.ndarray,
                  scale: float = 0.1,
                  K: np.ndarray = np.eye(3),
                  thickness: int = 3,
                  transparency: float = 0.0,
                  is_input_rgb: bool = False) -> np.ndarray:
    if is_input_rgb:
        base = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    else:
        base = color.copy()
    xx = np.array([1, 0, 0, 1], dtype=float)
    yy = np.array([0, 1, 0, 1], dtype=float)
    zz = np.array([0, 0, 1, 1], dtype=float)
    xx[:3] *= scale
    yy[:3] *= scale
    zz[:3] *= scale
    origin = tuple(project_3d_to_2d(np.array([0, 0, 0, 1], dtype=float), K, ob_in_cam))
    xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
    yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
    zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
    line_type = cv2.LINE_AA
    arrow_len = 0
    tmp = base.copy()
    tmp1 = cv2.arrowedLine(tmp.copy(), origin, xx, color=(0, 0, 255),
                           thickness=thickness, line_type=line_type, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = cv2.arrowedLine(tmp.copy(), origin, yy, color=(0, 255, 0),
                           thickness=thickness, line_type=line_type, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = cv2.arrowedLine(tmp.copy(), origin, zz, color=(255, 0, 0),
                           thickness=thickness, line_type=line_type, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp.astype(np.uint8), cv2.COLOR_BGR2RGB)
    else:
        tmp = tmp.astype(np.uint8)
    return tmp


def draw_posed_3d_box(K: np.ndarray,
                      img: np.ndarray,
                      ob_in_cam: np.ndarray,
                      bbox: np.ndarray,
                      line_color=(0, 255, 0),
                      linewidth: int = 2) -> np.ndarray:
    min_xyz = bbox.min(axis=0)
    xmin, ymin, zmin = min_xyz
    max_xyz = bbox.max(axis=0)
    xmax, ymax, zmax = max_xyz

    def draw_line3d(start, end, canvas):
        pts = np.stack((start, end), axis=0).reshape(-1, 3)
        pts = (ob_in_cam @ to_homo(pts).T).T[:, :3]
        projected = (K @ pts.T).T
        uv = np.round(projected[:, :2] / projected[:, 2].reshape(-1, 1)).astype(int)
        return cv2.line(canvas, uv[0].tolist(), uv[1].tolist(),
                        color=line_color, thickness=linewidth, lineType=cv2.LINE_AA)

    canvas = img.copy()
    for y in [ymin, ymax]:
        for z in [zmin, zmax]:
            start = np.array([xmin, y, z])
            end = start + np.array([xmax - xmin, 0, 0])
            canvas = draw_line3d(start, end, canvas)

    for x in [xmin, xmax]:
        for z in [zmin, zmax]:
            start = np.array([x, ymin, z])
            end = start + np.array([0, ymax - ymin, 0])
            canvas = draw_line3d(start, end, canvas)

    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            start = np.array([x, y, zmin])
            end = start + np.array([0, 0, zmax - zmin])
            canvas = draw_line3d(start, end, canvas)

    return canvas


def load_pose_matrix(pose_path: str) -> np.ndarray:
    pose_values = np.loadtxt(pose_path).astype(np.float32)
    if pose_values.ndim == 1:
        if pose_values.size == 16:
            pose_mat_cam = pose_values.reshape(4, 4)
        elif pose_values.size == 12:
            pose_mat_cam = np.vstack([pose_values.reshape(3, 4), np.array([0, 0, 0, 1], dtype=np.float32)])
        else:
            raise ValueError(f"Invalid SE3 vector length {pose_values.size} in {pose_path}")
    else:
        pose_mat_cam = pose_values
    if pose_mat_cam.shape == (3, 4):
        pose_mat_cam = np.vstack([pose_mat_cam, np.array([0, 0, 0, 1], dtype=np.float32)])
    if pose_mat_cam.shape != (4, 4):
        raise ValueError(f"Invalid SE3 matrix shape {pose_mat_cam.shape} in {pose_path}")
    return pose_mat_cam


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
                            add_legend: bool = False,
                            pose_sources: List[Dict] = None):
    seq_id = ds.seq_ids[idx]
    frame_ids = ds.action_frame_ids[idx]
    cam_intr = ds.seq_intrinsics[seq_id]
    #cam_intr[:2] *= 0.5 # for half resolution   
    demo_path = ds.data_paths[idx]
    rgb_dir = os.path.join(demo_path, "rgb")

    if pose_sources is None:
        pose_sources = [
            {
                "name": "Raw",
                "subdir": "ob_in_cam",
                "line_color": (0, 255, 0),
                "axis_alpha": 0.0,
                "thickness": 3,
                "label_color": (0, 255, 255)
            }
        ]

    resolved_sources = []
    for source in pose_sources:
        subdir = os.path.join(demo_path, source["subdir"])
        if not os.path.isdir(subdir):
            warnings.warn(f"[vis_gt] Pose directory missing: {subdir}")
            continue
        resolved = source.copy()
        resolved["path"] = subdir
        resolved_sources.append(resolved)

    if not resolved_sources:
        warnings.warn("[vis_gt] No pose sources available; returning empty overlays.")
        return []

    overlays = []
    for step, frame_id in enumerate(frame_ids):
        rgb_path_png = os.path.join(rgb_dir, f"{frame_id}.png")
        rgb_path_jpg = os.path.join(rgb_dir, f"{frame_id}.jpg")
        if os.path.exists(rgb_path_png):
            rgb = cv2.imread(rgb_path_png)
        else:
            rgb = cv2.imread(rgb_path_jpg)
        if rgb is None:
            continue

        overlay = rgb
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        vis = overlay_rgb
        info_lines = []

        for source in resolved_sources:
            pose_path = os.path.join(source["path"], f"{frame_id}.txt")
            if not os.path.exists(pose_path):
                warnings.warn(f"[vis_gt] Missing pose file: {pose_path}")
                continue
            pose_mat_cam = load_pose_matrix(pose_path)
            pose_center = pose_mat_cam @ POSE_DRAW_TRANSFORM
            vis = draw_posed_3d_box(
                cam_intr,
                img=vis,
                ob_in_cam=pose_center,
                bbox=BBOX,
                line_color=source.get("line_color", (0, 255, 0)),
                linewidth=2
            )
            vis = draw_xyz_axis(
                vis,
                ob_in_cam=pose_center,
                scale=0.1,
                K=cam_intr,
                thickness=source.get("thickness", 3),
                transparency=source.get("axis_alpha", 0.0),
                is_input_rgb=True
            )
            info_lines.append(source.get("name", "Pose"))

        overlay = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        if add_legend and step == 0:
            overlay = overlay.copy()
            legend_y = 30
            for idx, source in enumerate(resolved_sources):
                text = source.get("name", f"Pose {idx+1}")
                color = source.get("label_color", (0, 255, 255))
                cv2.putText(overlay, text, (15, legend_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                legend_y += 25
        overlays.append((frame_id, overlay))

    return overlays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data_back")
    ap.add_argument("--num_action", type=int, default=20)
    ap.add_argument("--demo_index", type=int, default=0, help="Index of sequence to visualize")
    ap.add_argument("--output_video", type=str, default=None, help="Optional mp4 path")
    ap.add_argument("--no_o3d", action="store_true", help="Skip Open3D visualization window")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "all"],
                    help="Dataset split to load")
    ap.add_argument("--full_episode", action="store_true", help="Use entire episode length for prediction")
    ap.add_argument("--fps", type=int, default=5, help="Frames per second for the rendered video")
    ap.add_argument("--compare_smooth", action="store_true",
                    help="Overlay smoothed poses from ob_in_cam_smooth in addition to raw poses.")
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

    frame_map: Dict[str, np.ndarray] = {}
    frame_sequence: List[str] = []
    legend_used = False

    for seq_idx in seq_indices:
        pose_sources = [
            {
                "name": "Raw GT",
                "subdir": "ob_in_cam",
                "line_color": (0, 255, 0),
                "axis_alpha": 0.0,
                "thickness": 3,
                "label_color": (0, 255, 0)
            }
        ]
        if args.compare_smooth:
            pose_sources.append(
                {
                    "name": "Smoothed GT",
                    "subdir": "ob_in_cam_smooth",
                    "line_color": (255, 215, 0),
                    "axis_alpha": 0.4,
                    "thickness": 2,
                    "label_color": (0, 215, 255)
                }
            )
        overlays = render_item_predictions(
            ds,
            seq_idx,
            add_legend=not legend_used,
            pose_sources=pose_sources
        )
        if overlays and not legend_used:
            legend_used = True
        for frame_id, overlay in overlays:
            if frame_id not in frame_map:
                frame_map[frame_id] = overlay
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
    main()
