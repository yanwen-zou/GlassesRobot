import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np

try:
    import rerun as rr
except ImportError:  # pragma: no cover - rerun may not be installed, fail later when needed
    rr = None




def load_traj_for_video(video_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load `outputs/<video_name>_traj.txt` (frame_id x y z) into numpy arrays.

    Returns:
        frame_ids: np.ndarray of shape (N,) with dtype=str containing frame ids.
        points: np.ndarray of shape (N, 3) with dtype=float32 containing XYZ coordinates.
    """
    traj_stem = os.path.basename(video_name)
    traj_path = os.path.join("outputs", f"{traj_stem}_traj.txt")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    frame_ids: List[str] = []
    points: List[List[float]] = []

    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 4:
                raise ValueError(f"Expected 'frame_id x y z' per line, got: {line}")
            frame_ids.append(parts[0])
            xyz = [float(val) for val in parts[1:4]]
            points.append(xyz)

    if not points:
        return np.empty((0,), dtype=str), np.empty((0, 3), dtype=np.float32)

    frame_ids_np = np.array(frame_ids, dtype=str)
    points_np = np.asarray(points, dtype=np.float32)
    return frame_ids_np, points_np


def visualize_video_with_traj(video_path: str,
                              frame_ids: np.ndarray,
                              points: np.ndarray,
                              spawn_viewer: bool = True,
                              point_radius: float = 0.01,
                              line_radius: float = 0.005) -> None:
    """Stream video frames while gradually revealing trajectory points in Rerun."""
    if rr is None:
        raise ImportError("rerun is not installed. Please `pip install rerun-sdk`.")
    if points.size == 0:
        print("No trajectory points to visualize.")
        return
    
    video_path = os.path.join("outputs", video_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")

    rr.init("traj_vis", spawn=spawn_viewer)

    num_pts = len(points)
    alphas = np.linspace(0.0, 1.0, num_pts, dtype=np.float32)
    colors = np.stack([
        (1 - alphas) * 255,
        alphas * 180 + (1 - alphas) * 50,
        255 - alphas * 120,
    ], axis=1).astype(np.uint8)

    frame_idx = 0
    while frame_idx < num_pts:
        ret, frame = cap.read()
        if not ret:
            print(f"Video ended after {frame_idx} frames; trajectory has {num_pts} points.")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rr.set_time_sequence("frame", frame_idx)
        rr.log("video/frame", rr.Image(rgb_frame))

        cur_points = points[:frame_idx + 1]
        cur_colors = colors[:frame_idx + 1]
        cur_labels = frame_ids[:frame_idx + 1].tolist()
        rr.log(
            "traj/points",
            rr.Points3D(
                positions=cur_points,
                radii=point_radius,
                colors=cur_colors,
                labels=cur_labels,
            ),
        )
        rr.log(
            "traj/line",
            rr.LineStrips3D(cur_points[np.newaxis, ...], radii=line_radius, colors=[255, 200, 0]),
        )
        frame_idx += 1

    cap.release()
    if frame_idx < num_pts:
        print("Warning: not all trajectory points were visualized due to short video.")


def resolve_video_path(video_name: str) -> Tuple[str, str]:
    """Return (base_name_without_ext, video_path) for the provided identifier."""
    if video_name.lower().endswith(".mp4"):
        video_path = video_name
        base = os.path.splitext(os.path.basename(video_name))[0]
    else:
        video_path = f"{video_name}.mp4"
        base = os.path.basename(video_name)
    return base, video_path


def main():
    parser = argparse.ArgumentParser(description="Visualize <video_name>.mp4 with its trajectory in Rerun.")
    parser.add_argument("video_name", help="Base name for video and trajectory (video_name.mp4 & outputs/video_name_traj.txt)")
    parser.add_argument("--no_spawn", action="store_true", help="Do not spawn Rerun Viewer window.")
    parser.add_argument("--point_radius", type=float, default=0.002, help="Radius for each point marker.")
    parser.add_argument("--line_radius", type=float, default=0.005, help="Radius for the connecting line strip.")
    args = parser.parse_args()

    traj_base, video_path = resolve_video_path(args.video_name)
    frame_ids, points = load_traj_for_video(traj_base)
    print(f"Loaded {len(points)} trajectory points from outputs/{traj_base}_traj.txt")
    if points.size:
        print("frame_ids shape:", frame_ids.shape, "dtype:", frame_ids.dtype)
        print("points shape:", points.shape, "dtype:", points.dtype)

    visualize_video_with_traj(
        video_path,
        frame_ids,
        points,
        spawn_viewer=not args.no_spawn,
        point_radius=args.point_radius,
        line_radius=args.line_radius,
    )


if __name__ == "__main__":
    main()
