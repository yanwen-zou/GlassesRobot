import argparse
import os

import numpy as np

try:
    import rerun as rr
except ImportError:  # pragma: no cover - rerun may not be installed, fail later when needed
    rr = None


def load_samples(input_dir: str):
    files = sorted(
        f for f in os.listdir(input_dir) if f.endswith(".npz")
    )
    for fname in files:
        path = os.path.join(input_dir, fname)
        data = np.load(path)
        action_obj = data.get("action_obj")
        current_obj = data.get("current_obj_pose")
        headpos = data.get("headpos")
        if action_obj is None or current_obj is None:
            continue
        yield fname, action_obj, current_obj, headpos


def main():
    parser = argparse.ArgumentParser(description="Visualize dataloader_output trajectories in Rerun.")
    parser.add_argument("--input-dir", default="MBA/dataset/dataloader_output")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--no-spawn", action="store_true")
    parser.add_argument("--point-radius", type=float, default=0.01)
    parser.add_argument("--line-radius", type=float, default=0.005)
    args = parser.parse_args()

    if rr is None:
        raise ImportError("rerun is not installed. Please `pip install rerun-sdk`.")

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    rr.init("dataloader_output_vis", spawn=not args.no_spawn)
    frame_idx = 0
    sample_count = 0

    for fname, action_obj, current_obj, headpos in load_samples(args.input_dir):
        if args.max_samples > 0 and sample_count >= args.max_samples:
            break
        sample_count += 1

        if action_obj.ndim != 2 or action_obj.shape[1] < 3:
            continue
        action_xyz = action_obj[:, :3].astype(np.float32)
        current_xyz = current_obj[:3].astype(np.float32)
        headpos = headpos[:3].astype(np.float32)

        for t in range(action_xyz.shape[0]):
            rr.set_time_sequence("frame", frame_idx)
            rr.log("action/points", rr.Points3D(action_xyz[: t + 1], radii=args.point_radius))
            rr.log("action/line", rr.LineStrips3D(action_xyz[: t + 1][np.newaxis, ...], radii=args.line_radius))
            rr.log("current/pose", rr.Points3D(current_xyz[None, :], radii=args.point_radius, colors=[255, 80, 80]))
            rr.log("sample/name", rr.TextLog(f"{fname} frame {t + 1}/{action_xyz.shape[0]}"))
            rr.log("headpose/pose", rr.Points3D(headpos[None, :], radii=args.point_radius, colors=[80, 255, 80]))
            frame_idx += 1


if __name__ == "__main__":
    main()
