import argparse
from typing import Iterable, Tuple

import h5py
import cv2
import numpy as np


def _iter_episodes(h5file: h5py.File) -> Iterable[Tuple[str, h5py.Group]]:
    if "episodes" in h5file:
        episodes = h5file["episodes"]
        for key in episodes.keys():
            yield key, episodes[key]
    else:
        for key in h5file.keys():
            obj = h5file[key]
            if isinstance(obj, h5py.Group):
                yield key, obj


def _print_dataset_info(name: str, dset: h5py.Dataset) -> None:
    shape = dset.shape
    dtype = dset.dtype
    print(f"  - {name}: shape={shape} dtype={dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect record.py HDF5 output.")
    parser.add_argument("--path", required=True, help="Path to replay_buffer_*.hdf5")
    parser.add_argument("--episode", default=None, help="Specific episode key to inspect (e.g., episode_0)")
    parser.add_argument("--show", action="store_true", help="Visualize left/right images (requires GUI).")
    parser.add_argument("--rerun", action="store_true", help="Visualize data with rerun.")
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames to show per episode.")
    args = parser.parse_args()

    with h5py.File(args.path, "r") as h5file:
        episodes = list(_iter_episodes(h5file))
        if not episodes:
            print("No episodes found.")
            return

        if args.episode is not None:
            episodes = [(k, g) for k, g in episodes if k == args.episode]
            if not episodes:
                print(f"Episode {args.episode} not found.")
                return

        print(f"Found {len(episodes)} episode(s) in {args.path}")

        if args.rerun:
            try:
                import rerun as rr
            except Exception as exc:
                print(f"[ERROR] Failed to import rerun: {exc}")
                return
            rr.init("inspect_record_hdf5", spawn=True)

        for key, group in episodes:
            print(f"\nEpisode: {key}")
            for name, obj in group.items():
                if isinstance(obj, h5py.Dataset):
                    _print_dataset_info(name, obj)
                else:
                    print(f"  - {name}: <{type(obj).__name__}>")

            # Print headpose content when episode is specified
            if args.episode is not None:
                headpose = group.get("headpose")
                if headpose is not None:
                    print(f"\nHeadpose content:")
                    headpose_data = headpose[:]
                    if len(headpose_data.shape) == 1:
                        # 1D array
                        print(f"  Shape: {headpose_data.shape}")
                        print(f"  Values: {headpose_data}")
                    elif len(headpose_data.shape) == 2:
                        # 2D array (frames x features)
                        print(f"  Shape: {headpose_data.shape}")
                        print(f"  Total frames: {headpose_data.shape[0]}")
                        print(f"  Features per frame: {headpose_data.shape[1]}")
                        print(f"  First frame: {headpose_data[0]}")
                        if headpose_data.shape[0] > 1:
                            print(f"  Last frame: {headpose_data[-1]}")
                        # Show a few sample frames if there are many
                        if headpose_data.shape[0] > 5:
                            print(f"  Sample frames (every {headpose_data.shape[0] // 5}):")
                            for i in range(0, headpose_data.shape[0], max(1, headpose_data.shape[0] // 5)):
                                print(f"    Frame {i}: {headpose_data[i]}")
                    else:
                        # Higher dimensional array
                        print(f"  Shape: {headpose_data.shape}")
                        print(f"  First element:\n{headpose_data.flat[:min(10, headpose_data.size)]}")
                else:
                    print(f"\nHeadpose: not found in episode {key}")

            if args.show:
                left = group.get("left_cam")
                right = group.get("right_cam")
                if left is None or right is None:
                    print("  - No left_cam/right_cam datasets found for visualization.")
                    continue

                num_frames = min(left.shape[0], right.shape[0], args.max_frames)
                for idx in range(num_frames):
                    left_img = left[idx]
                    right_img = right[idx]
                    try:
                        cv2.imshow("left_cam", left_img)
                        cv2.imshow("right_cam", right_img)
                        keycode = cv2.waitKey(30) & 0xFF
                        if keycode == ord("q") or keycode == 27:
                            cv2.destroyAllWindows()
                            return
                    except cv2.error as exc:
                        print(f"[WARN] OpenCV GUI not available: {exc}")
                        print("       Use --save-dir to write frames to disk.")
                        return
                cv2.destroyAllWindows()

            if args.rerun:
                left = group["left_cam"]
                right = group["right_cam"]
                joint_pos = group["joint_pos"]
                action = group["action"]
                robot_state = group["robot_state"]

                num_frames = min(
                    left.shape[0],
                    right.shape[0],
                    joint_pos.shape[0],
                    action.shape[0],
                    robot_state.shape[0],
                )

                robot_points = []
                for idx in range(num_frames):
                    rr.set_time_sequence("frame", idx)
                    rr.log("left_cam", rr.Image(left[idx]))
                    rr.log("right_cam", rr.Image(right[idx]))

                    jp = joint_pos[idx].astype(np.float32)
                    for j in range(jp.shape[0]):
                        rr.log(f"plots/joint_pos/{j}", rr.Scalars(jp[j]))

                    act = action[idx].astype(np.float32)
                    for j in range(act.shape[0]):
                        rr.log(f"plots/action/{j}", rr.Scalars(act[j]))
                    if act.shape[0] >= 3:
                        rr.log("action/position", rr.Points3D(act[:3][None, :], radii=0.03))

                    rs = robot_state[idx].astype(np.float32)
                    if rs.shape[0] >= 3:
                        robot_points.append(rs[:3])
                        rr.log(
                            "robot_state/position",
                            rr.Points3D(np.stack(robot_points, axis=0), radii=0.03),
                        )


if __name__ == "__main__":
    main()
