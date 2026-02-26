"""Visualize absolute headpose from one episode in an HDF5 file with Rerun.

Example:
  python baseline/openpi/scripts_dataset/visualize_hdf5_headpose_rerun.py \
    --hdf5-path /path/to/data.hdf5 \
    --episode-index 0
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List

import h5py
import numpy as np
import tyro


def _get_episode_keys(h5_file: h5py.File) -> List[str]:
    if "episodes" in h5_file:
        return list(h5_file["episodes"].keys())
    return list(h5_file.keys())


def _get_episode_group(h5_file: h5py.File, episode_key: str):
    if "episodes" in h5_file:
        return h5_file["episodes"][episode_key]
    return h5_file[episode_key]


@dataclass(frozen=True)
class Args:
    hdf5_path: str
    episode_index: int
    spawn: bool = True
    recording_name: str = "hdf5_headpose_abs_vis"
    sleep_sec: float = 0.0
    max_frames: int = -1


def main(args: Args) -> None:
    try:
        import rerun as rr  # type: ignore
    except Exception as exc:
        raise RuntimeError("Failed to import rerun. Please install rerun-sdk.") from exc

    with h5py.File(args.hdf5_path, "r") as h5_file:
        episode_keys = _get_episode_keys(h5_file)
        if len(episode_keys) == 0:
            raise ValueError(f"No episodes found in {args.hdf5_path}")
        if args.episode_index < 0 or args.episode_index >= len(episode_keys):
            raise IndexError(
                f"episode_index={args.episode_index} out of range [0, {len(episode_keys) - 1}]"
            )
        episode_key = episode_keys[args.episode_index]
        episode = _get_episode_group(h5_file, episode_key)
        if "headpose" not in episode:
            raise KeyError(f"Episode '{episode_key}' has no 'headpose' dataset")
        headpose = np.asarray(episode["headpose"][:], dtype=np.float32)
        left_cam = np.asarray(episode["left_cam"][:]) if "left_cam" in episode else None
        right_cam = np.asarray(episode["right_cam"][:]) if "right_cam" in episode else None

    if headpose.ndim != 2 or headpose.shape[1] != 7:
        raise ValueError(f"Expected headpose shape (T, 7), got {headpose.shape}")
    if left_cam is not None and left_cam.shape[0] != headpose.shape[0]:
        raise ValueError(f"left_cam length {left_cam.shape[0]} != headpose length {headpose.shape[0]}")
    if right_cam is not None and right_cam.shape[0] != headpose.shape[0]:
        raise ValueError(f"right_cam length {right_cam.shape[0]} != headpose length {headpose.shape[0]}")

    rr.init(args.recording_name, spawn=args.spawn)

    traj_xyz: list[np.ndarray] = []
    total = headpose.shape[0]
    limit = total if args.max_frames <= 0 else min(total, int(args.max_frames))
    for i in range(limit):
        pose = headpose[i]
        xyz = pose[:3].astype(np.float32)
        quat_xyzw = pose[3:7].astype(np.float32)
        traj_xyz.append(xyz.copy())
        xyz_arr = np.asarray(traj_xyz, dtype=np.float32)

        rr.set_time("frame", sequence=i)
        rr.log("headpose/current", rr.Points3D([xyz], colors=[[0, 255, 0]], radii=[0.008]))
        rr.log("headpose/trajectory", rr.LineStrips3D([xyz_arr], colors=[[80, 170, 255]]))
        rr.log("headpose/all_points", rr.Points3D(xyz_arr, colors=[[255, 255, 255]], radii=[0.0035]))
        rr.log("headpose/quat_xyzw", rr.TextLog(np.array2string(quat_xyzw, precision=5)))
        if left_cam is not None:
            rr.log("obs/left_cam", rr.Image(left_cam[i]))
        if right_cam is not None:
            rr.log("obs/right_cam", rr.Image(right_cam[i]))

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    print(
        f"Done. file={args.hdf5_path}, episode_index={args.episode_index}, "
        f"episode_key={episode_key}, frames={limit}/{total}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
