"""Visualize absolute headpose and robot TCP from one episode in an HDF5 file with Rerun.

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


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-12, None)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


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
        tcp_pose = np.asarray(episode["tcp_pose"][:], dtype=np.float32) if "tcp_pose" in episode else None
        robot_state = np.asarray(episode["robot_state"][:], dtype=np.float32) if "robot_state" in episode else None

    if headpose.ndim != 2 or headpose.shape[1] != 7:
        raise ValueError(f"Expected headpose shape (T, 7), got {headpose.shape}")
    if left_cam is not None and left_cam.shape[0] != headpose.shape[0]:
        raise ValueError(f"left_cam length {left_cam.shape[0]} != headpose length {headpose.shape[0]}")
    if right_cam is not None and right_cam.shape[0] != headpose.shape[0]:
        raise ValueError(f"right_cam length {right_cam.shape[0]} != headpose length {headpose.shape[0]}")
    if tcp_pose is not None:
        if tcp_pose.ndim != 2 or tcp_pose.shape[1] != 7:
            raise ValueError(f"Expected tcp_pose shape (T, 7), got {tcp_pose.shape}")
        if tcp_pose.shape[0] != headpose.shape[0]:
            raise ValueError(f"tcp_pose length {tcp_pose.shape[0]} != headpose length {headpose.shape[0]}")
    if robot_state is not None:
        if robot_state.ndim != 2 or robot_state.shape[1] < 7:
            raise ValueError(f"Expected robot_state shape (T, >=7), got {robot_state.shape}")
        if robot_state.shape[0] != headpose.shape[0]:
            raise ValueError(f"robot_state length {robot_state.shape[0]} != headpose length {headpose.shape[0]}")

    # record.py format: prefer tcp_pose; fallback to robot_state[:7].
    tcp_state = tcp_pose if tcp_pose is not None else (robot_state[:, :7] if robot_state is not None else None)

    rr.init(args.recording_name, spawn=args.spawn)

    traj_xyz: list[np.ndarray] = []
    tcp_traj_xyz: list[np.ndarray] = []
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
        axis_len = 0.06
        unit_axes = np.eye(3, dtype=np.float32)
        rotated_axes = _quat_rotate(np.repeat(quat_xyzw[None, :], 3, axis=0), unit_axes * axis_len)
        rr.log(
            "headpose/pose",
            rr.Transform3D(translation=xyz, quaternion=rr.Quaternion(xyzw=_quat_normalize(quat_xyzw))),
        )
        rr.log(
            "headpose/axes",
            rr.Arrows3D(
                origins=np.repeat(xyz[None, :], 3, axis=0),
                vectors=rotated_axes,
                colors=np.asarray([[255, 0, 0], [0, 255, 0], [0, 128, 255]], dtype=np.uint8),
            ),
        )
        if tcp_state is not None:
            tcp = tcp_state[i]
            tcp_xyz = tcp[:3].astype(np.float32)
            tcp_quat_xyzw = tcp[3:7].astype(np.float32)
            tcp_traj_xyz.append(tcp_xyz.copy())
            tcp_xyz_arr = np.asarray(tcp_traj_xyz, dtype=np.float32)
            rr.log("robot_tcp/current", rr.Points3D([tcp_xyz], colors=[[255, 100, 0]], radii=[0.008]))
            rr.log("robot_tcp/trajectory", rr.LineStrips3D([tcp_xyz_arr], colors=[[255, 180, 80]]))
            rr.log(
                "robot_tcp/pose",
                rr.Transform3D(translation=tcp_xyz, quaternion=rr.Quaternion(xyzw=_quat_normalize(tcp_quat_xyzw))),
            )
            tcp_rotated_axes = _quat_rotate(np.repeat(tcp_quat_xyzw[None, :], 3, axis=0), unit_axes * axis_len)
            rr.log(
                "robot_tcp/axes",
                rr.Arrows3D(
                    origins=np.repeat(tcp_xyz[None, :], 3, axis=0),
                    vectors=tcp_rotated_axes,
                    colors=np.asarray([[255, 80, 80], [80, 255, 80], [80, 160, 255]], dtype=np.uint8),
                ),
            )
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
