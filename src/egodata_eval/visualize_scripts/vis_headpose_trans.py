#!/usr/bin/env python3
"""Visualize headpose base-to-relative conversion with rerun."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt

here = Path(__file__).resolve()
project_root = here.parents[3]
src_root = project_root / "src"
mba_root = project_root / "MBA"
for path in (src_root, mba_root, project_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from MBA.utils.transformation import rotation_transform  # type: ignore
from egodata_eval.eval_utils import headpose_base_to_i2rt_rel


def _pitch_matrix(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c = np.cos(rad)
    s = np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _plot_frame(ax, T: np.ndarray, axis_len: float) -> None:
    origin = T[:3, 3]
    axes = T[:3, :3]
    colors = ["r", "g", "b"]
    for i in range(3):
        vec = axes[:, i] * axis_len
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            vec[0],
            vec[1],
            vec[2],
            color=colors[i],
            linewidth=2.0,
        )


def _pose_to_T(xyz: np.ndarray, rot6d: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = xyz.astype(np.float32)
    T[:3, :3] = rotation_transform(rot6d[None, :], "rotation_6d", "matrix").squeeze(0).astype(np.float32)
    return T


def _set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    half = max_range / 2.0
    ax.set_xlim3d([x_middle - half, x_middle + half])
    ax.set_ylim3d([y_middle - half, y_middle + half])
    ax.set_zlim3d([z_middle - half, z_middle + half])


def main() -> None:

    T_base_cam = np.array(
        [
            [0.25319037, 0.15298656, -0.9552433, 0.73196584],
            [0.96690476, -0.0079052, 0.25501522, -0.22405244],
            [0.03146251, -0.9881967, -0.14992495, 0.23414463],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    offset = 0.05
    pitch_deg = 3.0
    rel_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [offset, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rel_rot = np.stack(
        [
            np.eye(3, dtype=np.float32),
            _pitch_matrix(pitch_deg),
        ],
        axis=0,
    )
    rel_rot6d = rotation_transform(rel_rot, "matrix", "rotation_6d")
    rel_headpose_seq = np.concatenate([rel_xyz, rel_rot6d], axis=1).astype(np.float32)

    headpose_base_seq = rel_headpose_seq.copy()
    T_i2rt_tcp = np.eye(4, dtype=np.float32)
    rel_i2rt = headpose_base_to_i2rt_rel(headpose_base_seq, T_base_cam, T_i2rt_tcp)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        headpose_base_seq[:, 0],
        headpose_base_seq[:, 1],
        headpose_base_seq[:, 2],
        "o-",
        label="input_rel_base",
        color="tab:green",
    )
    ax.plot(
        rel_i2rt[:, 0],
        rel_i2rt[:, 1],
        rel_i2rt[:, 2],
        "o-",
        label="rel_i2rt",
        color="tab:blue",
    )
    _plot_frame(ax, np.eye(4, dtype=np.float32), axis_len=0.08)
    for idx in range(headpose_base_seq.shape[0]):
        T_input = _pose_to_T(headpose_base_seq[idx, :3], headpose_base_seq[idx, 3:9])
        _plot_frame(ax, T_input, axis_len=0.05)
        T_rel = _pose_to_T(rel_i2rt[idx, :3], rel_i2rt[idx, 3:9])
        _plot_frame(ax, T_rel, axis_len=0.05)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    _set_axes_equal(ax)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
