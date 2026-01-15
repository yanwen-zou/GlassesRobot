#!/usr/bin/env python3
"""Visualize headpose base-to-relative conversion with rerun."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
here = Path(__file__).resolve()
project_root = here.parents[3]
src_root = project_root / "src"
mba_root = project_root / "MBA"
for path in (src_root, mba_root, project_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from MBA.utils.transformation import rotation_transform  # type: ignore
from egodata_eval.eval_constant import DEFAULT_I2RT_ZED_TXT
from egodata_eval.eval_utils import headpose_base_to_i2rt_rel


def _pitch_matrix(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c = np.cos(rad)
    s = np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _pose_to_T(xyz: np.ndarray, rot6d: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = xyz.astype(np.float32)
    T[:3, :3] = rotation_transform(rot6d[None, :], "rotation_6d", "matrix").squeeze(0).astype(np.float32)
    return T


def _load_calib_mat_safe(path: Path) -> np.ndarray:
    arr = np.loadtxt(str(path), dtype=np.float32)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        pad = np.array([0, 0, 0, 1], dtype=np.float32)
        return np.vstack([arr, pad])
    raise ValueError(f"Unexpected calibration shape: {arr.shape} from {path}")


def _log_frame(rr, path: str, T: np.ndarray, axis_len: float) -> None:
    rr.log(
        path,
        rr.Transform3D(
            translation=T[:3, 3],
            mat3x3=T[:3, :3],
        ),
    )
    rr.log(
        f"{path}/axes",
        rr.Arrows3D(
            origins=np.zeros((3, 3), dtype=np.float32),
            vectors=(np.eye(3, dtype=np.float32) * axis_len),
            colors=np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8),
            radii=np.full(3, axis_len * 0.05, dtype=np.float32),
        ),
    )


def main() -> None:
    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("rerun package required. `pip install rerun-sdk`.") from exc

    rr.init("Headpose base->i2rt demo", spawn=True)
    try:
        rr.log("world", rr.ViewCoordinates.FRU)
    except Exception:
        pass

    T_base_cam = np.array(
        [
            [0.13304143, 0.07218286, -0.9884784, 0.7051812],
            [0.98942083, -0.06788499, 0.12821104, -0.19031298],
            [-0.0578482, -0.9950785, -0.08045074, 0.22179447],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    offset = 0.05
    pitch_deg = 0
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
    T_i2rt_tcp = np.array(
        [
            [0.00646259, -0.00903705, 0.99993828, 0.38464107],
            [-0.01851294, 0.9997867, 0.00915533, 0.00110604],
            [-0.99980773, -0.01857096, 0.00629391, 0.34892676],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    print(f"headpose_base_seq:\n{headpose_base_seq}")
    rel_i2rt = headpose_base_to_i2rt_rel(headpose_base_seq, T_base_cam, T_i2rt_tcp)
    print("Headpose base to i2rt relative:")
    print(rel_i2rt)

    axis_len = 0.08
    T_tcp_zed = _load_calib_mat_safe(Path(DEFAULT_I2RT_ZED_TXT))
    T_cam_base = np.linalg.inv(T_base_cam.astype(np.float32))
    T_i2rt_base = T_i2rt_tcp.astype(np.float32) @ T_tcp_zed.astype(np.float32) @ T_cam_base
    T_base_i2rt = np.linalg.inv(T_i2rt_base)
    rel_i2rt_h = np.concatenate(
        [rel_i2rt[:, :3], np.ones((rel_i2rt.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    rel_i2rt_in_base = (T_base_i2rt.astype(np.float32) @ rel_i2rt_h.T).T[:, :3]
    rr.log(
        "traj/input_rel_base",
        rr.LineStrips3D(
            [headpose_base_seq[:, :3]],
            colors=np.array([[0, 200, 0, 255]], dtype=np.uint8),
            radii=axis_len * 0.03,
        ),
    )
    rr.log(
        "traj/rel_i2rt",
        rr.LineStrips3D(
            [rel_i2rt[:, :3]],
            colors=np.array([[0, 120, 255, 255]], dtype=np.uint8),
            radii=axis_len * 0.03,
        ),
    )
    rr.log(
        "traj/rel_i2rt_in_base",
        rr.LineStrips3D(
            [rel_i2rt_in_base],
            colors=np.array([[220, 80, 255, 255]], dtype=np.uint8),
            radii=axis_len * 0.03,
        ),
    )
    _log_frame(rr, "frames/base", np.eye(4, dtype=np.float32), axis_len=axis_len)
    _log_frame(rr, "frames/base_i2rt", T_base_i2rt, axis_len=axis_len * 0.7)
    for idx in range(headpose_base_seq.shape[0]):
        rr.set_time_sequence("step", idx)
        T_input = _pose_to_T(headpose_base_seq[idx, :3], headpose_base_seq[idx, 3:9])
        _log_frame(rr, "frames/input_rel_base", T_input, axis_len=axis_len * 0.7)
        T_rel = _pose_to_T(rel_i2rt[idx, :3], rel_i2rt[idx, 3:9])
        _log_frame(rr, "frames/rel_i2rt", T_rel, axis_len=axis_len * 0.7)


if __name__ == "__main__":
    main()
