#!/usr/bin/env python3
"""Visualize a task-specific TCP->Object transform in Rerun."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def _setup_import_path() -> None:
    root = Path(__file__).resolve().parent
    src_root = root / "src"
    for p in (src_root, root):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def _parse_args(task_choices: tuple[str, ...]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize TASK_TCP_TO_OBJECT_SE3 for a given task.")
    parser.add_argument("--task", required=False, choices=task_choices, help="Task name.")
    parser.add_argument("--calib-npz", type=str, default=None, help="Calibration npz from calib_task_tcp_object_se3.py.")
    parser.add_argument("--axis-len", type=float, default=0.08, help="Axis length for frame rendering.")
    parser.add_argument("--spawn", action="store_true", help="Spawn Rerun viewer.")
    return parser.parse_args()


def _log_frame(rr, path: str, T: np.ndarray, axis_len: float) -> None:
    rr.log(path, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
    rr.log(
        f"{path}/axes",
        rr.Arrows3D(
            origins=np.zeros((3, 3), dtype=np.float32),
            vectors=np.eye(3, dtype=np.float32) * axis_len,
            colors=np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8),
            radii=np.full(3, axis_len * 0.05, dtype=np.float32),
        ),
    )


def main() -> None:
    _setup_import_path()
    from egodata_eval.eval_constant import TASK_CHOICES, TASK_TCP_TO_OBJECT_SE3

    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("rerun is required. Install with: pip install rerun-sdk") from exc

    args = _parse_args(TASK_CHOICES)
    if args.calib_npz is None and not args.task:
        raise ValueError("Either --task or --calib-npz must be provided.")

    rr.init(f"TASK_TCP_TO_OBJECT_SE3 ({args.task or 'calib'})", spawn=args.spawn)
    rr.log("world", rr.ViewCoordinates.FRU)
    rr.set_time_sequence("step", 0)
    _log_frame(rr, "frames/robot", np.eye(4, dtype=np.float32), axis_len=args.axis_len * 1.1)

    if args.calib_npz:
        data = np.load(args.calib_npz, allow_pickle=True)
        T_world_tcp = np.asarray(data["T_robot_tcp"], dtype=np.float32)
        T_tcp_obj = np.asarray(data["T_tcp_obj"], dtype=np.float32)
        meta_task = str(data["task"].item()) if "task" in data else (args.task or "unknown")
        if T_world_tcp.shape != (4, 4) or T_tcp_obj.shape != (4, 4):
            raise ValueError("calib npz contains invalid transform shapes; expected T_robot_tcp/T_tcp_obj as 4x4.")
        # Always derive object pose from the final T_tcp_obj used by calibration output.
        T_world_obj = (T_world_tcp @ T_tcp_obj).astype(np.float32)
    else:
        T_tcp_obj = np.asarray(TASK_TCP_TO_OBJECT_SE3[args.task], dtype=np.float32)
        if T_tcp_obj.shape != (4, 4):
            raise ValueError(f"TASK_TCP_TO_OBJECT_SE3[{args.task!r}] must be (4,4), got {T_tcp_obj.shape}")
        T_world_tcp = np.eye(4, dtype=np.float32)
        T_world_obj = T_world_tcp @ T_tcp_obj
        meta_task = args.task

    _log_frame(rr, "frames/tcp", T_world_tcp, axis_len=args.axis_len)
    _log_frame(rr, "frames/object", T_world_obj, axis_len=args.axis_len * 0.9)
    rr.log(
        "links/tcp_to_object",
        rr.LineStrips3D(
            [np.stack([T_world_tcp[:3, 3], T_world_obj[:3, 3]], axis=0)],
            colors=np.array([[255, 200, 0, 255]], dtype=np.uint8),
            radii=args.axis_len * 0.03,
        ),
    )
    rr.log("meta/task", rr.TextLog(f"task={meta_task}\nT_tcp_to_object=\n{T_tcp_obj}"))
    print(f"[ok] visualized task={meta_task}")
    print("T_tcp_to_object:")
    print(T_tcp_obj)


if __name__ == "__main__":
    main()
