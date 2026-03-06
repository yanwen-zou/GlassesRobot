#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize policy_inference_step_log_*.npz in Rerun.")
    parser.add_argument(
        "--log-npz",
        type=str,
        default=None,
        help="Path to policy_inference_step_log_*.npz. If omitted, auto-pick latest in --log-dir.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="baseline/openpi/logs",
        help="Directory containing policy_inference_step_log_*.npz.",
    )
    parser.add_argument("--spawn", action="store_true", help="Spawn Rerun Viewer.")
    parser.add_argument("--axis-len", type=float, default=0.04, help="Axis length for poses.")
    return parser.parse_args()


def _resolve_log_path(args: argparse.Namespace) -> Path:
    if args.log_npz is not None:
        p = Path(args.log_npz).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"log file not found: {p}")
        return p

    log_dir = Path(args.log_dir).expanduser().resolve()
    files = sorted(log_dir.glob("policy_inference_step_log_*.npz"))
    if not files:
        raise FileNotFoundError(f"No policy_inference_step_log_*.npz in: {log_dir}")
    return files[-1]


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-12, None)


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def _log_axes(rr, entity: str, t: np.ndarray, q_xyzw: np.ndarray, axis_len: float) -> None:
    basis = np.eye(3, dtype=np.float32) * axis_len
    vectors = _quat_rotate(np.repeat(q_xyzw[None, :], 3, axis=0), basis)
    colors = np.asarray([[255, 0, 0], [0, 255, 0], [0, 128, 255]], dtype=np.uint8)
    rr.log(entity, rr.Arrows3D(origins=np.repeat(t[None, :], 3, axis=0), vectors=vectors, colors=colors))


def main() -> None:
    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("rerun is required. Install with: pip install rerun-sdk") from exc

    args = _parse_args()
    log_path = _resolve_log_path(args)
    data = np.load(log_path, allow_pickle=False)

    infer_step = np.asarray(data["infer_step"], dtype=np.int32)
    robot_state = np.asarray(data["robot_state"], dtype=np.float32)
    headpose_state = np.asarray(data["headpose_state"], dtype=np.float32)
    action_abs = np.asarray(data["action_abs"], dtype=np.float32)

    if robot_state.ndim != 2 or robot_state.shape[1] < 7:
        raise ValueError(f"robot_state shape must be (N,>=7), got {robot_state.shape}")
    if headpose_state.ndim != 2 or headpose_state.shape[1] < 7:
        raise ValueError(f"headpose_state shape must be (N,>=7), got {headpose_state.shape}")
    if action_abs.ndim != 3 or action_abs.shape[2] < 15:
        raise ValueError(f"action_abs shape must be (N,K,>=15), got {action_abs.shape}")

    rr.init(f"policy_inference_step ({log_path.name})", spawn=args.spawn)
    rr.log("world", rr.ViewCoordinates.FRU)
    # Two separate world roots.
    rr.log("robot_world", rr.Transform3D(translation=np.zeros(3, dtype=np.float32)))
    rr.log("headpose_world", rr.Transform3D(translation=np.zeros(3, dtype=np.float32)))
    rr.log("meta/log_file", rr.TextLog(str(log_path)))

    robot_state_traj: list[np.ndarray] = []
    headpose_state_traj: list[np.ndarray] = []

    for i in range(infer_step.shape[0]):
        rr.set_time_sequence("step", int(infer_step[i]))

        rs = robot_state[i]
        hs = headpose_state[i]
        chunk = action_abs[i]

        robot_t = rs[:3]
        robot_q_xyzw = _quat_normalize(_quat_wxyz_to_xyzw(rs[3:7][None, :]))[0]
        rr.log("robot_world/state", rr.Transform3D(translation=robot_t, quaternion=rr.Quaternion(xyzw=robot_q_xyzw)))
        _log_axes(rr, "robot_world/state_axes", robot_t, robot_q_xyzw, args.axis_len)
        robot_state_traj.append(robot_t.copy())
        rr.log(
            "robot_world/state_traj",
            rr.LineStrips3D([np.asarray(robot_state_traj, dtype=np.float32)], colors=[[255, 255, 255]], radii=[0.003]),
        )

        valid_robot = np.all(np.isfinite(chunk[:, :7]), axis=1)
        if np.any(valid_robot):
            robot_chunk = chunk[valid_robot]
            r_chunk_t = robot_chunk[:, :3]
            r_chunk_q_xyzw = _quat_normalize(_quat_wxyz_to_xyzw(robot_chunk[:, 3:7]))
            rr.log("robot_world/action_chunk_points", rr.Points3D(r_chunk_t, colors=[[255, 200, 0]], radii=[0.004]))
            rr.log(
                "robot_world/action_chunk_first",
                rr.Transform3D(translation=r_chunk_t[0], quaternion=rr.Quaternion(xyzw=r_chunk_q_xyzw[0])),
            )
            _log_axes(rr, "robot_world/action_chunk_first_axes", r_chunk_t[0], r_chunk_q_xyzw[0], args.axis_len * 0.9)
        else:
            rr.log("robot_world/action_chunk_points", rr.Clear(recursive=False))
            rr.log("robot_world/action_chunk_first", rr.Clear(recursive=False))
            rr.log("robot_world/action_chunk_first_axes", rr.Clear(recursive=False))

        if np.all(np.isfinite(hs[:7])):
            head_t = hs[:3]
            head_q_xyzw = _quat_normalize(hs[3:7][None, :])[0]
            rr.log(
                "headpose_world/state",
                rr.Transform3D(translation=head_t, quaternion=rr.Quaternion(xyzw=head_q_xyzw)),
            )
            _log_axes(rr, "headpose_world/state_axes", head_t, head_q_xyzw, args.axis_len)
            headpose_state_traj.append(head_t.copy())
            rr.log(
                "headpose_world/state_traj",
                rr.LineStrips3D([np.asarray(headpose_state_traj, dtype=np.float32)], colors=[[180, 180, 180]], radii=[0.003]),
            )

            valid_head = np.all(np.isfinite(chunk[:, 8:15]), axis=1)
            if np.any(valid_head):
                head_chunk = chunk[valid_head]
                h_chunk_t = head_chunk[:, 8:11]
                h_chunk_q_xyzw = _quat_normalize(head_chunk[:, 11:15])
                rr.log("headpose_world/action_chunk_points", rr.Points3D(h_chunk_t, colors=[[80, 220, 255]], radii=[0.004]))
                rr.log(
                    "headpose_world/action_chunk_first",
                    rr.Transform3D(translation=h_chunk_t[0], quaternion=rr.Quaternion(xyzw=h_chunk_q_xyzw[0])),
                )
                _log_axes(
                    rr,
                    "headpose_world/action_chunk_first_axes",
                    h_chunk_t[0],
                    h_chunk_q_xyzw[0],
                    args.axis_len * 0.9,
                )
            else:
                rr.log("headpose_world/action_chunk_points", rr.Clear(recursive=False))
                rr.log("headpose_world/action_chunk_first", rr.Clear(recursive=False))
                rr.log("headpose_world/action_chunk_first_axes", rr.Clear(recursive=False))
        else:
            rr.log("headpose_world/state", rr.Clear(recursive=False))
            rr.log("headpose_world/state_axes", rr.Clear(recursive=False))
            rr.log("headpose_world/action_chunk_points", rr.Clear(recursive=False))
            rr.log("headpose_world/action_chunk_first", rr.Clear(recursive=False))
            rr.log("headpose_world/action_chunk_first_axes", rr.Clear(recursive=False))

    print(f"[ok] loaded {log_path}")
    print(f"[ok] visualized steps: {infer_step.shape[0]}")


if __name__ == "__main__":
    main()
