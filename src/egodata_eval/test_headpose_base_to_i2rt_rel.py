from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

here = Path(__file__).resolve()
project_root = here.parents[2]
src_root = project_root / "src"
mba_root = project_root / "MBA"
for path in (src_root, mba_root, project_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from MBA.utils.transformation import rotation_transform  # type: ignore


def _pose_row_to_mat(row: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = row[:3].astype(np.float32)
    T[:3, :3] = rotation_transform(
        row[3:9][None, :].astype(np.float32),
        "rotation_6d",
        "matrix",
    ).squeeze(0).astype(np.float32)
    return T


def _pose_mat_to_row(T: np.ndarray) -> np.ndarray:
    rot6d = rotation_transform(
        T[:3, :3][None, ...].astype(np.float32),
        "matrix",
        "rotation_6d",
    ).squeeze(0).astype(np.float32)
    return np.concatenate([T[:3, 3].astype(np.float32), rot6d], axis=0).astype(np.float32)


def _build_base_rel_traj(
    dx: float,
    dy: float,
    dz: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    repeat: int,
    bidirectional: bool,
) -> np.ndarray:
    repeat = max(1, int(repeat))
    base_rot = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix().astype(np.float32)
    base_rot6d = rotation_transform(
        base_rot[None, ...],
        "matrix",
        "rotation_6d",
    ).squeeze(0).astype(np.float32)
    forward_row = np.concatenate(
        [np.array([dx, dy, dz], dtype=np.float32), base_rot6d],
        axis=0,
    ).astype(np.float32)
    forward_T = _pose_row_to_mat(forward_row)
    inverse_row = _pose_mat_to_row(np.linalg.inv(forward_T).astype(np.float32))

    rows: list[np.ndarray] = []
    for _ in range(repeat):
        rows.append(forward_row.copy())
        if bidirectional:
            rows.append(inverse_row.copy())
    return np.stack(rows, axis=0).astype(np.float32)


def main() -> None:
    from egodata_eval.eval_constant import DEFAULT_BASE_TO_ROBOT_TXT, DEPTH_EST_SCALE

    ap = argparse.ArgumentParser(
        description="Test eval_utils.headpose_base_to_i2rt_rel with real I2RT execution."
    )
    ap.add_argument("--task", type=str, default="book", help="Task key used by move_i2rt_to_init_angles.")
    ap.add_argument("--base-to-robot-txt", type=str, default=DEFAULT_BASE_TO_ROBOT_TXT)
    ap.add_argument("--dx", type=float, default=0.00, help="Base-frame relative translation x (meter).")
    ap.add_argument("--dy", type=float, default=0.00, help="Base-frame relative translation y (meter).")
    ap.add_argument("--dz", type=float, default=0.00, help="Base-frame relative translation z (meter).")
    ap.add_argument("--roll-deg", type=float, default=0.0, help="Base-frame relative roll (deg).")
    ap.add_argument("--pitch-deg", type=float, default=0.0, help="Base-frame relative pitch (deg).")
    ap.add_argument("--yaw-deg", type=float, default=0.0, help="Base-frame relative yaw (deg).")
    ap.add_argument("--repeat", type=int, default=5, help="How many times to repeat the relative step.")
    ap.add_argument(
        "--one-way",
        action="store_true",
        help="Only execute forward relative step, do not append inverse step.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print converted trajectory, do not execute i2rt movement.",
    )
    ap.add_argument("--settle-sec", type=float, default=1.0, help="Wait time after move-to-init.")
    args = ap.parse_args()

    import rclpy
    from egodata_eval.eval_hardware import EvalHardware
    from egodata_eval.eval_utils import (
        calibrate_from_three_balls,
        headpose_base_to_i2rt_rel,
        move_i2rt_to_init_angles,
    )
    from egodata_eval.get_depth import DepthEstimator

    if not rclpy.ok():
        rclpy.init(args=None)

    hw = EvalHardware(
        base_to_robot_txt=args.base_to_robot_txt,
        task_name=args.task,
    )

    try:
        print(f"[INFO] Move I2RT to init pose for task={args.task}...")
        move_i2rt_to_init_angles(hw.i2rt_robot, task_name=args.task)
        time.sleep(float(args.settle_sec))

        print("[INFO] Running three-ball calibration to get T_base_cam...")
        depth_est = DepthEstimator(scale=DEPTH_EST_SCALE, camera=hw.camera)
        T_base_cam = calibrate_from_three_balls(
            hw.camera,
            depth_est,
            move_robot_fn=lambda: move_i2rt_to_init_angles(hw.i2rt_robot, task_name=args.task),
            centroid_log_dir=None,
        )
        if T_base_cam is None:
            raise RuntimeError("Failed to calibrate T_base_cam from three balls.")
        T_base_cam = T_base_cam.astype(np.float32)

        print("[INFO] Reading current I2RT TCP pose (FK) as T_i2rt_tcp...")
        q_now = hw.i2rt_robot.current_joint_pos().astype(np.float32)
        T_i2rt_tcp = hw.i2rt_kin.fk(q_now[:hw.i2rt_arm_dofs]).astype(np.float32)

        base_rel_traj = _build_base_rel_traj(
            dx=float(args.dx),
            dy=float(args.dy),
            dz=float(args.dz),
            roll_deg=float(args.roll_deg),
            pitch_deg=float(args.pitch_deg),
            yaw_deg=float(args.yaw_deg),
            repeat=int(args.repeat),
            bidirectional=not bool(args.one_way),
        )
        i2rt_rel_traj = headpose_base_to_i2rt_rel(
            base_rel_traj,
            T_base_cam,
            T_i2rt_tcp,
        )

        print(f"[INFO] T_base_cam:\n{T_base_cam}")
        print(f"[INFO] T_i2rt_tcp:\n{T_i2rt_tcp}")
        print(f"[INFO] base_rel_traj shape={base_rel_traj.shape}\n{base_rel_traj}")
        print(f"[INFO] i2rt_rel_traj shape={i2rt_rel_traj.shape}\n{i2rt_rel_traj}")

        if args.dry_run:
            print("[INFO] Dry run enabled, skip execute_pred_tcp_rel.")
            return

        print("[INFO] Executing converted i2rt relative trajectory via execute_pred_tcp_rel...")
        hw.execute_pred_tcp_rel(i2rt_rel_traj)
        print("[INFO] Done.")
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received, exiting.")
    finally:
        hw.close(timeout_s=5.0)
        if rclpy.ok():
            rclpy.shutdown()
        if hw.i2rt_server_proc is not None and hw.i2rt_server_proc.is_alive():
            hw.i2rt_server_proc.terminate()
            hw.i2rt_server_proc.join(timeout=2.0)


if __name__ == "__main__":
    main()
