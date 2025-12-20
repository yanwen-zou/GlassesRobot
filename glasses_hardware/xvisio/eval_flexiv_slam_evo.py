#!/usr/bin/env python3
"""
Compare Flexiv TCP and XVisio SLAM trajectories using evo, and visualize aligned 3D paths.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _import_evo():
    try:
        from evo.core import metrics, trajectory  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Please install evo: `pip install evo`") from exc
    return metrics, trajectory


def _import_rerun():
    try:
        import rerun as rr  # type: ignore
    except Exception:
        return None
    return rr


REQUIRED_COLUMNS = {
    "wall_time",
    "flexiv_x",
    "flexiv_y",
    "flexiv_z",
    "slam_x",
    "slam_y",
    "slam_z",
}


def load_positions(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, flexiv_xyz, slam_xyz = [], [], []
    with csv_path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            raise ValueError(f"{csv_path} missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                t = float(row["wall_time"])
                flexiv = [float(row["flexiv_x"]), float(row["flexiv_y"]), float(row["flexiv_z"])]
                slam = [float(row["slam_x"]), float(row["slam_y"]), float(row["slam_z"])]
            except ValueError:
                continue
            times.append(t)
            flexiv_xyz.append(flexiv)
            slam_xyz.append(slam)
    if not times:
        raise ValueError(f"No valid samples in {csv_path}")
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(flexiv_xyz, dtype=np.float64),
        np.asarray(slam_xyz, dtype=np.float64),
    )


def build_traj(trajectory_module, times: np.ndarray, positions: np.ndarray):
    orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (len(positions), 1))
    return trajectory_module.PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=orientations,
        timestamps=times,
    )


def umeyama_align(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return rotation R and translation t that aligns src -> dst (no scaling)."""
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = src_centered.T @ dst_centered / src.shape[0]
    U, _, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_mean - R @ src_mean
    return R, t


def print_stats(label: str, values: np.ndarray) -> None:
    print(f"\n[{label}]")
    print(f"  RMSE : {np.sqrt(np.mean(values ** 2)):.6f} m")
    print(f"  Mean : {np.mean(values):.6f} m")
    print(f"  Median: {np.median(values):.6f} m")
    print(f"  Std  : {np.std(values):.6f} m")
    print(f"  Min  : {np.min(values):.6f} m")
    print(f"  Max  : {np.max(values):.6f} m")


def visualize_rerun(csv_path: Path, flexiv: np.ndarray, slam_aligned: np.ndarray, spawn: bool) -> None:
    rr = _import_rerun()
    if rr is None:
        print("\n[Note] rerun-sdk not installed; skipping 3D visualization.")
        return
    rr.init(f"Flexiv vs SLAM (aligned) - {csv_path.name}", spawn=spawn)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass
    rr.log(
        "flexiv/trajectory",
        rr.LineStrips3D([flexiv.astype(np.float32)], colors=np.array([[255, 150, 0, 255]], dtype=np.uint8)),
    )
    slam_vis = slam_aligned.copy()
    #slam_vis[:, 1] *= -1.0
    rr.log(
        "slam/trajectory_aligned",
        rr.LineStrips3D([slam_vis.astype(np.float32)], colors=np.array([[0, 200, 255, 255]], dtype=np.uint8)),
    )


def evaluate(csv_path: Path, spawn: bool) -> None:
    metrics_mod, traj_mod = _import_evo()
    times, flexiv_pos, slam_pos = load_positions(csv_path)

    slam_eval = slam_pos.copy()
    slam_eval[:, 1] *= -1.0

    traj_ref = build_traj(traj_mod, times, flexiv_pos)
    traj_est = build_traj(traj_mod, times, slam_eval)

    ape_metric = metrics_mod.APE(metrics_mod.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))
    raw_errors = np.linalg.norm(flexiv_pos - slam_eval, axis=1)
    print_stats("Raw Flexiv-SLAM translation error (with Y flipped)", raw_errors)

    R, t = umeyama_align(slam_eval, flexiv_pos)
    slam_aligned = (slam_eval @ R.T) + t
    aligned_errors = np.linalg.norm(flexiv_pos - slam_aligned, axis=1)
    print_stats("Aligned error", aligned_errors)

    print("\nSample aligned errors (time, |flexiv-slam_aligned|):")
    for idx in range(min(10, len(aligned_errors))):
        print(f"  t={times[idx]:.3f} -> {aligned_errors[idx]:.6f} m")
    if len(aligned_errors) > 10:
        print(f"  ... ({len(aligned_errors)} samples)")

    visualize_rerun(csv_path, flexiv_pos, slam_aligned, spawn=spawn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Flexiv and SLAM trajectories (translation only) and visualize aligned paths."
    )
    parser.add_argument("data_path", type=Path, help="CSV from flexiv_slam_record.py")
    parser.add_argument("--spawn", action="store_true", help="Spawn a rerun viewer window if available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.data_path, spawn=args.spawn)


if __name__ == "__main__":
    main()
