#!/usr/bin/env python3
"""
Visualize paired Flexiv + XVisio SLAM trajectories recorded by flexiv_slam_record.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _import_rerun():
    try:
        import rerun as rr  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("rerun-sdk is required: pip install rerun-sdk") from exc
    return rr


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion (w, x, y, z) -> rotation matrix."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def load_samples(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                wall_time = float(row["wall_time"])
                confidence = float(row["slam_confidence"])
            except (ValueError, KeyError):
                continue
            if confidence <= 0:
                continue
            try:
                flexiv_pose = np.array(
                    [
                        float(row["flexiv_x"]),
                        float(row["flexiv_y"]),
                        float(row["flexiv_z"]),
                        float(row["flexiv_rw"]),
                        float(row["flexiv_rx"]),
                        float(row["flexiv_ry"]),
                        float(row["flexiv_rz"]),
                    ],
                    dtype=np.float32,
                )
                slam_pose = np.array(
                    [
                        float(row["slam_x"]),
                        float(row["slam_y"]),
                        float(row["slam_z"]),
                        float(row["slam_qw"]),
                        float(row["slam_qx"]),
                        float(row["slam_qy"]),
                        float(row["slam_qz"]),
                    ],
                    dtype=np.float32,
                )
            except ValueError:
                continue
            rows.append(
                {
                    "time": wall_time,
                    "repeat_idx": row.get("repeat_idx", ""),
                    "direction": row.get("direction", ""),
                    "flexiv_pose": flexiv_pose,
                    "slam_pose": slam_pose,
                }
            )
    if not rows:
        raise ValueError(f"No valid samples found in {csv_path}")
    return rows


def visualize(csv_path: Path, spawn: bool) -> None:
    samples = load_samples(csv_path)

    rr = _import_rerun()
    rr.init(f"Flexiv vs SLAM ({csv_path.name})", spawn=spawn)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    flexiv_path: List[np.ndarray] = []
    slam_path: List[np.ndarray] = []
    for row in samples:
        time_sec = row["time"]
        rr.set_time_seconds("wall_time", time_sec)
        flexiv_pose = row["flexiv_pose"]
        slam_pose = row["slam_pose"]
        flexiv_path.append(flexiv_pose[:3].astype(np.float32))
        slam_path.append(slam_pose[:3].astype(np.float32))
        flexiv_R = quaternion_to_matrix(
            flexiv_pose[3], flexiv_pose[4], flexiv_pose[5], flexiv_pose[6]
        )
        slam_R = quaternion_to_matrix(
            slam_pose[3], slam_pose[4], slam_pose[5], slam_pose[6]
        )
        rr.log(
            "flexiv/trajectory",
            rr.LineStrips3D(
                [np.stack(flexiv_path)],
                colors=np.array([[255, 150, 0, 255]], dtype=np.uint8),
                radii=np.full(1, 0.003, dtype=np.float32),
            ),
        )
        rr.log(
            "slam/trajectory",
            rr.LineStrips3D(
                [np.stack(slam_path)],
                colors=np.array([[0, 200, 255, 255]], dtype=np.uint8),
                radii=np.full(1, 0.003, dtype=np.float32),
            ),
        )
        rr.log(
            "flexiv/pose",
            rr.Transform3D(
                translation=flexiv_pose[:3],
                mat3x3=flexiv_R,
            ),
        )
        rr.log(
            "slam/pose",
            rr.Transform3D(
                translation=slam_pose[:3],
                mat3x3=slam_R,
            ),
        )
        rr.log(
            "flexiv/point",
            rr.Points3D(
                flexiv_pose[:3][None, :],
                colors=np.array([[255, 150, 0, 255]], dtype=np.uint8),
                radii=np.full(1, 0.005, dtype=np.float32),
            ),
        )
        rr.log(
            "slam/point",
            rr.Points3D(
                slam_pose[:3][None, :],
                colors=np.array([[0, 200, 255, 255]], dtype=np.uint8),
                radii=np.full(1, 0.005, dtype=np.float32),
            ),
        )
        rr.log(
            "annotations/direction",
            rr.TextLog(f"repeat={row['repeat_idx']} dir={row['direction']}"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize paired Flexiv and SLAM trajectories with rerun."
    )
    parser.add_argument(
        "log_csv",
        type=Path,
        help="CSV created by flexiv_slam_record.py",
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Spawn a separate rerun viewer window.",
    )
    args = parser.parse_args()
    visualize(args.log_csv, spawn=args.spawn)


if __name__ == "__main__":
    main()
