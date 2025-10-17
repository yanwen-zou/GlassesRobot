#!/usr/bin/env python3
"""Visualize trajectories saved in TUM pose format."""
import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def load_tum_file(path: Path) -> np.ndarray:
    """Load a TUM-style pose log (t, x, y, z, qx, qy, qz, qw)."""
    rows: List[List[float]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            try:
                rows.append([float(p) for p in parts[:8]])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f'No pose data could be read from {path}')
    return np.array(rows)


def plot_positions(ax, t: np.ndarray, data: np.ndarray, labels: Iterable[str]) -> None:
    for idx, label in enumerate(labels):
        ax.plot(t, data[:, idx], label=label)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.grid(True)
    ax.legend(loc='upper right')


def plot_trajectory(path: Path, show_orientation: bool = False) -> None:
    poses = load_tum_file(path)
    t = poses[:, 0] - poses[0, 0]
    positions = poses[:, 1:4]

    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(2, 2, (1, 3), projection='3d')
    ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label=path.name)
    ax3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', marker='o', label='start')
    ax3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', marker='x', label='end')
    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Z [m]')
    ax3d.legend()
    ax3d.set_title('3D trajectory')
    ax3d.grid(True)

    ax_pos = fig.add_subplot(2, 2, 2)
    plot_positions(ax_pos, t, positions, labels=('x', 'y', 'z'))
    ax_pos.set_title('Position vs. time')

    if show_orientation:
        orientations = poses[:, 4:8]
        ax_orient = fig.add_subplot(2, 2, 4)
        plot_positions(ax_orient, t, orientations, labels=('qx', 'qy', 'qz', 'qw'))
        ax_orient.set_ylabel('Quaternion')
        ax_orient.set_title('Orientation vs. time')
    else:
        ax_blank = fig.add_subplot(2, 2, 4)
        ax_blank.axis('off')

    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize a TUM pose file.')
    parser.add_argument('pose_file', type=Path, help='Path to the pose log (e.g. glasses_pose_*.txt).')
    parser.add_argument('--show-orientation', action='store_true', help='Plot quaternion components alongside position.')
    args = parser.parse_args()

    if not args.pose_file.exists():
        raise SystemExit(f'Pose file {args.pose_file} does not exist.')

    plot_trajectory(args.pose_file, show_orientation=args.show_orientation)


if __name__ == '__main__':
    main()
