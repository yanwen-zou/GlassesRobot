import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)


def list_sorted_stems(dir_path: str) -> List[str]:
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(".txt")]
    stems = [os.path.splitext(f)[0] for f in files]

    def key_fn(s: str):
        try:
            return int(s)
        except Exception:
            return s

    return sorted(list(set(stems)), key=key_fn)


def xyz_quat7_to_mat(vals: np.ndarray) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = vals.astype(np.float64)
    # normalize quaternion
    n = np.linalg.norm([qx, qy, qz, qw])
    if n == 0:
        raise ValueError("Zero-norm quaternion")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def load_head_traj(head_dir: str, relative: bool = True, downsample: int = 1) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Load head poses from directory.

    Returns:
      - Nx3 array of positions (meters)
      - Optional list of rotation matrices for each sample
    """
    stems = list_sorted_stems(head_dir)
    if not stems:
        raise FileNotFoundError(f"No head poses found in {head_dir}")

    Ts = []
    for s in stems:
        p = os.path.join(head_dir, f"{s}.txt")
        vals = np.loadtxt(p).astype(np.float64)
        if vals.ndim != 1 or vals.size != 7:
            raise ValueError(f"Expect 7 values in {p}, got shape {vals.shape}")
        Ts.append(xyz_quat7_to_mat(vals))

    if relative:
        T0_inv = np.linalg.inv(Ts[0])
        Ts = [T0_inv @ Ti for Ti in Ts]

    if downsample > 1:
        Ts = Ts[::downsample]

    pts = np.stack([T[:3, 3] for T in Ts], axis=0)
    Rs = [T[:3, :3].copy() for T in Ts]
    return pts, Rs


def set_equal_3d(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    if max_range == 0:
        max_range = 1.0
    Xb = 0.5 * max_range
    x_mid = (X.max() + X.min()) * 0.5
    y_mid = (Y.max() + Y.min()) * 0.5
    z_mid = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(x_mid - Xb, x_mid + Xb)
    ax.set_ylim(y_mid - Xb, y_mid + Xb)
    ax.set_zlim(z_mid - Xb, z_mid + Xb)


def plot_traj(pts: np.ndarray, Rs: List[np.ndarray], show_axes_every: int = 20, save: Path | None = None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Color by time along the path
    t = np.linspace(0, 1, len(pts))
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='gray', alpha=0.6, linewidth=1.5)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=t, cmap=cm.viridis, s=6)

    # Draw small orientation axes every N samples
    axis_len = max(1e-3, 0.05 * np.linalg.norm(pts[-1] - pts[0]))  # 5% of total displacement
    for i in range(0, len(pts), max(1, show_axes_every)):
        R = Rs[i]
        p = pts[i]
        for col, color in zip(range(3), ['r', 'g', 'b']):
            v = R[:, col] * axis_len
            ax.plot([p[0], p[0] + v[0]], [p[1], p[1] + v[1]], [p[2], p[2] + v[2]], color=color, linewidth=1)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Head Trajectory')
    set_equal_3d(ax, pts[:, 0], pts[:, 1], pts[:, 2])
    ax.view_init(elev=25, azim=-60)

    if save is not None:
        fig.tight_layout()
        fig.savefig(str(save), dpi=200)
        print(f"Saved: {save}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize head_pos trajectory in 3D')
    parser.add_argument('timestamp', type=str, help='Sequence under data/train, e.g., 20251029_192600')
    parser.add_argument('--data_root', type=str, default='data/train', help='Root folder of sequences')
    parser.add_argument('--absolute', action='store_true', help='Plot absolute poses (default: relative to first)')
    parser.add_argument('--downsample', type=int, default=1, help='Use every Nth frame for plotting')
    parser.add_argument('--axes_every', type=int, default=20, help='Draw small orientation axes every N frames')
    parser.add_argument('--save', type=str, default=None, help='Path to save the figure instead of showing')
    args = parser.parse_args()

    head_dir = Path(args.data_root) / args.timestamp / 'head_pos'
    if not head_dir.is_dir():
        raise FileNotFoundError(f"Missing head_pos directory: {head_dir}")

    pts, Rs = load_head_traj(str(head_dir), relative=not args.absolute, downsample=max(1, args.downsample))
    save_path = Path(args.save) if args.save else None
    plot_traj(pts, Rs, show_axes_every=max(1, args.axes_every), save=save_path)


if __name__ == '__main__':
    main()

