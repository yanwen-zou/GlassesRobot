#!/usr/bin/env python3
"""
Quick helper to inspect numpy .npy files from the command line.
"""

import argparse
import math
import pathlib
from typing import Iterable, Optional, Tuple

import numpy as np


def describe_array(path: pathlib.Path, verbose: bool) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    print(f"=== {path} ===")
    print(f"type: {type(data).__name__}")
    if isinstance(data, np.ndarray):
        print(f"dtype: {data.dtype}")
        print(f"shape: {data.shape}")
        if verbose:
            print("values:")
            print(data)
    else:
        print("value:")
        print(data)
    print()
    return data


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rot = np.eye(4, dtype=np.float64)
    rot[:3, :3] = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    return rot


def matrix_to_euler_xyz(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Return XYZ Euler angles (radians) from rotation matrix."""
    r = matrix[:3, :3]
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0
    return x, y, z


def print_transform_summary(name: str, matrix: np.ndarray) -> None:
    translation = matrix[:3, 3]
    rx, ry, rz = matrix_to_euler_xyz(matrix)
    print(f"[{name}] translation (m): {translation}")
    print(f"[{name}] rotation matrix:\n{matrix[:3, :3]}")
    print(
        f"[{name}] Euler XYZ (deg): "
        f"({math.degrees(rx):.2f}, {math.degrees(ry):.2f}, {math.degrees(rz):.2f})"
    )


def plot_frames(
    frames: Iterable[Tuple[str, np.ndarray]],
    output_path: Optional[pathlib.Path] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (register 3D projection)
    except ImportError as exc:  # pragma: no cover - plotting optional
        raise RuntimeError("matplotlib is required for plotting. Install with `pip install matplotlib`.") from exc

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Transform Frames")

    max_extent = 0.0
    for name, matrix in frames:
        origin = matrix[:3, 3]
        axes = matrix[:3, :3]
        length = 0.1
        colors = ["r", "g", "b"]
        for axis_idx in range(3):
            endpoint = origin + axes[:, axis_idx] * length
            ax.plot(
                [origin[0], endpoint[0]],
                [origin[1], endpoint[1]],
                [origin[2], endpoint[2]],
                color=colors[axis_idx],
            )
            ax.text(*(endpoint), f"{name}_{'xyz'[axis_idx]}", color=colors[axis_idx])
        ax.scatter(*origin, color="k", s=20)
        ax.text(*origin, name)
        max_extent = max(max_extent, np.linalg.norm(origin) + length)

    lim = max_extent if max_extent > 0 else 0.2
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.view_init(elev=25, azim=45)
    ax.set_box_aspect([1, 1, 1])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Plot saved to {output_path}")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect npy files by printing their type, shape, and optionally contents."
    )
    parser.add_argument("paths", nargs="*", help="Path(s) to .npy file(s).")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full array contents (may be large).",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Visualize transform frames (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-output",
        metavar="PNG_PATH",
        help="Save the plotted frames to a PNG (implies --plot).",
    )
    args = parser.parse_args()

    if args.plot_output:
        args.plot = True

    default_path = pathlib.Path("glasses_hardware/calib/eih_camT.npy").resolve()
    input_paths = args.paths if args.paths else [default_path]

    for path_str in input_paths:
        path = pathlib.Path(path_str).expanduser().resolve()
        if not path.exists():
            print(f"[WARN] {path} does not exist, skipping.")
            continue
        try:
            data = describe_array(path, args.verbose)
        except Exception as err:  # pylint: disable=broad-except
            print(f"[ERROR] Failed to read {path}: {err}")
            continue

        rotated_matrix = None
        if args.rotate_y90 and isinstance(data, np.ndarray) and data.shape == (4, 4):
            print_transform_summary("original", data)
            rot_y90 = rotation_matrix_y(math.radians(90.0))
            rotated_matrix = data @ rot_y90
            print_transform_summary("rotated_y+90deg", rotated_matrix)
            if args.save_rotated is not None:
                save_path = pathlib.Path(args.save_rotated).expanduser().resolve()
            else:
                save_path = path.with_name(f"{path.stem}_ry90.npy")
            np.save(save_path, rotated_matrix)
            print(f"[INFO] Rotated transform saved to {save_path}")
        elif isinstance(data, np.ndarray) and data.shape == (4, 4):
            print_transform_summary("transform", data)

        if args.plot and isinstance(data, np.ndarray) and data.shape == (4, 4):
            frames = [("original", data)]
            if rotated_matrix is not None:
                frames.append(("rotated", rotated_matrix))
            try:
                plot_frames(frames, pathlib.Path(args.plot_output).expanduser().resolve() if args.plot_output else None)
            except RuntimeError as exc:
                print(f"[WARN] {exc}")


if __name__ == "__main__":
    main()
