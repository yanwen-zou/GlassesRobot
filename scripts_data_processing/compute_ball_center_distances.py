#!/usr/bin/env python3
"""
Read ball_center.txt under a given data path and compute per-frame distances
between ball centers: (1-2) and (2-3).
"""
import argparse
import math
from pathlib import Path
from typing import Dict, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-frame distances between ball centers 1-2 and 2-3."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Episode directory or ball_center.txt path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output txt path; if omitted, results are printed to stdout.",
    )
    return parser.parse_args()


def load_ball_centers(ball_file: Path) -> Dict[int, Dict[int, Tuple[float, float, float]]]:
    frames: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
    with ball_file.open("r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            frame_id, ball_id = int(parts[0]), int(parts[1])
            x, y, z = map(float, parts[2:])
            frames.setdefault(frame_id, {})[ball_id] = (x, y, z)
    return frames


def dist(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def main():
    args = parse_args()

    ball_file = args.data_path
    if ball_file.is_dir():
        ball_file = ball_file / "ball_centers.txt"
    if not ball_file.exists():
        raise FileNotFoundError(f"ball_centers.txt not found at {ball_file}")

    frames = load_ball_centers(ball_file)
    lines = []
    sum_d12 = 0.0
    sum_d23 = 0.0
    count = 0
    for frame_id in sorted(frames.keys()):
        centers = frames[frame_id]
        missing = []
        for required in (1, 2, 3):
            if required not in centers:
                missing.append(required)
        if missing:
            lines.append(f"{frame_id} missing balls {missing}")
            continue
        d12 = dist(centers[1], centers[2])
        d23 = dist(centers[2], centers[3])
        lines.append(f"{frame_id} d12={d12:.6f} d23={d23:.6f}")
        sum_d12 += d12
        sum_d23 += d23
        count += 1

    if count > 0:
        avg_d12 = sum_d12 / count
        avg_d23 = sum_d23 / count
        lines.append(f"average d12={avg_d12:.6f} d23={avg_d23:.6f} over {count} frames")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n")
        print(f"Saved distances to {args.output}")
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
