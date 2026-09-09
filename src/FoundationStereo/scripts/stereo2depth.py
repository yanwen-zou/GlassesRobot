"""Batch stereo-to-depth conversion backed by Fast-FoundationStereo.

This file stays in the legacy FoundationStereo tools directory so existing
data-processing shell scripts keep working unchanged.
"""

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.egodata_eval.get_depth import DepthEstimator  # noqa: E402


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def sorted_image_files(directory: str) -> List[str]:
    files = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.splitext(name.lower())[1] in IMAGE_EXTS
    ]

    def sort_key(path: str) -> Tuple[int, object]:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    return sorted(files, key=sort_key)


def frame_source(
    left_path: str, right_path: str
) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    if os.path.isdir(left_path) and os.path.isdir(right_path):
        left_files = sorted_image_files(left_path)
        right_files = sorted_image_files(right_path)
        if not left_files or not right_files:
            raise RuntimeError("Left/right directories must contain image files")
        if len(left_files) != len(right_files):
            logging.warning(
                "Mismatch between left (%d) and right (%d) frame counts; using minimum",
                len(left_files),
                len(right_files),
            )
        for left_file, right_file in zip(left_files, right_files):
            left = cv2.imread(left_file, cv2.IMREAD_COLOR)
            right = cv2.imread(right_file, cv2.IMREAD_COLOR)
            if left is None or right is None:
                logging.warning("Failed to read pair (%s, %s); skipping", left_file, right_file)
                continue
            yield os.path.splitext(os.path.basename(left_file))[0], left, right
        return

    cap_left = cv2.VideoCapture(left_path)
    cap_right = cv2.VideoCapture(right_path)
    if not cap_left.isOpened() or not cap_right.isOpened():
        raise RuntimeError(f"Failed to open videos {left_path} / {right_path}")
    frame_id = 0
    try:
        while True:
            ret_left, left = cap_left.read()
            ret_right, right = cap_right.read()
            if not (ret_left and ret_right):
                break
            yield f"{frame_id:06d}", left, right
            frame_id += 1
    finally:
        cap_left.release()
        cap_right.release()


def main(
    left_file: str,
    right_file: str,
    out_dir: str,
    checkpoint: Optional[Path] = None,
    valid_iters: int = 4,
    max_disp: int = 192,
) -> None:
    os.makedirs(f"{out_dir}/rgb", exist_ok=True)
    os.makedirs(f"{out_dir}/depth", exist_ok=True)
    os.makedirs(f"{out_dir}/depth_vis", exist_ok=True)
    estimator = DepthEstimator(
        ckpt_path=checkpoint, valid_iters=valid_iters, max_disp=max_disp
    )

    processed = 0
    for frame_key, left, right in frame_source(left_file, right_file):
        print(f"Processing frame {frame_key}", flush=True)
        depth = estimator.depth(left, right)
        depth_mm = np.nan_to_num(
            depth * 1000.0, nan=0.0, posinf=0.0, neginf=0.0
        ).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)

        cv2.imwrite(os.path.join(out_dir, "rgb", f"{frame_key}.png"), left)
        cv2.imwrite(os.path.join(out_dir, "depth", f"{frame_key}.png"), depth_mm)
        depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(out_dir, "depth_vis", f"{frame_key}.png"), depth_color)
        processed += 1

    print(f"Saved {processed} frames to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_file", required=True, help="Left video or image directory")
    parser.add_argument("--right_file", required=True, help="Right video or image directory")
    parser.add_argument("--out_dir", required=True, help="Output episode directory")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--valid-iters", type=int, default=4)
    parser.add_argument("--max-disp", type=int, default=192)
    cli_args = parser.parse_args()
    main(
        cli_args.left_file,
        cli_args.right_file,
        cli_args.out_dir,
        checkpoint=cli_args.checkpoint,
        valid_iters=cli_args.valid_iters,
        max_disp=cli_args.max_disp,
    )
