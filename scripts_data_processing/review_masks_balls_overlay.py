#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np


def _sorted_frame_stems(files: List[str]) -> List[str]:
    stems = [os.path.splitext(f)[0] for f in files]

    def sort_key(name: str):
        try:
            return (0, int(name))
        except ValueError:
            return (1, name)

    return sorted(set(stems), key=sort_key)


def _load_mask_map(mask_dir: str) -> Dict[str, List[str]]:
    if not os.path.isdir(mask_dir):
        return {}
    mask_files = [
        f for f in os.listdir(mask_dir)
        if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg"}
    ]
    mask_map: Dict[str, List[str]] = {}
    for fname in mask_files:
        stem = os.path.splitext(fname)[0]
        frame_id = stem.split("_", 1)[0]
        mask_map.setdefault(frame_id, []).append(os.path.join(mask_dir, fname))
    return mask_map


def _overlay_mask(
    bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.4,
) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_bin = mask > 0
    if not np.any(mask_bin):
        return bgr
    overlay = bgr.copy()
    overlay[mask_bin] = color
    blended = bgr.copy()
    blended[mask_bin] = cv2.addWeighted(
        bgr[mask_bin], 1.0 - alpha, overlay[mask_bin], alpha, 0
    )
    return blended


def _find_image_path(dir_path: str, stem: str) -> Optional[str]:
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(dir_path, f"{stem}{ext}")
        if os.path.exists(path):
            return path
    return None


def review_episode(episode_dir: str, fps: float) -> str:
    rgb_dir = os.path.join(episode_dir, "rgb")
    mask_dir = os.path.join(episode_dir, "masks_balls")
    if not os.path.isdir(rgb_dir):
        return "skip"
    rgb_files = [
        f for f in os.listdir(rgb_dir)
        if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not rgb_files:
        return "skip"

    stems = _sorted_frame_stems(rgb_files)
    mask_map = _load_mask_map(mask_dir)
    # Review only the first frame in this episode.
    stem = stems[0]
    rgb_path = _find_image_path(rgb_dir, stem)
    if rgb_path is None:
        return "skip"
    frame = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if frame is None:
        return "skip"

    mask_paths = mask_map.get(stem)
    if mask_paths:
        colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
        ]
        for mask_path in mask_paths:
            if not os.path.exists(mask_path):
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            stem_name = os.path.splitext(os.path.basename(mask_path))[0]
            parts = stem_name.split("_", 1)
            color_idx = 0
            if len(parts) == 2 and parts[1].startswith("id"):
                try:
                    color_idx = int(parts[1][2:]) - 1
                except ValueError:
                    color_idx = 0
            color = colors[color_idx % len(colors)]
            frame = _overlay_mask(frame, mask, color)

    label = f"{os.path.basename(episode_dir)}  frame={stem}"
    cv2.putText(
        frame,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    delay_ms = max(1, int(1000.0 / fps)) if fps > 0 else 1
    while True:
        cv2.imshow("masks_balls review (p=pass, d=delete, q=quit)", frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("p"):
            return "pass"
        if key == ord("d"):
            return "delete"
        if key in (ord("q"), 27):
            return "quit"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Review masks_balls overlay on rgb frames; p=pass, d=delete, q=quit."
    )
    ap.add_argument(
        "--data-root",
        default="data",
        help="Root directory containing episode folders.",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback FPS (0 for fastest).",
    )
    args = ap.parse_args()

    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(os.getcwd(), data_root)
    data_root = os.path.realpath(data_root)
    if not os.path.isdir(data_root):
        print(f"[ERROR] Data root not found: {data_root}", file=sys.stderr)
        return 1

    episodes = [
        os.path.join(data_root, d)
        for d in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, d))
    ]
    if not episodes:
        print(f"[WARN] No episodes under {data_root}")
        return 0

    for episode_dir in episodes:
        action = review_episode(episode_dir, args.fps)
        if action == "delete":
            shutil.rmtree(episode_dir)
            print(f"[INFO] Deleted {episode_dir}")
        elif action == "quit":
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
