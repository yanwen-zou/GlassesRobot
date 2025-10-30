#!/usr/bin/env python3
"""
Quick visual QA for FoundationPose results.
Plays each episode's `foundationpose_vis.mp4` under data/train.
Press:
  p - mark as pass and continue to the next episode
  d - delete the entire episode directory and continue
  q - quit the script immediately
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2


def iter_episode_dirs(data_root: Path):
    for path in sorted(data_root.iterdir()):
        if path.is_dir():
            yield path


def play_video(video_path: Path) -> str | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ Failed to open video: {video_path}")
        return None

    window_name = f"FoundationPose QA - {video_path.parent.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    decision = None
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("p"):
            decision = "pass"
            break
        if key == ord("d"):
            decision = "delete"
            break
        if key == ord("q"):
            decision = "quit"
            break

    cap.release()
    cv2.destroyWindow(window_name)
    return decision


def delete_episode(episode_dir: Path):
    print(f"🗑️ Deleting episode: {episode_dir}")
    shutil.rmtree(episode_dir)


def main():
    parser = argparse.ArgumentParser(description="Review FoundationPose videos and prune failures.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/train"),
        help="Root directory containing per-episode subdirectories (default: data/train)",
    )
    args = parser.parse_args()

    data_root = args.data_root
    if not data_root.exists():
        print(f"❌ Data root not found: {data_root}")
        return 1

    for episode_dir in iter_episode_dirs(data_root):
        video_path = episode_dir / "foundationpose_vis.mp4"
        if not video_path.exists():
            print(f"⚠️ Skipping (video missing): {video_path}")
            continue

        print(f"▶️  Reviewing {video_path}")
        decision = play_video(video_path)
        if decision is None:
            continue
        if decision == "quit":
            print("👋 Exiting by user request.")
            return 0
        if decision == "delete":
            delete_episode(episode_dir)

    print("✅ Review complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
