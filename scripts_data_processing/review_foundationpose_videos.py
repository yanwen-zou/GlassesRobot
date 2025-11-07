#!/usr/bin/env python3
"""
Quickly review every data/train/<timestamp>/foundationpose_vis.mp4 file and
press:
  p - keep current sample and go to the next timestamp
  d - delete the entire <timestamp> directory
  q - quit the script immediately
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="遍历播放 data/train/<timestamp>/foundationpose_vis.mp4"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/train"),
        help="指向 data/train 的根目录",
    )
    parser.add_argument(
        "--video-name",
        default="foundationpose_vis.mp4",
        help="要播放的视频文件名，默认 foundationpose_vis.mp4",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=30,
        help="cv2.waitKey 的延迟（毫秒），默认 30ms",
    )
    return parser.parse_args()


def iter_videos(root: Path, video_name: str):
    for timestamp_dir in sorted(root.iterdir()):
        if not timestamp_dir.is_dir():
            continue
        video_path = timestamp_dir / video_name
        if video_path.is_file():
            yield timestamp_dir, video_path


def play_video(video_path: Path, delay: int) -> str:
    window_name = "foundationpose_vis"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 无法打开 {video_path}")
        return "skip"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    action = "skip"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("p"):
                action = "skip"
                break
            if key == ord("d"):
                action = "delete"
                break
            if key == ord("q"):
                action = "quit"
                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
    return action


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists():
        print(f"[ERR] {root} 不存在")
        sys.exit(1)

    print("按 'p' 跳到下一个，按 'd' 删除当前 <timestamp> 目录，按 'q' 退出脚本")

    try:
        for timestamp_dir, video_path in iter_videos(root, args.video_name):
            print(f"\n正在播放: {timestamp_dir.name}")
            action = play_video(video_path, args.delay)

            if action == "delete":
                print(f"删除目录: {timestamp_dir}")
                shutil.rmtree(timestamp_dir)
            elif action == "quit":
                print("用户退出。")
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
