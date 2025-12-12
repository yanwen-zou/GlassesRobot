#!/usr/bin/env python3
"""
Delete intermediate data folders under episode directories.

Usage:
    python cleanup_intermediate.py --data-root /path/to/data

By default removes: cam_pose_in_ball, masks_balls, jpg
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_TARGETS = ("cam_pose_in_ball", "masks_balls", "jpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean intermediate folders in data episodes.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing episode subdirectories.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="Folder names to delete under each episode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be deleted without removing anything.",
    )
    return parser.parse_args()


def clean_episode(ep_path: Path, targets: list[str], dry_run: bool) -> int:
    removed = 0
    for name in targets:
        tgt = ep_path / name
        if tgt.exists():
            print(f"{'[DRY-RUN] ' if dry_run else ''}Removing {tgt}")
            if not dry_run:
                shutil.rmtree(tgt, ignore_errors=False)
            removed += 1
    return removed


def main() -> None:
    args = parse_args()
    root = args.data_root
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")

    targets = [t.strip() for t in args.targets if t.strip()]
    if not targets:
        raise ValueError("No targets specified to delete.")

    total_removed = 0
    episodes = [p for p in root.iterdir() if p.is_dir()]
    if not episodes:
        print(f"No episode directories found under {root}")
        return

    for ep in sorted(episodes):
        removed = clean_episode(ep, targets, args.dry_run)
        total_removed += removed

    print(
        f"{'Would remove' if args.dry_run else 'Removed'} {total_removed} folder(s) "
        f"across {len(episodes)} episode(s)."
    )


if __name__ == "__main__":
    main()
