#!/usr/bin/env python3
"""
Remove the first 25 frames from zed_right* and head_pos* directories under each
sequence directory.

Directories are processed independently; if a directory has fewer than 26 frame
files nothing is deleted. Remaining frames are renumbered so indices stay
contiguous from 000000.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

FRAME_NAME_PATTERN = re.compile(r"^\d{6}\.[^.]+$")
TARGET_DIR_PREFIXES = ("zed_right", "head_pos")
FRAMES_TO_DROP = 25


def iter_target_dirs(seq_dir: Path) -> Iterable[Path]:
    for child in sorted(seq_dir.iterdir()):
        if not child.is_dir():
            continue
        if any(child.name.startswith(prefix) for prefix in TARGET_DIR_PREFIXES):
            yield child


def list_frame_files(frame_dir: Path) -> List[Path]:
    return sorted(
        (f for f in frame_dir.iterdir() if f.is_file() and FRAME_NAME_PATTERN.match(f.name)),
        key=lambda p: p.name,
    )


def delete_front_frames(files: List[Path], count: int, dry_run: bool) -> None:
    to_remove = files[:count]
    for path in to_remove:
        if dry_run:
            print(f"  DRY-RUN delete {path}")
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                print(f"  WARNING missing frame file {path.name}")


def rename_remaining(frame_dir: Path, dry_run: bool) -> None:
    remaining = list_frame_files(frame_dir)
    temp_paths: List[Path] = []
    for idx, original in enumerate(remaining):
        temp_path = original.with_name(f"__tmp_{idx:06d}{original.suffix}")
        if dry_run:
            print(f"  DRY-RUN rename {original} -> {temp_path.name}")
        else:
            original.rename(temp_path)
        temp_paths.append(temp_path)

    for idx, temp in enumerate(temp_paths):
        final_path = temp.with_name(f"{idx:06d}{temp.suffix}")
        if dry_run:
            print(f"  DRY-RUN rename {temp.name} -> {final_path.name}")
        else:
            temp.rename(final_path)


def process_sequence(seq_dir: Path, dry_run: bool) -> None:
    for frame_dir in iter_target_dirs(seq_dir):
        files = list_frame_files(frame_dir)
        frame_count = len(files)
        if frame_count <= FRAMES_TO_DROP:
            print(f"{seq_dir.name}/{frame_dir.name}: only {frame_count} frames, skipping")
            continue

        print(
            f"{seq_dir.name}/{frame_dir.name}: drop first {FRAMES_TO_DROP} of {frame_count} frames"
        )
        delete_front_frames(files, FRAMES_TO_DROP, dry_run=dry_run)
        rename_remaining(frame_dir, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim the first 25 frames from zed_right*/head_pos* directories."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data_lion/train"),
        help="Root directory containing sequence subdirectories (default: data_lion/train)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display planned operations without deleting or renaming files.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Root directory {root} does not exist")

    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        process_sequence(seq_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
