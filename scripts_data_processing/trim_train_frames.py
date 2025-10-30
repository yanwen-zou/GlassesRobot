#!/usr/bin/env python3
"""
Trim leading and trailing frames for each sequence under data_lion/train.

By default removes the first 25 and last 20 frames from every directory that
contains six-digit frame files (e.g. 000123.jpg) and then renames the remaining
files so they start again from 000000 while preserving the original order. Each
directory is processed independently even if other directories in the same
sequence have different frame counts.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple


FRAME_NAME_PATTERN = re.compile(r"^\d{6}\.[^.]+$")


def find_frame_directories(seq_dir: Path) -> List[Tuple[Path, List[Path]]]:
    """Return directories that contain six-digit frame files under seq_dir."""
    frame_dirs: List[Tuple[Path, List[Path]]] = []
    for child in sorted(seq_dir.iterdir()):
        if not child.is_dir():
            continue
        frame_files = sorted(
            (f for f in child.iterdir() if f.is_file() and FRAME_NAME_PATTERN.match(f.name)),
            key=lambda p: p.name,
        )
        if frame_files:
            frame_dirs.append((child, frame_files))
    return frame_dirs


def delete_frames(files: List[Path], indices_to_drop: set[int], dry_run: bool) -> None:
    missing: List[Path] = []
    for idx, file_path in enumerate(files):
        if idx not in indices_to_drop:
            continue
        if dry_run:
            print(f"  DRY-RUN delete {file_path}")
        else:
            try:
                file_path.unlink()
            except FileNotFoundError:
                missing.append(file_path)
    if missing:
        sample = ", ".join(p.name for p in missing[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        print(f"  WARNING missing frame files: {sample}{more}")


def rename_frames(frame_dir: Path, dry_run: bool) -> None:
    remaining = sorted(
        (f for f in frame_dir.iterdir() if f.is_file() and FRAME_NAME_PATTERN.match(f.name)),
        key=lambda p: p.name,
    )
    # Use temporary names to avoid collisions when renaming in place.
    temp_paths: List[Path] = []
    for idx, file_path in enumerate(remaining):
        temp_path = file_path.with_name(f"__tmp_{idx:06d}{file_path.suffix}")
        if dry_run:
            print(f"  DRY-RUN rename {file_path} -> {temp_path.name}")
        else:
            file_path.rename(temp_path)
        temp_paths.append(temp_path)

    for idx, temp_path in enumerate(temp_paths):
        final_path = temp_path.with_name(f"{idx:06d}{temp_path.suffix}")
        if dry_run:
            print(f"  DRY-RUN rename {temp_path.name} -> {final_path.name}")
        else:
            temp_path.rename(final_path)


def process_sequence(seq_dir: Path, front: int, back: int, dry_run: bool) -> None:
    frame_dirs = find_frame_directories(seq_dir)
    if not frame_dirs:
        return

    for frame_dir, files in frame_dirs:
        frame_count = len(files)
        if frame_count == 0:
            continue
        if frame_count <= front + back:
            print(
                f"{seq_dir.name}/{frame_dir.name}: only {frame_count} frames, skipping (need more than {front + back})"
            )
            continue

        indices_to_drop = set(range(front)) | set(range(frame_count - back, frame_count))
        kept = frame_count - len(indices_to_drop)
        print(
            f"{seq_dir.name}/{frame_dir.name}: {frame_count} frames -> drop {len(indices_to_drop)} keep {kept}"
        )
        delete_frames(files, indices_to_drop, dry_run=dry_run)
        rename_frames(frame_dir, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trim leading and trailing frames for each sequence under data_lion/train."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data_lion/train"),
        help="Path to the train directory (default: data_lion/train)",
    )
    parser.add_argument(
        "--front",
        type=int,
        default=25,
        help="Number of frames to remove from the beginning of each sequence.",
    )
    parser.add_argument(
        "--back",
        type=int,
        default=20,
        help="Number of frames to remove from the end of each sequence.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned operations without modifying any files.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Root directory {root} does not exist")

    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        process_sequence(seq_dir, args.front, args.back, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
