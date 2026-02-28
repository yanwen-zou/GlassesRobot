#!/usr/bin/env python3
"""Browse first-frame left_cam images across all episodes in test_record hdf5 files.

Keys:
  p: next episode
  q: quit
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import h5py
import numpy as np
import tyro


def _get_episode_keys(h5_file: h5py.File) -> List[str]:
    if "episodes" in h5_file:
        return list(h5_file["episodes"].keys())
    return list(h5_file.keys())


def _get_episode_group(h5_file: h5py.File, episode_key: str):
    if "episodes" in h5_file:
        return h5_file["episodes"][episode_key]
    return h5_file[episode_key]


def _collect_items(test_record_dir: Path) -> List[Tuple[Path, str]]:
    files = sorted(test_record_dir.glob("*.hdf5"))
    items: List[Tuple[Path, str]] = []
    for fp in files:
        with h5py.File(fp, "r") as h5_file:
            for ep_key in _get_episode_keys(h5_file):
                items.append((fp, ep_key))
    return items


@dataclass(frozen=True)
class Args:
    test_record_dir: str = "test_record"
    window_name: str = "first_left_cam"


def main(args: Args) -> None:
    test_record_dir = Path(args.test_record_dir)
    items = _collect_items(test_record_dir)
    if len(items) == 0:
        raise RuntimeError(f"No episodes found under {test_record_dir}")

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    idx = 0
    total = len(items)

    while idx < total:
        hdf5_path, ep_key = items[idx]
        with h5py.File(hdf5_path, "r") as h5_file:
            episode = _get_episode_group(h5_file, ep_key)
            if "left_cam" not in episode:
                print(f"[WARN] Skip {hdf5_path}::{ep_key} (no left_cam)")
                idx += 1
                continue
            left_cam = np.asarray(episode["left_cam"][0])

        # Stored left_cam is RGB in this codebase; convert to BGR for cv2 display.
        if left_cam.ndim == 3 and left_cam.shape[2] == 3:
            vis = left_cam[..., ::-1].copy()
        else:
            vis = left_cam.copy()

        label = f"[{idx + 1}/{total}] {hdf5_path.name} :: {ep_key} | key=p(next), q(quit)"
        cv2.putText(
            vis,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(args.window_name, vis)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            if key == ord("p"):
                idx += 1
                break

    cv2.destroyAllWindows()
    print("Done: reached last episode.")


if __name__ == "__main__":
    main(tyro.cli(Args))
