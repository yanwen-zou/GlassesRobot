#!/usr/bin/env python3
"""Review episodes from HDF5 files and save selected ones into a new HDF5.

Behavior:
- Recursively scan an input directory for .h5/.hdf5 files.
- Play each episode in a loop (prefer image datasets like left_cam/right_cam).
- During playback, press key to decide current episode:
  - k: keep current episode and continue
  - d: drop current episode (optionally delete from source file)
  - q or ESC: stop reviewing immediately
- Save all kept episodes into a new output HDF5 file under /episodes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

H5_SUFFIXES = {".h5", ".hdf5"}
IMAGE_KEYS_PRIORITY = ["left_cam", "right_cam", "image", "wrist_image", "rgb"]


@dataclass(frozen=True)
class EpisodeRef:
    src_path: Path
    episode_key: str
    src_group_path: str


def resolve_hdf5_paths(path: Path, recursive: bool = True) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in H5_SUFFIXES:
            raise ValueError(f"Expected .h5/.hdf5 file, got: {path}")
        return [path]

    if not path.is_dir():
        raise ValueError(f"Input path does not exist: {path}")

    iterator = path.rglob("*") if recursive else path.glob("*")
    files = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in H5_SUFFIXES)
    if not files:
        raise ValueError(f"No .h5/.hdf5 files found under: {path}")
    return files


def iter_episode_refs(h5_path: Path) -> Iterable[EpisodeRef]:
    with h5py.File(h5_path, "r") as f:
        if "episodes" in f and isinstance(f["episodes"], h5py.Group):
            for key in f["episodes"].keys():
                yield EpisodeRef(src_path=h5_path, episode_key=key, src_group_path=f"/episodes/{key}")
        else:
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    yield EpisodeRef(src_path=h5_path, episode_key=key, src_group_path=f"/{key}")


def _pick_image_datasets(group: h5py.Group) -> list[h5py.Dataset]:
    datasets: list[h5py.Dataset] = []
    for key in IMAGE_KEYS_PRIORITY:
        obj = group.get(key)
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3 and obj.shape[0] > 0:
            datasets.append(obj)

    if datasets:
        return datasets[:2]

    for obj in group.values():
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3 and obj.shape[0] > 0:
            datasets.append(obj)
            if len(datasets) >= 2:
                break
    return datasets


def _to_displayable(img: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not available in current environment.")
    if img.dtype != np.uint8:
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] >= 3:
        return img[:, :, :3]

    raise ValueError(f"Unsupported image shape: {img.shape}")


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not available in current environment.")
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / max(h, 1)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _play_episode(group: h5py.Group, fps: float, window_name: str, max_frames: int | None) -> tuple[int, str | None]:
    if cv2 is None:
        print("  [WARN] OpenCV not installed, cannot play frames. Continue with prompt only.")
        return 0, None
    image_sets = _pick_image_datasets(group)
    if not image_sets:
        print("  [WARN] No image-like dataset found, skip playback.")
        return 0, None

    total = min(int(ds.shape[0]) for ds in image_sets)
    if max_frames is not None and max_frames > 0:
        total = min(total, max_frames)

    delay_ms = max(1, int(round(1000.0 / max(fps, 1e-6))))
    frame_idx = 0
    loops = 0

    while True:
        panels = [_to_displayable(ds[frame_idx]) for ds in image_sets]
        target_h = min(img.shape[0] for img in panels)
        panels = [_resize_to_height(img, target_h) for img in panels]
        canvas = np.concatenate(panels, axis=1) if len(panels) > 1 else panels[0]

        cv2.putText(
            canvas,
            f"frame {frame_idx + 1}/{total}  [k=keep d=delete q=quit]",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("k"):
            cv2.destroyWindow(window_name)
            return total, "keep"
        if key == ord("d"):
            cv2.destroyWindow(window_name)
            return total, "delete"
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            return total, "quit"

        frame_idx += 1
        if frame_idx >= total:
            frame_idx = 0
            loops += 1
            if loops % 5 == 0:
                print("  [INFO] Still looping. Press k/d/q in window to choose this episode.")


def _prompt_action() -> str:
    while True:
        choice = input("Action [keep/delete/quit] (k/d/q): ").strip().lower()
        if choice in {"keep", "k"}:
            return "keep"
        if choice in {"delete", "d"}:
            return "delete"
        if choice in {"quit", "q"}:
            return "quit"
        print("Invalid input, please enter keep/delete/quit (or k/d/q).")


def _copy_episode(src_file: h5py.File, src_group_path: str, dst_episodes_group: h5py.Group, dst_name: str) -> None:
    src_group = src_file[src_group_path]
    src_file.copy(src_group, dst_episodes_group, name=dst_name)


def _delete_source_episode(src_path: Path, src_group_path: str) -> None:
    with h5py.File(src_path, "r+") as f:
        if src_group_path in f:
            del f[src_group_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Review/filter episodes from HDF5 files interactively.")
    parser.add_argument("--input", required=True, help="Input HDF5 file or directory")
    parser.add_argument("--output", required=True, help="Output HDF5 path for kept episodes")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS (default: 30)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per episode (default: all)")
    parser.add_argument("--non-recursive", action="store_true", help="Only scan top-level directory for HDF5")
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="If action=delete, also remove that episode from source HDF5 in-place",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if output_path.exists() and not args.overwrite_output:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite-output to replace it.")
    if output_path.exists() and args.overwrite_output:
        output_path.unlink()

    h5_paths = resolve_hdf5_paths(input_path, recursive=not args.non_recursive)
    episode_refs: list[EpisodeRef] = []
    for p in h5_paths:
        episode_refs.extend(iter_episode_refs(p))

    if not episode_refs:
        print("No episodes found.")
        return

    print(f"Found {len(h5_paths)} HDF5 files, {len(episode_refs)} episodes total.")
    print("Rule: press k/d/q in playback window. (fallback input if GUI unavailable)")

    kept_refs: list[EpisodeRef] = []
    deleted_refs: list[EpisodeRef] = []

    for idx, ref in enumerate(episode_refs, start=1):
        print(f"\n[{idx}/{len(episode_refs)}] file={ref.src_path} episode={ref.episode_key}")
        with h5py.File(ref.src_path, "r") as f:
            group = f[ref.src_group_path]
            played, action = _play_episode(
                group=group,
                fps=args.fps,
                window_name=f"episode_review::{ref.episode_key}",
                max_frames=args.max_frames,
            )
            print(f"  Played frames: {played}")

        if action is None:
            action = _prompt_action()

        if action == "keep":
            kept_refs.append(ref)
            print("  -> kept")
            continue

        if action == "delete":
            deleted_refs.append(ref)
            if args.delete_source:
                _delete_source_episode(ref.src_path, ref.src_group_path)
                print("  -> deleted from source")
            else:
                print("  -> dropped (source unchanged)")
            continue

        if action == "quit":
            print("Stop reviewing by user request.")
            break

    with h5py.File(output_path, "w") as out_f:
        out_eps = out_f.create_group("episodes")
        for new_idx, ref in enumerate(kept_refs):
            new_name = f"episode_{new_idx:06d}"
            with h5py.File(ref.src_path, "r") as src_f:
                _copy_episode(src_f, ref.src_group_path, out_eps, new_name)
            out_eps[new_name].attrs["source_file"] = str(ref.src_path)
            out_eps[new_name].attrs["source_episode_key"] = ref.episode_key

    print("\nDone.")
    print(f"Kept episodes: {len(kept_refs)}")
    print(f"Deleted/dropped episodes: {len(deleted_refs)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
