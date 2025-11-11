#!/usr/bin/env python3
"""Stream RealWorldDataset point clouds frame-by-frame in Rerun.

Example:
    python src/egodata_eval/vis_pointcloud_sequence.py \
        --data_path data/moving --sample_index 0 --fps 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MBA.dataset.realworld import RealWorldDataset, collate_fn  # type: ignore
from MBA.utils.constants import IMG_MEAN, IMG_STD  # type: ignore


def _import_rerun():
    try:
        import rerun as rr  # type: ignore
    except Exception as exc:  # pragma: no cover - rerun is optional at install time
        raise RuntimeError("Rerun package is required. Install with `pip install rerun-sdk`.") from exc
    return rr


def _denormalize_colors(norm_colors: np.ndarray) -> np.ndarray:
    colors = norm_colors * IMG_STD + IMG_MEAN
    colors = np.clip(colors, 0.0, 1.0)
    return (colors * 255).astype(np.uint8)


def _build_dataloader(dataset: RealWorldDataset,
                      num_workers: int = 0) -> torch.utils.data.DataLoader:
    sampler = torch.utils.data.SequentialSampler(dataset)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )


def load_data_point(dataset: RealWorldDataset,
                    sample_index: int,
                    num_workers: int = 0) -> Tuple[Dict, Dict]:
    """Load a single sample via DataLoader to honor collate_fn behavior."""
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index {sample_index} out of range (dataset has {len(dataset)} samples)")

    dataloader = _build_dataloader(dataset, num_workers)
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == sample_index:
            meta = {
                "seq_id": dataset.seq_ids[sample_index],
                "obs_frame_ids": dataset.obs_frame_ids[sample_index],
                "action_frame_ids": dataset.action_frame_ids[sample_index],
                "data_path": dataset.data_paths[sample_index],
            }
            return batch, meta
    raise RuntimeError(f"Failed to load sample_index {sample_index} via DataLoader.")


def visualize_sequence(frame_clouds: Sequence[Tuple[str, np.ndarray]],
                       seq_id: str,
                       fps: float,
                       point_radius: float,
                       spawn_viewer: bool) -> None:
    if not frame_clouds:
        raise ValueError("cloud_sequence is empty; nothing to visualize.")

    rr = _import_rerun()
    rr.init(f"DatasetClouds[{seq_id}]", spawn=spawn_viewer)
    try:
        rr.log("world", rr.ViewCoordinates.RDF)
    except Exception:
        pass

    dt = 1.0 / fps if fps > 1e-6 else 0.0
    for idx, (frame_id, cloud) in enumerate(frame_clouds):
        if cloud.size == 0:
            continue
        positions = cloud[:, :3].astype(np.float32)
        colors = _denormalize_colors(cloud[:, 3:6])
        try:
            time_idx = int(frame_id)
        except ValueError:
            time_idx = idx
        rr.set_time("frame", sequence=time_idx)
        rr.log(
            f"sample/{seq_id}/frame_{frame_id}",
            rr.Points3D(
                positions=positions,
                colors=colors,
                radii=point_radius,
            ),
        )
        if dt > 0:
            time.sleep(dt)


def _frame_sort_key(frame_id: str) -> Tuple[int, int | str]:
    try:
        return (0, int(frame_id))
    except ValueError:
        return (1, frame_id)


def gather_sequence_frames(dataset: RealWorldDataset,
                           seq_id: str,
                           num_workers: int = 0) -> Tuple[List[Tuple[str, np.ndarray]], Dict]:
    indices = [i for i, sid in enumerate(dataset.seq_ids) if sid == seq_id]
    if not indices:
        raise ValueError(f"Sequence id '{seq_id}' not found in dataset.")

    dataloader = _build_dataloader(dataset, num_workers)
    target_set = set(indices)
    frames: List[Tuple[str, np.ndarray]] = []
    seen_frames: set[str] = set()
    for idx, batch in enumerate(dataloader):
        if idx not in target_set:
            continue
        clouds_batch = batch.get("clouds_list")
        if not clouds_batch:
            continue
        frame_ids = dataset.obs_frame_ids[idx]
        cloud_sequence = clouds_batch[0]
        for fid, cloud in zip(frame_ids, cloud_sequence):
            if fid in seen_frames:
                continue
            frames.append((fid, cloud))
            seen_frames.add(fid)
        target_set.remove(idx)
        if not target_set:
            break

    frames.sort(key=lambda tup: _frame_sort_key(tup[0]))
    meta = {
        "seq_id": seq_id,
        "data_path": dataset.data_paths[indices[0]],
        "total_frames": len(frames),
    }
    return frames, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dataset point clouds per frame using Rerun.")
    parser.add_argument("--data_path", type=str, default="data/moving", help="Dataset root path.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval", "all"], help="Dataset split.")
    parser.add_argument("--mode", type=str, default="sample", choices=["sample", "sequence"],
                        help="'sample' visualizes a single dataset item, 'sequence' streams all frames from a demo.")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="Dataset sample index (used when --mode sample).")
    parser.add_argument("--seq_id", type=str, default=None,
                        help="Sequence folder name to visualize (overrides --seq_index) when --mode sequence.")
    parser.add_argument("--seq_index", type=int, default=0,
                        help="Sequence index from dataset listing when --seq_id is not provided.")
    parser.add_argument("--num_obs", type=int, default=1, help="Number of observation frames per sample.")
    parser.add_argument("--num_action", type=int, default=20, help="Number of action frames per sample (affects padding).")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="Voxel size passed to RealWorldDataset.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback speed for stepping through frames.")
    parser.add_argument("--point_radius", type=float, default=0.002, help="Point radius (meters) for Rerun markers.")
    parser.add_argument("--no_spawn", action="store_true", help="Do not spawn a separate Rerun viewer window.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = RealWorldDataset(
        path=args.data_path,
        split=args.split,
        num_obs=args.num_obs,
        num_action=args.num_action,
        voxel_size=args.voxel_size,
        with_cloud=True,
        with_obj_action=False,
        aug=False,
        aug_jitter=False,
    )
    if args.mode == "sample":
        batch, meta = load_data_point(dataset, args.sample_index, args.num_workers)
        clouds_batch = batch.get("clouds_list")
        if not clouds_batch:
            raise RuntimeError("Dataset was expected to return 'clouds_list'. Ensure with_cloud=True.")
        cloud_sequence = clouds_batch[0]
        frame_ids = meta["obs_frame_ids"]
        if len(cloud_sequence) != len(frame_ids):
            raise RuntimeError(f"Mismatched lengths: {len(cloud_sequence)} clouds vs {len(frame_ids)} frame ids.")
        frame_clouds = list(zip(frame_ids, cloud_sequence))
        print(f"[INFO] Visualizing seq={meta['seq_id']} sample_index={args.sample_index} from {meta['data_path']}")
    else:
        if args.seq_id is not None:
            seq_id = args.seq_id
        else:
            if args.seq_index < 0 or args.seq_index >= len(dataset.all_demos):
                raise IndexError(f"seq_index {args.seq_index} out of range (dataset has {len(dataset.all_demos)} demos)")
            seq_id = dataset.all_demos[args.seq_index]
        frame_clouds, meta = gather_sequence_frames(dataset, seq_id, args.num_workers)
        if not frame_clouds:
            raise RuntimeError(f"No frames collected for sequence {seq_id}.")
        print(f"[INFO] Visualizing entire seq={seq_id} ({meta['total_frames']} unique frames) from {meta['data_path']}")

    visualize_sequence(
        frame_clouds=frame_clouds,
        seq_id=meta["seq_id"],
        fps=args.fps,
        point_radius=args.point_radius,
        spawn_viewer=not args.no_spawn,
    )


if __name__ == "__main__":
    main()
