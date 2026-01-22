#!/usr/bin/env python3
"""
Load RealWorldDataset samples and cache point clouds + poses for visualization.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

here = Path(__file__).resolve()
project_root = here.parents[3]
mba_root = project_root / "MBA"
for path in (project_root, mba_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from MBA.dataset.realworld import RealWorldDataset


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache dataset samples for visualization.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument("--split", default="all", choices=["train", "eval", "all"])
    parser.add_argument("--num-obs", type=int, default=1)
    parser.add_argument("--num-action", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on samples.")
    parser.add_argument(
        "--episode-idx",
        type=int,
        default=None,
        help="Only load samples from the selected episode index (0-based, after sorting).",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Directory to store cached samples (default: visualize_scripts/tmp/<data-dir-name>).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    tmp_dir = args.tmp_dir
    if tmp_dir is None:
        tmp_dir = here.parent / "tmp" / data_dir.name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dataset = RealWorldDataset(
        path=str(data_dir),
        split=args.split,
        num_obs=args.num_obs,
        num_action=args.num_action,
        with_cloud=True,
        with_obj_action=True,
        with_headpose=True,
    )

    if args.episode_idx is not None:
        if args.episode_idx < 0 or args.episode_idx >= dataset.num_demos:
            raise ValueError(
                f"episode-idx {args.episode_idx} out of range [0, {dataset.num_demos - 1}]"
            )
        target_seq = dataset.all_demos[args.episode_idx]
        keep = [i for i, seq in enumerate(dataset.seq_ids) if seq == target_seq]
        dataset.data_paths = [dataset.data_paths[i] for i in keep]
        dataset.obs_frame_ids = [dataset.obs_frame_ids[i] for i in keep]
        dataset.action_frame_ids = [dataset.action_frame_ids[i] for i in keep]
        dataset.seq_ids = [dataset.seq_ids[i] for i in keep]
        if dataset.with_obj_action:
            dataset.obj_frame_ids = [dataset.obj_frame_ids[i] for i in keep]

    limit = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    for idx in range(limit):
        sample = dataset[idx]
        clouds_list = sample.get("clouds_list")
        current_obj_pose = _to_numpy(sample.get("current_obj_pose"))
        current_headpose = _to_numpy(sample.get("current_headpose"))
        if clouds_list is None:
            raise RuntimeError("clouds_list missing; ensure with_cloud=True.")

        out_path = tmp_dir / f"sample_{idx:06d}.npz"
        np.savez_compressed(
            out_path,
            clouds_list=np.array(clouds_list, dtype=object),
            current_obj_pose=current_obj_pose,
            current_headpose=current_headpose,
        )

    print(f"[INFO] Saved {limit} samples to {tmp_dir}")


if __name__ == "__main__":
    main()
