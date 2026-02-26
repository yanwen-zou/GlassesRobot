"""Quick checker for loading a LeRobot dataset through openpi data loader.

Example: (first export HF_LEROBOT_HOME)
    python baseline/openpi/scripts/check_lerobot_dataloader.py   --repo-id data/book_openpi   --config-name pi05_realworld   --num-batches 3   --batch-size 4
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import tyro

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclass(frozen=True)
class Args:
    # LeRobot dataset repo_id or local dataset path (e.g. data/book_openpi).
    repo_id: str # if use local dataset, export HF_LEROBOT_HOME first
    # Base openpi config to reuse transforms/model setup.
    config_name: str = "pi05_realworld"
    # Number of batches to iterate.
    num_batches: int = 1
    # Global batch size.
    batch_size: int = 1
    # Data loader workers.
    num_workers: int = 0
    # Shuffle dataset.
    shuffle: bool = False
    # Skip normalization stats lookup (recommended for quick validation).
    skip_norm_stats: bool = True
    # Inject prompt from LeRobot task field.
    prompt_from_task: bool = True
    # Raw dataset index to inspect before transforms.
    inspect_index: int = 0


def _shape_dtype(x: Any) -> str:
    arr = np.asarray(x)
    return f"shape={tuple(arr.shape)}, dtype={arr.dtype}"


def _print_tree(prefix: str, value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _print_tree(f"{prefix}/{key}" if prefix else str(key), child)
        return
    print(f"{prefix}: {_shape_dtype(value)}")


def main(args: Args) -> None:
    if "libero" in args.config_name.lower():
        raise ValueError(
            f"config_name={args.config_name!r} points to libero policy. "
            "Please use a realworld config such as 'pi05_realworld'."
        )

    cfg = _config.get_config(args.config_name)

    # Override dataset location and prompt loading behavior for quick validation.
    new_data_factory = dataclasses.replace(
        cfg.data,
        repo_id=args.repo_id,
        base_config=_config.DataConfig(prompt_from_task=args.prompt_from_task),
    )
    cfg = dataclasses.replace(
        cfg,
        data=new_data_factory,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("=== Effective Setup ===")
    print(f"config_name: {args.config_name}")
    print(f"repo_id: {args.repo_id}")
    print(f"batch_size: {cfg.batch_size}, num_workers: {cfg.num_workers}")
    print(f"action_horizon: {cfg.model.action_horizon}, action_dim: {cfg.model.action_dim}")
    print(f"skip_norm_stats: {args.skip_norm_stats}")

    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)

    print("\n=== Raw LeRobot Sample (before transforms) ===")
    raw_dataset = _data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
    print(f"dataset_len: {len(raw_dataset)}")
    raw_sample = raw_dataset[args.inspect_index]
    _print_tree("", raw_sample)

    print("\n=== openpi DataLoader Batches (after transforms) ===")
    loader = _data_loader.create_data_loader(
        cfg,
        num_batches=args.num_batches,
        shuffle=args.shuffle,
        skip_norm_stats=args.skip_norm_stats,
    )

    for i, (obs, actions) in enumerate(loader, start=1):
        state_np = np.asarray(obs.state)
        act_np = np.asarray(actions)
        print(f"[batch {i}] state={state_np.shape} actions={act_np.shape}")
        print(f"[batch {i}] state finite={np.isfinite(state_np).all()} actions finite={np.isfinite(act_np).all()}")
        for cam_key, cam_value in obs.images.items():
            cam_np = np.asarray(cam_value)
            print(f"[batch {i}] image/{cam_key}={cam_np.shape} dtype={cam_np.dtype}")
        if obs.tokenized_prompt is not None:
            print(f"[batch {i}] tokenized_prompt={np.asarray(obs.tokenized_prompt).shape}")

    print("\nCheck finished: data loader ran without exception.")


if __name__ == "__main__":
    main(tyro.cli(Args))
