from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


@dataclasses.dataclass
class Args:
    repo_id: str = tyro.MISSING
    episode_index: int = 0
    num_steps: int = 5
    state_key: str | None = None
    headpose_start: int = 8
    headpose_dim: int = 7


def _choose_state_key(sample: dict[str, Any], user_key: str | None) -> str:
    if user_key is not None:
        if user_key not in sample:
            raise KeyError(f"state key {user_key!r} not found in sample keys: {sorted(sample.keys())}")
        return user_key

    for key in ("state", "observation.state", "observation/state"):
        if key in sample:
            return key

    raise KeyError(f"Failed to infer state key from sample keys: {sorted(sample.keys())}")


def _collect_episode_indices(ds: LeRobotDataset, episode_index: int) -> list[int]:
    indices: list[int] = []
    for idx in range(len(ds)):
        sample = ds[idx]
        epi = int(sample.get("episode_index", -1))
        if epi == episode_index:
            indices.append(idx)
    if not indices:
        raise IndexError(f"Episode {episode_index} not found in dataset.")
    return indices


def main(args: Args) -> None:
    if args.num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {args.num_steps}")
    if args.headpose_dim <= 0:
        raise ValueError(f"headpose_dim must be > 0, got {args.headpose_dim}")
    if args.headpose_start < 0:
        raise ValueError(f"headpose_start must be >= 0, got {args.headpose_start}")

    ds = LeRobotDataset(args.repo_id)
    episode_indices = _collect_episode_indices(ds, args.episode_index)
    first_sample = ds[episode_indices[0]]
    state_key = _choose_state_key(first_sample, args.state_key)

    print("=== LeRobot Headpose State Dump ===")
    print(f"repo_id: {args.repo_id}")
    print(f"episode_index: {args.episode_index}")
    print(f"state_key: {state_key}")
    print(f"episode_length: {len(episode_indices)}")
    print(f"printing_steps: {min(args.num_steps, len(episode_indices))}")
    print(f"headpose_slice: [{args.headpose_start}:{args.headpose_start + args.headpose_dim}]")

    for local_step, sample_idx in enumerate(episode_indices[: args.num_steps]):
        sample = ds[sample_idx]
        state = np.asarray(sample[state_key], dtype=np.float32).reshape(-1)
        end = args.headpose_start + args.headpose_dim
        if state.shape[0] < end:
            raise ValueError(
                f"State dim {state.shape[0]} is smaller than requested headpose slice end {end}. "
                f"sample_idx={sample_idx}, episode_index={args.episode_index}"
            )

        headpose = state[args.headpose_start:end]
        frame_index = int(sample.get("frame_index", sample_idx))
        headpose_str = np.array2string(headpose, precision=6, separator=", ")
        print(f"step={local_step} sample_idx={sample_idx} frame_index={frame_index} headpose_state={headpose_str}")


if __name__ == "__main__":
    main(tyro.cli(Args))
