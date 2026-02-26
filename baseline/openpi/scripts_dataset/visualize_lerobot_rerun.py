"""Visualize a LeRobot-format dataset with Rerun.

Example:
  python baseline/openpi/scripts/visualize_lerobot_rerun.py \
    --repo-id data/book_openpi \
    --episode-index 0 \
    --max-frames 300
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


@dataclass(frozen=True)
class Args:
    # LeRobot repo id or local dataset path under $LEROBOT_HOME.
    repo_id: str
    # If set, only visualize this episode.
    episode_index: int | None = None
    # Maximum number of samples to visualize.
    max_frames: int = 500
    # Subsample steps (1 = every frame).
    stride: int = 1
    # Sleep time between frames for playback effect.
    sleep_sec: float = 0.0
    # Spawn rerun viewer process.
    spawn: bool = True
    # Recording name in rerun viewer.
    recording_name: str = "lerobot_dataset_vis"


def _to_hwc_uint8(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {arr.shape}")
    # CHW -> HWC
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        # Heuristic: dataset images are usually in [0,1] float.
        maxv = float(np.nanmax(arr)) if arr.size > 0 else 1.0
        if maxv <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _log_vector(rr: Any, base: str, vec: Any) -> None:
    arr = np.asarray(vec).reshape(-1)
    for i, v in enumerate(arr):
        rr.log(f"{base}/{i}", rr.Scalars(float(v)))


def _first_action_vec(sample: dict[str, Any]) -> np.ndarray | None:
    key = "actions" if "actions" in sample else ("action" if "action" in sample else None)
    if key is None:
        return None
    arr = np.asarray(sample[key], dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim >= 2:
        return arr.reshape(-1, arr.shape[-1])[0]
    return None


def main(args: Args) -> None:
    try:
        import rerun as rr  # type: ignore
    except Exception as exc:
        raise RuntimeError("Failed to import rerun. Please install rerun-sdk.") from exc

    ds = LeRobotDataset(args.repo_id)
    print(f"Loaded dataset: {args.repo_id}, len={len(ds)}")

    rr.init(args.recording_name, spawn=args.spawn)

    # Per-episode accumulated headpose xyz (action[8:11] are per-frame deltas).
    headpose_xyz_acc: dict[int, np.ndarray] = {}

    shown = 0
    for idx in range(0, len(ds), max(1, args.stride)):
        sample = ds[idx]

        epi = int(sample.get("episode_index", -1))
        if args.episode_index is not None and epi != args.episode_index:
            continue

        frame_idx = int(sample.get("frame_index", idx))
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("sample_idx", sequence=idx)
        rr.set_time("episode", sequence=epi)

        if "image" in sample:
            rr.log("obs/image", rr.Image(_to_hwc_uint8(sample["image"])))
        if "wrist_image" in sample:
            rr.log("obs/wrist_image", rr.Image(_to_hwc_uint8(sample["wrist_image"])))

        if "state" in sample:
            _log_vector(rr, "obs/state", sample["state"])

        if "actions" in sample:
            actions = np.asarray(sample["actions"])
            if actions.ndim == 2:
                _log_vector(rr, "act/step0", actions[0])
            else:
                _log_vector(rr, "act/step0", actions)
        elif "action" in sample:
            action = np.asarray(sample["action"])
            if action.ndim == 2:
                _log_vector(rr, "act/step0", action[0])
            else:
                _log_vector(rr, "act/step0", action)

        # Reconstruct headpose xyz by cumulatively summing per-frame delta xyz.
        action_vec = _first_action_vec(sample)
        if action_vec is not None and action_vec.shape[0] >= 11:
            delta_xyz = np.asarray(action_vec[8:11], dtype=np.float32)
            if epi not in headpose_xyz_acc:
                headpose_xyz_acc[epi] = np.zeros(3, dtype=np.float32)
            headpose_xyz_acc[epi] = headpose_xyz_acc[epi] + delta_xyz
            rr.log("headpose/delta_xyz", rr.Points3D([delta_xyz]))
            rr.log("headpose/accum_xyz", rr.Points3D([headpose_xyz_acc[epi]]))
            _log_vector(rr, "headpose/accum_xyz_scalar", headpose_xyz_acc[epi])

        task = sample.get("task", "")
        prompt = sample.get("prompt", "")
        rr.log("meta/text", rr.TextLog(f"episode={epi} frame={frame_idx} task={task} prompt={prompt}"))

        shown += 1
        if shown >= args.max_frames:
            break
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    print(f"Done. Visualized {shown} frames.")


if __name__ == "__main__":
    main(tyro.cli(Args))
