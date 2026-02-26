"""
Script to convert hdf5 data to LeRobot dataset format, **WITH Monte Carlo value labels!!!**

Usage: uv run examples/flexiv/convert_hdf5_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/flexiv/convert_hdf5_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

The resulting dataset will get saved to the $LEROBOT_HOME directory.
"""


import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation as R  # noqa: F401

# current unused
REPO_NAME = "shi-akihi/book_openpi" # Remember to change this to your own huggingface repo name


LOGGER = logging.getLogger(__name__)


def _relative_headpose_to_first(headpose_data: np.ndarray) -> np.ndarray:
    """Convert absolute head pose [x,y,z,qx,qy,qz,qw] to relative-to-first pose."""
    if headpose_data.ndim != 2 or headpose_data.shape[1] != 7:
        raise ValueError(f"Expected headpose shape (T, 7), got {headpose_data.shape}")

    t0 = headpose_data[0, :3]
    q0 = headpose_data[0, 3:7]
    q0 = q0 / np.linalg.norm(q0)
    r0_inv = R.from_quat(q0).inv()

    rel = np.empty_like(headpose_data, dtype=np.float32)
    for i in range(headpose_data.shape[0]):
        ti = headpose_data[i, :3]
        qi = headpose_data[i, 3:7]
        qi = qi / np.linalg.norm(qi)

        # Relative transform: T_rel = T0^{-1} * Ti
        rel_t = r0_inv.apply(ti - t0)
        rel_q = (r0_inv * R.from_quat(qi)).as_quat()

        rel[i, :3] = rel_t.astype(np.float32)
        rel[i, 3:7] = rel_q.astype(np.float32)
    return rel


def _relative_headpose_to_prev(headpose_data: np.ndarray) -> np.ndarray:
    """Convert absolute head pose [x,y,z,qx,qy,qz,qw] to relative-to-previous pose."""
    if headpose_data.ndim != 2 or headpose_data.shape[1] != 7:
        raise ValueError(f"Expected headpose shape (T, 7), got {headpose_data.shape}")

    rel = np.empty_like(headpose_data, dtype=np.float32)
    # First frame has no previous frame: use identity delta.
    rel[0, :3] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    rel[0, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    for i in range(1, headpose_data.shape[0]):
        t_prev = headpose_data[i - 1, :3]
        q_prev = headpose_data[i - 1, 3:7]
        q_prev = q_prev / np.linalg.norm(q_prev)
        r_prev_inv = R.from_quat(q_prev).inv()

        t_curr = headpose_data[i, :3]
        q_curr = headpose_data[i, 3:7]
        q_curr = q_curr / np.linalg.norm(q_curr)

        # Relative transform: T_rel = T_{i-1}^{-1} * T_i
        rel_t = r_prev_inv.apply(t_curr - t_prev)
        rel_q = (r_prev_inv * R.from_quat(q_curr)).as_quat()

        rel[i, :3] = rel_t.astype(np.float32)
        rel[i, 3:7] = rel_q.astype(np.float32)
    return rel


@dataclass
class ConversionConfig:
    data_dir: str
    output_path: str = str(HF_LEROBOT_HOME / REPO_NAME)
    push_to_hub: bool = False
    num_workers: int = 4
    headpose_mode: str = "abs"  # "abs" for absolute, "delta" for relative to previous headpose
    append_to_existing: bool = False


def _get_episode_keys(h5_file: h5py.File) -> List[str]:
    if "episodes" in h5_file:
        return list(h5_file["episodes"].keys())
    return list(h5_file.keys())


def _get_episode_group(h5_file: h5py.File, episode_key: str):
    if "episodes" in h5_file:
        return h5_file["episodes"][episode_key]
    return h5_file[episode_key]


def _compute_max_duration(hdf5_path: str) -> int:
    with h5py.File(hdf5_path, "r") as h5_file:
        episode_keys = _get_episode_keys(h5_file)
        if not episode_keys:
            raise ValueError(f"No episodes found in {hdf5_path}")
        return max(len(_get_episode_group(h5_file, key)["action"]) for key in episode_keys) - 1


def _resolve_hdf5_paths(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"Expected a .h5/.hdf5 file, got: {p}")
        return [str(p)]
    if not p.is_dir():
        raise ValueError(f"Path does not exist: {p}")

    files = sorted(
        [str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in {".h5", ".hdf5"}]
    )
    if not files:
        raise ValueError(f"No .h5/.hdf5 files found under: {p}")
    return files


def _load_episode_frames(hdf5_path: str, episode_key: str, max_duration: int, headpose_mode: str = "abs") -> List[Dict[str, np.ndarray]]:
    LOGGER.debug("Loading episode %s on %s", episode_key, threading.current_thread().name)
    with h5py.File(hdf5_path, "r") as h5_file:
        episode = _get_episode_group(h5_file, episode_key)
        num_steps = episode["action"].shape[0]
        values = np.arange(-num_steps + 1, 1, dtype=np.float32)[:, None] / max_duration
        frames: List[Dict[str, np.ndarray]] = []
        
        headpose = episode.get("headpose")
        if headpose is not None:
            headpose_data = headpose[:].astype(np.float32)
            if headpose_mode == "delta":
                headpose_data = _relative_headpose_to_first(headpose_data)
                # print(f"Episode {episode_key} headpose converted to relative-to-first frame:\n{headpose_data}")
                LOGGER.debug("Episode %s: Converted headpose to relative-to-first frame", episode_key)
        
        for step in range(num_steps):
            robot_state = episode["robot_state"][step].astype(np.float32)
            action = episode["action"][step].astype(np.float32)
            if headpose is not None:
                headpose_step = headpose_data[step]
                state = np.concatenate([robot_state, headpose_step], axis=0)
                actions = np.concatenate([action, headpose_step], axis=0)
            else:
                state = robot_state
                actions = action
            frames.append(
                {
                    # Convert RGB -> BGR for downstream consumers expecting OpenCV-style channel order.
                    "image": episode["left_cam"][step][..., ::-1],
                    "wrist_image": episode["right_cam"][step][..., ::-1],
                    "state": state,
                    "actions": actions,
                    "value": values[step],
                    "task": "Put the book in the shelf",
                }
            )
    return frames


def main(config: ConversionConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_path = Path(config.output_path)
    if output_path.exists() and not config.append_to_existing:
        shutil.rmtree(output_path)
    logging.info("Output dataset path: %s", output_path)

    state_dim = 8 if config.headpose_mode == "delta" else 15
    features = {
        "image": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (15,),
            "names": ["actions"],
        },
        "value": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["value"],
        },
    }

    if output_path.exists() and config.append_to_existing:
        logging.info("Appending to existing dataset at %s", output_path)
        dataset = LeRobotDataset(
            repo_id=REPO_NAME,
            root=output_path,
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            robot_type="single_flexiv_rizon4",
            fps=10,
            root=output_path,
            features=features,
            image_writer_threads=10,
            image_writer_processes=5,
        )

    hdf5_paths = _resolve_hdf5_paths(config.data_dir)
    logging.info("Found %d hdf5 files to process.", len(hdf5_paths))

    max_duration = max(_compute_max_duration(path) for path in hdf5_paths)
    logging.info("Max duration across all files: %s", max_duration)

    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as h5_file:
            episode_keys = _get_episode_keys(h5_file)
        logging.info(
            "Processing %s: %d episodes across %d threads",
            hdf5_path,
            len(episode_keys),
            config.num_workers,
        )

        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            futures = [
                executor.submit(_load_episode_frames, hdf5_path, episode_key, max_duration, config.headpose_mode)
                for episode_key in episode_keys
            ]
            for future in as_completed(futures):
                frames = future.result()
                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()

    if config.push_to_hub:
        dataset.push_to_hub()
        logging.info("Dataset pushed to hub at %s", REPO_NAME)


if __name__ == "__main__":
    main(tyro.cli(ConversionConfig))
