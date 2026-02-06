"""
Script to convert single arm data to LeRobot dataset format, **WITH Monte Carlo value labels!!!**

Usage: uv run examples/flexiv/convert_single_arm_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/flexiv/convert_single_arm_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

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
REPO_NAME = "shi-akihi/test" # Remember to change this to your own huggingface repo name


LOGGER = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    data_dir: str
    output_path: str = str(HF_LEROBOT_HOME / REPO_NAME)
    push_to_hub: bool = False
    num_workers: int = 4
    headpose_mode: str = "abs"  # "abs" for absolute, "delta" for relative to first headpose


def _get_episode_keys(h5_file: h5py.File) -> List[str]:
    if "episodes" in h5_file:
        return list(h5_file["episodes"].keys())
    return list(h5_file.keys())


def _get_episode_group(h5_file: h5py.File, episode_key: str):
    if "episodes" in h5_file:
        return h5_file["episodes"][episode_key]
    return h5_file[episode_key]


def _compute_max_duration(data_dir: str) -> int:
    with h5py.File(data_dir, "r") as h5_file:
        episode_keys = _get_episode_keys(h5_file)
        if not episode_keys:
            raise ValueError(f"No episodes found in {data_dir}")
        return max(len(_get_episode_group(h5_file, key)["action"]) for key in episode_keys) - 1


def _load_episode_frames(data_dir: str, episode_key: str, max_duration: int, headpose_mode: str = "abs") -> List[Dict[str, np.ndarray]]:
    LOGGER.debug("Loading episode %s on %s", episode_key, threading.current_thread().name)
    with h5py.File(data_dir, "r") as h5_file:
        episode = _get_episode_group(h5_file, episode_key)
        num_steps = episode["action"].shape[0]
        values = np.arange(-num_steps + 1, 1, dtype=np.float32)[:, None] / max_duration
        frames: List[Dict[str, np.ndarray]] = []
        
        headpose = episode.get("headpose")
        headpose_first = None
        if headpose is not None:
            headpose_data = headpose[:].astype(np.float32)
            if headpose_mode == "delta":
                # Store the first headpose as reference
                headpose_first = headpose_data[0]
                LOGGER.debug("Episode %s: Using delta mode, first headpose: %s", episode_key, headpose_first)
        
        for step in range(num_steps):
            robot_state = episode["robot_state"][step].astype(np.float32)
            action = episode["action"][step].astype(np.float32)
            if headpose is not None:
                if headpose_mode == "delta":
                    # Calculate relative to first headpose
                    headpose_step = headpose_data[step] - headpose_first
                else:  # abs mode
                    headpose_step = headpose_data[step]
                state = np.concatenate([robot_state, headpose_step], axis=0)
                actions = np.concatenate([action, headpose_step], axis=0)
            else:
                state = robot_state
                actions = action
            frames.append(
                {
                    "image": episode["left_cam"][step],
                    "wrist_image": episode["right_cam"][step],
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
    if output_path.exists():
        shutil.rmtree(output_path)
    logging.info("Output dataset path: %s", output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="single_flexiv_rizon4",
        fps=10,
        root=output_path,
        features={
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
                "shape": (15,),
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
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    max_duration = _compute_max_duration(config.data_dir)
    logging.info("Max duration across episodes: %s", max_duration)

    with h5py.File(config.data_dir, "r") as h5_file:
        episode_keys = _get_episode_keys(h5_file)
    logging.info("Scheduling %d episodes across %d threads", len(episode_keys), config.num_workers)

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = [
            executor.submit(_load_episode_frames, config.data_dir, episode_key, max_duration, config.headpose_mode)
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
