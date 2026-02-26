from __future__ import annotations

import dataclasses
import logging
from typing import Any

import cv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

from openpi_client import image_tools
from openpi_client import websocket_client_policy


@dataclasses.dataclass
class Args:
    # Policy server address.
    host: str = "localhost"
    port: int = 8000
    api_key: str | None = None

    # LeRobot dataset selection.
    repo_id: str = tyro.MISSING
    episode_index: int = tyro.MISSING

    # Optional explicit dataset keys.
    image_key: str | None = None
    state_key: str | None = None
    action_key: str | None = None
    prompt_key: str | None = None

    # Prompt behavior.
    prompt: str | None = None
    default_prompt: str = "Put the book in the shelf"

    # Preprocessing.
    image_size: int = 224
    # State dim expected by serving checkpoint norm stats.
    # Set to None to keep raw dataset state unchanged.
    state_dim: int | None = 8

    # Optional visualization for model input image.
    visualize_input_image: bool = False
    visualize_window_name: str = "pi05_lerobot_model_input"
    visualize_wait_ms: int = 1


def _choose_key(sample: dict[str, Any], candidates: list[str], user_key: str | None, desc: str) -> str:
    if user_key is not None:
        if user_key not in sample:
            raise KeyError(f"{desc} key {user_key!r} not found in sample keys: {sorted(sample.keys())}")
        return user_key
    for key in candidates:
        if key in sample:
            return key
    raise KeyError(f"Failed to infer {desc} key from sample keys: {sorted(sample.keys())}")


def _to_hwc_uint8(img: Any, image_size: int) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {arr.shape}")
    # CHW -> HWC
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        maxv = float(np.nanmax(arr)) if arr.size > 0 else 1.0
        if maxv <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = image_tools.resize_with_pad(arr, image_size, image_size)
    arr = image_tools.convert_to_uint8(arr)
    return np.asarray(arr, dtype=np.uint8)


def _visualize_model_input_image(image_rgb_uint8: np.ndarray, window_name: str, wait_ms: int) -> None:
    if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[2] != 3:
        raise ValueError(f"Unexpected model input image shape: {image_rgb_uint8.shape}")
    image_bgr = image_rgb_uint8[:, :, ::-1]
    cv2.imshow(window_name, image_bgr)
    cv2.waitKey(max(1, int(wait_ms)))


def _first_action_vec(action: np.ndarray) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim >= 2:
        return arr.reshape(-1, arr.shape[-1])[0]
    raise ValueError(f"Unexpected action shape: {arr.shape}")


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


def _align_state_dim(state: np.ndarray, target_dim: int | None) -> np.ndarray:
    if target_dim is None:
        return state.astype(np.float32, copy=False)
    if target_dim <= 0:
        raise ValueError(f"state_dim must be > 0 or None, got {target_dim}")
    cur_dim = int(state.shape[-1])
    if cur_dim == target_dim:
        return state.astype(np.float32, copy=False)
    if cur_dim > target_dim:
        return state[:target_dim].astype(np.float32, copy=False)
    pad = np.zeros((target_dim - cur_dim,), dtype=np.float32)
    return np.concatenate([state.astype(np.float32, copy=False), pad], axis=0)


def main(args: Args) -> None:
    ds = LeRobotDataset(args.repo_id)
    episode_indices = _collect_episode_indices(ds, args.episode_index)
    sample = ds[episode_indices[0]]

    image_key = _choose_key(sample, ["image", "observation.image", "observation/image"], args.image_key, "image")
    state_key = _choose_key(sample, ["state", "observation.state", "observation/state"], args.state_key, "state")
    action_key = _choose_key(sample, ["actions", "action"], args.action_key, "action")
    prompt_key = _choose_key(sample, ["task", "prompt"], args.prompt_key, "prompt")

    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logging.info("Server metadata: %s", client.get_server_metadata())

    print("=== Episode Info ===")
    print(f"repo_id: {args.repo_id}")
    print(f"episode_index: {args.episode_index}")
    print(f"num_frames: {len(episode_indices)}")
    print(f"image_key: {image_key}")
    print(f"state_key: {state_key}")
    print(f"action_key: {action_key}")
    print(f"prompt_key: {prompt_key}")

    for i, sample_idx in enumerate(episode_indices):
        sample = ds[sample_idx]
        image = _to_hwc_uint8(sample[image_key], args.image_size)
        raw_state = np.asarray(sample[state_key], dtype=np.float32).reshape(-1)
        state = _align_state_dim(raw_state, args.state_dim)
        dataset_action = np.asarray(sample[action_key], dtype=np.float32)

        prompt = args.prompt
        if prompt is None:
            raw_prompt = sample.get(prompt_key, "")
            prompt = str(raw_prompt) if raw_prompt is not None else ""
        if prompt == "":
            prompt = args.default_prompt

        if args.visualize_input_image:
            _visualize_model_input_image(
                image,
                window_name=args.visualize_window_name,
                wait_ms=args.visualize_wait_ms,
            )

        observation = {
            "observation/image": image,
            "observation/state": state,
            "prompt": prompt,
        }
        result = client.infer(observation)
        pred_action = np.asarray(result["actions"], dtype=np.float32)

        epi = int(sample.get("episode_index", -1))
        frame = int(sample.get("frame_index", sample_idx))
        pred_action_1d = _first_action_vec(pred_action)
        data_action_1d = _first_action_vec(dataset_action)
        pred_15 = np.round(pred_action_1d[:15], 5)
        data_15 = np.round(data_action_1d[:15], 5)
        # In 15-dim action, headpose is [8:15], so headpose xyz is [8:11].
        pred_headpose_xyz = np.round(pred_15[8:11], 5)
        data_headpose_xyz = np.round(data_15[8:11], 5)
        diff_headpose_xyz = np.round(pred_headpose_xyz - data_headpose_xyz, 5)

        print(f"\n=== Frame {i + 1}/{len(episode_indices)} ===")
        print(f"global_index: {sample_idx}")
        print(f"episode_index: {epi}")
        print(f"frame_index: {frame}")
        print(f"image_shape: {image.shape}, raw_state_shape: {raw_state.shape}, send_state_shape: {state.shape}")
        print("Model Output Action (first 15 / 32):")
        print(pred_15)
        print("Dataset Raw Action (first 15):")
        print(data_15)
        print(f"Headpose XYZ compare (model vs dataset): {pred_headpose_xyz} vs {data_headpose_xyz}")
        print(f"Headpose XYZ diff (model - dataset): {diff_headpose_xyz}")

    if args.visualize_input_image:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
