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
    state_dim: int | None = 15

    # Optional visualization for model input image.
    visualize_input_image: bool = False
    visualize_window_name: str = "pi05_lerobot_model_input"
    visualize_wait_ms: int = 1
    # Optional rerun visualization.
    use_rerun: bool = False
    rerun_spawn: bool = True
    rerun_recording_name: str = "pi05_lerobot_headpose_vis"


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


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-12, None)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def _compose_action_with_state(relative_action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
    rel = np.asarray(relative_action, dtype=np.float32).reshape(-1)
    base = np.asarray(current_state, dtype=np.float32).reshape(-1)
    if rel.shape[0] < 8 or base.shape[0] < 8:
        raise ValueError(f"Need at least 8 dims for rel/base, got rel={rel.shape[0]}, base={base.shape[0]}")
    if rel.shape[0] > base.shape[0]:
        raise ValueError(f"Relative action dim {rel.shape[0]} > state dim {base.shape[0]}")

    out = rel.copy()
    p0 = base[:3]
    q0_xyzw = _quat_normalize(_quat_wxyz_to_xyzw(base[3:7][None, :]))[0]
    out[:3] = _quat_rotate(q0_xyzw[None, :], rel[None, :3])[0] + p0
    out[3:7] = _quat_xyzw_to_wxyz(
        _quat_normalize(_quat_mul(q0_xyzw[None, :], _quat_wxyz_to_xyzw(rel[None, 3:7])))[0]
    )
    out[7] = rel[7] + base[7]

    if rel.shape[0] >= 15 and base.shape[0] >= 15:
        hp_t0 = base[8:11]
        hp_q0 = _quat_normalize(base[11:15][None, :])[0]
        out[8:11] = _quat_rotate(hp_q0[None, :], rel[None, 8:11])[0] + hp_t0
        out[11:15] = _quat_normalize(_quat_mul(hp_q0[None, :], rel[None, 11:15]))[0]
    return out


def main(args: Args) -> None:
    rr = None
    if args.use_rerun:
        try:
            import rerun as rr  # type: ignore
        except Exception as exc:
            raise RuntimeError("Failed to import rerun. Please install rerun-sdk.") from exc

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
    if rr is not None:
        rr.init(args.rerun_recording_name, spawn=args.rerun_spawn)
    # Keep per-episode trajectory history for headpose visualization.
    state_headpose_traj: dict[int, list[np.ndarray]] = {}

    for i, sample_idx in enumerate(episode_indices):
        sample = ds[sample_idx]
        image = _to_hwc_uint8(sample[image_key], args.image_size)
        raw_state = np.asarray(sample[state_key], dtype=np.float32).reshape(-1)
        print(f"Raw state shape: {raw_state.shape}, values: {raw_state}")
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
        pred_action_chunk = pred_action.reshape(-1, pred_action.shape[-1])[:, :15]
        pred_action_abs_chunk = np.stack(
            [_compose_action_with_state(chunk_step, state)[:15] for chunk_step in pred_action_chunk],
            axis=0,
        )
        pred_action_abs_1d = pred_action_abs_chunk[0]

        epi = int(sample.get("episode_index", -1))
        frame = int(sample.get("frame_index", sample_idx))
        pred_action_1d = _first_action_vec(pred_action_chunk)
        data_action_1d = _first_action_vec(dataset_action)
        pred_15 = np.round(pred_action_1d[:15], 5)
        pred_abs_15 = np.round(pred_action_abs_1d[:15], 5)
        data_15 = np.round(data_action_1d[:15], 5)
        # In 15-dim action, headpose is [8:15], so headpose xyz is [8:11].
        pred_headpose_xyz = np.round(pred_abs_15[8:11], 5)
        data_headpose_xyz = np.round(data_15[8:11], 5)
        diff_headpose_xyz = np.round(pred_headpose_xyz - data_headpose_xyz, 5)
        pred_gripper = np.round(pred_abs_15[7], 5)
        data_gripper = np.round(data_15[7], 5)
        diff_gripper = np.round(pred_gripper - data_gripper, 5)

        print(f"\n=== Frame {i + 1}/{len(episode_indices)} ===")
        print("Model Output Action RAW relative (first 15 / 32):")
        print(pred_15)
        print("Model Output Action converted to ABS (first 15, used for compare):")
        print(pred_abs_15)
        # print("Dataset Raw State (first 15):")
        # print(state[:15])
        print("Dataset Raw Action (first 15):")
        print(data_15)
        print(f"Gripper compare (model_abs vs dataset): {pred_gripper} vs {data_gripper} (diff={diff_gripper})")
        print(f"Headpose XYZ compare (model vs dataset): {pred_headpose_xyz} vs {data_headpose_xyz}")
        print(f"Headpose XYZ diff (model - dataset): {diff_headpose_xyz}")

        if rr is not None and state.shape[0] >= 15:
            rr.set_time("frame", sequence=frame)
            rr.set_time("sample_idx", sequence=sample_idx)
            rr.set_time("episode", sequence=epi)
            rr.log("obs/image", rr.Image(image))

            state_t = state[8:11].astype(np.float32)
            state_q = _quat_normalize(state[11:15][None, :])[0]
            pred_ts = pred_action_abs_chunk[:, 8:11].astype(np.float32)
            pred_qs = _quat_normalize(pred_action_abs_chunk[:, 11:15])
            axis_len = 0.05
            basis = np.eye(3, dtype=np.float32) * axis_len
            state_axes = _quat_rotate(np.repeat(state_q[None, :], 3, axis=0), basis)
            colors = np.asarray([[255, 0, 0], [0, 255, 0], [0, 128, 255]], dtype=np.uint8)
            chunk_axes = _quat_rotate(np.repeat(pred_qs, 3, axis=0), np.tile(basis, (pred_qs.shape[0], 1)))
            chunk_axes_origins = np.repeat(pred_ts, 3, axis=0)
            chunk_axes_colors = np.tile(colors, (pred_qs.shape[0], 1))
            chunk_point_colors = np.tile(np.asarray([[255, 220, 0]], dtype=np.uint8), (pred_ts.shape[0], 1))

            rr.log("headpose/current", rr.Transform3D(translation=state_t, quaternion=rr.Quaternion(xyzw=state_q)))
            rr.log(
                "headpose/action_chunk",
                rr.Transform3D(translation=pred_ts[0], quaternion=rr.Quaternion(xyzw=pred_qs[0])),
            )
            rr.log("headpose/current_axes", rr.Arrows3D(origins=np.repeat(state_t[None, :], 3, axis=0), vectors=state_axes, colors=colors))
            rr.log(
                "headpose/action_chunk_axes",
                rr.Arrows3D(origins=chunk_axes_origins, vectors=chunk_axes, colors=chunk_axes_colors),
            )
            rr.log("headpose/state_point", rr.Points3D([state_t], colors=[[255, 255, 255]], radii=[0.006]))
            rr.log("headpose/action_chunk_points", rr.Points3D(pred_ts, colors=chunk_point_colors, radii=[0.004]))

            if epi not in state_headpose_traj:
                state_headpose_traj[epi] = []
            state_headpose_traj[epi].append(state_t.copy())
            rr.log(
                "headpose/current_traj",
                rr.LineStrips3D([np.asarray(state_headpose_traj[epi], dtype=np.float32)], colors=[[200, 200, 200]]),
            )

    if args.visualize_input_image:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
