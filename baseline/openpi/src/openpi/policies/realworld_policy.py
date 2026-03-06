import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-12, None)


def _quat_inv(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    out = q.copy()
    out[..., :3] *= -1.0
    return out


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


def _relative_to_chunk_first_action(actions: np.ndarray) -> np.ndarray:
    """Convert [T, D] absolute action chunks to relative wrt the first action.

    Expected action layout when headpose is present:
    [robot_xyz, robot_quat_wxyz, gripper, head_tx, head_ty, head_tz, head_qx, head_qy, head_qz, head_qw]
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        return actions
    rel = actions.copy()

    p = actions[:, :3]
    q_wxyz = actions[:, 3:7]
    g = actions[:, 7:8]

    q_xyzw = _quat_normalize(_quat_wxyz_to_xyzw(q_wxyz))
    p0 = p[0]
    q0_inv = _quat_inv(q_xyzw[0:1])[0]
    g0 = g[0:1]

    rel[:, :3] = _quat_rotate(q0_inv[None, :], p - p0[None, :])
    q_rel_xyzw = _quat_mul(q0_inv[None, :], q_xyzw)
    rel[:, 3:7] = _quat_xyzw_to_wxyz(_quat_normalize(q_rel_xyzw))
    rel[:, 7:8] = g - g0

    # If headpose is present, it is stored in xyzw convention in the last 7 dims.
    if actions.shape[1] >= 15:
        t = actions[:, 8:11]
        hq = _quat_normalize(actions[:, 11:15])
        t0 = t[0]
        hq0_inv = _quat_inv(hq[0:1])[0]

        rel[:, 8:11] = _quat_rotate(hq0_inv[None, :], t - t0[None, :])
        rel[:, 11:15] = _quat_normalize(_quat_mul(hq0_inv[None, :], hq))

    return rel



@dataclasses.dataclass(frozen=True)
class RealWorldInputs(transforms.DataTransformFn):
    """Inputs for two-arm TCP + single wrist cam datasets."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        if "observation/image" in data: 
            right_wrist_image = _parse_image(data["observation/image"])
            base_image = np.zeros_like(right_wrist_image)
            left_wrist_image = np.zeros_like(right_wrist_image)
        else:
            raise KeyError('Missing image key: expected "observation/image", which represents left camera')


        if "observation/state" in data:
            state = np.asarray(data["observation/state"], dtype=np.float32)
        else:
            raise KeyError('Missing state key: expected ("observation/state").')

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # For pi0-fast, we don't mask missing images.
        if self.model_type == _model.ModelType.PI0_FAST:
            inputs["image_mask"]["left_wrist_0_rgb"] = np.True_
            inputs["image_mask"]["right_wrist_0_rgb"] = np.True_

        if "actions" in data:
            inputs["actions"] = _relative_to_chunk_first_action(data["actions"])
            # print("[DEBUG] Converted actions to relative-to-first chunk:\n", inputs["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RealWorldOutputs(transforms.DataTransformFn):
    """Outputs for two-arm TCP datasets."""

    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
