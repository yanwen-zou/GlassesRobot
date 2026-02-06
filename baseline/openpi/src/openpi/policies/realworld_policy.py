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


@dataclasses.dataclass(frozen=True)
class RealWorldInputs(transforms.DataTransformFn):
    """Inputs for two-arm TCP + single wrist cam datasets."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        wrist_image = _parse_image(datar["observation/wrist_image"]) # only head cam input
        base_image = np.zeros_like(wrist_image)
        right_wrist_image = np.zeros_like(wrist_image)

        tcp_arm = np.asarray(data["observation/tcp_arm"], dtype=np.float32)
        tcp_head = np.asarray(data["observation/tcp_head"], dtype=np.float32)
        state = np.concatenate([tcp_arm, tcp_head], axis=-1)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        # For pi0-fast, we don't mask missing images.
        if self.model_type == _model.ModelType.PI0_FAST:
            inputs["image_mask"]["base_0_rgb"] = np.True_
            inputs["image_mask"]["right_wrist_0_rgb"] = np.True_

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RealWorldOutputs(transforms.DataTransformFn):
    """Outputs for two-arm TCP datasets."""

    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
