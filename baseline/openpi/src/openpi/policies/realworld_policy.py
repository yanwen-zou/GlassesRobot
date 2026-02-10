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
        if "observation/image" in data: 
            base_image = _parse_image(data["observation/image"])
        else:
            raise KeyError('Missing image key: expected "observation/image", which represents left camera')
        
        left_wrist_image = np.zeros_like(base_image)
        right_wrist_image = np.zeros_like(base_image)

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
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        # For pi0-fast, we don't mask missing images.
        if self.model_type == _model.ModelType.PI0_FAST:
            inputs["image_mask"]["left_wrist_0_rgb"] = np.True_
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
