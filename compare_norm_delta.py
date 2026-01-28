#!/usr/bin/env python3
import torch

from MBA.dataset.realworld import RealWorldDataset
from MBA.policy.diffusion import DiffusionUNetPolicy


def main():
    torch.manual_seed(0)

    # Create a dummy dataset instance to access _normalize_obj without file I/O.
    dataset = RealWorldDataset.__new__(RealWorldDataset)

    # Minimal policy instance to access _absolute_to_delta.
    policy = DiffusionUNetPolicy(
        action_dim=9,
        horizon=4,
        n_obs_steps=1,
        obs_feature_dim=16,
        enable_mba=False,
        obj_dim=9,
        cond_extra_dim=0,
        obj_pose_mode="abs",
    )

    B, T, D = 2, 4, 9  # 3 trans + 6 rot
    abs_pose = torch.randn(B, T, D)
    base_pose = abs_pose[:, 0, :]

    # Path A: normalize first, then absolute->delta
    norm_abs = abs_pose.clone().reshape(B * T, D)
    norm_abs = dataset._normalize_obj(norm_abs).reshape(B, T, D)
    norm_base = norm_abs[:, 0, :]
    delta_after_norm = policy._absolute_to_delta(norm_abs, norm_base)

    # Path B: absolute->delta first, then normalize
    delta_before_norm = policy._absolute_to_delta(abs_pose, base_pose)
    norm_delta = delta_before_norm.clone().reshape(B * T, D)
    norm_delta = dataset._normalize_obj(norm_delta).reshape(B, T, D)

    diff = delta_after_norm - norm_delta
    diff_trans = diff[..., :3].abs()
    diff_rot = diff[..., 3:].abs()

    print("=== Compare (norm -> abs2delta) vs (abs2delta -> norm) ===")
    print(f"abs_pose shape: {abs_pose.shape}")
    print(f"delta_after_norm shape: {delta_after_norm.shape}")
    print(f"norm_delta shape: {norm_delta.shape}")
    print("")
    print("Translation diff: max {:.6f}, mean {:.6f}".format(
        diff_trans.max().item(), diff_trans.mean().item()
    ))
    print("Rotation diff:    max {:.6f}, mean {:.6f}".format(
        diff_rot.max().item(), diff_rot.mean().item()
    ))


if __name__ == "__main__":
    main()
