"""Validate realworld relative-action transform on a LeRobot dataset without pytest.

Example:
  uv run baseline/openpi/scripts_dataset/check_realworld_relative.py \
    --repo-id data/book_openpi --action-horizon 10 --num-samples 50
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro

from openpi.models import pi0_config
from openpi.policies import realworld_policy
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclass(frozen=True)
class Args:
    # LeRobot repo id or local dataset id under HF_LEROBOT_HOME (e.g. data/book_openpi).
    repo_id: str
    # Number of actions in one chunk.
    action_horizon: int = 10
    # Number of samples to check.
    num_samples: int = 50
    # Start index in dataset.
    start_index: int = 0
    # Absolute tolerance for allclose checks.
    atol: float = 2e-5


def _reference_relative(actions: np.ndarray) -> np.ndarray:
    """Reference implementation using scipy Rotation.

    Expected action format:
      robot: [x,y,z,qw,qx,qy,qz,gripper]
      head : [tx,ty,tz,qx,qy,qz,qw] (optional)
    """
    rel = actions.copy()

    # Robot (always expected).
    p = actions[:, :3]
    q_wxyz = actions[:, 3:7]
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
    g = actions[:, 7:8]

    r0_inv = R.from_quat(q_xyzw[0]).inv()
    rel[:, :3] = r0_inv.apply(p - p[0])
    rel[:, 3:7] = (r0_inv * R.from_quat(q_xyzw)).as_quat()[:, [3, 0, 1, 2]]
    rel[:, 7:8] = g - g[0:1]

    # Optional head pose.
    if actions.shape[1] >= 15:
        hp = actions[:, 8:11]
        hq = actions[:, 11:15]  # xyzw
        hr0_inv = R.from_quat(hq[0]).inv()
        rel[:, 8:11] = hr0_inv.apply(hp - hp[0])
        rel[:, 11:15] = (hr0_inv * R.from_quat(hq)).as_quat()

    return rel.astype(np.float32)


def _check_first_frame_constraints(rel: np.ndarray, atol: float) -> tuple[bool, str]:
    ok = True
    msgs: list[str] = []

    def _acc(cond: bool, msg: str) -> None:
        nonlocal ok
        if not cond:
            ok = False
            msgs.append(msg)

    _acc(np.allclose(rel[0, :3], 0.0, atol=atol), "robot xyz@t0 not zero")
    _acc(np.allclose(rel[0, 3:7], np.array([1.0, 0.0, 0.0, 0.0], np.float32), atol=atol), "robot quat@t0 not identity")
    _acc(np.allclose(rel[0, 7], 0.0, atol=atol), "gripper@t0 not zero")

    if rel.shape[1] >= 15:
        _acc(np.allclose(rel[0, 8:11], 0.0, atol=atol), "head xyz@t0 not zero")
        _acc(
            np.allclose(rel[0, 11:15], np.array([0.0, 0.0, 0.0, 1.0], np.float32), atol=atol),
            "head quat@t0 not identity",
        )

    return ok, "; ".join(msgs)


def main(args: Args) -> None:
    data_cfg = _config.DataConfig(repo_id=args.repo_id)
    model_cfg = pi0_config.Pi0Config(action_horizon=args.action_horizon)
    dataset = _data_loader.create_torch_dataset(data_cfg, args.action_horizon, model_cfg)

    n = min(args.num_samples, len(dataset) - args.start_index)
    if n <= 0:
        raise ValueError("No samples to check. Adjust start_index/num_samples.")

    pass_first = 0
    pass_norm = 0
    pass_ref = 0
    max_abs_err = 0.0
    failures: list[str] = []

    for i in range(args.start_index, args.start_index + n):
        sample = dataset[i]
        actions = np.asarray(sample["actions"], dtype=np.float32)
        rel = realworld_policy._relative_to_chunk_first_action(actions)
        ref = _reference_relative(actions)

        ok_first, msg_first = _check_first_frame_constraints(rel, args.atol)
        pass_first += int(ok_first)

        robot_norm_ok = np.allclose(np.linalg.norm(rel[:, 3:7], axis=-1), 1.0, atol=args.atol)
        head_norm_ok = True
        if rel.shape[1] >= 15:
            head_norm_ok = np.allclose(np.linalg.norm(rel[:, 11:15], axis=-1), 1.0, atol=args.atol)
        ok_norm = bool(robot_norm_ok and head_norm_ok)
        pass_norm += int(ok_norm)

        err = float(np.max(np.abs(rel - ref)))
        max_abs_err = max(max_abs_err, err)
        ok_ref = bool(np.allclose(rel, ref, atol=args.atol))
        pass_ref += int(ok_ref)

        if not (ok_first and ok_norm and ok_ref):
            failures.append(
                f"idx={i} first={ok_first} norm={ok_norm} ref={ok_ref} err={err:.3e} "
                f"{msg_first if not ok_first else ''}".strip()
            )

    print("=== RealWorld Relative Transform Check ===")
    print(f"repo_id: {args.repo_id}")
    print(f"checked_samples: {n}")
    print(f"action_horizon: {args.action_horizon}")
    print(f"atol: {args.atol}")
    print(f"first_frame_constraints: {pass_first}/{n}")
    print(f"quat_unit_norm: {pass_norm}/{n}")
    print(f"match_scipy_reference: {pass_ref}/{n}")
    print(f"max_abs_err: {max_abs_err:.3e}")

    if failures:
        print("\nFailures (up to 20):")
        for line in failures[:20]:
            print(f"- {line}")
        raise SystemExit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
