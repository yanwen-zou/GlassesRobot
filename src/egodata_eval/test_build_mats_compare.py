#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np


def _setup_import_path() -> None:
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    src_root = project_root / "src"
    mba_root = project_root / "MBA"
    for p in (src_root, mba_root, project_root):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def main() -> None:
    _setup_import_path()
    from egodata_eval.visualize_scripts.vis_eval import _build_pose_mats as build_mats_eval
    from egodata_eval.eval_utils import _build_pose_mats as build_mats_utils

    # Batch samples (N, 6): include orthogonal, near-orthogonal and generic cases.
    rot6d = np.array(
        [
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0],      # your sample
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],       # identity-like
            [0.0, 0.0, 1.0, 1.0, 0.1, 0.0],       # mixed axes
            [0.2, -0.8, 0.5, 0.4, 0.3, -0.7],     # generic non-orthogonal input
        ],
        dtype=np.float32,
    )
    trans = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, -0.2, 0.3],
            [0.3, 0.2, -0.1],
            [-0.05, 0.08, 0.12],
        ],
        dtype=np.float32,
    )

    mats_eval = build_mats_eval(trans, rot6d)
    mats_utils = build_mats_utils(trans, rot6d)

    diff = mats_eval - mats_utils

    np.set_printoptions(precision=6, suppress=True)
    print("Batch size:", rot6d.shape[0])
    for i in range(rot6d.shape[0]):
        print(f"\n=== Sample {i} ===")
        print("Input rot6d:", rot6d[i].tolist())
        print("[vis_eval._build_pose_mats] R:")
        print(mats_eval[i, :3, :3])
        print("[eval_utils._build_pose_mats] R:")
        print(mats_utils[i, :3, :3])
        print("sample max abs diff:", float(np.max(np.abs(diff[i]))))
    print("\nMax abs diff (pose 4x4):", float(np.max(np.abs(diff))))
    print("Mean abs diff (pose 4x4):", float(np.mean(np.abs(diff))))
    print("Allclose (atol=1e-6, rtol=1e-6):", bool(np.allclose(mats_eval, mats_utils, atol=1e-6, rtol=1e-6)))


if __name__ == "__main__":
    main()
