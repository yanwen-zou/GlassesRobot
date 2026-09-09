#!/usr/bin/env python3
"""Minimal i2rt FK example: compute end-effector pose from 6 joint angles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _setup_import_path() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    i2rt_root = repo_root / "glasses_hardware" / "i2rt"
    for path in (repo_root, i2rt_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute i2rt end-effector pose with FK.")
    parser.add_argument(
        "--q",
        type=float,
        nargs=6,
        required=True,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="6 joint angles in radians.",
    )
    parser.add_argument("--site", type=str, default="grasp_site", help="MuJoCo site name.")
    return parser.parse_args()


def main() -> None:
    _setup_import_path()

    from i2rt.robots.kinematics_mj import Kinematics
    from i2rt.robots.utils import YAM_XML_PATH

    args = parse_args()
    q = np.asarray(args.q, dtype=np.float32)

    kin = Kinematics(YAM_XML_PATH, args.site)
    T_base_ee = kin.fk(q).astype(np.float32)

    print("q(rad):", q)
    print("T_base_ee:")
    print(T_base_ee)
    print("xyz:", T_base_ee[:3, 3])


if __name__ == "__main__":
    main()
