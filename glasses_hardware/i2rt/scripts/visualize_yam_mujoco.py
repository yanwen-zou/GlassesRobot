#!/usr/bin/env python3
"""Visualize yam.xml in MuJoCo and show the grasp_site frame."""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from i2rt.robots.utils import YAM_XML_PATH,YAM_GLASS_PATH


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize YAM MuJoCo model with grasp_site frame.")
    ap.add_argument("--xml", type=str, default=YAM_GLASS_PATH, help="Path to MuJoCo XML.")
    ap.add_argument("--site-name", type=str, default="grasp_site", help="Site name to highlight.")
    ap.add_argument("--body-name", type=str, default="camera_payload", help="Body name to mark.")
    ap.add_argument("--marker-size", type=float, default=0.01, help="Marker sphere radius.")
    ap.add_argument("--dt", type=float, default=0.01, help="Viewer update interval.")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    site_id = None
    try:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.site_name)
    except Exception:
        site_id = -1
    if site_id is None or site_id < 0:
        print(f"[WARN] Site '{args.site_name}' not found; showing all site frames.")
        site_id = None

    body_id = -1
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.body_name)
    except Exception:
        body_id = -1
    if body_id < 0:
        print(f"[WARN] Body '{args.body_name}' not found; no body marker shown.")
        body_id = None

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_forward(model, data)
            if site_id is not None:
                viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
                if hasattr(viewer.opt, "framesite"):
                    viewer.opt.framesite = site_id
            if body_id is not None:
                viewer.user_scn.ngeom = 0
                pos = np.asarray(data.xpos[body_id], dtype=np.float64)
                geom = viewer.user_scn.geoms[0]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([args.marker_size, 0.0, 0.0], dtype=np.float64),
                    pos,
                    np.eye(3, dtype=np.float64).reshape(-1),
                    np.array([1.0, 0.2, 0.2, 0.9], dtype=np.float32),
                )
                viewer.user_scn.ngeom = 1
            viewer.sync()

            time_until_next_step = args.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
