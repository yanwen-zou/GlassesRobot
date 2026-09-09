"""
MuJoCo keyboard teleop for Panda TCP pose.

Controls:
    q / e : +X / -X
    a / d : +Y / -Y
    w / s : +Z / -Z
    i / k : +pitch / -pitch
    j / l : +yaw / -yaw
    r     : reset target TCP to current TCP
    x     : exit
"""
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def _prepare_panda_urdf_for_mujoco(urdf_path: Path) -> tuple[Path, Path]:
    """Flatten mesh filenames into a temporary workspace so MuJoCo can import the Panda URDF reliably."""
    asset_root = urdf_path.parent
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{urdf_path.stem}_mujoco_"))
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for mesh_node in root.findall(".//mesh"):
        filename = mesh_node.get("filename")
        if not filename:
            continue
        src_path = asset_root / filename
        if not src_path.exists():
            print(f"[WARN] Missing Panda mesh asset for MuJoCo import: {src_path}")
            continue
        alias_name = re.sub(r"[^A-Za-z0-9_.-]", "__", filename)
        alias_path = temp_dir / alias_name
        if not alias_path.exists():
            alias_path.symlink_to(src_path.resolve())
        mesh_node.set("filename", alias_name)

    temp_urdf_path = temp_dir / urdf_path.name
    tree.write(temp_urdf_path, encoding="utf-8", xml_declaration=True)
    return temp_urdf_path, temp_dir


def _compile_panda_model(mujoco_module) -> tuple[object, object, Path]:
    urdf_path = Path(__file__).resolve().parents[1] / "hardware" / "panda" / "panda_stick.urdf"
    resolved_urdf_path, temp_dir = _prepare_panda_urdf_for_mujoco(urdf_path)
    try:
        spec = mujoco_module.MjSpec.from_file(str(resolved_urdf_path))
        model = spec.compile()
        data = mujoco_module.MjData(model)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return model, data, temp_dir


def _rot_y(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotation_error(target_rot: np.ndarray, current_rot: np.ndarray) -> np.ndarray:
    rot_delta = target_rot @ current_rot.T
    skew = 0.5 * (rot_delta - rot_delta.T)
    err = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)
    trace = np.clip((np.trace(rot_delta) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(trace)
    if angle < 1e-8:
        return err
    return err * (angle / max(np.sin(angle), 1e-8))


def _hinge_qpos_indices(mujoco_module, model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_indices = []
    lower_limits = []
    upper_limits = []
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) != int(mujoco_module.mjtJoint.mjJNT_HINGE):
            continue
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        lower_limits.append(float(model.jnt_range[joint_id, 0]))
        upper_limits.append(float(model.jnt_range[joint_id, 1]))
    return (
        np.asarray(qpos_indices, dtype=np.int32),
        np.asarray(lower_limits, dtype=np.float64),
        np.asarray(upper_limits, dtype=np.float64),
    )


def _body_pose(data, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
    rot = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy()
    return pos, rot


def _tcp_local_transform() -> tuple[np.ndarray, np.ndarray]:
    # panda_link7 -> panda_link8 (0,0,0.107), panda_link8 -> panda_hand (Rz(-pi/4)),
    # panda_hand -> panda_hand_tcp (0,0,0.1034)
    local_pos = np.array([0.0, 0.0, 0.2104], dtype=np.float64)
    local_rot = _rot_z(-np.pi / 4.0)
    return local_pos, local_rot


def _tcp_world_pose(data, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    body_pos, body_rot = _body_pose(data, body_id)
    tcp_local_pos, tcp_local_rot = _tcp_local_transform()
    tcp_pos = body_pos + body_rot @ tcp_local_pos
    tcp_rot = body_rot @ tcp_local_rot
    return tcp_pos, tcp_rot


def _tcp_jacobian(mujoco_module, model, data, body_id: int, qpos_indices: np.ndarray) -> np.ndarray:
    tcp_pos, _ = _tcp_world_pose(data, body_id)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco_module.mj_jac(model, data, jacp, jacr, tcp_pos, body_id)
    return np.vstack([jacp[:, qpos_indices], jacr[:, qpos_indices]])


def _min_singular_value(mujoco_module, model, data, body_id: int, qpos_indices: np.ndarray) -> float:
    singular_values = np.linalg.svd(_tcp_jacobian(mujoco_module, model, data, body_id, qpos_indices), compute_uv=False)
    return float(singular_values[-1])


def _solve_tcp_ik_step(
    mujoco_module,
    model,
    data,
    body_id: int,
    qpos_indices: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    step_gain: float,
    damping: float,
) -> None:
    current_pos, current_rot = _tcp_world_pose(data, body_id)
    tcp_local_pos, _ = _tcp_local_transform()
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco_module.mj_jac(
        model,
        data,
        jacp,
        jacr,
        current_pos,
        body_id,
    )
    jac = np.vstack([jacp[:, qpos_indices], jacr[:, qpos_indices]])
    pos_err = target_pos - current_pos
    rot_err = _rotation_error(target_rot, current_rot)
    err = np.concatenate([pos_err, rot_err], axis=0)
    lhs = jac @ jac.T + damping * np.eye(6, dtype=np.float64)
    dq = jac.T @ np.linalg.solve(lhs, err)
    data.qpos[qpos_indices] = np.clip(data.qpos[qpos_indices] + step_gain * dq, lower_limits, upper_limits)
    data.qvel[:] = 0.0
    mujoco_module.mj_forward(model, data)


def _append_sphere_markers(mujoco_module, scene, points: list[np.ndarray], rgba: np.ndarray, radius: float) -> None:
    max_points = min(len(points), max(0, scene.maxgeom - scene.ngeom))
    for pos in points[-max_points:]:
        geom = scene.geoms[scene.ngeom]
        mujoco_module.mjv_initGeom(
            geom,
            mujoco_module.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0], dtype=np.float64),
            pos.astype(np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            rgba.astype(np.float32),
        )
        scene.ngeom += 1


def _interpolate_joint_path(waypoints: list[np.ndarray], steps_per_segment: int) -> list[np.ndarray]:
    path: list[np.ndarray] = []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        for alpha in np.linspace(0.0, 1.0, steps_per_segment, endpoint=False):
            path.append((1.0 - alpha) * start + alpha * end)
    path.append(waypoints[-1].copy())
    return path


class _VideoRecorder:
    def __init__(self, output_path: str | None, fps: float) -> None:
        self.output_path = Path(output_path).expanduser() if output_path else None
        self.fps = float(fps)
        self._writer = None
        self._rgb = None

    @property
    def enabled(self) -> bool:
        return self.output_path is not None

    def capture(self, mujoco_module, viewport, ctx) -> None:
        if not self.enabled:
            return
        width = int(viewport.width)
        height = int(viewport.height)
        if width <= 0 or height <= 0:
            return
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
            self._rgb = np.empty((height, width, 3), dtype=np.uint8)
        elif self._rgb is None or self._rgb.shape[:2] != (height, width):
            raise RuntimeError("Viewport size changed during recording; fixed-size recording is required.")

        depth = np.empty((height, width), dtype=np.float32)
        mujoco_module.mjr_readPixels(self._rgb, depth, viewport, ctx)
        frame_bgr = cv2.cvtColor(np.flipud(self._rgb), cv2.COLOR_RGB2BGR)
        self._writer.write(frame_bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def run_panda_singularity_demo_mujoco(
    control_hz: float = 60.0,
    end_effector_body_name: str = "panda_link7",
    record_path: str | None = None,
) -> None:
    try:
        import glfw  # type: ignore
        import mujoco  # type: ignore
    except ImportError as exc:
        raise ImportError("MuJoCo singularity demo requires `mujoco` and `glfw` in the active environment.") from exc

    model, data, temp_dir = _compile_panda_model(mujoco)
    qpos_indices, lower_limits, upper_limits = _hinge_qpos_indices(mujoco, model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body_name)
    if body_id < 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Body '{end_effector_body_name}' not found in Panda model.")

    if not glfw.init():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("GLFW init failed.")

    window = glfw.create_window(1280, 960, "MuJoCo Panda Singularity Demo", None, None)
    if window is None:
        glfw.terminate()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("Failed to create GLFW window.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    home_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
    pre_sing_q = np.array([0.0, -0.2, 0.0, -1.2, 0.0, 0.8, 0.785], dtype=np.float64)
    near_sing_q = np.array([0.0, 0.2, 0.0, -0.2, 0.0, 0.2, 0.785], dtype=np.float64)
    pass_sing_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.785], dtype=np.float64)
    retreat_q = np.array([0.0, -0.5, 0.0, -1.8, 0.0, 1.2, 0.785], dtype=np.float64)
    waypoints = [
        np.clip(home_q, lower_limits, upper_limits),
        np.clip(pre_sing_q, lower_limits, upper_limits),
        np.clip(near_sing_q, lower_limits, upper_limits),
        np.clip(pass_sing_q, lower_limits, upper_limits),
        np.clip(retreat_q, lower_limits, upper_limits),
        np.clip(home_q, lower_limits, upper_limits),
    ]
    segment_labels = [
        "home -> approach",
        "approach -> near singular",
        "near singular -> pass through",
        "pass through -> retreat",
        "retreat -> home",
    ]
    steps_per_segment = 180
    path = _interpolate_joint_path(waypoints, steps_per_segment)
    stage_breaks = [steps_per_segment * (idx + 1) for idx in range(len(segment_labels))]

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=4000)
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 1.4
    cam.azimuth = 45.0
    cam.elevation = -35.0
    cam.lookat[:] = np.array([0.4, 0.0, 0.35], dtype=np.float64)

    trail_points: list[np.ndarray] = []
    frame_dt = 1.0 / max(control_hz, 1.0)
    frame_idx = 0
    paused = False
    recorder = _VideoRecorder(record_path, fps=max(control_hz, 1.0))

    def key_callback(window_handle, key, scancode, action, mods) -> None:
        nonlocal paused, frame_idx
        del window_handle, scancode, mods
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_X:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            paused = not paused
        elif key == glfw.KEY_R:
            frame_idx = 0
            trail_points.clear()
            paused = False

    glfw.set_key_callback(window, key_callback)

    try:
        while not glfw.window_should_close(window):
            if not paused:
                q_target = path[frame_idx]
                data.qpos[qpos_indices] = q_target
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                tcp_pos, _ = _tcp_world_pose(data, body_id)
                trail_points.append(tcp_pos.copy())
                if len(trail_points) > 300:
                    trail_points.pop(0)
                frame_idx = (frame_idx + 1) % len(path)

            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            _append_sphere_markers(
                mujoco,
                scene,
                trail_points,
                rgba=np.array([0.95, 0.3, 0.15, 0.6], dtype=np.float32),
                radius=0.006,
            )
            mujoco.mjr_render(viewport, scene, ctx)

            stage_idx = min(frame_idx // steps_per_segment, len(segment_labels) - 1)
            min_sigma = _min_singular_value(mujoco, model, data, body_id, qpos_indices)
            tcp_pos, _ = _tcp_world_pose(data, body_id)
            controls_left = "\n".join(
                [
                    "Singularity Demo",
                    "space : pause/resume",
                    "r     : restart",
                    "x     : exit",
                ]
            )
            controls_right = "\n".join(["", "", "", ""])
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                controls_left,
                controls_right,
                ctx,
            )

            status_left = "\n".join(
                [
                    "Stage",
                    "Min singular value",
                    "TCP xyz",
                    "Paused",
                ]
            )
            status_right = "\n".join(
                [
                    segment_labels[stage_idx],
                    f"{min_sigma:.6f}",
                    f"[{tcp_pos[0]:+.3f}, {tcp_pos[1]:+.3f}, {tcp_pos[2]:+.3f}]",
                    "yes" if paused else "no",
                ]
            )
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                viewport,
                status_left,
                status_right,
                ctx,
            )
            recorder.capture(mujoco, viewport, ctx)

            glfw.swap_buffers(window)
            glfw.poll_events()
            time.sleep(frame_dt)
    finally:
        recorder.close()
        glfw.destroy_window(window)
        glfw.terminate()
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_panda_tcp_keyboard_teleop_mujoco(
    pos_step: float = 0.005,
    rot_step: float = 0.05,
    control_hz: float = 60.0,
    end_effector_body_name: str = "panda_link7",
    ik_step_gain: float = 0.7,
    ik_damping: float = 1e-4,
    record_path: str | None = None,
) -> None:
    try:
        import glfw  # type: ignore
        import mujoco  # type: ignore
    except ImportError as exc:
        raise ImportError("MuJoCo teleop requires `mujoco` and `glfw` in the active environment.") from exc

    model, data, temp_dir = _compile_panda_model(mujoco)
    qpos_indices, lower_limits, upper_limits = _hinge_qpos_indices(mujoco, model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body_name)
    if body_id < 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Body '{end_effector_body_name}' not found in Panda model.")

    if not glfw.init():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("GLFW init failed.")

    window = glfw.create_window(1280, 960, "MuJoCo Panda TCP Teleop", None, None)
    if window is None:
        glfw.terminate()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("Failed to create GLFW window.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    home_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
    data.qpos[qpos_indices] = np.clip(home_q, lower_limits, upper_limits)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    target_pos, target_rot = _tcp_world_pose(data, body_id)
    command_state = {"current": "idle"}

    def key_callback(window_handle, key, scancode, action, mods) -> None:
        del window_handle, scancode, mods
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        labels = {
            glfw.KEY_Q: "+X (q)",
            glfw.KEY_E: "-X (e)",
            glfw.KEY_A: "+Y (a)",
            glfw.KEY_D: "-Y (d)",
            glfw.KEY_W: "+Z (w)",
            glfw.KEY_S: "-Z (s)",
            glfw.KEY_I: "+pitch (i)",
            glfw.KEY_K: "-pitch (k)",
            glfw.KEY_J: "+yaw (j)",
            glfw.KEY_L: "-yaw (l)",
            glfw.KEY_R: "reset target TCP",
            glfw.KEY_X: "exit",
        }
        if key == glfw.KEY_X:
            glfw.set_window_should_close(window, True)
        if key in labels:
            command_state["current"] = labels[key]

    glfw.set_key_callback(window, key_callback)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=2000)
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 1.4
    cam.azimuth = 45.0
    cam.elevation = -35.0
    cam.lookat[:] = np.array([0.4, 0.0, 0.35], dtype=np.float64)

    frame_dt = 1.0 / max(control_hz, 1.0)
    recorder = _VideoRecorder(record_path, fps=max(control_hz, 1.0))
    try:
        while not glfw.window_should_close(window):
            pressed = None
            if glfw.get_key(window, glfw.KEY_R) in (glfw.PRESS, glfw.REPEAT):
                target_pos, target_rot = _tcp_world_pose(data, body_id)
                pressed = "reset target TCP"
            else:
                translations = {
                    glfw.KEY_Q: np.array([pos_step, 0.0, 0.0], dtype=np.float64),
                    glfw.KEY_E: np.array([-pos_step, 0.0, 0.0], dtype=np.float64),
                    glfw.KEY_A: np.array([0.0, pos_step, 0.0], dtype=np.float64),
                    glfw.KEY_D: np.array([0.0, -pos_step, 0.0], dtype=np.float64),
                    glfw.KEY_W: np.array([0.0, 0.0, pos_step], dtype=np.float64),
                    glfw.KEY_S: np.array([0.0, 0.0, -pos_step], dtype=np.float64),
                }
                for key, delta in translations.items():
                    if glfw.get_key(window, key) in (glfw.PRESS, glfw.REPEAT):
                        target_pos = target_pos + delta
                        pressed = command_state["current"]
                        break
                if pressed is None:
                    if glfw.get_key(window, glfw.KEY_I) in (glfw.PRESS, glfw.REPEAT):
                        target_rot = target_rot @ _rot_y(rot_step)
                        pressed = command_state["current"]
                    elif glfw.get_key(window, glfw.KEY_K) in (glfw.PRESS, glfw.REPEAT):
                        target_rot = target_rot @ _rot_y(-rot_step)
                        pressed = command_state["current"]
                    elif glfw.get_key(window, glfw.KEY_J) in (glfw.PRESS, glfw.REPEAT):
                        target_rot = target_rot @ _rot_z(rot_step)
                        pressed = command_state["current"]
                    elif glfw.get_key(window, glfw.KEY_L) in (glfw.PRESS, glfw.REPEAT):
                        target_rot = target_rot @ _rot_z(-rot_step)
                        pressed = command_state["current"]

            if pressed is not None:
                command_state["current"] = pressed

            _solve_tcp_ik_step(
                mujoco,
                model,
                data,
                body_id,
                qpos_indices,
                lower_limits,
                upper_limits,
                target_pos,
                target_rot,
                step_gain=ik_step_gain,
                damping=ik_damping,
            )

            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, ctx)

            controls_left = "\n".join(
                [
                    "Controls",
                    "q/e : +X / -X",
                    "a/d : +Y / -Y",
                    "w/s : +Z / -Z",
                    "i/k : +pitch / -pitch",
                    "j/l : +yaw / -yaw",
                    "r   : reset target",
                    "x   : exit",
                ]
            )
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                controls_left,
                "\n" * 7,
                ctx,
            )

            current_pos, _ = _tcp_world_pose(data, body_id)
            target_pitch = np.arctan2(target_rot[0, 2], target_rot[2, 2])
            target_yaw = np.arctan2(target_rot[1, 0], target_rot[0, 0])
            status_left = "\n".join(
                [
                    "Current command",
                    "Target TCP xyz",
                    "Target pitch/yaw",
                    "Current TCP xyz",
                ]
            )
            status_right = "\n".join(
                [
                    command_state["current"],
                    f"[{target_pos[0]:+.3f}, {target_pos[1]:+.3f}, {target_pos[2]:+.3f}]",
                    f"[pitch {target_pitch:+.3f}, yaw {target_yaw:+.3f}]",
                    f"[{current_pos[0]:+.3f}, {current_pos[1]:+.3f}, {current_pos[2]:+.3f}]",
                ]
            )
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                viewport,
                status_left,
                status_right,
                ctx,
            )
            recorder.capture(mujoco, viewport, ctx)

            glfw.swap_buffers(window)
            glfw.poll_events()
            time.sleep(frame_dt)
    finally:
        recorder.close()
        glfw.destroy_window(window)
        glfw.terminate()
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_panda_tcp_keyboard_teleop_pybullet(
    pos_step: float = 0.005,
    rot_step: float = 0.05,
    control_hz: float = 60.0,
    end_effector_link_name: str = "panda_link7",
) -> None:
    """Backward-compatible alias kept for existing callers."""
    run_panda_tcp_keyboard_teleop_mujoco(
        pos_step=pos_step,
        rot_step=rot_step,
        control_hz=control_hz,
        end_effector_body_name=end_effector_link_name,
    )


def run_yam_tcp_keyboard_teleop_pybullet(
    pos_step: float = 0.005,
    rot_step: float = 0.05,
    control_hz: float = 60.0,
    end_effector_link_name: str = "panda_link7",
) -> None:
    """Backward-compatible alias kept for old callers."""
    run_panda_tcp_keyboard_teleop_mujoco(
        pos_step=pos_step,
        rot_step=rot_step,
        control_hz=control_hz,
        end_effector_body_name=end_effector_link_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo keyboard teleop for Panda TCP pose.")
    parser.add_argument("--demo", choices=("teleop", "singularity"), default="teleop", help="Run keyboard teleop or a hardcoded singularity demo.")
    parser.add_argument("--pos-step", type=float, default=0.005, help="TCP translation step in meters.")
    parser.add_argument("--rot-step", type=float, default=0.05, help="TCP pitch/yaw step in radians.")
    parser.add_argument("--hz", type=float, default=60.0, help="Render/control frequency.")
    parser.add_argument("--ik-step-gain", type=float, default=0.7, help="Damped least-squares IK step gain.")
    parser.add_argument("--ik-damping", type=float, default=1e-4, help="Damped least-squares IK damping.")
    parser.add_argument("--record", type=str, default=None, help="Optional mp4 output path for recording the viewer.")
    args = parser.parse_args()
    if args.demo == "singularity":
        run_panda_singularity_demo_mujoco(control_hz=args.hz, record_path=args.record)
    else:
        run_panda_tcp_keyboard_teleop_mujoco(
            pos_step=args.pos_step,
            rot_step=args.rot_step,
            control_hz=args.hz,
            ik_step_gain=args.ik_step_gain,
            ik_damping=args.ik_damping,
            record_path=args.record,
        )


if __name__ == "__main__":
    main()
