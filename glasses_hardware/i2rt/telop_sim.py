import mujoco
import glfw
import numpy as np
import time
from i2rt.robots.utils import YAM_XML_PATH
MODEL_PATH = YAM_XML_PATH

# ----------------------
# Load model
# ----------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
qpos_target = data.qpos.copy()
jnt_limits = model.jnt_range.copy() if model.jnt_range.size else None

# ----------------------
# Keyboard mapping
# ----------------------
KEY_MAP = {
    glfw.KEY_Q: (0, +0.05),
    glfw.KEY_A: (0, -0.05),
    glfw.KEY_W: (1, +0.05),
    glfw.KEY_S: (1, -0.05),
    glfw.KEY_E: (2, +0.05),
    glfw.KEY_D: (2, -0.05),
}

# ----------------------
# GLFW init
# ----------------------
if not glfw.init():
    raise RuntimeError("GLFW init failed")

window = glfw.create_window(1200, 900, "MuJoCo Teleop", None, None)
glfw.make_context_current(window)

# ----------------------
# Keyboard callback
# ----------------------
def key_callback(window, key, scancode, action, mods):
    global qpos_target
    if action in (glfw.PRESS, glfw.REPEAT):
        if key in KEY_MAP:
            jid, delta = KEY_MAP[key]
            if jid >= qpos_target.shape[0]:
                return
            new_val = qpos_target[jid] + delta
            if jnt_limits is not None and jid < jnt_limits.shape[0]:
                low, high = jnt_limits[jid]
                new_val = np.clip(new_val, low, high)
            qpos_target[jid] = new_val

glfw.set_key_callback(window, key_callback)

# ----------------------
# Camera & scene
# ----------------------
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=1000)
ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

cam.distance = 2.5
cam.azimuth = 90
cam.elevation = -30

# ----------------------
# Main loop
# ----------------------
while not glfw.window_should_close(window):
    data.qpos[:] = qpos_target
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
    mujoco.mjv_updateScene(model, data, opt, None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, ctx)

    glfw.swap_buffers(window)
    glfw.poll_events()
    time.sleep(model.opt.timestep)

glfw.terminate()
