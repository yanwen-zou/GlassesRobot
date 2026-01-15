import mujoco
import glfw
import numpy as np
import time

MODEL_PATH = "arm.xml"

# ----------------------
# Load model
# ----------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

ctrl = np.zeros(model.nu)

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
    global ctrl
    if action in (glfw.PRESS, glfw.REPEAT):
        if key in KEY_MAP:
            jid, delta = KEY_MAP[key]
            ctrl[jid] = np.clip(ctrl[jid] + delta, -2.5, 2.5)

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
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)

    viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
    mujoco.mjv_updateScene(model, data, opt, None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, ctx)

    glfw.swap_buffers(window)
    glfw.poll_events()
    time.sleep(model.opt.timestep)

glfw.terminate()
