from __future__ import annotations

import numpy as np

# I2RT initial joint configuration.
I2RT_TARGET_DEG = [-0.1314, 59.9407, 50.3597, 10.8160, -0.1314, -1.1748]
I2RT_TARGET_RAD = np.deg2rad(I2RT_TARGET_DEG).astype(np.float32)
I2RT_INIT_DURATION = 2.0
I2RT_INIT_STEPS = 80

# I2RT headpose-following parameters.
I2RT_MAX_ROT = 0.1
I2RT_CMD_DURATION = 0.2
I2RT_CMD_STEPS = 100

# Trajectory execution parameters.
UPDATE_INTERVAL = 5
STEPS_TO_EXECUTE = 3
GRIP_OPEN_THRESH = 0.8
GRIPPER_OPEN_WIDTH_DEFAULT = 0.085
LOOP_SLEEP_SEC = 0.05

# ZED + UI defaults.
ZED_RESOLUTION = "WVGA"
ZED_FPS = 30
DEPTH_EST_SCALE = 0.75
VIDEO_FPS = 30
WIN_CALIB = "Ball Calibration"
WIN_STREAM = "ZED Stream (click to segment)"

# Default resource paths/topics.
DEFAULT_POSE_TOPIC = "/glasses_pose"
DEFAULT_BASE_TO_ROBOT_TXT = "glasses_hardware/calib/T_robot_base.txt"
DEFAULT_GLASSES_ZED_TXT = "glasses_hardware/calib/T_glasses_zed.txt"
DEFAULT_I2RT_ZED_TXT = "glasses_hardware/calib/T_i2rt_zed.txt"
DEFAULT_MESH_NAME = "book"
CALIB_DIR_REL = "glasses_hardware/calib"
I2RT_SERVER_CHANNEL = "can0"
