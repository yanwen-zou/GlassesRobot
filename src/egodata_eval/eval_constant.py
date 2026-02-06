from __future__ import annotations

import numpy as np

I2RT_INIT_DURATION = 2.0
I2RT_INIT_STEPS = 80
TASK_CHOICES = ("teapot", "book", "sword", "cup")
TASK_I2RT_TARGET_RAD = {
    "teapot": np.array([0.0536, 0.3998, 0.6120, 0.0002, -0.0032, -0.0036, 0.0012], dtype=np.float32),
    "book": np.array([-0.2170, 0.6498, 0.5620, 0.1040, 0.0040, -0.0002, 0.0013], dtype=np.float32),
    "sword": np.deg2rad(np.array([
        -0.2292,
        51.5547,
        70.4222,
        -22.0302,
        0.8824,
        -1.9538,
        0.0917,
    ], dtype=np.float32)),
    "cup": np.array([
        -0.8216,
        0.5913,
        0.7326,
        -0.1097,
        -0.819,
        -0.0861, 
        0.0013
        ], dtype=np.float32),
}

# I2RT headpose-following parameters.
I2RT_MAX_ROT = 0.1
I2RT_CMD_DURATION = 0.2
I2RT_CMD_STEPS = 100

# Trajectory execution parameters.
UPDATE_INTERVAL = 2
STEPS_TO_EXECUTE = 5
STEPS_HEAD_TO_EXECUTE = 6
GRIP_OPEN_THRESH = {
    "teapot": 0.85,
    "book": 0.7,
    "sword": 0.7,
    "cup": 0.7,
}
GRIPPER_OPEN_WIDTH_DEFAULT = 0.085
LOOP_SLEEP_SEC = 0.05

# ZED + UI defaults.
ZED_RESOLUTION = "WVGA"
ZED_FPS = 30
DEPTH_EST_SCALE = 0.75
VIDEO_FPS = 30
WIN_CALIB = "Ball Calibration"
WIN_STREAM = "ZED Stream (click to segment)"

# Base-frame point cloud crop (x, y range).
BASE_CLOUD_X_MIN = -0.1
BASE_CLOUD_Y_MAX = 0.32
BASE_CLOUD_Y_MIN = -0.75

# Default resource paths/topics.
DEFAULT_POSE_TOPIC = "/glasses_pose"
DEFAULT_BASE_TO_ROBOT_TXT = "glasses_hardware/calib/T_robot_base.txt"
DEFAULT_GLASSES_ZED_TXT = "glasses_hardware/calib/T_glasses_zed.txt"
DEFAULT_I2RT_ZED_TXT = "glasses_hardware/calib/T_i2rt_zed.txt"
DEFAULT_MESH_NAME = "book"
CALIB_DIR_REL = "glasses_hardware/calib"
I2RT_SERVER_CHANNEL = "can0"
