from __future__ import annotations

import numpy as np

I2RT_INIT_DURATION = 2.0
I2RT_INIT_STEPS = 80
TASK_CHOICES = ("teapot", "book", "sword", "cup", "bread")
TASK_I2RT_TARGET_RAD = {
    "teapot": np.array([-0.0799, 0.5902, 0.9523, -0.5479, 0.1274, -0.0082, 0.0012], dtype=np.float32),
    "book": np.array([-0.2170, 0.6498, 0.5620, 0.1040, 0.0040, -0.0002, 0.0013], dtype=np.float32),
    "sword": np.array([
        0.0609,
        0.5528,
        0.9009,
        -0.4874,
        0.2257,
        0.0313,
        np.deg2rad(0.0917),
    ], dtype=np.float32),
    "cup": np.array([
        -0.8216,
        0.5913,
        0.7326,
        -0.1097,
        -0.819,
        -0.0861, 
        0.0013
        ], dtype=np.float32),
    "bread": np.array([
        -0.4992,
        0.6492,
        0.5575,
        0.2149,
        -0.3936,
        0.0303,
        0.0013,
    ], dtype=np.float32),
}

# Task-specific calibration init pose (ball_base <- cam).
# Keep values hardcoded to avoid runtime file dependency.
# T_base_cam
calib_init_pose = {
    "teapot": None,
    "book": np.array([
        [0.251526689, 0.088033824, -0.963838353, 0.968090760],
        [0.964353586, 0.061780472, 0.257303971, -0.292739480],
        [0.082197841, -0.994199788, -0.069356296, 0.231739740],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32),
    "sword": None,
    "cup": None,
    "bread": None,
}

# I2RT headpose-following parameters.
I2RT_MAX_ROT = 0.2
I2RT_CMD_DURATION = 0.2
I2RT_CMD_STEPS = 100

# Trajectory execution parameters.
UPDATE_INTERVAL = 2
STEPS_TO_EXECUTE = 5
STEPS_HEAD_TO_EXECUTE = 5
GRIP_OPEN_THRESH = {
    "teapot": 0.85,
    "book": 0.7,
    "sword": 0.7,
    "cup": 0.7,
    "bread": 0.7,
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
