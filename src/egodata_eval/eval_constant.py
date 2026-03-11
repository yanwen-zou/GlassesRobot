from __future__ import annotations

import numpy as np

I2RT_INIT_DURATION = 2.0
I2RT_INIT_STEPS = 80
TASK_CHOICES = ("teapot", "book", "sword", "cup", "bread")
TASK_I2RT_TARGET_RAD = {
    "teapot": np.array([-9.9400e-02,  8.9980e-01,  1.7590e+00, -1.4464e+00,  5.9000e-03, -1.0630e-01, 1.2000e-03], dtype=np.float32),
    # "teapot": np.array([-9.9400e-02,  8.9980e-01,  1.7590e+00, -1.1864e+00,  5.9000e-03, -1.0630e-01, 1.2000e-03], dtype=np.float32),
    "book": np.array([-0.0657, 0.5615, 0.7590, -0.2887, 0.1330, 0.0114, 0.0013], dtype=np.float32),
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
        -0.2397,
        -0.819,
        -0.0861, 
        0.0013
        ], dtype=np.float32),
    # "bread": np.array([
    #     -1.529e-01,
    #     8.998e-01,
    #     1.459e+00,
    #     -9.960e-01,
    #     2.500e-03,
    #     -6.240e-02,
    #     1.200e-03,
    # ], dtype=np.float32),
    "bread": np.array([
        -1.529e-01,
        8.998e-01,
        1.459e+00,
        -9.960e-01,
        2.500e-03,
        -6.240e-02,
        1.200e-03,
    ], dtype=np.float32),
}

# Task-specific tcp->object transform in robot frame convention.
# Placeholder values: identity matrices. Replace per-task after hand-eye calibration.
TASK_TCP_TO_OBJECT_SE3 = {
    "teapot": np.array(
        [
            [0.81253284, -0.57030559, -0.12059095, -0.08000000],
            [-0.00574543, -0.21470079, 0.97666276, 0.00000000],
            [-0.58288735, -0.79287803, -0.17772797, 0.15000010],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ],
        dtype=np.float32,
    ),
    "book": np.array(
        [
            [1.00000000, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 1.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 1.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ],
        dtype=np.float32,
    ),
    "sword": np.array(
        [
            [1.00000000, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 1.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 1.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ],
        dtype=np.float32,
    ),
    "cup": np.array(
        [
            [1.00000000, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 1.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 1.00000000, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ],
        dtype=np.float32,
    ),
    "bread": np.array(
        [
            [0.95991260, -0.21833697, -0.17577569, 0.00000000],
            [-0.17876323, -0.95987475, 0.21606553, 0.00000000],
            [-0.21589755, -0.17598169, -0.96042633, 0.15000001],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ],
        dtype=np.float32,
    ),
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
STEPS_TO_EXECUTE = 7
STEPS_HEAD_TO_EXECUTE = 7
GRIP_OPEN_THRESH = {
    "teapot": 0.7,
    "book": 0.8,
    "sword": 0.7,
    "cup": 0.7,
    "bread": 0.85,
}
GRIPPER_OPEN_WIDTH_DEFAULT = 0.085
LOOP_SLEEP_SEC = 0.2

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
