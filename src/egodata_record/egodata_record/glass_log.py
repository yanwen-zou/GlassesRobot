import socket
import pyzed.sl as sl
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# ========== 初始化 ZED ==========
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
init_params.coordinate_units = sl.UNIT.METER

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Camera Open : " + repr(err) + ". Exit program.")
    exit(1)

tracking_params = sl.PositionalTrackingParameters()
err = zed.enable_positional_tracking(tracking_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Enable tracking error:", err)
    zed.close()
    exit(1)

runtime_params = sl.RuntimeParameters()
zed_pose = sl.Pose()

# ========== 存储轨迹 ==========
glasses_pose = []  # AI眼镜 pose
zed_traj = []      # ZED tracking pose

recording = False
running = True

def start_recording():
    global recording, glasses_pose, zed_traj
    recording = True
    glasses_pose = []
    zed_traj = []
    print("🎬 Start recording trajectories...")

def stop_recording():
    global recording, running
    recording = False
    running = False
    print("🛑 Stop recording")

while running:
    # 1) 接收 UDP 消息
    try:
        data, addr = sock.recvfrom(1024)
        msg = data.decode().strip().split(",")
        if msg[0] == "start":
            start_recording()
        elif msg[0] == "stop":
            stop_recording()
        elif msg[0] == "pose" and recording:
            x, y, z ,rx, ry, rz, rw= map(float, msg[1:8])
            print(f"Received pose: x={x}, y={y}, z={z}, rx={rx}, ry={ry}, rz={rz}, rw={rw}")
            # glasses_pose.append([x, y, z])
    except BlockingIOError:
        pass
# 清理
sock.close()
