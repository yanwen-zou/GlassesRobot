import socket

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)


# ========== 存储轨迹 ==========
glasses_pose = []  # [timestamp, x,y,z,qx,qy,qz,qw]
zed_traj = []      # 同上

recording = False
running = True

print("waiting for msg...")

while running:
    timestamp = time.time()

    # 1) 接收 UDP 消息
    try:
        data, addr = sock.recvfrom(1024)
        msg = data.decode().strip().split(",")
        print(msg)

    except BlockingIOError:
        pass


sock.close()
