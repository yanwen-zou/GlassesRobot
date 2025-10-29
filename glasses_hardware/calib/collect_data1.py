from device.robot import FlexivRobot
from device.camera import CameraD400
import time
import numpy as np
from PIL import Image
import os
import cv2
from scipy.spatial.transform.rotation import Rotation as R
from device.keyboard import Keyboard

from calib.utils import checkChessboard

ROBOT_IP = '192.168.2.100'
CAM_HAND = '135122070361'
CAM_SIDE = '135122075425'
FPS = 30
DIR = os.path.join("/home/mzc/calib", "cali_hand")

if __name__ == '__main__':
    robot = FlexivRobot(ROBOT_IP)
    robot.robot.setMode(robot.mode.NRT_PLAN_EXECUTION)
    robot.robot.executePlan("PLAN-FreeDriveAuto")

    cam = CameraD400(CAM_HAND, fps=FPS)
    os.makedirs(DIR)
    k = Keyboard()
    for i in range(30):
        # time.sleep(0.1)
        while True:
            k.done = False
            data = None
            while not k.done:
                curr_time = int(time.time() * 1000)
                color_image, depth_image = cam.get_data()
                tcpPose, jointPose, tcpVel, jointVel = robot.get_robot_state()
                flag, result_img = checkChessboard(color_image)
                cv2.imshow('color', result_img)
                cv2.waitKey(100)
                if flag:
                    data = curr_time, tcpPose, jointPose, color_image, depth_image
            if data is not None:
                break
        curr_time, tcpPose, jointPose, color_image, depth_image = data
        Image.fromarray(color_image).save(
            os.path.join(DIR, f'{curr_time}c.png'))
        Image.fromarray(depth_image).save(
            os.path.join(DIR, f'{curr_time}d.png'))
        np.savetxt(os.path.join(DIR, f'{curr_time}t.txt'), tcpPose)
        np.savetxt(os.path.join(DIR, f'{curr_time}j.txt'), jointPose)
        print(f'saved: {i}')
