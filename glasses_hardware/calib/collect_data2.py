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
import glob

ROBOT_IP = '192.168.2.100'
CAM_HAND = '244222073667'
CAM_SIDE = '750612070265'
FPS = 30
DIR = os.path.join("/home/mzc/calib", "0915")
replay = '/home/mzc/calib/0814'
# replay = None

if __name__ == '__main__':
    robot = FlexivRobot(ROBOT_IP)
    if replay:
        flangeposList = glob.glob(f"{replay}/*.txt")
        flangeposList.sort()
        poses = [np.loadtxt(filename) for filename in flangeposList]

    else:
        robot.send_tcp_pose((0.4,0,0.5,0,0,1,0))
        # robot.robot.SwitchMode(robot.mode.NRT_PLAN_EXECUTION)
        # robot.robot.ExecutePlan("PLAN-FreeDriveAuto")
    

    cam_hand = CameraD400(CAM_HAND, fps=FPS)
    cam_side = CameraD400(CAM_SIDE, fps=FPS)

    os.makedirs(DIR)
    dir_hand = os.path.join(DIR, 'hand')
    dir_side = os.path.join(DIR, 'side')
    os.mkdir(dir_hand)
    os.mkdir(dir_side)
    k = Keyboard()
    for i in range(30):
        if replay:
            if i>= len(poses):
                break
            robot.send_tcp_pose(poses[i])
            # input()
            # continue
        # time.sleep(0.1)
        while True:
            k.done = False
            data = None
            while not k.done:
                curr_time = int(time.time() * 1000)
                color_image_hand, depth_image_hand = cam_hand.get_data()
                color_image, depth_image = cam_side.get_data()
                # color_image, depth_image = color_image_hand, depth_image_hand
                tcpPose, jointPose, tcpVel, jointVel = robot.get_robot_state()
                flag1, result_img = checkChessboard(color_image)
                flag2, result_img_hand = checkChessboard(color_image_hand)
                cv2.imshow('side', result_img)
                cv2.imshow('hand', result_img_hand)
                cv2.waitKey(100)
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                result_img_hand = cv2.cvtColor(
                    result_img_hand, cv2.COLOR_BGR2RGB)

                if flag1 and flag2:
                    data = curr_time, color_image_hand, depth_image_hand, color_image, depth_image, result_img, result_img_hand, tcpPose, jointPose
            if data is not None:
                break

        curr_time, color_image_hand, depth_image_hand, color_image, depth_image, result_img, result_img_hand, tcpPose, jointPose = data
        print(i, 'saved')
        Image.fromarray(result_img_hand).save(
            os.path.join(dir_hand, f'{curr_time}r.png'))
        Image.fromarray(color_image_hand).save(
            os.path.join(dir_hand, f'{curr_time}c.png'))
        Image.fromarray(depth_image_hand).save(
            os.path.join(dir_hand, f'{curr_time}d.png'))

        Image.fromarray(result_img).save(
            os.path.join(dir_side, f'{curr_time}r.png'))
        Image.fromarray(color_image).save(
            os.path.join(dir_side, f'{curr_time}c.png'))
        Image.fromarray(depth_image).save(
            os.path.join(dir_side, f'{curr_time}d.png'))
        np.savetxt(os.path.join(DIR, f'{curr_time}.txt'), tcpPose)
