import numpy as np  
import transforms3d as t3d
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger
from pydantic import BaseModel
from typing import List
from glasses_hardware.hardware.my_device.robot import FlexivRobot, FlexivGripper


class HandMes(BaseModel):
    q: List[float]

    pos: List[float]
    quat: List[float]

    thumbTip: List[float]
    indexTip: List[float]
    middleTip: List[float]
    ringTip: List[float]
    pinkyTip: List[float]

    squeeze: int

    cmd:int

class UnityMes(BaseModel):
    valid:bool
    leftHand:HandMes
    rightHand:HandMes

def unity2zup_right_frame(pos_quat):
    pos_quat*=np.array([1,-1,1,1,-1,1,-1])
    rot_mat = t3d.quaternions.quat2mat(pos_quat[3:])
    pos_vec = pos_quat[:3]
    T=np.eye(4)
    T[:3,:3]= rot_mat
    T[:3,3]=pos_vec
    fit_mat = t3d.euler.axangle2mat([0,1,0],np.pi/2)
    fit_mat = fit_mat@t3d.euler.axangle2mat([0,0,1],-np.pi/2)
    target_rot_mat=fit_mat@rot_mat
    target_rot_mat= target_rot_mat @ t3d.euler.axangle2mat([0, 0, 1], -np.pi / 2)
    target_pos_vec=fit_mat@pos_vec
    target = np.array(target_pos_vec.tolist()+t3d.quaternions.mat2quat(target_rot_mat).tolist())
    return target

robot = FlexivRobot()
gripper = FlexivGripper(robot)

tracking_state = False
start_unity_tcp = None
start_robot_tcp = None


def get_relative_target(unity_tcp):
    global start_unity_tcp, start_robot_tcp
    if start_unity_tcp is None or start_robot_tcp is None:
        return robot.get_tcp_pose()
    target = start_robot_tcp.copy()
    target[:3] = start_robot_tcp[:3] + (unity_tcp[:3] - start_unity_tcp[:3])
    target[3:] = unity_tcp[3:]
    return target

from queue import Queue
qq = Queue()
app = FastAPI()
# @app.post('/unity')
# def unity(mes:UnityMes):
#     qq.put(mes)
#     print(f"mes put in queue: {mes}")
#     return {'status':'ok'}

@app.post('/unity')
async def unity(mes: UnityMes):
    logger.info(f"receive mes: {mes.leftHand.squeeze},{mes.leftHand.cmd}")
    qq.put(mes)
    return {'status': 'ok'}

@app.get('/')
async def root():
    return {'status': 'ok'}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("Validation Error:")
    logger.error(f"Errors: {exc.errors()}")
    logger.error(f"Request Body: {await request.body()}")

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": await request.json()
        },
    )


def MainThread():
    global tracking_state, start_unity_tcp, start_robot_tcp
    while True:
        try:
            while qq.qsize()>1:
                mes: UnityMes = qq.get()
            if not qq.empty():
                mes: UnityMes = qq.get()
            else:
                # print("no mes received")
                continue
            print(f"received mes: {mes}")
        except Exception as e:
            print(e)
            continue

        gripper.move(0.1 - mes.leftHand.squeeze / 9)

        if mes.leftHand.cmd == 3:
            robot.send_joint_pose(robot.home_joint_pos)
            tracking_state = False
            continue

        l_pos_from_unity = unity2zup_right_frame(np.array(mes.leftHand.pos + mes.leftHand.quat))

        if mes.leftHand.cmd == 2:
            if tracking_state:
                print("robot stop tracking")
                tracking_state = False
            else:
                print("robot start tracking")
                start_unity_tcp = l_pos_from_unity.copy()
                start_robot_tcp = robot.get_tcp_pose().copy()
                tracking_state = True

        threshold = 0.5
        target = get_relative_target(l_pos_from_unity)
        current_tcp = robot.get_tcp_pose()
        if np.linalg.norm(target[:3] - current_tcp[:3]) > threshold:
            if tracking_state:
                print("robot lost sync")
            tracking_state = False

        if tracking_state:
            robot.send_tcp_pose(target)


if __name__ == "__main__":
    import threading
    threading.Thread(target=MainThread,daemon=True).start()
    print("===========================================")
    print("start teleoperation")
    uvicorn.run(app,host="192.168.43.14", port=8082, log_level="info")
