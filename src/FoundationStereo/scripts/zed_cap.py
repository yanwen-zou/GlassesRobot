import pyzed.sl as sl
import cv2
import numpy as np

def main():
    # 创建 ZED 相机对象
    zed = sl.Camera()

    # 初始化参数
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720   # 可改成 HD1080, AUTO 等
    init_params.camera_fps = 30

    # 打开相机
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err) + ". Exit program.")
        exit(1)

    # 创建图像容器
    left_image = sl.Mat()
    right_image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    print("Capturing one stereo pair...")

    # 抓取一帧
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # 获取左右图像
        zed.retrieve_image(left_image, sl.VIEW.LEFT)
        zed.retrieve_image(right_image, sl.VIEW.RIGHT)

        # 转 numpy 格式
        frame_left = left_image.get_data()
        frame_right = right_image.get_data()

        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGRA2BGR)
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGRA2BGR)
        cv2.imwrite("left.png", frame_left)
        cv2.imwrite("right.png", frame_right)
        
        print("Saved left.png and right.png")

    # 关闭相机
    zed.close()

if __name__ == "__main__":
    main()
