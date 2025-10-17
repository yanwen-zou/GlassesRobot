import pyzed.sl as sl
import cv2
import os
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, required=True, help="输出文件夹名")
    args = parser.parse_args()

    out_dir = os.path.join("demo_data_for_pose", args.video_name)
    os.makedirs(out_dir, exist_ok=True)

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err))
        exit(1)

    left_image = sl.Mat()
    right_image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    # 保存视频到新文件夹（缩小分辨率，比如640x360）
    target_size = (640, 360)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_left = cv2.VideoWriter(os.path.join(out_dir, "left.mp4"), fourcc, 30.0, target_size)
    out_right = cv2.VideoWriter(os.path.join(out_dir, "right.mp4"), fourcc, 30.0, target_size)

    print(f"Press 'q' to stop recording... Output folder: {out_dir}")
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            left = left_image.get_data()
            right = right_image.get_data()

            left_bgr = cv2.cvtColor(left, cv2.COLOR_BGRA2BGR)
            right_bgr = cv2.cvtColor(right, cv2.COLOR_BGRA2BGR)

            left_bgr = cv2.resize(left_bgr, target_size)
            right_bgr = cv2.resize(right_bgr, target_size)

            out_left.write(left_bgr)
            out_right.write(right_bgr)

            cv2.imshow("Left", left)
            cv2.imshow("Right", right)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    out_left.release()
    out_right.release()
    zed.close()
    cv2.destroyAllWindows()
    print(f"Recording stopped. Videos saved to {out_dir}/left.mp4 and {out_dir}/right.mp4")

if __name__ == "__main__":
    main()
