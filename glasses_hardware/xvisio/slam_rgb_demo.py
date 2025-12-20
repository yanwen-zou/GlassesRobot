"""
Minimal demo that uses the xvsdk bindings to record SLAM poses while
visualizing the RGB camera stream.
"""

import csv
import datetime as dt
import time
from pathlib import Path

import cv2
import numpy as np

import xvsdk


def yuv_to_bgr(frame_width, frame_height, yuv_buffer):
    """Convert IYUV (YUV420 planar) data emitted by the SDK into a BGR image."""
    nheight = int(frame_height * 3 / 2)
    yuv = np.ctypeslib.as_array(yuv_buffer)
    yuv = yuv[: frame_width * nheight]
    yuv = yuv.reshape((nheight, frame_width))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_IYUV)


def main():
    log_path = Path(__file__).with_name(
        f"slam_log_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    xvsdk.init()
    xvsdk.slam_start()
    xvsdk.stereo_start()
    xvsdk.imu_start()
    xvsdk.rgb_start()

    def wait_for_valid_pose(timeout=5.0):
        start = time.time()
        while time.time() - start < timeout:
            (
                _,
                _,
                _,
                _,
                slam_hostTimestamp,
                slam_confidence,
            ) = xvsdk.xv_get_6dof()
            if slam_hostTimestamp.value > 0 and slam_confidence.value > 0:
                return
            time.sleep(0.05)
        print(
            "Warning: SLAM pose not stable after warmup; "
            "logs may contain invalid timestamps."
        )

    wait_for_valid_pose()

    with log_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "host_timestamp",
                "edge_timestamp",
                "confidence",
                "pos_x",
                "pos_y",
                "pos_z",
                "roll",
                "pitch",
                "yaw",
                "quat_w",
                "quat_x",
                "quat_y",
                "quat_z",
            ]
        )

        try:
            while True:
                (
                    position,
                    orientation,
                    quaternion,
                    slam_edgeTimestamp,
                    slam_hostTimestamp,
                    slam_confidence,
                ) = xvsdk.xv_get_6dof()

                if (
                    slam_hostTimestamp.value <= 0
                    or slam_edgeTimestamp.value <= 0
                    or slam_confidence.value <= 0
                ):
                    time.sleep(0.01)
                    continue

                writer.writerow(
                    [
                        slam_hostTimestamp.value,
                        slam_edgeTimestamp.value,
                        slam_confidence.value,
                        position.x,
                        position.y,
                        position.z,
                        orientation.z,
                        orientation.x,
                        orientation.y,
                        quaternion.q0,
                        quaternion.q1,
                        quaternion.q2,
                        quaternion.q3,
                    ]
                )

                (
                    rgb_width,
                    rgb_height,
                    _,
                    _,
                    _,
                    rgb_data,
                    rgb_data_size,
                ) = xvsdk.xv_get_rgb()

                if rgb_data_size.value > 0:
                    bgr = yuv_to_bgr(rgb_width.value, rgb_height.value, rgb_data)
                    cv2.imshow("RGB camera", bgr)

                if cv2.waitKey(1) == ord("q"):
                    break
        finally:
            cv2.destroyAllWindows()
            xvsdk.slam_stop()
            xvsdk.stop()

    print(f"SLAM log written to {log_path}")


if __name__ == "__main__":
    main()
