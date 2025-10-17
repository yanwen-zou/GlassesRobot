#!/usr/bin/env python3
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import pyzed.sl as sl
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String


class StereoVideoRecorder(Node):
    def __init__(self):
        super().__init__('stereo_video_recorder')

        self.recording = False
        self.recording_id = None

        # Declare configurable parameters for output location and frame rate
        workspace_root = self._find_workspace_root(Path(__file__).resolve())
        default_root = str(workspace_root / 'data')

        self.output_dir = Path(
            self.declare_parameter('output_dir', default_root).get_parameter_value().string_value
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recording_dir: Path | None = None

        self.frame_rate = self.declare_parameter('frame_rate', 30.0).get_parameter_value().double_value
        self.downscale_factor = self.declare_parameter('downscale_factor', 0.5).get_parameter_value().double_value
        if self.downscale_factor <= 0:
            raise ValueError('downscale_factor must be positive')

        # ROS interfaces
        self.create_subscription(String, '/control_cmd', self.cmd_callback, 10)
        self.create_subscription(PoseStamped, '/glasses_pose', self.glasses_callback, 10)

        # ZED camera setup
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = int(self.frame_rate)
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.coordinate_units = sl.UNIT.METER

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error('Failed to open ZED camera.')
            raise RuntimeError('ZED camera open failed')

        self.runtime_params = sl.RuntimeParameters()
        self.left_image = sl.Mat()
        self.right_image = sl.Mat()

        self.pose_queue: deque[list[float]] = deque()
        self.last_pose: list[float] | None = None
        self.frame_index = 0
        self.left_dir: Path | None = None
        self.right_dir: Path | None = None
        self.head_pos_dir: Path | None = None

        timer_period = 1.0 / self.frame_rate if self.frame_rate > 0 else 0.033
        self.timer = self.create_timer(timer_period, self.capture_frames)

        self.get_logger().info('Stereo video recorder node ready.')

    def cmd_callback(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd == 'start' and not self.recording:
            self.start_recording()
        elif cmd == 'stop' and self.recording:
            self.stop_recording()

    def glasses_callback(self, msg: PoseStamped):
        if not self.recording:
            return

        pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
        self.pose_queue.append(pose)

    def capture_frames(self):
        if not self.recording:
            return

        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            return

        self.zed.retrieve_image(self.left_image, sl.VIEW.LEFT)
        self.zed.retrieve_image(self.right_image, sl.VIEW.RIGHT)

        left_frame = self.left_image.get_data()
        right_frame = self.right_image.get_data()

        # ZED returns BGRA, convert to BGR before writing
        left_bgr = cv2.cvtColor(left_frame, cv2.COLOR_BGRA2BGR)
        right_bgr = cv2.cvtColor(right_frame, cv2.COLOR_BGRA2BGR)

        if self.downscale_factor != 1.0:
            target_size = (
                max(1, int(left_bgr.shape[1] * self.downscale_factor)),
                max(1, int(left_bgr.shape[0] * self.downscale_factor)),
            )
            left_bgr = cv2.resize(left_bgr, target_size, interpolation=cv2.INTER_AREA)
            right_bgr = cv2.resize(right_bgr, target_size, interpolation=cv2.INTER_AREA)

        if self.left_dir is None or self.right_dir is None or self.head_pos_dir is None:
            self.get_logger().error('Output directories not ready; dropping frame.')
            return

        frame_name = f'{self.frame_index:06d}'
        left_path = self.left_dir / f'{frame_name}.png'
        right_path = self.right_dir / f'{frame_name}.png'
        if not cv2.imwrite(str(left_path), left_bgr):
            self.get_logger().error(f'Failed to write left frame {left_path}.')
        if not cv2.imwrite(str(right_path), right_bgr):
            self.get_logger().error(f'Failed to write right frame {right_path}.')

        pose = None
        while self.pose_queue:
            pose = self.pose_queue.popleft()
        if pose is not None:
            self.last_pose = pose
        elif self.last_pose is not None:
            pose = self.last_pose
        else:
            pose = [float('nan')] * 7
            self.get_logger().warning(
                f'No head pose available for frame {frame_name}; writing NaNs.'
            )

        pose_path = self.head_pos_dir / f'{frame_name}.txt'
        pose_line = ' '.join(f'{value:.6f}' if value == value else 'nan' for value in pose)
        pose_path.write_text(pose_line + '\n')

        self.frame_index += 1

    def start_recording(self):
        self.recording_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.recording_dir = self.output_dir / self.recording_id
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        self.left_dir = self.recording_dir / 'zed_left'
        self.right_dir = self.recording_dir / 'zed_right'
        self.head_pos_dir = self.recording_dir / 'head_pos'
        self.left_dir.mkdir(parents=True, exist_ok=True)
        self.right_dir.mkdir(parents=True, exist_ok=True)
        self.head_pos_dir.mkdir(parents=True, exist_ok=True)
        self.pose_queue.clear()
        self.last_pose = None
        self.frame_index = 0
        self.get_logger().info(f'Start recording session {self.recording_id}.')
        self.recording = True

    def stop_recording(self):
        self.recording = False
        self.get_logger().info(f'Stopping recording session {self.recording_id}.')

        self.recording_dir = None
        self.recording_id = None
        self.left_dir = None
        self.right_dir = None
        self.head_pos_dir = None
        self.pose_queue.clear()
        self.last_pose = None

    @staticmethod
    def _find_workspace_root(start: Path) -> Path:
        for candidate in (start,) + tuple(start.parents):
            if (candidate / 'src').is_dir() and (candidate / 'install').is_dir():
                return candidate
        return start.parent

    def destroy_node(self):
        if self.recording:
            self.stop_recording()
        if self.zed.is_opened():
            self.zed.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StereoVideoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
