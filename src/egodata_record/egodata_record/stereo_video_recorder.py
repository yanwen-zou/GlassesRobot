#!/usr/bin/env python3
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String

from .udp_sender import UDPSender



class StereoVideoRecorder(Node):
    def __init__(self):
        super().__init__('stereo_video_recorder')

        self.recording = False
        self.recording_id = None
        self._udp_sender = UDPSender()
        self._aruco_message_sent = False

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

        self.marker_length = self.declare_parameter('marker_length', 0.045).get_parameter_value().double_value
        if self.marker_length <= 0:
            raise ValueError('marker_length must be positive')

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

        self.camera_matrix, self.dist_coeffs = self._load_camera_intrinsics()

        if not hasattr(cv2, 'aruco'):
            raise RuntimeError('OpenCV ArUco module is unavailable; install opencv-contrib-python')
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        detector_params_ctor = getattr(cv2.aruco, 'DetectorParameters_create', None)
        if detector_params_ctor is not None:
            self.aruco_params = detector_params_ctor()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters()

        self.last_marker_transform: np.ndarray | None = None
        self.last_detected_marker_id: int | None = None
        self.latest_head_pose: np.ndarray | None = None
        self.calibration_cached = False
        self.waiting_for_head_pose = False
        self.calibrated_transform: np.ndarray | None = None
        self.calibrated_head_pose: np.ndarray | None = None
        self._calibration_log_emitted = False

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
        pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ], dtype=np.float32)
        self.latest_head_pose = pose

        if self.recording:
            self.pose_queue.append(pose.tolist())

    def capture_frames(self):
        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            return

        self.zed.retrieve_image(self.left_image, sl.VIEW.LEFT)
        self.zed.retrieve_image(self.right_image, sl.VIEW.RIGHT)

        left_frame = self.left_image.get_data()
        right_frame = self.right_image.get_data()

        # ZED returns BGRA, convert to BGR before writing
        left_bgr = cv2.cvtColor(left_frame, cv2.COLOR_BGRA2BGR)
        right_bgr = cv2.cvtColor(right_frame, cv2.COLOR_BGRA2BGR)

        self._detect_and_cache_marker(left_bgr)

        if not self.recording:
            return

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

        if isinstance(pose, np.ndarray):
            pose_for_dump = pose
        else:
            pose_for_dump = np.array(pose, dtype=np.float32)

        pose_path = self.head_pos_dir / f'{frame_name}.txt'
        pose_line = ' '.join(f'{value:.6f}' if value == value else 'nan' for value in pose_for_dump)
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
        self._calibration_log_emitted = False
        self._persist_calibration_to_disk()
        self._send_udp('start')

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
        self._send_udp('stop')

    def _send_udp(self, message: str) -> None:
        if self._udp_sender is None:
            return
        try:
            self._udp_sender.send(message)
        except OSError as exc:
            self.get_logger().warning(f'Failed to send UDP message {message!r}: {exc}')

    @staticmethod
    def _find_workspace_root(start: Path) -> Path:
        for candidate in (start,) + tuple(start.parents):
            if (candidate / 'src').is_dir() and (candidate / 'install').is_dir():
                return candidate
        return start.parent

    def _load_camera_intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        info = self.zed.get_camera_information()
        config = getattr(info, 'camera_configuration', None)
        calibration = config.calibration_parameters if config else info.calibration_parameters
        left_cam = calibration.left_cam

        fx = float(getattr(left_cam, 'fx'))
        fy = float(getattr(left_cam, 'fy'))
        cx = float(getattr(left_cam, 'cx'))
        cy = float(getattr(left_cam, 'cy'))

        camera_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        disto = getattr(left_cam, 'disto', None)
        if disto is None:
            disto = getattr(left_cam, 'distortion', None)
        if disto is None:
            disto = getattr(left_cam, 'distortion_coefficients', None)
        dist_list = []
        if disto is not None:
            try:
                dist_list = [float(val) for val in disto]
            except TypeError:
                self.get_logger().warning(f'Unable to parse distortion coefficients: {disto}')
                dist_list = []
        if not dist_list:
            dist_list = [0.0, 0.0, 0.0, 0.0, 0.0]

        dist_coeffs = np.array(dist_list, dtype=np.float32).reshape(-1, 1)
        return camera_matrix, dist_coeffs

    def _detect_and_cache_marker(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None or frame_bgr.size == 0:
            return

        if self.calibration_cached:
            return

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.marker_length,
            self.camera_matrix,
            self.dist_coeffs,
        )
        if rvecs is None or len(rvecs) == 0:
            return

        rvec = rvecs[0].reshape(3)
        tvec = tvecs[0].reshape(3)
        marker_id = int(ids[0])

        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation
        transform[:3, 3] = tvec

        head_pose = self.latest_head_pose.copy() if self.latest_head_pose is not None else None

        self.last_marker_transform = transform.copy()
        self.last_detected_marker_id = marker_id

        if head_pose is None:
            if not self.waiting_for_head_pose:
                self.get_logger().info(
                    'Detected ArUco marker but head pose not yet available; waiting to cache transform.'
                )
                self.waiting_for_head_pose = True
            return

        self.waiting_for_head_pose = False
        self._cache_calibration(transform, head_pose, marker_id)
        self.calibration_cached = True

    def _cache_calibration(self, transform: np.ndarray, head_pose: np.ndarray | None, marker_id: int) -> None:
        self.calibrated_transform = transform.copy()
        self.calibrated_head_pose = head_pose.copy() if head_pose is not None else None

        self.get_logger().info(
            f'Detected ArUco marker {marker_id}; cached transform and head pose in memory.'
        )
        self._persist_calibration_to_disk()
        self._log_calibration_save_path()
        if not self._aruco_message_sent:
            self._send_udp('aruco')
            self._aruco_message_sent = True

    def _persist_calibration_to_disk(self) -> None:
        if self.recording_dir is None:
            return

        if self.calibrated_transform is not None:
            transform_path = self.recording_dir / 'calibrated_transform.txt'
            np.savetxt(transform_path, self.calibrated_transform, fmt='%.6f')

        if self.calibrated_head_pose is not None:
            pose_path = self.recording_dir / 'calibrated_head_pose.txt'
            pose_line = ' '.join(f'{value:.6f}' for value in self.calibrated_head_pose)
            pose_path.write_text(pose_line + '\n')

    def _log_calibration_save_path(self) -> None:
        if self._calibration_log_emitted:
            return
        if self.recording_dir is None:
            return
        self.get_logger().info(f'Persisted calibration outputs under {self.recording_dir}.')
        self._calibration_log_emitted = True

    def destroy_node(self):
        if self.recording:
            self.stop_recording()
        if self.zed.is_opened():
            self.zed.close()
        if self._udp_sender is not None:
            self._udp_sender.close()
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
