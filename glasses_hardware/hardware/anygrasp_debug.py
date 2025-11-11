import argparse
import os
import sys
import time
import numpy as np
import cv2
from PIL import Image  # noqa: F401 (kept for parity with demo dependencies)

# Add AnyGrasp SDK path
ANYGRASP_DIR = os.path.join(os.path.dirname(__file__), "..", "anygrasp_sdk", "grasp_detection")
if ANYGRASP_DIR not in sys.path:
    sys.path.insert(0, ANYGRASP_DIR)

from gsnet import AnyGrasp  # type: ignore  # noqa: E402
from graspnetAPI import GraspGroup  # type: ignore  # noqa: E402

FlexivRobotType = None
FlexivGripperType = None

try:
    from my_device.camera import CameraD400  # type: ignore  # noqa: E402
    from my_device.macros import CAM_SERIAL  # type: ignore  # noqa: E402
    from my_device.robot import FlexivRobot as FlexivRobotType, FlexivGripper as FlexivGripperType  # type: ignore  # noqa: E402
    from utils import load_cam_to_tcp, matrix_to_pose, pose_to_matrix  # type: ignore  # noqa: E402
except ImportError:
    try:
        from glasses_hardware.hardware.my_device.camera import CameraD400  # type: ignore  # noqa: E402
        from glasses_hardware.hardware.my_device.macros import CAM_SERIAL  # type: ignore  # noqa: E402
        from glasses_hardware.hardware.my_device.robot import (  # type: ignore  # noqa: E402
            FlexivRobot as FlexivRobotType,
            FlexivGripper as FlexivGripperType,
        )
        from glasses_hardware.hardware.utils import load_cam_to_tcp, matrix_to_pose, pose_to_matrix  # type: ignore  # noqa: E402
    except ImportError:
        from glasses_hardware.hardware.my_device.camera import CameraD400  # type: ignore  # noqa: E402
        from glasses_hardware.hardware.my_device.macros import CAM_SERIAL  # type: ignore  # noqa: E402
        from glasses_hardware.hardware.utils import load_cam_to_tcp, matrix_to_pose, pose_to_matrix  # type: ignore  # noqa: E402


class RealFlexivRobot:
    """Thin wrapper to provide a unified interface for hardware and simulation control."""

    def __init__(self) -> None:
        if FlexivRobotType is None or FlexivGripperType is None:
            raise RuntimeError(
                "Flexiv hardware interfaces are unavailable. Run with simulation (default) or install flexiv RDK."
            )
        self.robot = FlexivRobotType()

    def get_tcp_pose(self) -> np.ndarray:
        return self.robot.get_tcp_pose()

    def send_tcp_pose(self, tcp: np.ndarray) -> None:
        self.robot.send_tcp_pose(tcp)

    def wait(self, duration: float) -> None:
        time.sleep(duration)

    def stop(self) -> None:
        self.robot.stop()


class SimFlexivGripper:
    """Simple gripper stub for PyBullet playback."""

    def __init__(self, max_width: float = 0.085) -> None:
        self.max_width = max_width
        self.current_width = max_width

    def move(self, width: float) -> None:
        width = max(0.0, min(self.max_width, float(width)))
        self.current_width = width
        print(f"[SimGripper] target width: {width:.3f} m")


class SimFlexivRobot:
    """Basic PyBullet simulation for the Flexiv arm."""

    def __init__(self, gui: bool = False, motion_time: float = 2.5) -> None:
        try:
            import pybullet as p  # type: ignore
            import pybullet_data  # type: ignore
        except ImportError as exc:
            raise ImportError("PyBullet is required for simulation. Install with `pip install pybullet`.") from exc

        self._p = p
        self.motion_time = max(0.1, float(motion_time))
        self.dt = 1.0 / 240.0
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.resetSimulation(physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        resource_dir = os.path.join(os.path.dirname(__file__), "my_device", "flexiv_rdk", "resources")
        p.setAdditionalSearchPath(resource_dir, physicsClientId=self.client)
        urdf_path = os.path.join(resource_dir, "flexiv_rizon4s_kinematics.urdf")

        flags = p.URDF_USE_SELF_COLLISION | p.URDF_MAINTAIN_LINK_ORDER
        self.robot_id = p.loadURDF(
            urdf_path,
            useFixedBase=True,
            flags=flags,
            physicsClientId=self.client,
        )

        self.joint_indices = []
        self.ee_link_index = -1
        self.home_joint_pos = np.array([0.218, 0.211, -0.035, 2.181, 0.021, 0.884, 0.169], dtype=np.float32)

        for idx in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            joint_info = p.getJointInfo(self.robot_id, idx, physicsClientId=self.client)
            if joint_info[2] != p.JOINT_FIXED:
                self.joint_indices.append(idx)
            link_name = joint_info[12].decode("utf-8")
            if link_name == "flange":
                self.ee_link_index = idx

        if self.ee_link_index < 0:
            raise RuntimeError("Failed to locate flange link in URDF.")

        for joint_id, joint_angle in zip(self.joint_indices, self.home_joint_pos):
            p.resetJointState(self.robot_id, joint_id, float(joint_angle), physicsClientId=self.client)

        for _ in range(20):
            p.stepSimulation(physicsClientId=self.client)

    def _xyzw(self, wxyz: np.ndarray) -> tuple:
        return float(wxyz[1]), float(wxyz[2]), float(wxyz[3]), float(wxyz[0])

    def _step(self, duration: float) -> None:
        steps = max(1, int(duration / self.dt))
        for _ in range(steps):
            self._p.stepSimulation(physicsClientId=self.client)
            time.sleep(self.dt)

    def get_tcp_pose(self) -> np.ndarray:
        state = self._p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        pos = state[4]
        orn = state[5]
        return np.array([pos[0], pos[1], pos[2], orn[3], orn[0], orn[1], orn[2]], dtype=np.float64)

    def send_joint_pose(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        target = []
        for idx in range(len(self.joint_indices)):
            target.append(float(q[idx] if idx < len(q) else q[-1]))

        self._p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            self._p.POSITION_CONTROL,
            targetPositions=target,
            forces=[150.0] * len(self.joint_indices),
            physicsClientId=self.client,
        )
        self._step(self.motion_time)

    def send_tcp_pose(self, tcp: np.ndarray) -> None:
        tcp = np.asarray(tcp, dtype=np.float64)
        target_pos = tcp[:3]
        target_orn = self._xyzw(tcp[3:])
        ik_solution = self._p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            target_pos,
            target_orn,
            physicsClientId=self.client,
            maxNumIterations=200,
            residualThreshold=1e-4,
        )
        self.send_joint_pose(np.array(ik_solution[: len(self.joint_indices)], dtype=np.float64))

    def wait(self, duration: float) -> None:
        if duration <= 0:
            return
        self._step(duration)

    def stop(self) -> None:
        if self.client is not None and self._p.isConnected(self.client):
            self._p.disconnect(self.client)
            self.client = None



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AnyGrasp detection using cam2 RealSense stream and execute grasp.")
    parser.add_argument(
        "--checkpoint_path",
        default="glasses_hardware/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar",
        help="Path to AnyGrasp detection checkpoint.",
    )
    parser.add_argument("--max_gripper_width", type=float, default=0.085, help="Maximum gripper width (<=0.1m)")
    parser.add_argument("--gripper_height", type=float, default=0.03, help="Gripper height")
    parser.add_argument("--top_down_grasp", default=True, help="Output top-down grasps only")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Depth scale (default millimeters)")
    parser.add_argument("--z_min", type=float, default=0.0, help="Minimum Z (meters) for workspace filtering")
    parser.add_argument("--z_max", type=float, default=1.5, help="Maximum Z (meters) for workspace filtering")
    parser.add_argument("--debug", action="store_true", help="Enable AnyGrasp debug visualization")
    parser.add_argument("--lift_distance", type=float, default=0.2, help="Retract distance (m) after closing gripper")
    parser.add_argument("--grasp_index", type=int, default=0, help="Index of grasp candidate to execute after sorting")
    parser.add_argument(
        "--cam_to_tcp",
        type=str,
        default="glasses_hardware/calib/eih_camT.npy",
        help="Path to camera-to-TCP transform (eye-in-hand calibration).",
    )
    parser.add_argument(
        "--no_camera_preview",
        action="store_true",
        help="Disable camera preview before capturing a frame.",
    )
    parser.add_argument(
        "--sim_gui",
        action="store_true",
        help="Show PyBullet GUI when running in simulation.",
    )
    parser.add_argument(
        "--real_robot",
        dest="simulate",
        action="store_false",
        help="Execute motion on the physical robot instead of PyBullet simulation.",
    )
    parser.set_defaults(simulate=True)
    parser.add_argument(
        "--sim_motion_time",
        type=float,
        default=2.5,
        help="Seconds to allocate for each simulated Cartesian motion.",
    )
    parser.add_argument(
        "--save_capture",
        action="store_true",
        help="Save captured RGB-D frames to example_data/color_r.png and depth_r.png.",
    )
    return parser


def depth_to_points(depths: np.ndarray, intrinsics: np.ndarray, scale: float) -> np.ndarray:
    """Project depth image to 3D points (shared with demo.py logic)."""
    h, w = depths.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    return np.stack((points_x, points_y, points_z), axis=-1)


def main() -> None:
    parser = build_argparser()
    cfgs = parser.parse_args()
    cfgs.max_gripper_width = max(0.0, min(0.1, cfgs.max_gripper_width))

    detector_cfg = argparse.Namespace(
        checkpoint_path=cfgs.checkpoint_path,
        max_gripper_width=cfgs.max_gripper_width,
        gripper_height=cfgs.gripper_height,
        top_down_grasp=cfgs.top_down_grasp,
        debug=cfgs.debug,
        depth_scale=cfgs.depth_scale,
        z_min=cfgs.z_min,
        z_max=cfgs.z_max,
    )

    anygrasp = AnyGrasp(detector_cfg)
    anygrasp.load_net()

    if cfgs.simulate:
        robot = SimFlexivRobot(gui=cfgs.sim_gui, motion_time=cfgs.sim_motion_time)
        gripper = SimFlexivGripper()
    else:
        robot = RealFlexivRobot()
        gripper = FlexivGripperType(robot.robot)

    tcp_to_cam = load_cam_to_tcp(cfgs.cam_to_tcp)

    cam2_serial = CAM_SERIAL[1] if len(CAM_SERIAL) > 1 else None
    camera = CameraD400(serial=cam2_serial)
    intrinsics = np.array(camera.mtx, dtype=np.float32)

    color_image, depth_image = camera.get_data()

    if cfgs.save_capture:
        example_dir = os.path.join(ANYGRASP_DIR, "example_data")
        os.makedirs(example_dir, exist_ok=True)
        cv2.imwrite(os.path.join(example_dir, "color_r.png"), color_image)
        depth_path = os.path.join(example_dir, "depth_r.png")
        cv2.imwrite(depth_path, depth_image.astype(np.uint16))

    del camera

    color_rgb = color_image[..., ::-1].astype(np.float32) / 255.0
    points = depth_to_points(depth_image.astype(np.float32), intrinsics=intrinsics, scale=cfgs.depth_scale)

    mask = (points[..., 2] > 0) & (points[..., 2] < cfgs.z_max)
    points = points[mask]
    colors = color_rgb[mask]

    if points.size == 0:
        raise RuntimeError("No valid points captured from cam2.")

    points = np.ascontiguousarray(points.astype(np.float32))
    colors = np.ascontiguousarray(colors.astype(np.float32))

    xmin, ymin, _ = points.min(axis=0)
    xmax, ymax, _ = points.max(axis=0)
    lims = [float(xmin), float(xmax), float(ymin), float(ymax), cfgs.z_min, cfgs.z_max]

    gg, cloud = anygrasp.get_grasp(
        points,
        colors,
        lims=lims,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True,
    )

    if len(gg) == 0:
        print("No grasps detected.")
        return
    if cloud is None:
        print("Warning: AnyGrasp returned no point cloud geometry; proceeding with locally reconstructed point cloud.")
        if cfgs.debug:

            import open3d as o3d

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))


    gg = gg.nms().sort_by_score()
    gg_pick = gg[:20]
    print("Top grasp scores:", gg_pick.scores)

    grasp_idx = int(np.clip(cfgs.grasp_index, 0, len(gg_pick) - 1))
    chosen_grasp = gg_pick[grasp_idx]
    print(f"Executing grasp #{grasp_idx} with score {chosen_grasp.score:.4f}")

    tool_rot_y90= np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    cam_T_grasp = np.eye(4, dtype=np.float64)
    cam_T_grasp[:3, :3] = chosen_grasp.rotation_matrix @ tool_rot_y90 # Apply extra Y+90° rotation to align gripper
    cam_T_grasp[:3, 3] = chosen_grasp.translation

    tcp_T_cam = tcp_to_cam
    base_T_tcp = pose_to_matrix(robot.get_tcp_pose())
    base_T_cam = base_T_tcp @ tcp_T_cam
    base_T_grasp = base_T_cam @ cam_T_grasp

    # Preview current camera image and the grasp candidate (projected axes like AnyGrasp demo)
    try:
        img_disp = color_image.copy()
        # Build grasp frame in camera coordinates
        Rg = (chosen_grasp.rotation_matrix @ tool_rot_y90).astype(np.float64)
        tg = chosen_grasp.translation.astype(np.float64)
        # Define short axis segments (meters) in grasp local frame
        L = 0.05  # 5 cm for visualization
        pts_cam = [
            tg,                      # origin
            tg + Rg[:, 0] * L,       # +X (red)
            tg + Rg[:, 1] * L,       # +Y (green)
            tg + Rg[:, 2] * L,       # +Z (blue, approach)
        ]
        def proj(pt):
            X, Y, Z = float(pt[0]), float(pt[1]), float(pt[2])
            if Z <= 1e-6:
                return None
            u = int(intrinsics[0, 0] * X / Z + intrinsics[0, 2])
            v = int(intrinsics[1, 1] * Y / Z + intrinsics[1, 2])
            return (u, v)
        pix = [proj(p) for p in pts_cam]
        # Draw axes if valid
        if all(p is not None for p in pix):
            o, px, py, pz = pix
            cv2.line(img_disp, o, px, (0, 0, 255), 2)   # X - red
            cv2.line(img_disp, o, py, (0, 255, 0), 2)   # Y - green
            cv2.line(img_disp, o, pz, (255, 0, 0), 2)   # Z - blue
            cv2.circle(img_disp, o, 3, (255, 255, 255), -1)
        cv2.putText(img_disp, "Preview grasp candidate: 'y' proceed, 'q' cancel", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 200, 10), 2)
        cv2.imshow("AnyGrasp Preview", img_disp)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("AnyGrasp Preview")
        if key in (ord('q'), 27):
            print("[Preview] Canceled by user.")
            return
    except Exception as e:
        print(f"[Preview] Failed to show preview: {e}")

    pregrasp_pose = base_T_grasp.copy()
    pregrasp_pose[:3, 3] = pregrasp_pose[:3, 3] + np.array([0.0, 0.0, 0.1], dtype=np.float64)

    lift_pose = base_T_grasp.copy()
    lift_pose[:3, 3] = lift_pose[:3, 3] + np.array([0, 0, cfgs.lift_distance])

    target_pose = matrix_to_pose(base_T_grasp) - np.array([0, 0, -0.05, 0, 0, 0, 0], dtype=np.float64) # Higher approach by 5 cm
    pregrasp_tcp = matrix_to_pose(pregrasp_pose)
    lift_tcp = matrix_to_pose(lift_pose)

    open_width = gripper.max_width
    #grasp_width = min(max(chosen_grasp.width * 0.2, 0.0), gripper.max_width)
    grasp_width = 0.0  # fully close gripper

    print("[Grasp] Moving to pre-grasp pose...")
    gripper.move(open_width)
    robot.send_tcp_pose(pregrasp_tcp)
    robot.wait(3.0)

    print("[Grasp] Approaching grasp pose...")
    robot.send_tcp_pose(target_pose)
    robot.wait(2.0)

    print("[Grasp] Closing gripper...")
    gripper.move(grasp_width)
    robot.wait(1.5)

    print("[Grasp] Lifting object...")
    robot.send_tcp_pose(lift_tcp)
    robot.wait(3.0)

    print("[Grasp] Completed.")
    robot.stop()


if __name__ == "__main__":
    main()
