from re import M
import sys
from pathlib import Path
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# Make repo modules importable when run as a script.
here = Path(__file__).resolve()
repo_root = here.parents[2]  # unity_comm/glasses_hardware/
project_root = repo_root.parent
for path in (project_root, repo_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from glasses_hardware.hardware.my_device.i2rt_robo import I2RT
from i2rt.robots.kinematics_mj import Kinematics
from i2rt.robots.utils import YAM_XML_PATH


def pose_to_transform_matrix(pos: np.ndarray, quat: np.ndarray, quat_format: str = "xyzw") -> np.ndarray:
    """Convert position and quaternion to a 4x4 homogeneous transformation matrix.
    
    Args:
        pos: Position vector [x, y, z] (shape: (3,))
        quat: Quaternion in specified format (shape: (4,))
        quat_format: Format of quaternion, either "xyzw" (scipy default) or "wxyz"
    
    Returns:
        4x4 homogeneous transformation matrix
    """
    pos = np.array(pos).flatten()
    quat = np.array(quat).flatten()
    
    if len(pos) != 3:
        raise ValueError(f"Position must have 3 elements, got {len(pos)}")
    if len(quat) != 4:
        raise ValueError(f"Quaternion must have 4 elements, got {len(quat)}")
    
    # Convert quaternion format if needed
    # scipy expects [x, y, z, w] format
    if quat_format == "wxyz":
        # Convert from [w, x, y, z] to [x, y, z, w]
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    elif quat_format == "xyzw":
        quat_xyzw = quat
    else:
        raise ValueError(f"quat_format must be 'xyzw' or 'wxyz', got {quat_format}")
    
    # Convert quaternion to rotation matrix
    rotation = R.from_quat(quat_xyzw)
    
    # Get 3x3 rotation matrix
    rot_matrix = rotation.as_matrix()
    
    # Create 4x4 homogeneous transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = pos
    
    return transform

def slerp(q1, q2, t):
    """SLERP quaternion interpolation
    
    Args:
        q1: First quaternion in [x, y, z, w] format
        q2: Second quaternion in [x, y, z, w] format
        t: Interpolation parameter in [0, 1]
    
    Returns:
        Interpolated quaternion in [x, y, z, w] format
    """
    q1 = np.array(q1)
    q2 = np.array(q2)
    
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product to determine if we need to flip q2
    dot = np.dot(q1, q2)
    
    # If dot product is negative, flip q2 to take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # Clamp dot product to avoid numerical issues
    dot = np.clip(dot, -1.0, 1.0)
    
    # Compute angle between quaternions
    theta = np.arccos(dot)
    
    # If quaternions are very close, use linear interpolation
    if abs(theta) < 1e-6:
        return (1 - t) * q1 + t * q2
    
    # Standard SLERP formula
    sin_theta = np.sin(theta)
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    result = w1 * q1 + w2 * q2
    # Normalize result
    return result / np.linalg.norm(result)

def interpolate_pose(p1, p2, q1, q2, num=50):
    """生成插值 pose 序列"""
    poses = []
    for i in range(num):
        t = i / (num - 1)
        pos = (1 - t) * p1 + t * p2
        quat = slerp(q1, q2, t)
        poses.append((quat, pos))
    return poses

def vec6_to_transform(vec6: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, rx, ry, rz] (XYZ + rad euler) to 4x4 SE3."""
    vec6 = np.asarray(vec6, dtype=np.float32).flatten()
    if vec6.shape[0] != 6:
        raise ValueError(f"vec6 must have 6 elements, got {vec6.shape}")
    pos = vec6[:3]
    rpy = vec6[3:]
    rot = R.from_euler("xyz", rpy)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot.as_matrix().astype(np.float32)
    T[:3, 3] = pos
    return T

def matrix_to_pos_quat(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 4x4 pose to (pos, quat_xyzw)."""
    pos = np.asarray(T[:3, 3], dtype=np.float32)
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    return pos, quat_xyzw

def main():
    robot = I2RT(channel="can0",home=True, zero_gravity_mode=True)
    #model = Kinematics(YAM_XML_PATH, "link_6") # for pyroki
    model = Kinematics(YAM_XML_PATH, "grasp_site") #for mujoco
    """执行 IK 插值路径"""
    TEST_HANDLE_POSE_1 = {
        "position": [0.16, 0.13, 0.375],
        "quaternion_wxyz": [0.51325869, -0.48638001, 0.48638001, -0.51325869]
    }
    
    TEST_HANDLE_POSE_2 = {
        "position": [0.154, 0.0, 0.48],
        "quaternion_wxyz": [0.51325869, -0.48638001, 0.48638001, -0.51325869]
    }

    TEST_HANDLE_POSE_3 = {
        "position": [0.16, -0.147, 0.375],
        "quaternion_wxyz": [0.51325869, -0.48638001, 0.48638001, -0.51325869]
    }

    q_test_1 = pose_to_transform_matrix(TEST_HANDLE_POSE_1["position"], TEST_HANDLE_POSE_1["quaternion_wxyz"], "wxyz")
    q_test_2 = pose_to_transform_matrix(TEST_HANDLE_POSE_2["position"], TEST_HANDLE_POSE_2["quaternion_wxyz"], "wxyz")
    q_test_3 = pose_to_transform_matrix(TEST_HANDLE_POSE_3["position"], TEST_HANDLE_POSE_3["quaternion_wxyz"], "wxyz")

    q_tests = [q_test_1, q_test_2, q_test_3]

    while True:
        for idx in range(len(q_tests)):
            T_start = q_tests[idx]
            T_end = q_tests[(idx + 1) % len(q_tests)]
            pos_s, quat_s = matrix_to_pos_quat(T_start)
            pos_e, quat_e = matrix_to_pos_quat(T_end)
            for quat, pos in interpolate_pose(pos_s, pos_e, quat_s, quat_e, num=50):
                pose_mat = pose_to_transform_matrix(pos, quat, quat_format="xyzw")
                success, q_ik = model.ik(pose_mat, "grasp_site", verbose=False)
                if not success:
                    continue
                robot.send_joint_pos_rad(q_ik, duration=0.2, steps=10)
                time.sleep(0.01)


if __name__ == "__main__":
    main()
