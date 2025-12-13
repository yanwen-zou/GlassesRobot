from re import M
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
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
    """SLERP 四元数插值"""
    return R.slerp(0, 1, [R.from_quat(q1), R.from_quat(q2)])(t).as_quat()

def interpolate_pose(p1, p2, q1, q2, num=50):
    """生成插值 pose 序列"""
    poses = []
    for i in range(num):
        t = i / (num - 1)
        pos = (1 - t) * p1 + t * p2
        quat = slerp(q1, q2, t)
        poses.append((quat, pos))
    return poses


def main():
    robot = get_yam_robot(channel="can0", zero_gravity_mode=True)
    mj_model = Kinematics(YAM_XML_PATH, "grasp_site")
    """执行 IK 插值路径"""
    
    q_init = np.array([0.0, 0.15, 0.25, -0.2, 0.0, 0.0])
    pose_init = mj_model.fk(q_init)
    # pose = pose_to_transform_matrix([0.16, 0.13, 0.375], [0.51325869, -0.48638001, 0.48638001, -0.51325869], "wxyz")
    q_ik = mj_model.ik(pose_init, "grasp_site", verbose=True)
    print(q_ik)

if __name__ == "__main__":
    main()