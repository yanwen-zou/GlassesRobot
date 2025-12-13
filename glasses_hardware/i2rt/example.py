import numpy as np
import time
from scipy.spatial.transform import Rotation as R


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
    """Generate interpolated pose sequence"""
    poses = []
    for i in range(num):
        t = i / (num - 1) if num > 1 else 0
        pos = (1 - t) * p1 + t * p2
        quat = slerp(q1, q2, t)
        poses.append((quat, pos))
    return poses


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


def run_interpolated_path(robot, kinematics_solver, site_name="grasp_site"):
    """Execute IK interpolated path
    
    Args:
        robot: Robot instance (e.g., from get_yam_robot)
        kinematics_solver: Kinematics instance for IK solving
        site_name: Name of the site for IK solving
    """
    # Define test poses as dictionaries with position and quaternion_wxyz
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

    poses = [TEST_HANDLE_POSE_1, TEST_HANDLE_POSE_2, TEST_HANDLE_POSE_3]

    # Convert to numpy arrays and convert quaternions from wxyz to xyzw format
    positions = [np.array(p["position"]) for p in poses]
    quaternions_wxyz = [np.array(p["quaternion_wxyz"]) for p in poses]
    # Convert wxyz to xyzw for scipy
    quaternions_xyzw = []
    for q_wxyz in quaternions_wxyz:
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])  # [x, y, z, w]
        quaternions_xyzw.append(q_xyzw)

    print("Ready to start interpolation")

    time.sleep(3.0)

    # Get initial joint configuration
    current_q = robot.get_joint_pos()[:6]  # Exclude gripper if present

    # Loop execution (1→2→3→1)
    idx = 0
    while True:
        p1 = positions[idx]
        q1 = quaternions_xyzw[idx]
        p2 = positions[(idx + 1) % 3]
        q2 = quaternions_xyzw[(idx + 1) % 3]

        # Generate 50 interpolated points
        interp_seq = interpolate_pose(p1, p2, q1, q2, num=50)

        print(f"Executing segment {idx+1}: pose {idx} -> pose {(idx+1)%3}")

        # Execute IK for each interpolated pose
        for step, (quat, pos) in enumerate(interp_seq):
            # Convert quaternion and position to transformation matrix
            target_pose = pose_to_transform_matrix(pos, quat, quat_format="xyzw")
            current_q = robot.get_joint_pos()[:6]
            # Solve IK
            success, q_ik = kinematics_solver.ik(
                target_pose, 
                site_name, 
                init_q=current_q,
                verbose=False
            )

            if success:
                # Command robot to the IK solution (add gripper position if needed)
                if len(robot.get_joint_pos()) == 7:
                    # Include gripper position
                    q_command = np.append(q_ik, robot.get_joint_pos()[6])
                else:
                    q_command = q_ik
                # print(f"q_ik: {q_ik}")
            else:
                if len(robot.get_joint_pos()) == 7:
                    # Include gripper position
                    q_command = np.append(q_ik, robot.get_joint_pos()[6])
                else:
                    q_command = q_ik
                print(f"  Warning: IK failed at step {step+1}/50")

            robot.command_joint_pos(q_command)
            
            # print(f"  Step {step+1}/50: pos={pos}, quat={quat}")
            time.sleep(0.05)  # Control trajectory smoothness

        idx = (idx + 1) % 3
        time.sleep(1.0)


def main():
    """Main function to run the interpolated path"""
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.kinematics import Kinematics
    from i2rt.robots.utils import YAM_XML_PATH
    
    # Initialize robot and kinematics solver
    robot = get_yam_robot(channel="can0", zero_gravity_mode=False)
    kinematics_solver = Kinematics(YAM_XML_PATH, "grasp_site")
    
    # Run the interpolated path
    run_interpolated_path(robot, kinematics_solver, site_name="grasp_site")


if __name__ == "__main__":
    main()