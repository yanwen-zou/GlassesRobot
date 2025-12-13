from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import YAM_XML_PATH
import numpy as np
import time


def move_to_pose(robot, mj_model, target_pose, site_name="grasp_site", num_steps=50, duration=2.0):
    """Move robot to target pose using IK with interpolation
    
    Args:
        robot: Robot instance
        mj_model: Kinematics solver
        target_pose: 4x4 transformation matrix
        site_name: Name of the site for IK
        num_steps: Number of interpolation steps
        duration: Total duration of movement in seconds
    """
    # Get initial joint configuration
    current_joint_pos = robot.get_joint_pos()
    current_q = current_joint_pos[:6]
    gripper_pos = current_joint_pos[6] if len(current_joint_pos) == 7 else None
    
    # Get current pose
    current_pose = mj_model.fk(current_q)
    
    # Interpolate position
    current_pos = current_pose[:3, 3]
    target_pos = target_pose[:3, 3]
    
    # Interpolate rotation (using rotation matrix)
    current_rot = current_pose[:3, :3]
    target_rot = target_pose[:3, :3]
    
    # Convert rotation matrices to quaternions for interpolation
    from scipy.spatial.transform import Rotation as R
    current_rot_obj = R.from_matrix(current_rot)
    target_rot_obj = R.from_matrix(target_rot)
    
    for i in range(num_steps + 1):
        alpha = i / num_steps
        
        # Linear interpolation of position
        interp_pos = (1 - alpha) * current_pos + alpha * target_pos
        
        # SLERP interpolation of rotation
        q1 = current_rot_obj.as_quat()
        q2 = target_rot_obj.as_quat()
        q_interp = slerp_quat(q1, q2, alpha)
        rot_interp = R.from_quat(q_interp).as_matrix()
        
        # Construct interpolated pose
        interp_pose = np.eye(4)
        interp_pose[:3, :3] = rot_interp
        interp_pose[:3, 3] = interp_pos
        
        # Solve IK
        success, q_ik = mj_model.ik(
            interp_pose,
            site_name,
            init_q=current_q,
            verbose=False
        )
        
        if success:
            current_q = q_ik
            # Add gripper position if needed
            if gripper_pos is not None:
                q_command = np.append(q_ik, gripper_pos)
            else:
                q_command = q_ik
            robot.command_joint_pos(q_command)
        else:
            print(f"Warning: IK failed at step {i}/{num_steps}")
        
        time.sleep(duration / num_steps)


def slerp_quat(q1, q2, t):
    """SLERP quaternion interpolation"""
    q1 = np.array(q1)
    q2 = np.array(q2)
    
    # Normalize
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.dot(q1, q2)
    
    # If dot product is negative, flip q2
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    
    # If quaternions are very close, use linear interpolation
    if abs(theta) < 1e-6:
        return (1 - t) * q1 + t * q2
    
    # Standard SLERP formula
    sin_theta = np.sin(theta)
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    result = w1 * q1 + w2 * q2
    return result / np.linalg.norm(result)


def main():
    robot = get_yam_robot(channel="can0", zero_gravity_mode=False)
    mj_model = Kinematics(YAM_XML_PATH, "grasp_site")
    
    # Get initial joint configuration
    q_init = np.array([0.0, 0.15, 0.25, -0.2, 0.0, 0.0])
    
    # Get initial pose
    pose_init = mj_model.fk(q_init)
    print("Initial pose:")
    print(pose_init)
    
    # Move to initial position first
    print("Moving to initial position...")
    current_joint_pos = robot.get_joint_pos()
    if len(current_joint_pos) == 7:
        q_init_with_gripper = np.append(q_init, current_joint_pos[6])
    else:
        q_init_with_gripper = q_init
    robot.command_joint_pos(q_init_with_gripper)
    time.sleep(2.0)
    
    # Movement distance in meters
    move_distance = 0.08
    
    # Get initial position and orientation
    init_pos = pose_init[:3, 3].copy()
    init_rot = pose_init[:3, :3].copy()
    
    # Define movement directions (in world frame)
    directions = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0])
    }
    
    print("Starting movement sequence...")
    time.sleep(1.0)
    
    # Move in each direction: forward then back, then return to init
    for axis_name, direction in directions.items():
        print(f"\n=== Moving in {axis_name.upper()} direction ===")
        
        # Forward movement
        target_pos_forward = init_pos + direction * move_distance
        target_pose_forward = np.eye(4)
        target_pose_forward[:3, :3] = init_rot
        target_pose_forward[:3, 3] = target_pos_forward
        
        print(f"Moving forward in {axis_name} direction by {move_distance}m...")
        move_to_pose(robot, mj_model, target_pose_forward, num_steps=50, duration=2.0)
        time.sleep(0.5)
        
        # Backward movement (back to init)
        print(f"Moving back to initial position...")
        move_to_pose(robot, mj_model, pose_init, num_steps=50, duration=2.0)
        time.sleep(0.5)
        
        # Backward movement (negative direction)
        target_pos_backward = init_pos - direction * move_distance
        target_pose_backward = np.eye(4)
        target_pose_backward[:3, :3] = init_rot
        target_pose_backward[:3, 3] = target_pos_backward
        
        print(f"Moving backward in {axis_name} direction by {move_distance}m...")
        move_to_pose(robot, mj_model, target_pose_backward, num_steps=50, duration=2.0)
        time.sleep(0.5)
        
        # Return to initial position
        print(f"Returning to initial position...")
        move_to_pose(robot, mj_model, pose_init, num_steps=50, duration=2.0)
        time.sleep(1.0)
    
    print("\n=== Movement sequence completed ===")
    print("Robot is back at initial position.")


if __name__ == "__main__":
    main()