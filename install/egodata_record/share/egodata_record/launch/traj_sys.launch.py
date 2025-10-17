from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='egodata_record', executable='udp_listener', output='screen'),
        Node(package='egodata_record', executable='zed_tracker', output='screen'),
        Node(package='egodata_record', executable='zed_xreal_traj_recorder', output='screen'),
    ])
