from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

os.environ["PATH"] = "/home/yanwen/miniconda3/envs/zed/bin:" + os.environ["PATH"]
os.environ["PYTHONPATH"] = "/home/yanwen/miniconda3/envs/zed/lib/python3.10/site-packages:" + os.environ.get("PYTHONPATH", "")


def _find_workspace_root(start: Path) -> Path:
    for candidate in (start,) + tuple(start.parents):
        if (candidate / 'src').is_dir() and (candidate / 'install').is_dir():
            return candidate
    return start.parent


def generate_launch_description():
    script_path = Path(__file__).resolve()
    workspace_root = _find_workspace_root(script_path)
    data_dir = workspace_root / 'data'

    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value=str(data_dir),
        description='Directory to save stereo MP4 files and pose logs (defaults to current working directory).'
    )

    frame_rate_arg = DeclareLaunchArgument(
        'frame_rate',
        default_value='30.0',
        description='Frame rate for the recorded MP4 files.'
    )

    downscale_factor_arg = DeclareLaunchArgument(
        'downscale_factor',
        default_value='0.5',
        description='Factor to downscale captured frames before recording.'
    )

    udp_listener_node = Node(
        package='egodata_record',
        executable='udp_listener',
        output='screen'
    )

    stereo_recorder_node = Node(
        package='egodata_record',
        executable='stereo_video_recorder',
        output='screen',
        parameters=[{
            'output_dir': LaunchConfiguration('output_dir'),
            'frame_rate': LaunchConfiguration('frame_rate'),
            'downscale_factor': LaunchConfiguration('downscale_factor'),
        }]
    )

    return LaunchDescription([
        output_dir_arg,
        frame_rate_arg,
        downscale_factor_arg,
        udp_listener_node,
        stereo_recorder_node,
    ])
