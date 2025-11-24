from pathlib import Path
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Keep ZED in its own conda env to avoid numpy ABI mismatch
os.environ["PATH"] = "/home/yanwen/miniconda3/envs/zed/bin:" + os.environ["PATH"]
os.environ["PYTHONPATH"] = "/home/yanwen/miniconda3/envs/zed/lib/python3.10/site-packages:" + os.environ.get("PYTHONPATH", "")


def _find_workspace_root(start: Path) -> Path:
    for candidate in (start,) + tuple(start.parents):
        if (candidate / "src").is_dir() and (candidate / "install").is_dir():
            return candidate
    return start.parent


def generate_launch_description():
    script_path = Path(__file__).resolve()
    workspace_root = _find_workspace_root(script_path)

    intrinsics_arg = DeclareLaunchArgument(
        "intrinsics",
        default_value=str(workspace_root / "src/FoundationStereo/assets/K_ZED.txt"),
        description="ZED 左相机内参路径（npy/npz/txt）。默认使用 FoundationStereo 的 K_ZED.txt。",
    )
    intrinsic_scale_arg = DeclareLaunchArgument(
        "intrinsic_scale",
        default_value="2.0",
        description="若内参下采样过，放大 fx,fy,cx,cy 的倍率（K_ZED.txt 需 *2）。",
    )
    output_arg = DeclareLaunchArgument(
        "output",
        default_value=str(workspace_root / "glasses_hardware/calib/T_zed_tcp.npy"),
        description="标定输出路径（T_zed_to_tcp）。",
    )
    pose_topic_arg = DeclareLaunchArgument(
        "pose_topic",
        default_value="/glasses_pose",
        description="头显/眼镜 PoseStamped 话题名。",
    )
    frame_rate_arg = DeclareLaunchArgument(
        "frame_rate",
        default_value="30.0",
        description="ZED 采集帧率。",
    )
    pattern_cols_arg = DeclareLaunchArgument(
        "pattern_cols",
        default_value="11",
        description="棋盘列数（内角点）。",
    )
    pattern_rows_arg = DeclareLaunchArgument(
        "pattern_rows",
        default_value="8",
        description="棋盘行数（内角点）。",
    )
    square_size_arg = DeclareLaunchArgument(
        "square_size_m",
        default_value="0.024",
        description="棋盘单元边长（米）。",
    )
    min_samples_arg = DeclareLaunchArgument(
        "min_samples",
        default_value="10",
        description="最少采样数量。",
    )

    headpos_listener_node = Node(
        package="egodata_record",
        executable="headpos_listener",
        output="screen",
    )

    zed_handeye_node = Node(
        package="egodata_record",
        executable="zed_handeye_node",
        output="screen",
        arguments=[
            "--intrinsics",
            LaunchConfiguration("intrinsics"),
            "--intrinsic-scale",
            LaunchConfiguration("intrinsic_scale"),
            "--output",
            LaunchConfiguration("output"),
            "--pose-topic",
            LaunchConfiguration("pose_topic"),
            "--frame-rate",
            LaunchConfiguration("frame_rate"),
            "--pattern-cols",
            LaunchConfiguration("pattern_cols"),
            "--pattern-rows",
            LaunchConfiguration("pattern_rows"),
            "--square-size-m",
            LaunchConfiguration("square_size_m"),
            "--min-samples",
            LaunchConfiguration("min_samples"),
        ],
    )

    return LaunchDescription(
        [
            intrinsics_arg,
            intrinsic_scale_arg,
            output_arg,
            pose_topic_arg,
            frame_rate_arg,
            pattern_cols_arg,
            pattern_rows_arg,
            square_size_arg,
            min_samples_arg,
            headpos_listener_node,
            zed_handeye_node,
        ]
    )
