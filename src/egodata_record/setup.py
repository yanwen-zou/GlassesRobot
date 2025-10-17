from setuptools import setup
import os
from glob import glob

package_name = 'egodata_record'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 让 launch 文件夹里的所有 .py 都被安装
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yanwen',
    maintainer_email='you@example.com',
    description='Trajectory recording system',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'traj_recorder = egodata_record.zed_xreal_traj_recorder:main',
            'udp_listener = egodata_record.udp_listener:main',
            'zed_tracker = egodata_record.zed_tracker:main',
            'stereo_video_recorder = egodata_record.stereo_video_recorder:main',
            'pose_visualizer = egodata_record.pose_visualizer:main',
        ],
    },
)
