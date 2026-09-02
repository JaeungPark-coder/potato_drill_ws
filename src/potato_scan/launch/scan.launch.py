import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('potato_scan')
    params_file = os.path.join(pkg_share, 'config', 'params.yaml')
    rviz_config = os.path.join(pkg_share, 'rviz', 'potato_scan.rviz')

    robot_backend = LaunchConfiguration('robot_backend')

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true',
                               description='launch RViz2 with the scan-progress view'),
        DeclareLaunchArgument('robot_backend', default_value='rtde',
                               description="'rtde' (real UR5e / URSim) or 'isaac_sim' "
                                           "(needs ../isaac/isaac_scene.py running separately)"),
        Node(
            package='potato_scan',
            executable='pointcloud_accumulator',
            name='pointcloud_accumulator',
            parameters=[params_file],
            output='screen',
        ),
        Node(
            package='potato_scan',
            executable='scan_controller',
            name='scan_controller',
            parameters=[params_file, {'robot_backend': robot_backend}],
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            condition=IfCondition(LaunchConfiguration('rviz')),
            output='screen',
        ),
    ])
