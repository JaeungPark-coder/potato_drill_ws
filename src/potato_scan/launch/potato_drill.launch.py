"""Full pipeline: scan -> eye detection -> drilling.

scan_controller and drill_controller both hold a connection to the robot
(RTDE by default, or Isaac Sim over ROS2 topics/TF -- see robot_backend
below); drill_controller only acts once /potato_scan/start_drilling is
published (after you've reviewed the eye markers in RViz), so it is safe
to bring all four nodes up together.

For Isaac Sim: start ../isaac/isaac_scene.py separately first (with
Isaac Sim's own python.sh -- it needs the Kit runtime, so it isn't
launched from here), then run this with robot_backend:=isaac_sim.
"""
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

    nodes = [
        DeclareLaunchArgument('rviz', default_value='true',
                               description='launch RViz2 with the scan/eye view'),
        DeclareLaunchArgument('robot_backend', default_value='rtde',
                               description="'rtde' (real UR5e / URSim) or 'isaac_sim' "
                                           "(needs ../isaac/isaac_scene.py running separately)"),
    ]
    for pkg_exec in ('pointcloud_accumulator', 'scan_controller', 'eye_detector', 'drill_controller'):
        node_params = [params_file]
        if pkg_exec in ('scan_controller', 'drill_controller'):
            node_params.append({'robot_backend': robot_backend})
        nodes.append(Node(
            package='potato_scan',
            executable=pkg_exec,
            name=pkg_exec,
            parameters=node_params,
            output='screen',
        ))
    nodes.append(Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen',
    ))
    return LaunchDescription(nodes)
