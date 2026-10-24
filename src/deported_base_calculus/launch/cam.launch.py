import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = "deported_base_calculus"
    pkg = get_package_share_directory(pkg_name)

    cam_tracking = Node(
        package=pkg_name,
        executable='cam_tracking',
    )
    
    aruco_params = os.path.join(
        pkg,
        'config',
        'aruco_parameters.yaml'
    )

    aruco_node = Node(
        package='ros2_aruco',
        executable='aruco_node',
        parameters=[aruco_params]
    )

    return LaunchDescription([
        cam_tracking,
        aruco_node, 
    ])
