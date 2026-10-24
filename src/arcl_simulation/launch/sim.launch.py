import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import ExecuteProcess


def generate_launch_description():
    pkg_name = "arcl_simulation"
    pkg = get_package_share_directory(pkg_name)

    world = os.path.join(pkg, 'worlds', 'arene_omni.sdf')
    world_camera = os.path.join(pkg, 'worlds', 'arene_omni_deported_base.sdf')

    with open(os.path.join(pkg, 'urdf', 'robot_omni.urdf')) as f:
        robot_description = f.read()

    gz = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {world}'}.items()
        #launch_arguments={'gz_args': f'-r {world_camera}'}.items()
    )

    bridge = Node(
        package='ros_gz_bridge', 
        executable='parameter_bridge', 
        output='screen',
        parameters=[{'config_file': os.path.join(pkg, 'config', 'ros_bridge.yaml'), 'use_sim_time': True}]
    )

    rsp = Node(
        package='robot_state_publisher', 
        executable='robot_state_publisher', 
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}]
    )

    rviz = Node(
        package='rviz2', 
        executable='rviz2', 
        output='screen',
        arguments=['-d', os.path.join(pkg, 'rviz', 'arcl.rviz')],
        parameters=[{'use_sim_time': True, "--params-file": os.path.join(pkg, 'config', 'ekf_params.yaml')}],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    robot_control = Node(
        package=pkg_name,
        executable="robot_control",
        parameters=[{'use_sim_time': True, 'publish_command_events': ParameterValue(LaunchConfiguration('command_events'), value_type=bool)}]
    )

    # Localisation : odométrie roues + OTOS -> ekf_node (robot_localization) -> recalage balises
    wheel_odometry = Node(
        package=pkg_name,
        executable="wheel_odometry",
        parameters=[{'use_sim_time': True}]
    )

    otos_simulator = Node(
        package=pkg_name,
        executable="otos_simulator",
        parameters=[{'use_sim_time': True}]
    )

    ekf_odom = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_odom',
        output='screen',
        parameters=[os.path.join(pkg, 'config', 'ekf_odom.yaml')]
    )

    beacon_localization = Node(
        package=pkg_name,
        executable="beacon_localization",
        parameters=[{'use_sim_time': True}]
    )

    rock_centering = Node(
        package=pkg_name,
        executable="rock_centering",
        parameters=[{'use_sim_time': True}]
    )

    table_markers = Node(
        package=pkg_name,
        executable="table_markers"
    )

    ventouse_service = Node(
        package=pkg_name,
        executable='ventouse_service',
        parameters=[{'use_sim_time': True, 'publish_command_events': ParameterValue(LaunchConfiguration('command_events'), value_type=bool)}],
        output='screen'
    )
    

    aruco_params = os.path.join(
        get_package_share_directory('ros2_aruco'),
        'config',
        'aruco_parameters.yaml'
    )

    aruco_node = Node(
        package='ros2_aruco',
        executable='aruco_node',
        parameters=[aruco_params]
    )

    command_interface = ExecuteProcess(
        cmd=['python3', os.path.join(pkg, 'launch', 'move.launch.py')],
        condition=IfCondition(LaunchConfiguration('pupitre')),
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('command_events', default_value='false'),
        DeclareLaunchArgument('gz_flags', default_value='-r'),
        DeclareLaunchArgument('pupitre', default_value='true'),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(pkg, 'worlds')),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH',os.path.join(pkg, 'models')),
        gz, bridge, rsp, rviz, 
        robot_control, rock_centering, table_markers, ventouse_service,
        wheel_odometry, otos_simulator, ekf_odom, beacon_localization,
        aruco_node,
        command_interface
    ])
