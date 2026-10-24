"""Explicit ROS check: ordinary controllers work without command telemetry.

Run after building and sourcing the workspace. This test uses temporary files,
synthetic sensor streams, and the actual controller/service executables.
"""
import builtins
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile
import time

import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from robot_msgs.action import GoToPose
from robot_msgs.msg import RobotSpeed
from std_msgs.msg import Float64
from std_srvs.srv import SetBool
from rcl_interfaces.srv import DescribeParameters, GetParameters, SetParameters
from rclpy.parameter import Parameter
from agent.architecture.command_telemetry import require_command_events


def main():
    root = Path(__file__).resolve().parents[2]
    # Import the real console without starting Tk, rejecting every agent import.
    original_import = builtins.__import__
    def independent_import(name, *args, **kwargs):
        if name == 'agent' or name.startswith('agent.'):
            raise AssertionError(f'console imports {name}')
        return original_import(name, *args, **kwargs)
    try:
        builtins.__import__ = independent_import
        console = runpy.run_path(str(root / 'src/arcl_simulation/launch/move.launch.py'))
        assert not hasattr(console['Interface'], 'start_recording')
    finally:
        builtins.__import__ = original_import

    with tempfile.TemporaryDirectory(prefix='arcl-optional-validation-') as temporary:
        directory = Path(temporary)
        os.environ['ROS_DOMAIN_ID'] = str(50 + os.getpid() % 100)
        environment = {**os.environ, 'ROS_LOG_DIR': str(directory / 'logs')}
        processes = []
        streams = []
        def launch(command):
            stream = (directory / f'process-{len(processes)}.log').open('w')
            streams.append(stream)
            process = subprocess.Popen(command, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            processes.append(process)
            return process

        rclpy.init()
        node = rclpy.create_node('optional_telemetry_validation')
        pose_publisher = node.create_publisher(PoseStamped, '/pose_estimation', 10)
        speed_publisher = node.create_publisher(RobotSpeed, '/speed_command', 10)
        wheels = []
        node.create_subscription(Float64, '/robot_omni/roue_1/cmd_vel', lambda message: wheels.append(message.data), 100)
        def pump(seconds):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.pose.position.x = pose.pose.position.y = 1.
                pose.pose.orientation.w = 1.
                pose_publisher.publish(pose)
                rclpy.spin_once(node, timeout_sec=.01)
        def result(future):
            deadline = time.monotonic() + 5.
            while not future.done() and time.monotonic() < deadline:
                pump(.02)
            assert future.done(), 'ROS request timed out'
            return future.result()
        def speed(vx):
            message = RobotSpeed()
            message.vx = vx
            speed_publisher.publish(message)
        try:
            control = launch([str(root / 'install/lib/arcl_simulation/robot_control')])
            service = launch([str(root / 'install/lib/arcl_simulation/ventouse_service'),
                              '--ros-args', '-p', 'duree_poignet:=0.15'])
            pump(2.)
            assert control.poll() is None and service.poll() is None
            assert node.count_publishers('/agent/command_events') == 0
            for remote in ('/robot_control', '/ventouse_service'):
                get = node.create_client(GetParameters, remote + '/get_parameters')
                assert get.wait_for_service(timeout_sec=3.)
                request = GetParameters.Request()
                request.names = ['publish_command_events']
                assert result(get.call_async(request)).values[0].bool_value is False
                describe = node.create_client(DescribeParameters, remote + '/describe_parameters')
                request = DescribeParameters.Request()
                request.names = ['publish_command_events']
                assert result(describe.call_async(request)).descriptors[0].read_only
                set_parameters = node.create_client(SetParameters, remote + '/set_parameters')
                request = SetParameters.Request()
                request.parameters = [Parameter('publish_command_events', value=True).to_parameter_msg()]
                assert not result(set_parameters.call_async(request)).results[0].successful
            try:
                require_command_events(node, ['/robot_control', '/ventouse_service'])
            except RuntimeError as error:
                assert 'command_events:=true' in str(error)
            else:
                raise AssertionError('disabled telemetry was accepted by the observer check')

            speed(.1)
            pump(.08)
            assert any(abs(value) > .1 for value in wheels), 'direct motion failed without telemetry'
            pump(.35)
            assert abs(wheels[-1]) < 1e-8, 'watchdog failed without telemetry'
            navigation = ActionClient(node, GoToPose, '/go_to_pose')
            assert navigation.wait_for_server(timeout_sec=3.)
            goal = GoToPose.Goal()
            goal.x, goal.y, goal.theta = 1.5, 1., 0.
            handle = result(navigation.send_goal_async(goal))
            assert handle.accepted
            pump(.1)
            result(handle.cancel_goal_async())
            assert result(handle.get_result_async()).status == 5
            wrist = node.create_client(SetBool, '/robot_omni/poignet/set')
            assert wrist.wait_for_service(timeout_sec=3.)
            request = SetBool.Request()
            request.data = True
            assert result(wrist.call_async(request)).success
            assert node.count_publishers('/agent/command_events') == 0

            collector = launch([sys.executable, '-m', 'agent.architecture.collect_ros',
                                '--output', str(directory / 'session')])
            deadline = time.monotonic() + 10.
            while collector.poll() is None and time.monotonic() < deadline:
                pump(.1)
            assert collector.poll() is not None and collector.returncode != 0
            assert 'command_events:=true' in (directory / 'process-2.log').read_text()
            observations = directory / 'session/observations.jsonl'
            assert not observations.exists() or not observations.read_text().strip()
            manifest = json.loads((directory / 'session/manifest.json').read_text())
            assert not manifest['complete']
            print('PASS: console independent, telemetry off by default/read-only, no event publishers, '
                  'motion/watchdog/cancel/wrist functional, disabled collection fails without observations')
        finally:
            speed(0.)
            for process in reversed(processes):
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=7.)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            for stream in streams:
                stream.close()


if __name__ == '__main__':
    main()
