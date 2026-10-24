"""Record authorized ROS sources regardless of the client driving the robot."""
import argparse
import json
import math
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from .capture import SessionWriter
from .ros_sources import load_topics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/sessions') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    parser.add_argument('--topics-config')
    parser.add_argument('--command-event-nodes', nargs='+', default=['/robot_control', '/ventouse_service'])
    parser.add_argument('--carton-source', default='gazebo_sim', choices=['gazebo_sim'])
    args = parser.parse_args()
    topics = load_topics(args.topics_config)
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from robot_msgs.msg import RobotSpeed
    session = SessionWriter(args.output, {'topics': topics, 'carton_source': args.carton_source, 'sample_hz': 10})
    rclpy.init()
    node = Node('arcl_ros_collector')
    def command(message):
        received = time.monotonic()
        try:
            session.write('events', json.loads(message.data), received)
        except ValueError as error:
            node.get_logger().error(f'invalid command event: {error}')
    def speed(message):
        values = {k: float(getattr(message, k)) for k in ('vx', 'vy', 'omega')}
        session.write('events', {'kind': 'requested_speed',
            'parameters': {k: v if math.isfinite(v) else None for k, v in values.items()},
            'valid': all(math.isfinite(v) for v in values.values())})
    node.create_subscription(String, topics['commands'], command, 1000)
    node.create_subscription(RobotSpeed, topics['speed'], speed, 1000)
    cmd = [sys.executable, '-m', 'agent.architecture.world_observer', '--carton-source', args.carton_source,
           '--image-directory', str(session.path / 'images'),
           '--command-event-nodes', *args.command_event_nodes]
    if args.topics_config:
        cmd += ['--topics-config', args.topics_config]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    failure = []
    def observations():
        try:
            for line in process.stdout:
                snapshot = json.loads(line)
                image = snapshot.get('front_camera') or {}
                if image.get('path'):
                    image['path'] = str(Path(image['path']).relative_to(session.path))
                session.write('observations', snapshot)
        except Exception as error:
            failure.append(error)
    reader = threading.Thread(target=observations, daemon=True)
    reader.start()
    print(f'Collecte ROS : {session.path}', flush=True)
    complete = False
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if failure or process.poll() is not None:
                raise RuntimeError(f'observer stopped: {failure or process.returncode}; '
                                   'check that both servers were started with command_events:=true')
    except KeyboardInterrupt:
        complete = True
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        reader.join(timeout=5)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        session.close(complete=complete and not failure)
        print(f'Session sauvegardée : {session.path}', flush=True)

if __name__ == '__main__':
    main()
