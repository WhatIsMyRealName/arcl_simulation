"""Declared ROS observation sources; simulation truth is never a robot feature."""
import json
from pathlib import Path
DEFAULT_TOPICS = {
    'pose': '/pose_estimation', 'velocity': '/odometry/filtered',
    'joints': '/joint_states', 'obstacles': '/obstacles_scan', 'markers': '/aruco_markers',
    'clock': '/clock', 'image': '/camera/image', 'camera_info': '/camera/camera_info',
    'commands': '/agent/command_events', 'speed': '/speed_command',
    **{f'suction_{i}': f'/robot_omni/ventouse_{i}/etat' for i in range(1, 4)},
}

def load_topics(path=None):
    overrides = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(overrides, dict) or set(overrides) - set(DEFAULT_TOPICS):
        raise ValueError('unknown ROS source roles')
    topics = {**DEFAULT_TOPICS, **overrides}
    for value in topics.values():
        if not isinstance(value, str) or not value.startswith('/') or any(
            forbidden in value.lower() for forbidden in ('deported', 'deportee', 'pose_vraie', 'rock_centering')):
            raise ValueError(f'Forbidden or invalid observation topic: {value!r}')
    return topics
