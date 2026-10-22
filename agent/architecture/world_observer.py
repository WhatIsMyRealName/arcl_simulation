"""Assemble partial robot observations and publish them as NDJSON."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .gazebo_pose_source import CartonPoseProvider, GazeboCliPoseSource


DEFAULT_MAP = Path(__file__).with_name("arena_map.json")
SUCTION_STATES = {
    "libre": "free", "aspiration": "aspirating",
    "tenu": "holding", "descente": "lowering",
}
DEFAULT_FRESHNESS = {
    "robot_pose": 0.75,
    "obstacles": 0.75,
    "aruco": 0.75,
    "cartons": 1.0,
    "velocity": 0.25,
    "wrist": 1.0,
    "suction": 2.0,
}


def load_map_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("map config must contain a JSON object")
    arena = config.get("arena")
    bounds = arena.get("bounds") if isinstance(arena, dict) else None
    if not isinstance(bounds, dict):
        raise ValueError("map config must define arena.bounds")
    bound_keys = ("x_min", "x_max", "y_min", "y_max")
    bound_values = {key: bounds.get(key) for key in bound_keys}
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and _finite(value)
        for value in bound_values.values()
    ):
        raise ValueError("arena.bounds must contain finite x_min/x_max/y_min/y_max")
    bound_values = {key: float(value) for key, value in bound_values.items()}
    if bound_values["x_max"] <= bound_values["x_min"] or bound_values["y_max"] <= bound_values["y_min"]:
        raise ValueError("arena.bounds must have positive width and height")
    for key in ("zones", "fixed_obstacles"):
        if not isinstance(config.get(key, []), list):
            raise ValueError(f"map config {key} must be a list")
    return config


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _source_age(sample: dict[str, Any] | None, now: float) -> float | None:
    if sample is None:
        return None
    return max(0.0, now - float(sample["received_at"]))


def _source_status(
    sample: dict[str, Any] | None, now: float, ttl: float
) -> dict[str, Any]:
    if sample is None:
        return {"status": "unknown", "fresh": False, "age_s": None, "stamp": None}
    age = _source_age(sample, now)
    declared = sample.get("status", "fresh")
    fresh = declared == "fresh" and age is not None and age <= ttl
    status = "fresh" if fresh else (declared if declared != "fresh" else "stale")
    return {
        "status": status,
        "fresh": fresh,
        "age_s": age,
        "stamp": sample.get("stamp"),
    }


class MatchTimer:
    """Count down from the first simulation-clock value observed by the publisher."""

    def __init__(self, duration_s: float = 90.0) -> None:
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("match duration must be finite and positive")
        self.duration_s = float(duration_s)
        self.started_at_s: float | None = None
        self.elapsed_s = 0.0

    def remaining(self, sim_time_s: float | None) -> float | None:
        if sim_time_s is None or not math.isfinite(sim_time_s):
            return None
        if self.started_at_s is None:
            self.started_at_s = float(sim_time_s)
        self.elapsed_s = max(self.elapsed_s, max(0.0, float(sim_time_s) - self.started_at_s))
        return max(0.0, self.duration_s - self.elapsed_s)


def _quaternion_matrix(quaternion: Mapping[str, Any]) -> np.ndarray:
    x, y, z, w = (float(quaternion[key]) for key in ("x", "y", "z", "w"))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12 or not math.isfinite(norm):
        raise ValueError("quaternion must be finite and non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _rpy_matrix(rpy: list[float] | tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def _velocity_to_map(linear_x: float, linear_y: float, angular_z: float,
                     robot_yaw: float) -> dict[str, float]:
    values = (float(linear_x), float(linear_y), float(angular_z), float(robot_yaw))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("odometry velocity and robot heading must be finite")
    vx, vy, omega, yaw = values
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return {"x": cosine * vx - sine * vy, "y": sine * vx + cosine * vy,
            "yaw": omega}


class SnapshotAssembler:
    """Pure snapshot builder; ROS callbacks only translate messages to dicts."""

    def __init__(
        self,
        map_config: dict[str, Any],
        *,
        robot_id: str = "robot_omni",
        freshness: Mapping[str, float] | None = None,
        association_radius_m: float = 0.32,
        association_margin_m: float = 0.04,
        max_pose_marker_skew_s: float = 0.25,
    ) -> None:
        self.map_config = copy.deepcopy(map_config)
        self.robot_id = robot_id
        self.freshness = {**DEFAULT_FRESHNESS, **dict(freshness or {})}
        self.association_radius_m = float(association_radius_m)
        self.association_margin_m = float(association_margin_m)
        self.max_pose_marker_skew_s = float(max_pose_marker_skew_s)
        self.pose: dict[str, Any] | None = None
        self.obstacles: dict[str, Any] | None = None
        self.markers: dict[str, Any] | None = None
        self.suctions: dict[str, dict[str, Any]] = {}
        self.velocity: dict[str, Any] | None = None
        self.wrist: dict[str, Any] | None = None

    def update_actuator_suction(self, index: int, value: str | None, *, received_at: float) -> None:
        state = SUCTION_STATES.get(value)
        self.suctions[str(index)] = {
            "value": state, "received_at": float(received_at),
            "status": "fresh" if state is not None else "unknown",
        }

    def update_velocity(self, value: dict[str, Any] | None, *, received_at: float,
                        status: str = "fresh") -> None:
        self.velocity = {"value": copy.deepcopy(value), "received_at": float(received_at), "status": status}

    def update_wrist(self, value: float | None, *, received_at: float, status: str = "fresh") -> None:
        self.wrist = {"value": value, "received_at": float(received_at), "status": status}

    def update_pose(
        self,
        pose: dict[str, Any] | None,
        *,
        stamp: float | None,
        received_at: float,
        status: str = "fresh",
    ) -> None:
        self.pose = {
            "value": copy.deepcopy(pose),
            "stamp": stamp,
            "received_at": float(received_at),
            "status": status,
        }

    def update_obstacles(
        self,
        obstacles: list[dict[str, Any]],
        *,
        stamp: float | None,
        received_at: float,
        status: str = "fresh",
    ) -> None:
        self.obstacles = {
            "value": copy.deepcopy(obstacles), "stamp": stamp,
            "received_at": float(received_at), "status": status,
        }

    def update_markers(
        self,
        markers: list[dict[str, Any]],
        *,
        frame_id: str,
        stamp: float | None,
        received_at: float,
        status: str = "fresh",
    ) -> None:
        self.markers = {
            "value": copy.deepcopy(markers), "frame_id": frame_id,
            "stamp": stamp, "received_at": float(received_at), "status": status,
        }

    def build(
        self,
        *,
        sequence: int,
        timestamp: float,
        now: float,
        carton_source: dict[str, Any],
        clock_age_s: float | None = None,
    ) -> dict[str, Any]:
        pose_meta = _source_status(self.pose, now, self.freshness["robot_pose"])
        velocity_meta = _source_status(self.velocity, now, self.freshness["velocity"])
        wrist_meta = _source_status(self.wrist, now, self.freshness["wrist"])
        obstacle_meta = _source_status(self.obstacles, now, self.freshness["obstacles"])
        marker_meta = _source_status(self.markers, now, self.freshness["aruco"])
        if self.markers is not None:
            marker_meta["frame_id"] = self.markers.get("frame_id")
        pose_marker_skew = None
        if self.pose is not None and self.markers is not None:
            pose_stamp, marker_stamp = self.pose.get("stamp"), self.markers.get("stamp")
            if _finite(pose_stamp) and _finite(marker_stamp):
                pose_marker_skew = abs(float(pose_stamp) - float(marker_stamp))
            else:
                pose_marker_skew = abs(
                    float(self.pose["received_at"]) - float(self.markers["received_at"])
                )
        marker_meta["pose_skew_s"] = pose_marker_skew
        marker_meta["map_transform_status"] = (
            "unknown" if pose_marker_skew is None
            else "time_skew" if pose_marker_skew > self.max_pose_marker_skew_s
            else "aligned"
        )
        carton_age_value = carton_source.get("age_s")
        carton_age = float(carton_age_value) if _finite(carton_age_value) else None
        carton_fresh = carton_source.get("fresh") is True and (
            carton_age is not None and carton_age <= self.freshness["cartons"]
        )
        carton_meta = {
            "status": carton_source.get("status", "unknown") if carton_fresh else (
                carton_source.get("status", "unknown")
                if carton_source.get("status") in ("error", "unknown") else "stale"
            ),
            "fresh": carton_fresh,
            "age_s": carton_age,
            "provider": carton_source.get("provider"),
            "error": carton_source.get("error"),
        }

        raw_cartons = carton_source.get("poses", {}) if carton_fresh else {}
        carton_ages = carton_source.get("carton_ages_s", {})
        cartons = self._carton_tracks(
            raw_cartons, carton_ages, provider=carton_source.get("provider")
        )
        aruco_detections = []
        robot_pose = copy.deepcopy(self.pose.get("value")) if pose_meta["fresh"] and self.pose else None
        velocity = copy.deepcopy(self.velocity.get("value")) if velocity_meta["fresh"] and self.velocity else None
        wrist_angle = self.wrist.get("value") if wrist_meta["fresh"] and self.wrist else None
        suction_meta = {
            str(index): _source_status(self.suctions.get(str(index)), now, self.freshness["suction"])
            for index in range(1, 4)
        }
        suction_states = {
            index: self.suctions[index]["value"] if suction_meta[index]["fresh"] else None
            for index in self.suctions
        }
        obstacles = []
        if obstacle_meta["fresh"] and self.obstacles:
            for index, item in enumerate(self.obstacles["value"]):
                obstacle = copy.deepcopy(item)
                obstacle.setdefault("id", f"lidar-{sequence}-{index}")
                obstacle["age_s"] = obstacle_meta["age_s"]
                obstacles.append(obstacle)
        for index, item in enumerate(self.map_config.get("fixed_obstacles", [])):
            obstacle = copy.deepcopy(item)
            obstacle.setdefault("id", f"map-obstacle-{index}")
            obstacle.setdefault("source", "map")
            obstacle["age_s"] = 0.0
            obstacles.append(obstacle)

        if (
            marker_meta["fresh"] and pose_meta["fresh"]
            and pose_marker_skew is not None
            and pose_marker_skew <= self.max_pose_marker_skew_s
            and self.markers is not None and self.markers.get("frame_id") == "camera_optical"
        ):
            aruco_detections = self._transform_marker_detections(
                cartons if carton_fresh else [],
                self.markers["value"],
                robot_pose,
                age_s=marker_meta["age_s"],
            )

        arena = copy.deepcopy(self.map_config["arena"])
        robot = {
            "id": self.robot_id,
            "pose": robot_pose,
            "pose_age_s": pose_meta["age_s"],
            "velocity": velocity,
            "wrist_angle_rad": wrist_angle,
            "suction_states": suction_states,
            "actuator_state_ages_s": {
                index: _source_age(sample, now) for index, sample in self.suctions.items()
            },
            "payload_state": None,
            "held_object": None,
            "contact_confirmed": None,
            "contact_target_id": None,
        }
        return {
            "schema_version": 2,
            "sequence": int(sequence),
            "timestamp": float(timestamp),
            "game_time_s": float(timestamp),
            "frame_id": self.map_config.get("frame_id", "map"),
            "arena": arena,
            "robot": robot,
            "cartons": cartons,
            "aruco_detections": aruco_detections,
            "obstacles": obstacles,
            "other_robots": [],
            "active_operation": None,
            "last_action_result": None,
            "zones": copy.deepcopy(self.map_config.get("zones", [])),
            "mission": {
                "remaining_s": None,
                "score": None,
                "cartons_delivered": None,
                "cartons_remaining": None,
            },
            "sources": {
                "robot_pose": {"topic": "/pose_estimation", **pose_meta},
                "velocity": {"topic": "/odometry/filtered", **velocity_meta},
                "wrist": {"topic": "/joint_states", **wrist_meta},
                "obstacles": {"topic": "/obstacles_scan", **obstacle_meta},
                "aruco": {"topic": "/aruco_markers", **marker_meta},
                "cartons": carton_meta,
                "map": {"provider": "config", "status": "fresh", "age_s": 0.0},
                "clock": {"topic": "/clock", "age_s": clock_age_s},
                **{
                    f"suction_{index}": {
                        "topic": f"/robot_omni/ventouse_{index}/etat",
                        "status": suction_meta[str(index)]["status"],
                        "fresh": suction_meta[str(index)]["fresh"],
                        "age_s": suction_meta[str(index)]["age_s"],
                    } for index in range(1, 4)
                },
            },
        }

    def _carton_tracks(
        self,
        poses: Mapping[str, Any],
        ages: Mapping[str, Any],
        *,
        provider: str | None,
    ) -> list[dict[str, Any]]:
        model = self.map_config.get("carton_model", {})
        dimensions = model.get("size_xyz_m", [0.11, 0.11, 0.32])
        marker_id = model.get("marker_id", 13)
        result = []
        for carton_id, raw in sorted(poses.items()):
            if provider == "gazebo_sim" and (
                not str(carton_id).startswith("pierre_")
                or str(carton_id) == "pierre_camelot"
            ):
                continue
            age_value = ages.get(carton_id, 0.0)
            if not _finite(age_value):
                continue
            age = float(age_value)
            if age > self.freshness["cartons"] or not isinstance(raw, dict):
                continue
            position = raw.get("position", {})
            orientation = raw.get("orientation", {})
            if not isinstance(position, dict) or not isinstance(orientation, dict):
                continue
            try:
                position_vector = np.asarray([
                    float(position["x"]), float(position["y"]), float(position["z"])
                ], dtype=np.float64)
                rotation = _quaternion_matrix(orientation)
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(position_vector).all():
                continue
            model_size = np.asarray(dimensions, dtype=np.float64)
            if model_size.shape != (3,) or not np.isfinite(model_size).all():
                continue
            long_axis_index = int(np.argmax(model_size))
            long_axis = rotation[:, long_axis_index]
            horizontal_length = math.hypot(float(long_axis[0]), float(long_axis[1]))
            if horizontal_length > 0.5:
                posture = "horizontal"
                yaw = math.atan2(float(long_axis[1]), float(long_axis[0]))
                planar_length = float(model_size[long_axis_index]) * horizontal_length
                other_dimensions = [
                    float(model_size[index]) for index in range(3)
                    if index != long_axis_index
                ]
                planar_width = max(other_dimensions)
            else:
                posture = "vertical"
                short_axis = rotation[:, 0]
                yaw = math.atan2(float(short_axis[1]), float(short_axis[0]))
                planar_length = float(model_size[0])
                planar_width = float(model_size[1])
            vertical_size = float(np.abs(rotation[2, :]) @ model_size)
            result.append({
                "id": str(carton_id),
                "model_name": raw.get("model_name", str(carton_id)),
                "marker_id": marker_id,
                "posture": posture,
                "pose": {
                    "x": float(position_vector[0]),
                    "y": float(position_vector[1]),
                    "z": float(position_vector[2]),
                    "yaw": yaw,
                },
                "size": {
                    "x": planar_length,
                    "y": planar_width,
                    "z": vertical_size,
                },
                "visible": None,
                "camera_distance": None,
                "age_s": age,
                "accessible": None,
                "approach_pose": None,
                "position_source": raw.get("source", provider),
            })
        return result

    def _transform_marker_detections(
        self,
        cartons: list[dict[str, Any]],
        detections: list[dict[str, Any]],
        robot_pose: dict[str, Any] | None,
        *,
        age_s: float | None,
    ) -> list[dict[str, Any]]:
        if robot_pose is None:
            return []
        extrinsics = self.map_config.get("camera_optical_extrinsics", {})
        translation = np.asarray(extrinsics.get("translation_m", [0.11, 0.0, 0.0855]), dtype=float)
        rpy = extrinsics.get("rpy_rad", [-math.pi / 2.0, 0.0, -math.pi / 2.0])
        if translation.shape != (3,) or len(rpy) != 3:
            return []
        yaw = float(robot_pose["yaw"])
        c, s = math.cos(yaw), math.sin(yaw)
        robot_rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        camera_rotation = _rpy_matrix(rpy)
        robot_translation = np.asarray([
            float(robot_pose["x"]), float(robot_pose["y"]), float(robot_pose.get("z", 0.0))
        ])
        camera_origin = robot_translation + robot_rotation @ translation
        candidates: dict[str, tuple[float, float]] = {}
        transformed_detections = []
        marker_id = self.map_config.get("carton_model", {}).get("marker_id", 13)
        for carton in cartons:
            carton["visible"] = False
        for index, marker in enumerate(detections):
            point = marker.get("position")
            if not isinstance(point, dict):
                continue
            try:
                camera_point = np.asarray([float(point[key]) for key in ("x", "y", "z")])
                map_point = camera_origin + robot_rotation @ camera_rotation @ camera_point
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(map_point).all():
                continue
            camera_position = float(np.linalg.norm(camera_point))
            associated_id = None
            association_error = None
            if marker.get("id") == marker_id and cartons:
                ranked = [
                    (
                        float(np.linalg.norm(map_point - np.asarray([
                            carton["pose"]["x"], carton["pose"]["y"], carton["pose"]["z"]
                        ]))), carton
                    )
                    for carton in cartons
                ]
                ranked.sort(key=lambda item: item[0])
                if ranked and ranked[0][0] <= self.association_radius_m and (
                    len(ranked) == 1
                    or ranked[1][0] - ranked[0][0] >= self.association_margin_m
                ):
                    associated_id = ranked[0][1]["id"]
                    association_error = ranked[0][0]
                    previous = candidates.get(associated_id)
                    if previous is None or camera_position < previous[0]:
                        candidates[associated_id] = (camera_position, association_error)

            pose = {
                "x": float(map_point[0]),
                "y": float(map_point[1]),
                "z": float(map_point[2]),
            }
            orientation = marker.get("orientation")
            normal = None
            if isinstance(orientation, dict):
                try:
                    marker_rotation = _quaternion_matrix(orientation)
                    map_rotation = robot_rotation @ camera_rotation @ marker_rotation
                    pose["yaw"] = _yaw_from_rotation(map_rotation)
                    normal = dict(zip(("x", "y", "z"), map_rotation[:, 2].tolist()))
                except (KeyError, TypeError, ValueError):
                    pass
            transformed_detections.append({
                "id": f"aruco-{marker.get('id', 'unknown')}-{index}",
                "marker_id": marker.get("id"),
                "pose": pose,
                "normal": normal,
                "camera_depth": float(camera_point[2]),
                "camera_distance": camera_position,
                "associated_carton_id": associated_id,
                "association_error_m": association_error,
                "age_s": age_s,
                "source": "/aruco_markers",
            })
        for carton in cartons:
            association = candidates.get(carton["id"])
            if association is not None:
                carton["visible"] = True
                carton["camera_distance"] = association[0]
                carton["marker_association_error_m"] = association[1]
        return transformed_detections


def _yaw_from_rotation(matrix: np.ndarray) -> float:
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))


def _seconds(stamp: Any) -> float | None:
    if stamp is None:
        return None
    try:
        value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return value if math.isfinite(value) else None
    except (AttributeError, TypeError, ValueError):
        return None


def _quaternion_yaw(quaternion: Any) -> float:
    x, y, z, w = (float(getattr(quaternion, field)) for field in ("x", "y", "z", "w"))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("pose estimate has an invalid orientation quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _run_ros(
    args: argparse.Namespace,
    map_config: dict[str, Any],
    source: CartonPoseProvider,
) -> None:
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from robot_msgs.msg import ObstacleArray
        from ros2_aruco_interfaces.msg import ArucoMarkers
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 Python messages are unavailable; source the ROS workspace before running the observer"
        ) from error

    class WorldObserverNode(Node):
        def __init__(self) -> None:
            super().__init__("agent_world_observer")
            self.assembler = SnapshotAssembler(
                map_config,
                robot_id=args.robot_id,
                association_radius_m=args.association_radius,
                association_margin_m=args.association_margin,
                max_pose_marker_skew_s=args.max_pose_marker_skew,
            )
            self.started_at = time.monotonic()
            self.sequence = 0
            self.sim_time: float | None = None
            self.clock_received_at: float | None = None
            self.match_timer = MatchTimer(args.match_duration_s)
            self.warned_missing_clock = False
            self.warned_stale_clock = False
            self.last_timestamp = 0.0
            self.logged_source_error: str | None = None
            clock_qos = QoSProfile(depth=1)
            clock_qos.reliability = ReliabilityPolicy.BEST_EFFORT
            self.create_subscription(PoseStamped, "/pose_estimation", self.on_pose, 10)
            self.create_subscription(Odometry, "/odometry/filtered", self.on_velocity, 10)
            self.create_subscription(JointState, "/joint_states", self.on_joint_states, 10)
            self.create_subscription(ObstacleArray, "/obstacles_scan", self.on_obstacles, 10)
            self.create_subscription(ArucoMarkers, "/aruco_markers", self.on_markers, 10)
            self.create_subscription(Clock, "/clock", self.on_clock, clock_qos)
            for index in range(1, 4):
                self.create_subscription(
                    String, f"/robot_omni/ventouse_{index}/etat",
                    lambda message, actuator=index: self.on_actuator_suction(actuator, message), 10,
                )
            self.create_timer(1.0 / args.publish_hz, self.on_timer)

        def on_actuator_suction(self, index: int, message: Any) -> None:
            self.assembler.update_actuator_suction(index, message.data, received_at=time.monotonic())

        def on_velocity(self, message: Any) -> None:
            received = time.monotonic()
            twist = message.twist.twist
            pose_sample = self.assembler.pose
            pose = pose_sample.get("value") if pose_sample else None
            pose_age = _source_age(pose_sample, received)
            if (pose is None or pose_sample.get("status") != "fresh" or pose_age is None
                    or pose_age > self.assembler.freshness["robot_pose"]):
                self.assembler.update_velocity(None, received_at=received, status="unknown")
                return
            try:
                values = _velocity_to_map(twist.linear.x, twist.linear.y,
                                          twist.angular.z, pose["yaw"])
            except (KeyError, TypeError, ValueError):
                self.assembler.update_velocity(None, received_at=received, status="invalid")
                return
            self.assembler.update_velocity(values, received_at=received)

        def on_joint_states(self, message: Any) -> None:
            received = time.monotonic()
            try:
                index = next(i for i, name in enumerate(message.name)
                             if name.split("::")[-1] == "poignet_joint")
                angle = float(message.position[index])
            except (ValueError, IndexError, TypeError):
                self.assembler.update_wrist(None, received_at=received, status="invalid")
                return
            if not math.isfinite(angle):
                self.assembler.update_wrist(None, received_at=received, status="invalid")
                return
            self.assembler.update_wrist(angle, received_at=received)

        def on_clock(self, message: Any) -> None:
            self.sim_time = _seconds(message.clock)
            self.clock_received_at = time.monotonic()

        def on_pose(self, message: Any) -> None:
            received = time.monotonic()
            stamp = _seconds(message.header.stamp)
            if message.header.frame_id != "map":
                self.assembler.update_pose(
                    None, stamp=stamp, received_at=received, status="wrong_frame"
                )
                return
            pose_message = message.pose
            try:
                pose = {
                    "x": float(pose_message.position.x),
                    "y": float(pose_message.position.y),
                    "z": float(pose_message.position.z),
                    "yaw": _quaternion_yaw(pose_message.orientation),
                }
            except (TypeError, ValueError) as error:
                self.get_logger().warning(f"ignoring invalid /pose_estimation: {error}")
                self.assembler.update_pose(
                    None, stamp=stamp, received_at=received, status="invalid"
                )
                return
            if not all(math.isfinite(value) for value in pose.values()):
                self.assembler.update_pose(
                    None, stamp=stamp, received_at=received, status="invalid"
                )
                return
            self.assembler.update_pose(
                pose, stamp=stamp, received_at=received
            )

        def on_obstacles(self, message: Any) -> None:
            received = time.monotonic()
            stamp = _seconds(message.header.stamp)
            if message.header.frame_id != "map":
                self.assembler.update_obstacles(
                    [], stamp=stamp, received_at=received, status="wrong_frame"
                )
                return
            obstacles = []
            invalid = False
            for item in message.obstacles:
                values = (float(item.x), float(item.y), float(item.radius))
                if not all(math.isfinite(value) for value in values) or values[2] < 0:
                    invalid = True
                    break
                obstacles.append({
                    "pose": {"x": values[0], "y": values[1], "yaw": 0.0},
                    "radius": values[2],
                    "source": "lidar",
                })
            self.assembler.update_obstacles(
                [] if invalid else obstacles,
                stamp=stamp,
                received_at=received,
                status="invalid" if invalid else "fresh",
            )

        def on_markers(self, message: Any) -> None:
            received = time.monotonic()
            stamp = _seconds(message.header.stamp)
            frame_id = message.header.frame_id.lstrip("/")
            markers = []
            for marker_id, pose in zip(message.marker_ids, message.poses):
                markers.append({
                    "id": int(marker_id),
                    "position": {
                        "x": float(pose.position.x),
                        "y": float(pose.position.y),
                        "z": float(pose.position.z),
                    },
                    "orientation": {
                        "x": float(pose.orientation.x),
                        "y": float(pose.orientation.y),
                        "z": float(pose.orientation.z),
                        "w": float(pose.orientation.w),
                    },
                })
            self.assembler.update_markers(
                markers,
                frame_id=frame_id,
                stamp=stamp,
                received_at=received,
                status="fresh" if frame_id == "camera_optical" else "wrong_frame",
            )

        def on_timer(self) -> None:
            now = time.monotonic()
            timestamp = self.sim_time if self.sim_time is not None else now - self.started_at
            timestamp = max(self.last_timestamp, float(timestamp))
            self.last_timestamp = timestamp
            carton_state = source.snapshot(now=now)
            if carton_state.get("error") and carton_state["error"] != self.logged_source_error:
                print(f"agent observer: carton source error: {carton_state['error']}", file=sys.stderr, flush=True)
                self.logged_source_error = carton_state["error"]
            snapshot = self.assembler.build(
                sequence=self.sequence,
                timestamp=timestamp,
                now=now,
                carton_source=carton_state,
                clock_age_s=(None if self.clock_received_at is None else now - self.clock_received_at),
            )
            remaining_s = self.match_timer.remaining(self.sim_time)
            snapshot["mission"]["total_s"] = self.match_timer.duration_s
            snapshot["mission"]["remaining_s"] = remaining_s
            snapshot["observer_monotonic_s"] = now
            clock_age_s = None if self.clock_received_at is None else now - self.clock_received_at
            snapshot["sources"]["clock"] = {
                "topic": "/clock",
                "status": "unknown" if clock_age_s is None else ("stale" if clock_age_s > 2.0 else "fresh"),
                "fresh": clock_age_s is not None and clock_age_s <= 2.0,
                "age_s": clock_age_s,
            }
            if clock_age_s is None and now - self.started_at >= 2.0 and not self.warned_missing_clock:
                self.get_logger().warning(
                    "no /clock received; match countdown remains unknown and the live runner cannot time out"
                )
                self.warned_missing_clock = True
            elif clock_age_s is not None and clock_age_s > 2.0 and not self.warned_stale_clock:
                self.get_logger().warning(
                    "/clock has not advanced recently; match countdown is paused until clock updates resume"
                )
                self.warned_stale_clock = True
            self.sequence += 1
            print(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False), flush=True)

    rclpy.init()
    node = WorldObserverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble partial ROS 2 observations and stream NDJSON on stdout."
    )
    parser.add_argument(
        "--carton-source", required=True,
        choices=("gazebo_sim", "external_camera"),
        help="explicit carton pose provider; external_camera is not connected yet",
    )
    parser.add_argument("--map-config", default=str(DEFAULT_MAP))
    parser.add_argument("--robot-id", default="robot_omni")
    parser.add_argument("--publish-hz", type=float, default=10.0)
    parser.add_argument("--match-duration-s", type=float, default=90.0,
                        help="match countdown duration, measured in Gazebo simulation seconds")
    parser.add_argument("--carton-timeout", type=float, default=1.0)
    parser.add_argument("--association-radius", type=float, default=0.32)
    parser.add_argument("--association-margin", type=float, default=0.04)
    parser.add_argument("--max-pose-marker-skew", type=float, default=0.25)
    parser.add_argument("--gz-topic", default="/world/arene_omni/pose/info")
    args = parser.parse_args()
    if args.publish_hz <= 0:
        parser.error("--publish-hz must be positive")
    if not math.isfinite(args.match_duration_s) or args.match_duration_s <= 0:
        parser.error("--match-duration-s must be finite and positive")
    if args.association_radius <= 0 or args.association_margin < 0:
        parser.error("association radius must be positive and margin non-negative")
    if args.max_pose_marker_skew < 0:
        parser.error("--max-pose-marker-skew must be non-negative")
    if args.carton_source == "external_camera":
        parser.error("external_camera is an interface placeholder and is not connected yet")
    try:
        map_config = load_map_config(args.map_config)
        source = GazeboCliPoseSource(topic=args.gz_topic, stale_after_s=args.carton_timeout)
        print(
            "WARNING: --carton-source gazebo_sim uses Gazebo carton poses for development; "
            "they are marked in snapshot provenance and must not represent the real robot input.",
            file=sys.stderr,
            flush=True,
        )
        source.start()
        try:
            _run_ros(args, map_config, source)
        finally:
            source.close()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"agent observer: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()

