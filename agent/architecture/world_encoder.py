"""Versioned absolute-map observations encoded as permutation-invariant sets."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .actions import ACTIONS


EMBEDDING_SIZE = 32
CONTEXT_SIZE = 64
ENTITY_GROUPS = ("cartons", "obstacles")
OBSERVATION_VERSION = 2
OPERATION_TYPES = ACTIONS


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _pose(item: dict[str, Any]) -> dict[str, Any]:
    return _object(item.get("pose", item))


def _put(row: dict[str, float], key: str, value: Any, scale: float = 1.0) -> None:
    number = _number(value)
    row[key] = 0.0 if number is None else number / scale
    row[key + "_known"] = float(number is not None)


def _flag(row: dict[str, float], key: str, value: Any) -> None:
    row[key] = float(value is True)
    row[key + "_known"] = float(isinstance(value, bool))


def _category(row: dict[str, float], key: str, value: Any, values: tuple[str, ...]) -> None:
    for choice in (*values, "unknown"):
        row[f"{key}_{choice}"] = float(
            value == choice if choice != "unknown" else value not in values
        )


def _map_coordinate(value: Any, lower: float, upper: float) -> float | None:
    number = _number(value)
    return None if number is None else (number - lower) / (upper - lower)


class WorldEncoder(nn.Module):
    """Encode map-frame robot state and unordered carton/obstacle sets."""

    def __init__(self, *, velocity_scale_m_s: float = 1.0,
                 arena_bounds: tuple[float, float, float, float] = (0.0, 3.0, 0.0, 2.0)) -> None:
        super().__init__()
        self.velocity_scale_m_s = velocity_scale_m_s
        self.arena_bounds = arena_bounds
        if not math.isfinite(self.velocity_scale_m_s) or self.velocity_scale_m_s <= 0:
            raise ValueError("velocity_scale_m_s must be finite and positive")
        if (len(arena_bounds) != 4 or not all(math.isfinite(value) for value in arena_bounds)
                or arena_bounds[0] >= arena_bounds[1] or arena_bounds[2] >= arena_bounds[3]):
            raise ValueError("arena_bounds must contain finite, increasing map limits")
        templates = {
            "robot": self._robot_features({}, None, self.arena_bounds),
            "context": self._context_features({}),
            "cartons": self._entity_features({}, "cartons", self.arena_bounds),
            "obstacles": self._entity_features({}, "obstacles", self.arena_bounds),
        }
        self.feature_names = {key: tuple(row) for key, row in templates.items()}
        self.robot_mlp = self._mlp(len(templates["robot"]))
        self.entity_mlps = nn.ModuleDict({
            group: self._mlp(len(templates[group])) for group in ENTITY_GROUPS
        })
        self.attention = nn.ModuleDict({
            group: nn.MultiheadAttention(EMBEDDING_SIZE, 4, batch_first=True)
            for group in ENTITY_GROUPS
        })
        self.attention_query = self._mlp(EMBEDDING_SIZE + len(templates["context"]))
        self.context_mlp = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE * 3 + len(templates["context"]), 96),
            nn.Tanh(), nn.Linear(96, CONTEXT_SIZE), nn.Tanh(),
        )

    @staticmethod
    def _mlp(inputs: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(),
                             nn.Linear(64, EMBEDDING_SIZE), nn.Tanh())

    def _bounds(self, world: dict[str, Any]) -> tuple[float, float, float, float]:
        arena = _object(world.get("arena"))
        bounds = _object(arena.get("bounds", arena))
        values = tuple(_number(bounds.get(key)) for key in ("x_min", "x_max", "y_min", "y_max"))
        if (all(value is not None for value in values) and values[0] < values[1]
                and values[2] < values[3]):
            return float(values[0]), float(values[1]), float(values[2]), float(values[3])
        return self.arena_bounds

    def _robot_features(self, world: dict[str, Any], pose: dict[str, Any] | None,
                        bounds: tuple[float, float, float, float]) -> dict[str, float]:
        row: dict[str, float] = {}
        pose = pose or {}
        x_min, x_max, y_min, y_max = bounds
        _put(row, "position_x", _map_coordinate(pose.get("x"), x_min, x_max))
        _put(row, "position_y", _map_coordinate(pose.get("y"), y_min, y_max))
        _put(row, "position_z", pose.get("z"))
        yaw = _number(pose.get("yaw", pose.get("theta")))
        row["yaw_sin"] = 0.0 if yaw is None else math.sin(yaw)
        row["yaw_cos"] = 0.0 if yaw is None else math.cos(yaw)
        row["yaw_known"] = float(yaw is not None)
        robot = _object(world.get("robot"))
        velocity = _object(robot.get("velocity"))
        _put(row, "velocity_x", velocity.get("x"), self.velocity_scale_m_s)
        _put(row, "velocity_y", velocity.get("y"), self.velocity_scale_m_s)
        _put(row, "angular_velocity", velocity.get("yaw"), 2.0)
        _put(row, "pose_age", robot.get("pose_age_s"), 90.0)
        _put(row, "velocity_age", _object(_object(world.get("sources")).get("velocity")).get("age_s"), 90.0)
        suction = _object(robot.get("suction_states"))
        for index in range(1, 4):
            state = suction.get(str(index))
            _category(row, f"suction_{index}", state,
                      ("free", "aspirating", "holding", "lowering"))
            _put(row, f"suction_{index}_age", _object(_object(world.get("sources")).get("suction_{index}")).get("age_s"), 90.0)
        wrist = robot.get("wrist_angle_rad")
        wrist_number = _number(wrist)
        row["wrist_angle_sin"] = 0.0 if wrist_number is None else math.sin(wrist_number)
        row["wrist_angle_cos"] = 0.0 if wrist_number is None else math.cos(wrist_number)
        row["wrist_angle_known"] = float(wrist_number is not None)
        _put(row, "wrist_age", _object(_object(world.get("sources")).get("wrist")).get("age_s"), 90.0)
        _category(row, "payload", robot.get("payload_state"), ("empty", "loaded"))
        return row

    def _context_features(self, world: dict[str, Any]) -> dict[str, float]:
        row: dict[str, float] = {}
        arena = _object(world.get("arena"))
        bounds = _object(arena.get("bounds", arena))
        _put(row, "arena_width", _number(bounds.get("x_max")) - _number(bounds.get("x_min"))
             if _number(bounds.get("x_max")) is not None and _number(bounds.get("x_min")) is not None else None)
        _put(row, "arena_height", _number(bounds.get("y_max")) - _number(bounds.get("y_min"))
             if _number(bounds.get("y_max")) is not None and _number(bounds.get("y_min")) is not None else None)
        mission = _object(world.get("mission"))
        for key in ("remaining_s", "score", "cartons_delivered", "cartons_remaining"):
            _put(row, key, mission.get(key), 90.0 if key == "remaining_s" else 1.0)
        sources = _object(world.get("sources"))
        for name in ("cartons", "obstacles"):
            source = _object(sources.get(name))
            _flag(row, f"{name}_source_fresh", source.get("fresh"))
            source_age = _number(source.get("age_s"))
            _put(row, f"{name}_source_age",
                 None if source_age is None else min(2.0, max(0.0, source_age)), 2.0)
            _category(row, f"{name}_source_status", source.get("status"),
                      ("fresh", "stale", "error", "unknown", "invalid", "wrong_frame"))
        active = _object(world.get("active_operation"))
        last = _object(world.get("last_action_result"))
        statuses = ("idle", "issuing", "running", "cancel_requested", "canceling",
                    "succeeded", "failed", "canceled", "rejected")
        active_is_idle = "active_operation" in world and world.get("active_operation") is None
        map_bounds = self._bounds(world)
        for prefix, operation in (("active", active), ("last", last)):
            known_idle = prefix == "active" and active_is_idle
            _category(row, prefix + "_type", "IDLE" if known_idle else operation.get("type"),
                      OPERATION_TYPES)
            _category(row, prefix + "_status", "idle" if known_idle else operation.get("status"), statuses)
            _flag(row, prefix + "_failed", False if known_idle else operation.get("failed"))
            _flag(row, prefix + "_cancelable", False if known_idle else operation.get("cancelable"))
            target = _object(operation.get("target_pose"))
            for key in ("x", "y"):
                low, high = (map_bounds[:2] if key == "x" else map_bounds[2:])
                _put(row, f"{prefix}_target_{key}", _map_coordinate(target.get(key), low, high))
            yaw = _number(target.get("yaw"))
            sine, cosine = _number(target.get("sin")), _number(target.get("cos"))
            if yaw is None and sine is not None and cosine is not None:
                yaw = math.atan2(sine, cosine)
            row[prefix + "_target_yaw_sin"] = 0.0 if yaw is None else math.sin(yaw)
            row[prefix + "_target_yaw_cos"] = 0.0 if yaw is None else math.cos(yaw)
            row[prefix + "_target_yaw_known"] = float(yaw is not None)
        return row

    def _entity_features(self, item: dict[str, Any], group: str,
                         bounds: tuple[float, float, float, float]) -> dict[str, float]:
        row: dict[str, float] = {}
        pose = _pose(item)
        _put(row, "pose_x", _map_coordinate(pose.get("x"), bounds[0], bounds[1]))
        _put(row, "pose_y", _map_coordinate(pose.get("y"), bounds[2], bounds[3]))
        _put(row, "pose_z", pose.get("z"))
        yaw = _number(pose.get("yaw", pose.get("theta")))
        row["pose_yaw_sin"] = 0.0 if yaw is None else math.sin(yaw)
        row["pose_yaw_cos"] = 0.0 if yaw is None else math.cos(yaw)
        row["pose_yaw_known"] = float(yaw is not None)
        _put(row, "age", item.get("age_s"), 90.0)
        _put(row, "confidence", item.get("confidence"))
        if group == "cartons":
            size = _object(item.get("size"))
            for key in ("x", "y", "z"):
                _put(row, "size_" + key, size.get(key))
            _category(row, "posture", item.get("posture"), ("vertical", "horizontal"))
            _flag(row, "visible", item.get("visible"))
            _flag(row, "accessible", item.get("accessible"))
            _category(row, "source", item.get("position_source"), ("gazebo_sim", "camera", "map"))
        else:
            _put(row, "radius", item.get("radius"))
            size = _object(item.get("size"))
            _put(row, "size_x", size.get("x", item.get("length")))
            _put(row, "size_y", size.get("y", item.get("width")))
            _flag(row, "is_opponent", item.get("kind") == "opponent")
        return row

    def feature_dicts(self, world: dict[str, Any]) -> dict[str, Any]:
        if world.get("schema_version") != OBSERVATION_VERSION:
            raise ValueError(f"observation schema_version must be {OBSERVATION_VERSION}")
        bounds = self._bounds(world)
        robot = _object(world.get("robot"))
        pose = robot.get("pose")
        rows: dict[str, Any] = {
            "robot": self._robot_features(world, pose if isinstance(pose, dict) else None, bounds),
            "context": self._context_features(world),
        }
        for group in ENTITY_GROUPS:
            items = world.get(group, [])
            if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
                raise ValueError(f"world.{group} must be a list of objects")
            if group == "obstacles":
                items = list(items)
                opponents = world.get("other_robots", [])
                if isinstance(opponents, list):
                    for opponent in opponents:
                        if isinstance(opponent, dict):
                            items.append({"id": opponent.get("id"), "pose": _pose(opponent),
                                          "radius": opponent.get("radius", 0.20), "kind": "opponent"})
            rows[group] = [self._entity_features(item, group, bounds) for item in items]
        return rows

    def forward(self, world: dict[str, Any]) -> dict[str, Any]:
        rows = self.feature_dicts(world)
        device = next(self.parameters()).device

        def tensor(values: list[dict[str, float]], group: str) -> torch.Tensor:
            matrix = [[row[key] for key in self.feature_names[group]] for row in values]
            return torch.as_tensor(matrix, dtype=torch.float32, device=device).reshape(1, len(values), len(self.feature_names[group]))

        robot = self.robot_mlp(tensor([rows["robot"]], "robot").squeeze(1))
        context = tensor([rows["context"]], "context").squeeze(1)
        entities = {group: self.entity_mlps[group](tensor(rows[group], group)) for group in ENTITY_GROUPS}
        query = self.attention_query(torch.cat([robot, context], dim=-1)).unsqueeze(1)
        pooled = []
        for group in ENTITY_GROUPS:
            values = entities[group]
            if values.shape[1] == 0:
                pooled.append(torch.zeros_like(robot))
            else:
                attended, _ = self.attention[group](query, values, values, need_weights=False)
                pooled.append(attended.squeeze(1))
        combined = self.context_mlp(torch.cat([robot, *pooled, context], dim=-1))
        return {"context": combined, "entities": entities, "feature_dicts": rows}
