"""Action vocabulary and validation shared by policy, execution and RL."""

from __future__ import annotations

import math
from typing import Any


DEFAULT_POSE_FRESHNESS_S = 0.75
MIN_MOVE_DISTANCE_M = 1e-4

ACTIONS = (
    "IDLE", "CANCEL", "MOVE", "SUCTION_1_ON", "SUCTION_2_ON", "SUCTION_3_ON",
    "SUCTION_1_OFF", "SUCTION_2_OFF", "SUCTION_3_OFF", "TOGGLE_WRIST",
    "SUCTION_ALL_ON", "SUCTION_ALL_OFF",
)


def pose_available(world: dict[str, Any], freshness_s: float = DEFAULT_POSE_FRESHNESS_S) -> bool:
    robot = world.get("robot")
    pose = robot.get("pose") if isinstance(robot, dict) else None
    sources = world.get("sources")
    source = sources.get("robot_pose") if isinstance(sources, dict) else None
    age = _number(robot.get("pose_age_s"), math.inf) if isinstance(robot, dict) else math.inf
    return (isinstance(pose, dict) and (not isinstance(source, dict) or source.get("fresh", True) is True)
            and age <= freshness_s
            and all(math.isfinite(_number(pose.get(key), math.nan)) for key in ("x", "y", "yaw")))


def wrist_available(world: dict[str, Any]) -> bool:
    robot = world.get("robot")
    if not isinstance(robot, dict) or not math.isfinite(_number(robot.get("wrist_angle_rad"), math.nan)):
        return False
    sources = world.get("sources")
    source = sources.get("wrist") if isinstance(sources, dict) else {}
    if not isinstance(source, dict):
        source = {}
    return source.get("fresh", True) is True and _number(source.get("age_s"), math.inf) <= 1.0


def operation_allows(action: str, world: dict[str, Any]) -> bool:
    if "active_operation" not in world:
        return action == "IDLE"
    operation = world["active_operation"]
    if operation is None:
        return action != "CANCEL"
    if not isinstance(operation, dict):
        return action == "IDLE"
    status = operation.get("status")
    if status in ("issuing", "running"):
        return action == "IDLE" or (
            action == "CANCEL" and operation.get("type") == "MOVE"
            and operation.get("cancelable") is True
        )
    if status in ("succeeded", "failed", "canceled", "rejected"):
        return action != "CANCEL"
    return action == "IDLE"


def action_mask(world: dict[str, Any], pose_freshness_s: float = DEFAULT_POSE_FRESHNESS_S) -> list[bool]:
    mask = [operation_allows(action, world) for action in ACTIONS]
    if not pose_available(world, pose_freshness_s):
        mask[ACTIONS.index("MOVE")] = False
    if not wrist_available(world):
        mask[ACTIONS.index("TOGGLE_WRIST")] = False
    if not any(mask):
        mask[ACTIONS.index("IDLE")] = True
    return mask


def move_bounds(world: dict[str, Any], robot_margin_m: float) -> tuple[float, float, float, float]:
    arena = world.get("arena")
    bounds = arena.get("bounds", arena) if isinstance(arena, dict) else {}
    values = tuple(_number(bounds.get(key), default) for key, default in
                   (("x_min", 0.0), ("x_max", 3.0), ("y_min", 0.0), ("y_max", 2.0)))
    x_min, x_max, y_min, y_max = (values[0] + robot_margin_m, values[1] - robot_margin_m,
                                   values[2] + robot_margin_m, values[3] - robot_margin_m)
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("arena is too small for the configured robot margin")
    return x_min, x_max, y_min, y_max


def validate_move_target(
    world: dict[str, Any], target: Any, *, robot_margin_m: float,
    pose_freshness_s: float = DEFAULT_POSE_FRESHNESS_S,
) -> dict[str, float]:
    if not pose_available(world, pose_freshness_s):
        raise ValueError("MOVE requires a fresh robot pose")
    if not isinstance(target, dict):
        raise ValueError("MOVE requires target_pose")
    x, y, sine, cosine = (_number(target.get(key), math.nan) for key in ("x", "y", "sin", "cos"))
    if not all(math.isfinite(value) for value in (x, y, sine, cosine)):
        raise ValueError("MOVE target requires finite x, y, sin and cos")
    x_min, x_max, y_min, y_max = move_bounds(world, robot_margin_m)
    if not (x_min <= x <= x_max and y_min <= y <= y_max):
        raise ValueError("MOVE target lies outside the robot-center arena bounds")
    pose = world["robot"]["pose"]
    if math.hypot(x - pose["x"], y - pose["y"]) < MIN_MOVE_DISTANCE_M:
        raise ValueError("MOVE target is too close to the current position")
    norm = math.hypot(sine, cosine)
    if norm < 1e-6:
        raise ValueError("MOVE target orientation vector has near-zero norm")
    return {"x": x, "y": y, "sin": sine / norm, "cos": cosine / norm}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default
