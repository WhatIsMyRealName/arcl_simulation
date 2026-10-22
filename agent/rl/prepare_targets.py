"""Convert rewarded self-play episode logs into canonical supervised targets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from ..architecture.actions import ACTIONS
from ..architecture.targets import (
    DEFAULT_SELFPLAY_STEP_SIZE,
    DEFAULT_TARGET_DATASET_PATH,
    TARGET_DATASET_FORMAT,
    append_target_samples,
    iter_jsonl,
    selfplay_target_distribution,
)
from ..records import RECORD_SCHEMA_VERSION


def read_selfplay_episodes(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    episodes: dict[str, list[dict[str, Any]]] = {}
    locations: dict[str, Path] = {}
    for path, line_number, row in iter_jsonl(paths, "self-play record"):
        if not isinstance(row, dict) or row.get("record_schema_version") != RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"{path}:{line_number}: expected episode record schema v{RECORD_SCHEMA_VERSION}"
            )
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str):
            raise ValueError(f"{path}:{line_number}: missing episode_id")
        if episode_id in locations and locations[episode_id] != path:
            raise ValueError(f"duplicate episode {episode_id!r} in multiple files")
        locations[episode_id] = path
        episodes.setdefault(episode_id, []).append(row)
    return episodes


def prepare_episode_targets(
    episode_id: str,
    rows: list[dict[str, Any]],
    *,
    robot_id: str | None = None,
    step_size: float = DEFAULT_SELFPLAY_STEP_SIZE,
) -> list[dict[str, Any]]:
    ends = [row for row in rows if row.get("kind") == "episode_end"]
    if len(ends) != 1 or ends[0].get("complete") is not True:
        raise ValueError(f"episode {episode_id}: episode is incomplete or has no unique end marker")
    end = ends[0]
    if end.get("rewards_complete") is not True:
        raise ValueError(f"episode {episode_id}: rewards are incomplete")
    gamma = _finite(end.get("gamma"), math.nan)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"episode {episode_id}: invalid discount factor")
    transitions = [row for row in rows if row.get("kind") == "transition"]
    if not transitions:
        raise ValueError(f"episode {episode_id}: no transitions")

    robot_ids = end.get("robot_ids")
    if not isinstance(robot_ids, list) or not robot_ids:
        raise ValueError(f"episode {episode_id}: missing robot IDs")
    selected_robots = [robot_id] if robot_id is not None else [str(value) for value in robot_ids]
    if any(robot not in robot_ids for robot in selected_robots):
        raise ValueError(f"episode {episode_id}: requested robot is not in this episode")

    prepared: list[dict[str, Any]] = []
    for robot in selected_robots:
        returns: list[float] = [0.0] * len(transitions)
        total_return = 0.0
        for index in range(len(transitions) - 1, -1, -1):
            reward_map = transitions[index].get("reward")
            if not isinstance(reward_map, dict) or robot not in reward_map:
                raise ValueError(f"episode {episode_id}: missing reward for {robot}")
            reward = _finite(reward_map[robot], math.nan)
            if not math.isfinite(reward):
                raise ValueError(f"episode {episode_id}: non-finite reward for {robot}")
            total_return = reward + gamma * total_return
            returns[index] = total_return
        scale = max((abs(value) for value in returns), default=0.0)

        for index, (row, return_value) in enumerate(zip(transitions, returns)):
            actions = row.get("actions")
            observations = row.get("agent_observations")
            decision = actions.get(robot) if isinstance(actions, dict) else None
            observation = observations.get(robot) if isinstance(observations, dict) else None
            if not isinstance(decision, dict) or not isinstance(observation, dict):
                raise ValueError(f"episode {episode_id}: missing decision or observation for {robot}")
            action = decision.get("action")
            if action not in ACTIONS:
                raise ValueError(f"episode {episode_id}: unsupported action {action!r}")
            if observation.get("schema_version") != 2:
                raise ValueError(f"episode {episode_id}: expected observation schema v2")
            behavior = decision.get("behavior")
            probabilities = behavior.get("action_probs") if isinstance(behavior, dict) else None
            if not isinstance(probabilities, list):
                raise ValueError(f"episode {episode_id}: missing recorded action distribution for {robot}")
            signal = max(-1.0, min(1.0, return_value / scale)) if scale > 0.0 else 0.0
            target_probs = selfplay_target_distribution(
                probabilities, action, signal, step_size=step_size
            )
            step = row.get("step", index + 1)
            prepared.append({
                "format": TARGET_DATASET_FORMAT,
                "source": "selfplay",
                "group_id": episode_id,
                "sample_id": f"{episode_id}:{robot}:{step}",
                "observation": observation,
                "chosen_action": action,
                "target_action_probs": target_probs,
                "target_move_pose": None,
                "move_target_valid": False,
                "metadata": {
                    "episode_id": episode_id,
                    "robot_id": robot,
                    "step": step,
                    "discounted_return": return_value,
                    "normalized_return": signal,
                    "target_step_size": step_size,
                },
            })
    return prepared


def prepare_targets(
    paths: list[Path],
    output: Path,
    *,
    robot_id: str | None = None,
    step_size: float = DEFAULT_SELFPLAY_STEP_SIZE,
) -> int:
    if not math.isfinite(step_size) or not 0.0 <= step_size <= 1.0:
        raise ValueError("target step size must be between 0 and 1")
    episodes = read_selfplay_episodes(paths)
    samples = [
        sample
        for episode_id, rows in sorted(episodes.items())
        for sample in prepare_episode_targets(
            episode_id, rows, robot_id=robot_id, step_size=step_size
        )
    ]
    if not samples:
        raise ValueError("no self-play targets were produced")
    return append_target_samples(output, samples)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path, help="complete rewarded episode logs")
    parser.add_argument("--output", type=Path, default=DEFAULT_TARGET_DATASET_PATH,
                        help="canonical target JSONL file (appended; default: %(default)s)")
    parser.add_argument("--robot-id", help="prepare one robot only; default prepares both")
    parser.add_argument("--target-step-size", type=float, default=DEFAULT_SELFPLAY_STEP_SIZE)
    args = parser.parse_args()
    try:
        count = prepare_targets(args.records, args.output, robot_id=args.robot_id,
                                step_size=args.target_step_size)
        print(f"prepared {count} self-play targets: {args.output}")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"prepare_targets: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
