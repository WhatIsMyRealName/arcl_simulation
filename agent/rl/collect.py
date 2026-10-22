"""Collect stochastic two-robot episodes through a user-provided simulator backend."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from ..architecture.policy import (
    AgentPolicy,
    DEFAULT_CHECKPOINT_PATH,
)
from .environment import TwoRobotArenaEnv, assign_action_points, assign_winner_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, help="module:factory returning reset()/advance() backend")
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--opponent-id", required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--opponent-model", type=Path, help="default: same policy for self-play")
    parser.add_argument("--records", type=Path, default=Path("runs/episodes"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-seconds", type=float, default=90)
    parser.add_argument("--robot-margin", type=float,
                        help="override checkpoint map-boundary clearance, in meters")
    parser.add_argument("--pose-freshness", type=float,
                        help="override checkpoint robot-pose freshness, in seconds")
    parser.add_argument("--velocity-scale", type=float, default=1.0,
                        help="linear velocity feature scale in meters per second")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--record-only", action="store_true", help="collect without the unimplemented reward functions")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    backend = env = None
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        module, factory = args.backend.split(":", 1)
        backend = getattr(importlib.import_module(module), factory)()
        policy, created = AgentPolicy.from_checkpoint(
            args.model, device=args.device, create_if_missing=True,
            velocity_scale_m_s=args.velocity_scale, robot_margin_m=args.robot_margin,
            pose_freshness_s=args.pose_freshness,
        )
        if created:
            print(f"WARNING: checkpoint was absent; created randomly initialized policy at {args.model}. "
                  "This policy has not been trained.", file=sys.stderr, flush=True)
        opponent = policy
        if args.opponent_model is not None:
            opponent, _ = AgentPolicy.from_checkpoint(
                args.opponent_model, device=args.device, velocity_scale_m_s=args.velocity_scale,
                robot_margin_m=args.robot_margin, pose_freshness_s=args.pose_freshness,
            )
        env = TwoRobotArenaEnv(
            backend.advance, controlled_robot_id=args.robot_id, opponent_robot_id=args.opponent_id,
            episode_seconds=args.episode_seconds, records_path=args.records, gamma=args.gamma,
            robot_margin_m={args.robot_id: policy.robot_margin_m,
                            args.opponent_id: opponent.robot_margin_m},
            pose_freshness_s={args.robot_id: policy.pose_freshness_s,
                              args.opponent_id: opponent.pose_freshness_s},
            action_rewarder=None if args.record_only else assign_action_points,
            terminal_rewarder=None if args.record_only else assign_winner_points,
        )
        for episode in range(args.episodes):
            env.reset(backend.reset())
            while not env.done:
                before = env.agent_observations
                actions = (policy.decide(before[args.robot_id], deterministic=False),
                           opponent.decide(before[args.opponent_id], deterministic=False))
                _, _, _, _, info = env.step(*actions)
            print(json.dumps({"episode": episode + 1, "winner_id": info["winner_id"],
                              "rewards_pending": info["rewards_pending"], "records_path": info["records_path"]}), flush=True)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, ImportError, AttributeError) as error:
        print(f"collect: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    finally:
        if env is not None:
            env.close()
        if backend is not None and callable(getattr(backend, "close", None)):
            backend.close()


if __name__ == "__main__":
    main()
