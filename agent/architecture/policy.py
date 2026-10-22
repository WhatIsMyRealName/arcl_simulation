"""Masked hybrid actor for single-operation robot control."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from .actions import (
    ACTIONS, DEFAULT_POSE_FRESHNESS_S, action_mask, move_bounds,
)
from .world_encoder import CONTEXT_SIZE, WorldEncoder


CHECKPOINT_FORMAT = "arcl-agent-policy-v2"
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "policy.pt"
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
DEFAULT_ROBOT_MARGIN_M = 0.20
class PolicyNetwork(nn.Module):
    """Shared observation encoder with categorical, continuous, and value heads."""

    def __init__(self, *, velocity_scale_m_s: float = 1.0,
                 arena_bounds: tuple[float, float, float, float] = (0.0, 3.0, 0.0, 2.0)) -> None:
        super().__init__()
        self.encoder = WorldEncoder(velocity_scale_m_s=velocity_scale_m_s, arena_bounds=arena_bounds)
        self.action_head = nn.Sequential(
            nn.Linear(CONTEXT_SIZE, 64), nn.Tanh(), nn.Linear(64, len(ACTIONS))
        )
        self.move_mean_head = nn.Sequential(
            nn.Linear(CONTEXT_SIZE, 64), nn.Tanh(), nn.Linear(64, 4)
        )
        self.move_log_std = nn.Parameter(torch.full((4,), -0.5))
        self.value_head = nn.Sequential(
            nn.Linear(CONTEXT_SIZE, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def forward(self, world: dict[str, Any]) -> dict[str, Any]:
        context = self.encoder(world)["context"]
        move_mean = self.move_mean_head(context)
        return {
            "action_logits": self.action_head(context),
            "move_mean": move_mean,
            "move_log_std": self.move_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(
                move_mean
            ),
            "value": self.value_head(context).squeeze(-1),
        }


class AgentPolicy:
    """Sample valid intents and absolute map-frame move targets."""

    def __init__(self, *, device: str = "cpu", robot_margin_m: float = DEFAULT_ROBOT_MARGIN_M,
                 velocity_scale_m_s: float = 1.0,
                 pose_freshness_s: float = DEFAULT_POSE_FRESHNESS_S,
                 arena_bounds: tuple[float, float, float, float] = (0.0, 3.0, 0.0, 2.0)) -> None:
        self.device = torch.device(device)
        self.robot_margin_m = float(robot_margin_m)
        if not math.isfinite(self.robot_margin_m) or self.robot_margin_m < 0:
            raise ValueError("robot_margin_m must be finite and non-negative")
        self.pose_freshness_s = float(pose_freshness_s)
        if not math.isfinite(self.pose_freshness_s) or self.pose_freshness_s <= 0:
            raise ValueError("pose_freshness_s must be finite and positive")
        self.velocity_scale_m_s = float(velocity_scale_m_s)
        self.arena_bounds = tuple(float(value) for value in arena_bounds)
        self.training_status = "untrained"
        self.network = PolicyNetwork(velocity_scale_m_s=self.velocity_scale_m_s,
                                     arena_bounds=self.arena_bounds).to(self.device)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
            found = checkpoint.get("format") if isinstance(checkpoint, dict) else type(checkpoint).__name__
            raise ValueError(f"unsupported checkpoint format {found!r}; expected {CHECKPOINT_FORMAT}")
        configuration = checkpoint["observation_config"]
        if not isinstance(configuration, dict):
            raise ValueError("checkpoint observation_config must be an object")
        self.velocity_scale_m_s = float(configuration["velocity_scale_m_s"])
        self.arena_bounds = tuple(float(value) for value in configuration["arena_bounds"])
        self.robot_margin_m = float(configuration["robot_margin_m"])
        self.pose_freshness_s = float(configuration["pose_freshness_s"])
        if (len(self.arena_bounds) != 4 or not all(math.isfinite(value) for value in self.arena_bounds)
                or self.arena_bounds[0] >= self.arena_bounds[1]
                or self.arena_bounds[2] >= self.arena_bounds[3]
                or not math.isfinite(self.velocity_scale_m_s) or self.velocity_scale_m_s <= 0
                or not math.isfinite(self.robot_margin_m) or self.robot_margin_m < 0
                or not math.isfinite(self.pose_freshness_s) or self.pose_freshness_s <= 0):
            raise ValueError("checkpoint observation configuration is invalid")
        self.network.encoder.velocity_scale_m_s = self.velocity_scale_m_s
        self.network.encoder.arena_bounds = self.arena_bounds
        self.network.load_state_dict(checkpoint["state_dict"], strict=True)
        self.training_status = checkpoint["training_status"]
        self.network.eval()

    @classmethod
    def from_checkpoint(
        cls, path: str | Path | None, *, device: str = "cpu", create_if_missing: bool = False,
        velocity_scale_m_s: float = 1.0,
        robot_margin_m: float | None = None, pose_freshness_s: float | None = None,
    ) -> tuple[AgentPolicy, bool]:
        policy = cls(
            device=device,
            velocity_scale_m_s=velocity_scale_m_s,
            robot_margin_m=DEFAULT_ROBOT_MARGIN_M if robot_margin_m is None else robot_margin_m,
            pose_freshness_s=DEFAULT_POSE_FRESHNESS_S if pose_freshness_s is None else pose_freshness_s,
        )
        if path is None:
            return policy, False
        checkpoint_path = Path(path)
        created = not checkpoint_path.is_file()
        if created:
            if not create_if_missing:
                raise FileNotFoundError(checkpoint_path)
            policy.save_checkpoint(checkpoint_path)
        else:
            policy.load_checkpoint(checkpoint_path)
        for name, override in (("robot_margin_m", robot_margin_m), ("pose_freshness_s", pose_freshness_s)):
            if override is not None:
                if override != getattr(policy, name):
                    warnings.warn(f"explicit {name} overrides checkpoint configuration", RuntimeWarning)
                setattr(policy, name, float(override))
        return policy, created

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": CHECKPOINT_FORMAT,
            "observation_config": {
                "velocity_scale_m_s": self.velocity_scale_m_s,
                "arena_bounds": list(self.arena_bounds),
                "robot_margin_m": self.robot_margin_m,
                "pose_freshness_s": self.pose_freshness_s,
            },
            "training_status": self.training_status,
            "state_dict": self.network.state_dict(),
        }, checkpoint_path)

    @classmethod
    def valid_action_mask(
        cls, world: dict[str, Any], device: torch.device,
        pose_freshness_s: float = DEFAULT_POSE_FRESHNESS_S,
    ) -> torch.Tensor:
        return torch.tensor(action_mask(world, pose_freshness_s), dtype=torch.bool, device=device)

    def _distributions(self, world: dict[str, Any], output: dict[str, Any]) -> tuple[Categorical, Normal]:
        mask = self.valid_action_mask(world, self.device, self.pose_freshness_s)
        action = Categorical(logits=output["action_logits"][0].masked_fill(~mask, -torch.inf))
        move = Normal(output["move_mean"][0], output["move_log_std"][0].exp())
        return action, move

    def _move_bounds(self, world: dict[str, Any]) -> tuple[float, float, float, float]:
        return move_bounds(world, self.robot_margin_m)

    def decode_move_tensor(self, world: dict[str, Any], latent: torch.Tensor) -> torch.Tensor:
        x_min, x_max, y_min, y_max = self._move_bounds(world)
        x = x_min + (x_max - x_min) * torch.sigmoid(latent[..., 0])
        y = y_min + (y_max - y_min) * torch.sigmoid(latent[..., 1])
        direction = latent[..., 2:4]
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        robot = world.get("robot") if isinstance(world.get("robot"), dict) else {}
        pose = robot.get("pose") if isinstance(robot.get("pose"), dict) else {}
        yaw = _finite(pose.get("yaw"), 0.0)
        fallback = torch.tensor((math.sin(yaw), math.cos(yaw)), dtype=latent.dtype,
                                device=latent.device).expand_as(direction)
        normalized = direction / norm.clamp_min(1e-12)
        direction = torch.where(norm < 1e-6, fallback, normalized)
        return torch.stack((x, y, direction[..., 0], direction[..., 1]), dim=-1)

    def _decode_move(self, world: dict[str, Any], latent: torch.Tensor) -> dict[str, float]:
        decoded = self.decode_move_tensor(world, latent)
        x, y, sine, cosine = decoded.tolist()
        return {"x": float(x), "y": float(y), "sin": float(sine), "cos": float(cosine)}

    def _evaluate_output(self, world: dict[str, Any], decision: dict[str, Any], output: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_dist, move_dist = self._distributions(world, output)
        action = decision.get("action")
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        action_index = ACTIONS.index(action)
        if not torch.isfinite(action_dist.logits[action_index]).item():
            raise ValueError(f"recorded action {action} is incompatible with its observation")
        log_prob = action_dist.log_prob(torch.tensor(action_index, device=self.device))
        entropy = action_dist.entropy()
        if action == "MOVE":
            latent_values = decision.get("behavior", {}).get("move_latent", decision.get("move_latent"))
            if not isinstance(latent_values, list) or len(latent_values) != 4:
                raise ValueError("MOVE decision must record its four latent Gaussian samples")
            latent = torch.tensor(latent_values, dtype=torch.float32, device=self.device)
            log_prob = log_prob + move_dist.log_prob(latent).sum()
            entropy = entropy + move_dist.entropy().sum()
        return log_prob, entropy, output["value"][0]

    def evaluate_action(self, world: dict[str, Any], decision: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._evaluate_output(world, decision, self.network(world))

    def decide(self, world: dict[str, Any], *, deterministic: bool = True) -> dict[str, Any]:
        with torch.no_grad():
            output = self.network(world)
            action_dist, move_dist = self._distributions(world, output)
            action_index = int(torch.argmax(action_dist.logits).item()) if deterministic else int(action_dist.sample().item())
            action = ACTIONS[action_index]
            decision: dict[str, Any] = {
                "action": action,
                "world_sequence": int(world.get("sequence", -1)),
                "world_timestamp": _finite(world.get("timestamp")),
            }
            if action == "MOVE":
                latent = move_dist.mean if deterministic else move_dist.sample()
                latent_list = latent.tolist()
                target = self._decode_move(world, latent)
                decision.update({"target_pose": target, "move_latent": latent_list})
            log_prob, _, value = self._evaluate_output(world, decision, output)
            decision["behavior"] = {
                "sampled": not deterministic,
                "log_prob": float(log_prob),
                "value": float(value),
                "action_probs": action_dist.probs.tolist(),
                **({"move_latent": decision.pop("move_latent")} if action == "MOVE" else {}),
            }
        return decision


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default
