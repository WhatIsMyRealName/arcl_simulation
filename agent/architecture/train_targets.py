"""Train a robot policy from source-independent prepared target datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from datetime import datetime, timezone
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .policy import (
    ACTIONS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_POSE_FRESHNESS_S,
    DEFAULT_ROBOT_MARGIN_M,
    AgentPolicy,
)
from .targets import TARGET_DATASET_FORMAT, iter_jsonl


def read_target_datasets(
    paths: list[Path],
    *,
    pose_freshness_s: float = DEFAULT_POSE_FRESHNESS_S,
    speed_linear_limit: float = 0.25, speed_angular_limit: float = 1.0,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    sample_ids: set[str] = set()
    source_counts: dict[str, int] = {}
    for path, line_number, sample in iter_jsonl(paths, "target dataset"):
        if not isinstance(sample, dict) or sample.get("format") != TARGET_DATASET_FORMAT:
            raise ValueError(
                f"{path}:{line_number}: expected canonical {TARGET_DATASET_FORMAT} sample; "
                "prepare raw demonstration/self-play records first"
            )
        group_id = sample.get("group_id")
        sample_id = sample.get("sample_id")
        source = sample.get("source")
        observation = sample.get("observation")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"{path}:{line_number}: missing group_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}:{line_number}: missing sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id!r}")
        sample_ids.add(sample_id)
        if not isinstance(source, str) or not source:
            raise ValueError(f"{path}:{line_number}: missing source metadata")
        chosen_action = sample.get("chosen_action")
        if chosen_action not in ACTIONS:
            raise ValueError(f"{path}:{line_number}: unsupported chosen_action {chosen_action!r}")
        if not isinstance(observation, dict):
            raise ValueError(f"{path}:{line_number}: observation must be an object")
        probabilities = sample.get("target_action_probs")
        _validate_distribution(probabilities, f"{path}:{line_number}")
        mask = AgentPolicy.valid_action_mask(
            observation, torch.device("cpu"), pose_freshness_s
        )
        incompatible = [action for index, action in enumerate(ACTIONS)
                        if float(probabilities[index]) > 0.0 and not bool(mask[index])]
        if incompatible:
            raise ValueError(
                f"{path}:{line_number}: target assigns probability to masked actions: "
                + ", ".join(incompatible)
            )
        move_valid = sample.get("move_target_valid")
        target_pose = sample.get("target_move_pose")
        if not isinstance(move_valid, bool):
            raise ValueError(f"{path}:{line_number}: move_target_valid must be boolean")
        if move_valid:
            if not isinstance(target_pose, dict):
                raise ValueError(f"{path}:{line_number}: MOVE target pose is missing")
            if chosen_action != "MOVE":
                raise ValueError(f"{path}:{line_number}: pose target requires chosen_action MOVE")
            if probabilities[ACTIONS.index("MOVE")] <= 0.0:
                raise ValueError(f"{path}:{line_number}: pose target has no MOVE target mass")
            _validate_move_target(target_pose, path, line_number)
        elif target_pose is not None:
            raise ValueError(f"{path}:{line_number}: masked MOVE pose target must be null")
        valid_speed = sample.get("speed_target_valid", False)
        if not isinstance(valid_speed, bool):
            raise ValueError(f"{path}:{line_number}: invalid speed_target_valid")
        if valid_speed:
            from .actions import validate_speed
            if chosen_action != "SPEED":
                raise ValueError("speed targets require SPEED")
            validate_speed(observation, sample.get("target_speed"), speed_linear_limit, speed_angular_limit)
        elif sample.get("target_speed") is not None:
            raise ValueError("masked speed target must be null")
        image = observation.get("front_camera") or {}
        if image.get("path"):
            image_path = Path(image['path'])
            image['path'] = str((image_path if image_path.is_absolute() else path.parent / image_path).resolve())
            from .vision import validate_image_reference
            validate_image_reference(observation)
        groups.setdefault(group_id, []).append(sample)
        source_counts[source] = source_counts.get(source, 0) + 1
    if not groups:
        raise ValueError("no prepared training targets found")
    return groups, source_counts


def _validate_distribution(value: Any, location: str) -> None:
    if not isinstance(value, list) or len(value) != len(ACTIONS):
        raise ValueError(f"{location}: target_action_probs must have {len(ACTIONS)} entries")
    numbers = [_finite(item, math.nan) for item in value]
    if any(not math.isfinite(item) or item < 0.0 for item in numbers):
        raise ValueError(f"{location}: target action probabilities must be finite and non-negative")
    if not math.isclose(sum(numbers), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"{location}: target action probabilities must sum to one")


def _validate_move_target(target: dict[str, Any], path: Path, line_number: int) -> None:
    values = [_finite(target.get(key), math.nan) for key in ("x", "y", "sin", "cos")]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{path}:{line_number}: MOVE target must be finite")
    _, _, sine, cosine = values
    if math.hypot(sine, cosine) < 1e-6:
        raise ValueError(f"{path}:{line_number}: MOVE orientation target is degenerate")


def split_groups(
    groups: dict[str, list[dict[str, Any]]], validation_fraction: float, seed: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must be strictly between zero and one")
    if len(groups) < 2:
        raise ValueError("at least two sessions/episodes are required for train/validation")
    identifiers = sorted(groups)
    random.Random(seed).shuffle(identifiers)
    count = min(len(identifiers) - 1, max(1, round(len(identifiers) * validation_fraction)))
    validation_ids = set(identifiers[:count])
    return (
        {key: value for key, value in groups.items() if key not in validation_ids},
        {key: value for key, value in groups.items() if key in validation_ids},
    )


def _flatten(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [sample for samples in groups.values() for sample in samples]


def balance_action_samples(
    samples: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    by_action = {action: [] for action in ACTIONS}
    for sample in samples:
        by_action[sample["chosen_action"]].append(sample)
    target_count = max((len(rows) for rows in by_action.values()), default=0)
    if target_count == 0:
        raise ValueError("training split contains no target samples")

    generator = random.Random(seed)
    balanced: list[dict[str, Any]] = []
    for action in ACTIONS:
        rows = by_action[action]
        if not rows:
            continue
        balanced.extend(rows)
        for _ in range(target_count - len(rows)):
            balanced.append(generator.choice(rows))

    generator.shuffle(balanced)
    return balanced


def mirror_sample_along_length(sample: dict[str, Any]) -> dict[str, Any]:
    mirrored = copy.deepcopy(sample)
    observation = mirrored["observation"]
    arena = observation.get("arena") if isinstance(observation.get("arena"), dict) else {}
    bounds = arena.get("bounds") if isinstance(arena.get("bounds"), dict) else arena
    values = tuple(_finite(bounds.get(key), math.nan) for key in
                   ("x_min", "x_max", "y_min", "y_max"))
    if (not all(math.isfinite(value) for value in values)
            or values[0] >= values[1] or values[2] >= values[3]):
        x_min, x_max = 0.0, 3.0
    else:
        x_min, x_max = values[:2]
    reflection_sum = x_min + x_max

    def mirror_pose(pose: Any) -> None:
        if not isinstance(pose, dict):
            return
        position_x = _finite(pose.get("x"), math.nan)
        if math.isfinite(position_x):
            pose["x"] = reflection_sum - position_x
        for key in ("yaw", "theta"):
            angle = _finite(pose.get(key), math.nan)
            if math.isfinite(angle):
                pose[key] = math.pi - angle
        cosine = _finite(pose.get("cos"), math.nan)
        if math.isfinite(cosine):
            pose["cos"] = -cosine

    robot = observation.get("robot") if isinstance(observation.get("robot"), dict) else {}
    robot_pose = robot.get("pose")
    mirror_pose(robot_pose)
    velocity = robot.get("velocity")
    if isinstance(velocity, dict):
        for key in ("x", "yaw"):
            value = _finite(velocity.get(key), math.nan)
            if math.isfinite(value):
                velocity[key] = -value

    for group in ("cartons", "obstacles", "other_robots"):
        items = observation.get(group)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    mirror_pose(item.get("pose", item))

    for key in ("active_operation", "last_action_result"):
        operation = observation.get(key)
        if isinstance(operation, dict):
            mirror_pose(operation.get("target_pose"))

    mirror_pose(mirrored.get("target_move_pose"))
    speed = mirrored.get('target_speed')
    if isinstance(speed, dict):
        # Reflection of the world with heading pi-yaw preserves body forward and reverses lateral/yaw.
        speed['vy'] = -speed['vy']
        speed['omega'] = -speed['omega']
    return mirrored


def augment_length_symmetry(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [*samples, *(mirror_sample_along_length(sample) for sample in samples
                        if not (sample['observation'].get('front_camera') or {}).get('path'))]


def _class_weights(samples: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    counts = torch.zeros(len(ACTIONS), dtype=torch.float32, device=device)
    for sample in samples:
        counts += torch.tensor(sample["target_action_probs"], dtype=torch.float32, device=device)
    present = counts > 0
    weights = torch.ones_like(counts)
    weights[present] = torch.sqrt(counts[present].sum() / (present.sum() * counts[present]))
    weights[present] = weights[present] / weights[present].mean()
    return weights


def target_loss(
    policy: AgentPolicy,
    sample: dict[str, Any],
    class_weights: torch.Tensor,
    *,
    position_weight: float = 1.0,
    orientation_weight: float = 1.0,
    output: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    observation = sample["observation"]
    output = policy.network(observation) if output is None else output
    action_distribution, _ = policy._distributions(observation, output)
    target = torch.tensor(sample["target_action_probs"], dtype=torch.float32, device=policy.device)
    log_probs = action_distribution.logits
    active = target > 0.0
    if not torch.isfinite(log_probs[active]).all():
        raise ValueError("target assigns probability to an action masked by its observation")
    classification = -(target[active] * class_weights[active] * log_probs[active]).sum()

    position_loss = output["action_logits"].new_zeros(())
    orientation_loss = output["action_logits"].new_zeros(())
    if sample["move_target_valid"]:
        target_pose = sample["target_move_pose"]
        predicted_pose = policy.decode_move_tensor(observation, output["move_mean"][0])
        target_xy = torch.tensor(
            (float(target_pose["x"]), float(target_pose["y"])),
            dtype=torch.float32,
            device=policy.device,
        )
        position_loss = F.smooth_l1_loss(predicted_pose[:2], target_xy)
        target_direction = torch.tensor(
            (float(target_pose["sin"]), float(target_pose["cos"])),
            dtype=torch.float32,
            device=policy.device,
        )
        target_direction = F.normalize(target_direction, dim=0, eps=1e-6)
        predicted_direction = output["move_mean"][0, 2:4]
        orientation_loss = 1.0 - F.cosine_similarity(
            predicted_direction.unsqueeze(0), target_direction.unsqueeze(0), dim=-1, eps=1e-6
        )[0]
    speed_loss = output['action_logits'].new_zeros(())
    if sample.get('speed_target_valid', False):
        values = sample['target_speed']
        target_speed = output['speed_mean'].new_tensor([values[k] for k in ('vx', 'vy', 'omega')])
        bounds = target_speed.new_tensor([policy.speed_linear_limit, policy.speed_linear_limit, policy.speed_angular_limit])
        speed_loss = F.smooth_l1_loss(policy.decode_speed_tensor(output['speed_mean'][0]) / bounds, target_speed / bounds)
        magnitude = torch.linalg.vector_norm(target_speed[:2])
        speed_loss = speed_loss + F.smooth_l1_loss(torch.sigmoid(output['speed_mean'][0, 0]), magnitude / policy.speed_linear_limit)
        if float(magnitude) > 1e-6:
            direction = torch.stack((target_speed[1], target_speed[0])) / magnitude
            speed_loss = speed_loss + 1.0 - F.cosine_similarity(output['speed_mean'][0, 1:3], direction, dim=0)
    total = classification + position_weight * position_loss + orientation_weight * orientation_loss + speed_loss
    if not torch.isfinite(total):
        raise ValueError("non-finite target-supervised loss")
    return total, {"classification": classification, "position": position_loss,
                   "orientation": orientation_loss, "speed": speed_loss}


def evaluate(
    policy: AgentPolicy,
    samples: list[dict[str, Any]],
    class_weights: torch.Tensor,
    *,
    position_weight: float,
    orientation_weight: float,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("validation/test split contains no target samples")
    policy.network.eval()
    totals = {"loss": 0.0, "expected_action_accuracy": 0.0}
    move_distance = 0.0
    move_heading = 0.0
    move_count = 0
    speed_count = image_count = 0
    speed_linear_error = speed_angular_error = 0.0
    free_count = 0
    free_accuracy = 0.0
    action_counts = {action: 0 for action in ACTIONS}
    action_accuracy = {action: 0.0 for action in ACTIONS}
    with torch.no_grad():
        for sample in samples:
            output = policy.network(sample["observation"])
            loss, _ = target_loss(policy, sample, class_weights, output=output,
                                  position_weight=position_weight,
                                  orientation_weight=orientation_weight)
            distribution, _ = policy._distributions(sample["observation"], output)
            predicted_action = int(distribution.probs.argmax())
            target = sample["target_action_probs"]
            totals["loss"] += float(loss)
            totals["expected_action_accuracy"] += float(target[predicted_action])
            chosen = sample["chosen_action"]
            action_counts[chosen] += 1
            action_accuracy[chosen] += float(target[predicted_action])
            operation = sample["observation"].get("active_operation")
            if "active_operation" in sample["observation"] and (
                operation is None or isinstance(operation, dict) and operation.get("status") in (
                    "succeeded", "failed", "canceled", "rejected"
                )
            ):
                free_count += 1
                free_accuracy += float(target[predicted_action])
            from .actions import image_available
            image_count += int(image_available(sample['observation']))
            if sample.get('speed_target_valid', False):
                predicted_speed = policy.decode_speed_tensor(output['speed_mean'][0])
                target_speed = sample['target_speed']
                speed_linear_error += math.hypot(float(predicted_speed[0]) - target_speed['vx'], float(predicted_speed[1]) - target_speed['vy'])
                speed_angular_error += abs(float(predicted_speed[2]) - target_speed['omega'])
                speed_count += 1
            if sample["move_target_valid"]:
                target_pose = sample["target_move_pose"]
                prediction = policy.decode_move_tensor(sample["observation"], output["move_mean"][0])
                move_distance += math.hypot(float(prediction[0]) - float(target_pose["x"]),
                                            float(prediction[1]) - float(target_pose["y"]))
                predicted_yaw = math.atan2(float(prediction[2]), float(prediction[3]))
                target_yaw = math.atan2(float(target_pose["sin"]), float(target_pose["cos"]))
                move_heading += abs(_wrap_angle(predicted_yaw - target_yaw))
                move_count += 1
    count = len(samples)
    metrics = {key: value / count for key, value in totals.items()}
    metrics["move_position_error_m"] = move_distance / move_count if move_count else 0.0
    metrics["move_heading_error_rad"] = move_heading / move_count if move_count else 0.0
    metrics["speed_samples"] = speed_count
    metrics["image_samples"] = image_count
    metrics["image_availability"] = image_count / count
    metrics["speed_linear_error_m_s"] = speed_linear_error / speed_count if speed_count else None
    metrics["speed_angular_error_rad_s"] = speed_angular_error / speed_count if speed_count else None
    metrics["move_samples"] = move_count
    metrics["free_action_samples"] = free_count
    metrics["free_expected_action_accuracy"] = free_accuracy / free_count if free_count else None
    metrics["per_action"] = {
        action: {"samples": count, "expected_action_accuracy": action_accuracy[action] / count}
        for action, count in action_counts.items() if count
    }
    return metrics


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    robot_margin_override = getattr(args, "robot_margin", None)
    pose_freshness_override = getattr(args, "pose_freshness", None)
    if args.model is not None and not args.model.is_file():
        raise ValueError(f"initial checkpoint does not exist: {args.model}")
    torch.set_num_threads(args.cpu_threads)
    policy, _ = AgentPolicy.from_checkpoint(
        args.model, device=args.device,
        robot_margin_m=robot_margin_override, pose_freshness_s=pose_freshness_override,
        speed_linear_limit=args.speed_linear_limit, speed_angular_limit=args.speed_angular_limit,
    )

    groups, source_counts = read_target_datasets(
        args.datasets, pose_freshness_s=policy.pose_freshness_s,
        speed_linear_limit=policy.speed_linear_limit, speed_angular_limit=policy.speed_angular_limit
    )
    if args.validation_data:
        training = groups
        validation, _ = read_target_datasets(
            args.validation_data, pose_freshness_s=policy.pose_freshness_s,
            speed_linear_limit=policy.speed_linear_limit, speed_angular_limit=policy.speed_angular_limit
        )
    else:
        training, validation = split_groups(groups, args.validation_fraction, args.seed)
    test, _ = read_target_datasets(
        args.test_data, pose_freshness_s=policy.pose_freshness_s,
        speed_linear_limit=policy.speed_linear_limit, speed_angular_limit=policy.speed_angular_limit
    ) if args.test_data else ({}, {})
    if set(training) & set(validation) or set(test) & (set(training) | set(validation)):
        raise ValueError("train, validation and test groups must be disjoint")

    run_directory = args.run_directory or Path('runs/training') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    run_directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    original_train_samples = _flatten(training)
    balanced_samples = balance_action_samples(original_train_samples, args.seed)
    train_samples = augment_length_symmetry(balanced_samples)
    validation_samples = _flatten(validation)
    validation_weights = _class_weights(balanced_samples, policy.device)
    options = {"position_weight": args.position_weight,
               "orientation_weight": args.orientation_weight}
    baseline = evaluate(policy, validation_samples, validation_weights, **options)
    optimizer = torch.optim.Adam(policy.network.parameters(), lr=args.learning_rate)
    report: dict[str, Any] = {
        "train_group_ids": sorted(training), "validation_group_ids": sorted(validation),
        "test_group_ids": sorted(test), "train_samples": len(train_samples),
        "original_train_samples": len(original_train_samples),
        "balanced_train_samples": len(balanced_samples),
        "train_action_counts": {action: sum(
            sample["chosen_action"] == action for sample in balanced_samples
        ) for action in ACTIONS if any(
            sample["chosen_action"] == action for sample in balanced_samples
        )},
        "validation_samples": len(validation_samples), "source_counts": source_counts,
        "seed": args.seed, "baseline_validation": baseline, "epochs": [], "best_epoch": 0,
        "configuration": {"initial_checkpoint": str(args.model) if args.model else None,
                          "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                          "epochs": args.epochs, "device": args.device, **options},
    }
    report["configuration"].update({
        "speed_linear_limit": policy.speed_linear_limit, "speed_angular_limit": policy.speed_angular_limit,
        "robot_margin_m": policy.robot_margin_m,
        "pose_freshness_s": policy.pose_freshness_s,
    })
    report['run_directory'] = str(run_directory)
    report['datasets'] = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                          for values in (args.datasets, args.validation_data or [], args.test_data or [])
                          for value in values for path in ([value] if value.is_file() else sorted(value.rglob('*.jsonl')))]
    (run_directory / 'configuration.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    policy.save_checkpoint(run_directory / "best.pt")
    best = baseline["loss"]
    best_state = copy.deepcopy(policy.network.state_dict())
    best_status = policy.training_status
    for epoch in range(1, args.epochs + 1):
        policy.network.train()
        random.shuffle(train_samples)
        for start in range(0, len(train_samples), args.batch_size):
            batch = train_samples[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            outputs = policy.network.forward_batch([sample['observation'] for sample in batch])
            losses = [target_loss(policy, sample, validation_weights, output=output, **options)[0]
                      for sample, output in zip(batch, outputs)]
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.network.parameters(), 0.5)
            optimizer.step()
        metrics = {
            "epoch": epoch,
            "train": evaluate(policy, train_samples, validation_weights, **options),
            "validation": evaluate(policy, validation_samples, validation_weights, **options),
        }
        report["epochs"].append(metrics)
        linear = metrics["validation"]["speed_linear_error_m_s"]
        angular = metrics["validation"]["speed_angular_error_rad_s"]
        linear_text = f"{linear:.3f}" if linear is not None else "—"
        angular_text = f"{angular:.3f}" if angular is not None else "—"
        line = (f"Époque {epoch}/{args.epochs} | perte train {metrics['train']['loss']:.4f} | "
                f"validation {metrics['validation']['loss']:.4f} | "
                f"pose {metrics['validation']['move_position_error_m'] * 100:.2f} cm | "
                f"vitesse {linear_text} m/s | "
                f"rotation {angular_text} rad/s | "
                f"précision attendue {metrics['validation']['expected_action_accuracy']:.1%} | "
                f"meilleure {epoch if metrics['validation']['loss'] < best else report['best_epoch']} | "
                f"{time.monotonic() - started:.1f} s")
        print(line, flush=True)
        with (run_directory / 'train.log').open('a') as log:
            log.write(line + '\n')
        with (run_directory / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + '\n')
        flat = {'epoch': epoch, **{f'{split}_{key}': value for split in ('train', 'validation')
                                  for key, value in metrics[split].items() if key != 'per_action'}}
        with (run_directory / 'metrics.csv').open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(flat)
        if metrics["validation"]["loss"] < best:
            best = metrics["validation"]["loss"]
            report["best_epoch"] = epoch
            best_state = copy.deepcopy(policy.network.state_dict())
            best_status = "trained_targets"
            policy.training_status = best_status
            policy.save_checkpoint(run_directory / "best.pt")
    policy.network.load_state_dict(best_state)
    policy.training_status = best_status
    policy.network.eval()
    if test:
        report["test"] = evaluate(policy, _flatten(test), validation_weights, **options)
    report["selected_training_status"] = best_status
    policy.save_checkpoint(args.output)
    policy.save_checkpoint(run_directory / "best.pt")
    (run_directory / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path, help="canonical target JSONL files or directories")
    parser.add_argument("--run-directory", type=Path, help="new training artifact directory")
    parser.add_argument("--model", type=Path, help="optional initial checkpoint")
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--validation-data", nargs="+", type=Path)
    parser.add_argument("--test-data", nargs="+", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=1.0)
    parser.add_argument("--speed-linear-limit", type=float, help="translation magnitude bound in m/s")
    parser.add_argument("--speed-angular-limit", type=float, help="angular velocity bound in rad/s")
    parser.add_argument("--cpu-threads", type=int, default=1, help="PyTorch CPU workers")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--robot-margin", type=float,
                        help="override checkpoint robot margin (default: checkpoint or 0.20 m)")
    parser.add_argument("--pose-freshness", type=float,
                        help="override checkpoint pose freshness (default: checkpoint or 0.75 s)")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0.0):
        parser.error("epochs, batch size and learning rate must be positive")
    if (not 0.0 < args.validation_fraction < 1.0 or any(
        not math.isfinite(value) or value < 0.0
        for value in (args.position_weight, args.orientation_weight)
    )):
        parser.error("invalid validation fraction or loss weights")
    if ((args.robot_margin is not None
         and (not math.isfinite(args.robot_margin) or args.robot_margin < 0.0))
            or (args.pose_freshness is not None
                and (not math.isfinite(args.pose_freshness) or args.pose_freshness <= 0.0))):
        parser.error("robot margin must be non-negative and pose freshness must be positive")
    try:
        report = train(args)
        selection = (f"epoch {report['best_epoch']}" if report["best_epoch"]
                     else "initial baseline retained; no epoch improved validation loss")
        print(f"saved checkpoint: {args.output} ({selection})")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"train_targets: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
