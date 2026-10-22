"""Shared target-data format and self-play target construction."""

from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

from .actions import ACTIONS


TARGET_DATASET_FORMAT = "arcl-agent-targets-v1"
DEFAULT_SELFPLAY_STEP_SIZE = 0.1
DEFAULT_TARGET_DATASET_PATH = Path("runs/targets.jsonl")


def iter_jsonl(paths: Sequence[str | Path], label: str) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    files: set[Path] = set()
    for value in paths:
        path = Path(value)
        if not path.exists():
            raise ValueError(f"{label} path does not exist: {path}")
        if path.is_dir():
            files.update(item.resolve() for item in path.rglob("*.jsonl"))
        else:
            files.add(path.resolve())
    if not files:
        raise ValueError(f"no {label} JSONL files found")
    for path in sorted(files):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield path, line_number, json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def append_target_samples(path: str | Path, samples: Sequence[dict[str, Any]]) -> int:
    """Append canonical samples, skipping identical sample IDs already present."""
    target_path = Path(path)
    rows = list(samples)
    if not rows:
        return 0
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with target_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            existing: dict[str, dict[str, Any]] = {}
            stream.seek(0)
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{target_path}:{line_number}: invalid JSON") from error
                if not isinstance(sample, dict) or sample.get("format") != TARGET_DATASET_FORMAT:
                    raise ValueError(
                        f"{target_path}:{line_number}: expected canonical target sample"
                    )
                sample_id = sample.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"{target_path}:{line_number}: missing sample_id")
                if sample_id in existing:
                    raise ValueError(f"{target_path}:{line_number}: duplicate sample_id {sample_id!r}")
                existing[sample_id] = sample

            to_append: list[dict[str, Any]] = []
            for sample in rows:
                if sample.get("format") != TARGET_DATASET_FORMAT:
                    raise ValueError("only canonical target samples can be appended")
                sample_id = sample.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError("target sample is missing sample_id")
                previous = existing.get(sample_id)
                if previous is not None:
                    if previous != sample:
                        raise ValueError(f"sample_id {sample_id!r} already exists with different data")
                    continue
                existing[sample_id] = sample
                to_append.append(sample)

            if not to_append:
                return 0
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            if end:
                stream.seek(end - 1)
                if stream.read(1) != "\n":
                    raise ValueError(f"{target_path} does not end with a newline")
                stream.seek(0, os.SEEK_END)
            payload = "".join(
                json.dumps(sample, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                for sample in to_append
            )
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            return len(to_append)
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def one_hot_action(action: str) -> list[float]:
    if action not in ACTIONS:
        raise ValueError(f"unsupported target action: {action!r}")
    return [1.0 if candidate == action else 0.0 for candidate in ACTIONS]


def selfplay_target_distribution(
    action_probabilities: Sequence[float],
    action: str,
    reward_signal: float,
    *,
    step_size: float = DEFAULT_SELFPLAY_STEP_SIZE,
) -> list[float]:
    """Move the recorded policy distribution slightly toward/away from its sampled action."""
    if action not in ACTIONS:
        raise ValueError(f"unsupported self-play action: {action!r}")
    if len(action_probabilities) != len(ACTIONS):
        raise ValueError(f"action distribution must have {len(ACTIONS)} entries")
    probabilities = [_finite(value, math.nan) for value in action_probabilities]
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError("action probabilities must be finite and non-negative")
    total = sum(probabilities)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("action probabilities must have positive mass")
    probabilities = [value / total for value in probabilities]
    action_index = ACTIONS.index(action)
    if probabilities[action_index] <= 0.0:
        raise ValueError("sampled action has zero probability under its recorded mask")
    reward_signal = _finite(reward_signal, math.nan)
    step_size = _finite(step_size, math.nan)
    if not math.isfinite(reward_signal) or not -1.0 <= reward_signal <= 1.0:
        raise ValueError("normalized reward signal must be between -1 and 1")
    if not math.isfinite(step_size) or not 0.0 <= step_size <= 1.0:
        raise ValueError("self-play target step size must be between 0 and 1")

    target = [
        max(0.0, probability + step_size * reward_signal *
            ((1.0 if index == action_index else 0.0) - probability))
        for index, probability in enumerate(probabilities)
    ]
    target_total = sum(target)
    if not math.isfinite(target_total) or target_total <= 0.0:
        raise ValueError("self-play update produced an empty target distribution")
    return [value / target_total for value in target]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default
