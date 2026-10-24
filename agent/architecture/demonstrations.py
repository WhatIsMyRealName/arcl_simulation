"""Record operator choices as directly trainable JSONL examples."""

from __future__ import annotations

import copy
import json
import math
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .targets import (
    DEFAULT_TARGET_DATASET_PATH,
    TARGET_DATASET_FORMAT,
    append_target_samples,
    one_hot_action,
)


DATASET_FORMAT = TARGET_DATASET_FORMAT


def operator_choice_to_decision(
    command: str,
    parameters: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Convert one operator-console choice to the policy's action contract."""
    if command == "move":
        x = _finite(parameters.get("x"))
        y = _finite(parameters.get("y"))
        yaw = _finite(parameters.get("theta"))
        if not all(math.isfinite(value) for value in (x, y, yaw)):
            raise ValueError("move choice requires finite x, y and theta")
        return {
            "action": "MOVE",
            "target_pose": {"x": x, "y": y, "sin": math.sin(yaw), "cos": math.cos(yaw)},
        }
    if command == "suction":
        enabled = parameters.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError("suction choice requires a boolean enabled value")
        group = parameters.get("group")
        if group == "all":
            return {"action": "SUCTION_ALL_ON" if enabled else "SUCTION_ALL_OFF"}
        if group not in (1, 2, 3):
            raise ValueError("suction group must be 'all' or an actuator number from 1 to 3")
        return {"action": f"SUCTION_{group}_{'ON' if enabled else 'OFF'}"}
    if command == "wrist":
        target = _finite(parameters.get("target_angle_rad"))
        robot = observation.get("robot") if isinstance(observation.get("robot"), dict) else {}
        current = _finite(robot.get("wrist_angle_rad"))
        if not math.isfinite(target) or not math.isfinite(current):
            return {"action": "UNSUPPORTED_WRIST_TARGET"}
        action = "IDLE" if abs(_wrap_angle(target - current)) < 0.10 else "TOGGLE_WRIST"
        return {"action": action}
    if command == "cancel":
        return {"action": "CANCEL" if parameters.get("operation_type") == "MOVE" else
                "UNSUPPORTED_ROCK_CENTERING_CANCEL"}
    if command == "idle":
        return {"action": "IDLE"}
    if command == "rock_centering":
        return {"action": "UNSUPPORTED_ROCK_CENTERING"}
    raise ValueError(f"unsupported operator command: {command}")


class DemonstrationRecorder:
    """Buffer one example per choice, then publish a trainable JSONL file."""

    def __init__(
        self,
        destination: str | Path = DEFAULT_TARGET_DATASET_PATH,
        *,
        session_id: str | None = None,
        source: str = "operator",
    ) -> None:
        self.destination = Path(destination)
        self.append_to_file = self.destination.suffix == ".jsonl"
        if self.append_to_file:
            self.destination.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.destination.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or str(uuid.uuid4())
        if not isinstance(source, str) or not source:
            raise ValueError("demonstration source must be a non-empty string")
        self.source = source
        self.path = (self.destination if self.append_to_file else
                     self.destination / f"{self.session_id}.jsonl")
        if not self.append_to_file and self.path.exists():
            raise FileExistsError(self.path)
        self._lock = threading.RLock()
        self._samples: list[dict[str, Any]] = []
        self._samples_by_id: dict[str, dict[str, Any]] = {}
        self._pending: set[str] = set()
        self._stopping = False
        self._closed = False
        self._published_samples = 0

    @property
    def published_samples(self) -> int:
        with self._lock:
            return self._published_samples

    @property
    def pending_results(self) -> int:
        with self._lock:
            return len(self._pending)

    def record_decision(
        self,
        observation: dict[str, Any],
        decision: dict[str, Any],
        *,
        command: str,
        parameters: dict[str, Any] | None = None,
        requires_result: bool = True,
    ) -> str:
        with self._lock:
            if self._stopping or self._closed:
                raise RuntimeError("demonstration recording is stopping")
            decision_id = str(uuid.uuid4())
            action = decision.get("action")
            target_move_pose = decision.get("target_pose") if action == "MOVE" else None
            sample = {
                "format": DATASET_FORMAT,
                "source": self.source,
                "group_id": self.session_id,
                "session_id": self.session_id,
                "sample_id": decision_id,
                "recorded_at": _timestamp(),
                "observation": copy.deepcopy(observation),
                "chosen_action": action,
                "target_action_probs": one_hot_action(action),
                "target_move_pose": copy.deepcopy(target_move_pose),
                "move_target_valid": target_move_pose is not None,
                "speed_target_valid": False, "target_speed": None,
                "source_command": command,
                "source_parameters": copy.deepcopy(parameters or {}),
                "result": None if requires_result else {"status": "not_applicable"},
            }
            self._samples.append(sample)
            self._samples_by_id[decision_id] = sample
            if requires_result:
                self._pending.add(decision_id)
            return decision_id

    def record_result(self, decision_id: str | None, result: dict[str, Any]) -> bool:
        if decision_id is None:
            return False
        with self._lock:
            if decision_id not in self._pending or self._closed:
                return False
            sample = self._samples_by_id[decision_id]
            sample["result"] = copy.deepcopy(result)
            sample["result_recorded_at"] = _timestamp()
            self._pending.remove(decision_id)
            if self._stopping and not self._pending:
                self._finalize_locked()
            return True

    def stop(self, *, reason: str = "operator stopped recording") -> None:
        del reason
        with self._lock:
            if self._closed:
                return
            self._stopping = True
            if not self._pending:
                self._finalize_locked()

    def abort_pending(self, *, reason: str) -> None:
        with self._lock:
            for decision_id in tuple(self._pending):
                sample = self._samples_by_id[decision_id]
                sample["result"] = {
                    "status": "unknown", "failed": None, "reason": reason,
                }
                sample["result_recorded_at"] = _timestamp()
                self._pending.remove(decision_id)
            if self._stopping and not self._pending:
                self._finalize_locked()

    def _finalize_locked(self) -> None:
        if self._closed:
            return
        if not self._samples:
            self._closed = True
            return
        if self.append_to_file:
            self._published_samples = append_target_samples(self.path, self._samples)
            self._closed = True
            return
        temporary = self.destination / f".{self.session_id}.jsonl.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                for sample in self._samples:
                    stream.write(json.dumps(
                        sample, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                    ) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            self._published_samples = len(self._samples)
            self._closed = True
        except Exception:
            temporary.unlink(missing_ok=True)
            self._closed = False
            raise


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
