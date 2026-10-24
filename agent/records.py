"""Finalize an entire episode before publishing its JSONL training records."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np



def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


class EpisodeRecords:
    def __init__(self, destination: str | Path | None, episode_id: str,
                 robot_ids: tuple[str, ...], *, gamma: float = 0.99) -> None:
        self.destination = Path(destination) if destination is not None else None
        self.episode_id = episode_id
        self.robot_ids = robot_ids
        self.gamma = gamma
        self.rows: list[dict[str, Any]] = []
        self.finished = False
        self.path: Path | None = None

    def add(self, row: dict[str, Any]) -> None:
        if self.finished:
            raise RuntimeError("episode records are already finalized")
        self.rows.append(copy.deepcopy({
            **row, "episode_id": self.episode_id,
        }))

    def finish(self, *, complete: bool, winner_id: str | None,
               winner_targets: dict[str, float] | None,
               rewards_complete: bool, reason: str = "episode finished") -> Path | None:
        if self.finished:
            return self.path
        transitions = [row for row in self.rows if row.get("kind") == "transition"]
        ready = complete and rewards_complete and winner_targets is not None and bool(transitions) and all(
            isinstance(row.get("reward"), dict) for row in transitions
        )
        self.add({"kind": "episode_end", "reason": reason})
        for row in self.rows:
            row.update(complete=complete, winner_id=winner_id,
                       winner_known=winner_targets is not None, winner_targets=winner_targets,
                       rewards_complete=ready, gamma=self.gamma, robot_ids=self.robot_ids)
        # Serialize only after all labels have been propagated, including early steps.
        payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                     default=_json_default, allow_nan=False) + "\n"
                          for row in self.rows)
        # Finalization is irreversible even if publishing fails: a caller closing
        # after an I/O error must not append a second end marker or erase labels.
        self.finished = True
        if self.destination is not None:
            if self.destination.suffix == ".jsonl":
                target = self.destination
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8") as stream:
                    stream.write(payload)
                self.path = target
            else:
                self.destination.mkdir(parents=True, exist_ok=True)
                target = self.destination / f"{self.episode_id}.jsonl"
                # A directory reader never sees an unfinished episode file.
                temporary: str | None = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                                     dir=self.destination, delete=False) as stream:
                        temporary = stream.name
                        stream.write(payload)
                    os.replace(temporary, target)
                    self.path = target
                finally:
                    if temporary is not None and os.path.exists(temporary):
                        os.unlink(temporary)
        return self.path
