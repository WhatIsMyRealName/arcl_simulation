"""NDJSON bridge between world snapshots, policy inference and typed ROS clients."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .action_executor import ActionExecutor
from .actions import pose_available
from .policy import DEFAULT_CHECKPOINT_PATH, AgentPolicy
from ..records import EpisodeRecords
from ..rl.environment import winner_from_world, winner_targets


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def _match_finished(world: dict[str, Any]) -> bool:
    match = world.get("match") if isinstance(world.get("match"), dict) else {}
    mission = world.get("mission") if isinstance(world.get("mission"), dict) else {}
    remaining = mission.get("remaining_s")
    return match.get("finished") is True or world.get("match_finished") is True or (
        isinstance(remaining, (int, float)) and not isinstance(remaining, bool)
        and math.isfinite(remaining) and remaining <= 0
    )


def _snapshot_age_s(world: dict[str, Any], now: float) -> float:
    """Age of a locally produced snapshot, including time queued in the pipe."""
    produced_at = world.get("observer_monotonic_s")
    if produced_at is None:
        return 0.0
    if (isinstance(produced_at, bool) or not isinstance(produced_at, (int, float))
            or not math.isfinite(produced_at)):
        return math.inf
    return max(0.0, now - float(produced_at))


class SnapshotSafetyGate:
    """Lock decisions across a stale stream until observations and operation state recover."""

    def __init__(self, timeout_s: float, pose_freshness_s: float, *, started_at: float) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("snapshot timeout must be finite and positive")
        self.timeout_s = float(timeout_s)
        self.pose_freshness_s = float(pose_freshness_s)
        self.last_fresh_at = float(started_at)
        self.locked = False

    def poll(self, now: float) -> bool:
        """Return true once when a silent stream first crosses its timeout."""
        if not self.locked and now - self.last_fresh_at >= self.timeout_s:
            self.locked = True
            return True
        return False

    def observe(
        self, world: dict[str, Any], now: float, *, operation_terminal: bool,
    ) -> tuple[bool, bool, bool]:
        """Return (may_decide, recovered_from_lock, newly_locked) for a snapshot."""
        if _snapshot_age_s(world, now) > self.timeout_s:
            newly_locked = not self.locked
            self.locked = True
            return False, False, newly_locked
        self.last_fresh_at = now
        if self.locked:
            if (not operation_terminal
                    or not pose_available(world, self.pose_freshness_s)):
                return False, False, False
            self.locked = False
            return True, True, False
        return True, False, False


def _outcome(world: dict[str, Any], robot_id: str, opponent_id: str | None) -> tuple[str | None, dict | None]:
    match = world.get("match") if isinstance(world.get("match"), dict) else {}
    scoreboard = world.get("scoreboard") if isinstance(world.get("scoreboard"), dict) else {}
    if opponent_id == robot_id:
        raise ValueError("opponent ID must differ from the controlled robot ID")
    if opponent_id is None:
        explicit = match.get("winner_id", world.get("winner_id"))
        opponent_id = explicit if isinstance(explicit, str) and explicit not in (robot_id, "draw", "tie") else "opponent"
        for scores in (match.get("scores"), scoreboard.get("scores"), world.get("scores")):
            if isinstance(scores, dict) and robot_id in scores and len(scores) == 2:
                opponent_id = next(key for key in scores if key != robot_id)
                break
    winner, known = winner_from_world(world, (robot_id, opponent_id))
    targets = winner_targets(winner, (robot_id, opponent_id), result_known=known)
    return winner, None if targets is None else {robot_id: targets[robot_id]}


async def _run(args: argparse.Namespace) -> int:
    policy, created = AgentPolicy.from_checkpoint(
        args.model, device=args.device, create_if_missing=True,
        velocity_scale_m_s=args.velocity_scale,
        robot_margin_m=args.robot_margin, pose_freshness_s=args.pose_freshness,
    )
    if created:
        print(
            f"WARNING: checkpoint was absent; created randomly initialized policy at {args.model}. "
            "This policy has not been trained.",
            file=sys.stderr, flush=True,
        )

    executor = ActionExecutor(
        command_timeout_s=args.command_timeout,
        navigation_timeout_s=args.navigation_timeout,
        pose_freshness_s=policy.pose_freshness_s,
        robot_margin_m=policy.robot_margin_m,
    )
    last_sequence: int | None = None
    last_timestamp: float | None = None
    records: EpisodeRecords | None = None
    robot_id = "robot_omni"
    last_world: dict[str, Any] = {}
    complete = False
    loop = asyncio.get_running_loop()
    input_reader = asyncio.StreamReader(limit=args.max_snapshot_bytes)
    input_protocol = asyncio.StreamReaderProtocol(input_reader)
    input_transport, _ = await loop.connect_read_pipe(lambda: input_protocol, sys.stdin.buffer)
    safety_gate = SnapshotSafetyGate(
        args.snapshot_timeout, policy.pose_freshness_s, started_at=time.monotonic()
    )
    read_task = asyncio.create_task(input_reader.readline())

    try:
        while True:
            done, _ = await asyncio.wait((read_task,), timeout=0.1)
            if not done:
                if last_world:
                    await executor.update(last_world)
                now = time.monotonic()
                if safety_gate.poll(now):
                    print(
                        f"WARNING: snapshot stream stale for at least {args.snapshot_timeout:g}s; "
                        "pausing decisions and canceling active MOVE.",
                        file=sys.stderr, flush=True,
                    )
                    executor.request_cancel_active_move("snapshot stream became stale")
                continue
            try:
                raw_line = read_task.result()
            except ValueError as error:
                raise ValueError(
                    f"snapshot exceeds --max-snapshot-bytes ({args.max_snapshot_bytes})"
                ) from error
            read_task = asyncio.create_task(input_reader.readline())
            if raw_line == b"":
                break
            line = raw_line.decode("utf-8")
            if not line.strip():
                continue
            try:
                world = json.loads(line)
                if not isinstance(world, dict):
                    raise ValueError("each input line must be a JSON object")
                sequence = int(world.get("sequence", -1))
                if last_sequence is not None and sequence <= last_sequence:
                    raise ValueError("world sequence must increase for every snapshot")
                timestamp = float(world.get("timestamp", math.nan))
                if not math.isfinite(timestamp):
                    raise ValueError("world timestamp must be finite seconds")
                if last_timestamp is not None and timestamp < last_timestamp:
                    raise ValueError("world timestamp must not move backwards")
                received_at = time.monotonic()
                last_sequence = sequence
                last_timestamp = timestamp
                last_world = copy.deepcopy(world)
                await executor.update(world)
                may_decide, recovered, newly_locked = safety_gate.observe(
                    world, received_at, operation_terminal=executor.active is None
                )
                if not may_decide:
                    if newly_locked:
                        print(
                            "WARNING: received a queued snapshot older than the stream timeout; "
                            "pausing decisions and canceling active MOVE.",
                            file=sys.stderr, flush=True,
                        )
                    executor.request_cancel_active_move("received snapshot older than stream timeout")
                    continue
                if recovered:
                    print("INFO: fresh snapshot and terminal operation status received; resuming agent.",
                          file=sys.stderr, flush=True)
                if records is None:
                    robot_id = str((world.get("robot") or {}).get("id", "robot_omni"))
                    records = EpisodeRecords(None if args.no_records else args.records, str(uuid.uuid4()), (robot_id,))
                    records.add({"kind": "reset", "observation": world})
                if _match_finished(world):
                    complete = True
                    break
                world = copy.deepcopy(world)
                local_operation = executor.observation()
                if local_operation is not None:
                    world["active_operation"] = local_operation
                if executor.last_result is not None:
                    world["last_action_result"] = executor.last_result

                decision = policy.decide(world, deterministic=not args.sample_actions)
                execution = {"status": "dry_run"} if args.dry_run else await executor.dispatch(decision, world)
                records.add({"kind": "decision", "agent_observations": {robot_id: world},
                             "actions": {robot_id: decision}, "responses": {robot_id: execution},
                             "active_operation": executor.observation(), "last_action_result": executor.last_result})
                _emit({
                    "world_sequence": world.get("sequence"),
                    "decision": decision,
                    "execution": execution,
                    "active_operation": executor.observation(),
                    "last_action_result": executor.last_result,
                })
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                _emit({"error": str(error)})
    finally:
        if not read_task.done():
            read_task.cancel()
            try:
                await read_task
            except asyncio.CancelledError:
                pass
        input_transport.close()
        await executor.close()
        if records is not None:
            winner, targets = _outcome(last_world, robot_id, args.opponent_id) if complete else (None, None)
            records.add({"kind": "runner_end", "final_observation": last_world,
                         "active_operation": executor.observation(), "last_action_result": executor.last_result})
            records.finish(complete=complete, winner_id=winner, winner_targets=targets,
                           rewards_complete=False, reason="match finished" if complete else "input interrupted or ended before match result")
            _emit({"kind": "episode_end", "episode_id": records.episode_id, "complete": complete,
                   "winner_targets": targets, "records_path": str(records.path) if records.path else None})
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the world policy over newline-delimited JSON snapshots."
    )
    parser.add_argument(
        "--model", "--weights", dest="model", type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="checkpoint to load or create with random weights (default: %(default)s)",
    )
    parser.add_argument("--device", default="cpu", help="torch device, default: cpu")
    parser.add_argument("--command-timeout", type=float, default=35.0)
    parser.add_argument("--navigation-timeout", type=float, default=90.0)
    parser.add_argument("--pose-freshness", type=float,
                        help="override checkpoint pose freshness; default: checkpoint or 0.75 s")
    parser.add_argument("--robot-margin", type=float,
                        help="map-boundary clearance for the robot center, in meters")
    parser.add_argument("--velocity-scale", type=float, default=1.0,
                        help="linear velocity feature scale in meters per second")
    parser.add_argument("--records", type=Path, default=Path("runs/agent"), help="episode directory or JSONL file")
    parser.add_argument("--no-records", action="store_true")
    parser.add_argument("--opponent-id", help="opponent ID for final score comparison, otherwise inferred")
    parser.add_argument("--sample-actions", action="store_true", help="sample actions/targets instead of argmax")
    parser.add_argument("--dry-run", action="store_true", help="record decisions without issuing ROS commands")
    parser.add_argument("--max-snapshot-bytes", type=int, default=4 * 1024 * 1024,
                        help="maximum bytes per NDJSON snapshot, excluding newline (default: %(default)s)")
    parser.add_argument("--snapshot-timeout", type=float, default=1.0,
                        help="lock decisions after this long without a fresh snapshot (default: %(default)s s)")
    args = parser.parse_args()
    if args.max_snapshot_bytes < 1:
        parser.error("--max-snapshot-bytes must be positive")
    if (args.snapshot_timeout <= 0
            or args.robot_margin is not None and (not math.isfinite(args.robot_margin) or args.robot_margin < 0)
            or args.pose_freshness is not None and (not math.isfinite(args.pose_freshness) or args.pose_freshness <= 0)):
        parser.error("snapshot timeout and pose freshness must be positive; robot margin must be non-negative")
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"agent: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
