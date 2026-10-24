"""Asynchronous ROS action/service client for the hybrid robot policy."""

from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid
from typing import Any

from .actions import ACTIONS, operation_allows, validate_move_target, wrist_available, validate_speed


class ActionExecutor:
    """Run one move action or actuator service at a time using typed ROS APIs."""

    def __init__(self, *, navigation_timeout_s: float = 90.0,
                 command_timeout_s: float = 35.0, robot_margin_m: float = 0.20,
                 pose_freshness_s: float = 0.75) -> None:
        self.navigation_timeout_s = float(navigation_timeout_s)
        self.command_timeout_s = float(command_timeout_s)
        self.robot_margin_m = float(robot_margin_m)
        self.pose_freshness_s = float(pose_freshness_s)
        if (not math.isfinite(self.navigation_timeout_s) or self.navigation_timeout_s <= 0
                or not math.isfinite(self.command_timeout_s) or self.command_timeout_s <= 0
                or not math.isfinite(self.robot_margin_m) or self.robot_margin_m < 0
                or not math.isfinite(self.pose_freshness_s) or self.pose_freshness_s <= 0):
            raise ValueError("timeouts and pose freshness must be positive; robot margin must be non-negative")
        self.active: dict[str, Any] | None = None
        self.last_result: dict[str, Any] | None = None
        self._rclpy: Any = None
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None
        self._owns_rclpy = False
        self._move_client: Any = None
        self._speed_pub: Any = None
        self._speed_timer: Any = None
        self._speed_until = 0.0
        self._speed_lock = threading.RLock()
        self.speed_linear_limit = 0.25
        self.speed_angular_limit = 1.0
        self._suction_clients: dict[int, Any] = {}
        self._suction_all_client: Any = None
        self._wrist_client: Any = None
        self._last_world: dict[str, Any] | None = None

    def start(self) -> None:
        if self._node is not None:
            return
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.action import ActionClient
        from robot_msgs.action import GoToPose
        from robot_msgs.msg import RobotSpeed
        from std_srvs.srv import SetBool

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        self._node = Node("arcl_agent_executor")
        self._speed_pub = self._node.create_publisher(RobotSpeed, "/speed_command", 10)
        self._speed_timer = self._node.create_timer(0.01, self._expire_speed)
        self._move_client = ActionClient(self._node, GoToPose, "/go_to_pose")
        self._suction_clients = {
            index: self._node.create_client(SetBool, f"/robot_omni/ventouse_{index}/set")
            for index in range(1, 4)
        }
        self._suction_all_client = self._node.create_client(SetBool, "/robot_omni/ventouse/set")
        self._wrist_client = self._node.create_client(SetBool, "/robot_omni/poignet/set")
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def observation(self) -> dict[str, Any] | None:
        if self.active is None:
            return None
        started_at = self.active.get("started_at", time.monotonic())
        return {
            key: self.active.get(key) for key in (
                "id", "type", "target_pose", "status", "cancelable", "reason",
                "started_sequence", "failed",
            )
        } | {
            "elapsed_s": max(0.0, time.monotonic() - started_at),
            "timeout_s": self.navigation_timeout_s if self.active["type"] == "MOVE" else self.command_timeout_s,
        }

    async def update(self, world: dict[str, Any]) -> None:
        self._last_world = world
        operation = self.active
        if operation is None:
            return
        if operation["type"] == "MOVE":
            self._update_move(operation, world)
            if self.active is None:
                return
        terminal_future = operation.get("terminal_future")
        if operation["status"] == "unknown" and terminal_future is not None and terminal_future.done():
            try:
                response = terminal_future.result()
            except Exception as error:
                operation["reason"] = f"late remote result remains uncertain: {error}"
                return
            status = "succeeded" if response.success else "failed"
            reason = response.message or ("service succeeded" if response.success else "service reported failure")
            message = response.message
            self._finish(status, reason, world, message)
            return
        task = operation.get("task")
        if task is None or not task.done():
            return
        try:
            result = task.result()
        except Exception as error:
            result = {"status": "unknown", "reason": f"ROS operation outcome uncertain: {error}"}
        if result["status"] == "unknown":
            operation["status"] = "unknown"
            operation["cancelable"] = False
            operation["reason"] = result["reason"]
            return
        if result["status"] == "running":
            operation["status"] = "running"
            operation["reason"] = result["reason"]
            return
        self._finish(result["status"], result["reason"], world, result.get("message", ""))

    async def dispatch(self, decision: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
        action = decision.get("action")
        if action == "IDLE":
            return {"status": "waiting", "active_operation": self.observation()}
        if action not in ACTIONS:
            return {"status": "rejected", "reason": f"unsupported action: {action!r}"}
        if int(decision.get("world_sequence", -1)) != int(world.get("sequence", -2)):
            return {"status": "rejected", "reason": "decision is based on a stale world sequence"}
        if action == "CANCEL":
            if self.active is None or self.active.get("type") != "MOVE" or not self.active.get("cancelable"):
                return {"status": "rejected", "reason": "there is no cancelable move"}
            if self.active["status"] in ("cancel_requested", "canceling", "unknown"):
                return {"status": "waiting", "active_operation": self.observation()}
            self.active["cancel_requested"] = True
            self.active["status"] = "cancel_requested"
            self.active["reason"] = "waiting for /go_to_pose cancellation and terminal result"
            self._request_move_cancel(self.active)
            return {"status": "cancel_requested", "operation": self.observation()}
        if self.active is not None:
            return {"status": "rejected", "reason": f"{self.active['type']} is still {self.active['status']}",
                    "active_operation": self.observation()}
        if "active_operation" not in world:
            return {"status": "rejected", "reason": "active-operation state is unknown"}
        external = world.get("active_operation")
        if external is not None and not operation_allows(action, {"active_operation": external}):
            return {"status": "rejected", "reason": "another controller reports an active operation",
                    "active_operation": external}
        try:
            self._validate_action(action, decision, world)
        except (TypeError, ValueError, KeyError) as error:
            return {"status": "rejected", "reason": str(error)}
        if self._node is None:
            try:
                self.start()
            except (ImportError, RuntimeError) as error:
                return {"status": "failed", "reason": f"ROS client unavailable: {error}"}
        if action == "SPEED":
            from robot_msgs.msg import RobotSpeed
            values = validate_speed(world, decision.get("target_speed"), self.speed_linear_limit, self.speed_angular_limit)
            msg = RobotSpeed()
            msg.vx, msg.vy, msg.omega = (values[k] for k in ("vx", "vy", "omega"))
            with self._speed_lock:
                self._speed_until = time.monotonic() + 0.1
                self._speed_pub.publish(msg)
            return {"status": "running", "duration_s": 0.1, "target_speed": values}
        self.stop_speed()
        operation_id = str(uuid.uuid4())
        is_move = action == "MOVE"
        operation = {
            "id": operation_id, "type": action,
            "status": "issuing", "cancelable": is_move, "failed": False,
            "started_at": time.monotonic(), "started_sequence": int(world["sequence"]),
            "target_pose": decision.get("target_pose") if is_move else None,
            "reason": "ROS request is being sent",
        }
        self.active = operation
        if is_move:
            from robot_msgs.action import GoToPose

            if not self._move_client.server_is_ready():
                self._finish("failed", "/go_to_pose action server is unavailable", world)
                return {"status": "failed", "reason": self.last_result["reason"]}
            target = decision["target_pose"]
            goal = GoToPose.Goal()
            goal.x, goal.y = float(target["x"]), float(target["y"])
            goal.theta = math.atan2(target["sin"], target["cos"])
            try:
                operation["goal_future"] = self._move_client.send_goal_async(goal)
            except Exception as error:
                self._finish("failed", f"could not send MOVE goal: {error}", world)
                return {"status": "failed", "reason": str(error)}
            return {"status": "issuing", "operation": self.observation()}
        operation["task"] = asyncio.create_task(self._run_service(operation, world))
        return {"status": "issuing", "operation": self.observation()}

    def _validate_action(self, action: str, decision: dict[str, Any], world: dict[str, Any]) -> None:
        if action not in ACTIONS:
            raise ValueError(f"unsupported action: {action!r}")
        if action == "MOVE":
            validate_move_target(
                world, decision.get("target_pose"), robot_margin_m=self.robot_margin_m,
                pose_freshness_s=self.pose_freshness_s,
            )
        elif action == "SPEED":
            validate_speed(world, decision.get("target_speed"), self.speed_linear_limit, self.speed_angular_limit)
        elif action == "TOGGLE_WRIST":
            if not wrist_available(world):
                raise ValueError("TOGGLE_WRIST requires a fresh measured wrist angle")
        elif action in ("IDLE", "CANCEL"):
            raise ValueError(f"action {action} cannot start an operation")

    def stop_speed(self) -> None:
        with self._speed_lock:
            if self._speed_pub is not None and self._speed_until:
                from robot_msgs.msg import RobotSpeed
                self._speed_pub.publish(RobotSpeed())
            self._speed_until = 0.0

    def _expire_speed(self) -> None:
        with self._speed_lock:
            if self._speed_until and time.monotonic() >= self._speed_until:
                self.stop_speed()

    def request_cancel_active_move(self, reason: str) -> bool:
        """Request cancellation while keeping the operation locked until its terminal result."""
        self.stop_speed()
        operation = self.active
        if operation is None or operation.get("type") != "MOVE":
            return False
        operation["cancel_requested"] = True
        operation["reason"] = reason
        self._request_move_cancel(operation)
        return True

    def _request_move_cancel(self, operation: dict[str, Any]) -> None:
        handle = operation.get("goal_handle")
        if handle is not None and "cancel_future" not in operation:
            try:
                operation["cancel_future"] = handle.cancel_goal_async()
            except Exception as error:
                operation["reason"] = f"could not request cancellation: {error}"
                return
            operation["status"] = "canceling"
            operation["cancelable"] = False

    def _update_move(self, operation: dict[str, Any], world: dict[str, Any]) -> None:
        try:
            future = operation.get("goal_future")
            if future is not None and future.done() and "goal_handle" not in operation:
                handle = future.result()
                if not handle.accepted:
                    self._finish("rejected", "/go_to_pose goal rejected", world)
                    return
                operation["goal_handle"] = handle
                operation["result_future"] = handle.get_result_async()
                if not operation.get("cancel_requested"):
                    operation["status"] = "running"
                    operation["reason"] = "/go_to_pose goal accepted; awaiting terminal result"
            if operation.get("cancel_requested"):
                self._request_move_cancel(operation)
            cancel_future = operation.get("cancel_future")
            if cancel_future is not None and cancel_future.done() and not operation.get("cancel_checked"):
                operation["cancel_checked"] = True
                if not cancel_future.result().goals_canceling:
                    operation["reason"] = "cancellation rejected; awaiting original goal result"
                    operation["status"] = "running"
                    operation["cancelable"] = True
                    operation["cancel_requested"] = False
                    operation.pop("cancel_future")
                    operation.pop("cancel_checked")
            result_future = operation.get("result_future")
            if result_future is not None and result_future.done():
                response = result_future.result()
                # action_msgs/GoalStatus: SUCCEEDED=4, CANCELED=5, ABORTED=6.
                status = ("canceled" if response.status == 5 else
                          "succeeded" if response.status == 4 and response.result.success else "failed")
                self._finish(status, response.result.message or status, world)
                return
            if time.monotonic() - operation["started_at"] > self.navigation_timeout_s:
                operation["cancel_requested"] = True
                self._request_move_cancel(operation)
                operation["status"] = "unknown"
                operation["reason"] = "navigation timed out; cancellation requested; awaiting terminal result"
        except Exception as error:
            operation["status"] = "unknown"
            operation["reason"] = f"MOVE outcome uncertain: {error}"

    async def _run_service(self, operation: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
        from std_srvs.srv import SetBool

        action = operation["type"]
        if action in ("SUCTION_ALL_ON", "SUCTION_ALL_OFF"):
            client = self._suction_all_client
            request = SetBool.Request()
            request.data = action.endswith("_ON")
        elif action.startswith("SUCTION_"):
            index, state = action.removeprefix("SUCTION_").split("_")
            client = self._suction_clients[int(index)]
            request = SetBool.Request()
            request.data = state == "ON"
        else:
            client = self._wrist_client
            current = _finite(world.get("robot", {}).get("wrist_angle_rad"), math.nan)
            request = SetBool.Request()
            request.data = current < math.pi / 4.0
        if not client.service_is_ready():
            return {"status": "failed", "reason": f"service for {action} is unavailable"}
        try:
            operation["terminal_future"] = client.call_async(request)
            response = await _wait_ros_future(operation["terminal_future"], self.command_timeout_s)
        except asyncio.TimeoutError:
            return {"status": "unknown", "reason": f"{action} service timed out; physical outcome is uncertain"}
        return {"status": "succeeded" if response.success else "failed",
                "reason": response.message or ("service succeeded" if response.success else "service reported failure"),
                "message": response.message}

    def _finish(self, status: str, reason: str, world: dict[str, Any], message: str = "") -> None:
        operation = self.active
        if operation is None:
            return
        self.last_result = {
            "id": operation["id"], "type": operation["type"], "status": status,
            "failed": status in ("failed", "rejected"), "reason": reason,
            "cancelable": operation["cancelable"], "target_pose": operation.get("target_pose"),
            "message": message, "finished_sequence": int(world.get("sequence", 0)),
        }
        self.active = None

    async def close(self) -> None:
        self.stop_speed()
        operation = self.active
        if operation is not None and operation.get("type") == "MOVE":
            self.request_cancel_active_move("runner stopped; canceling active MOVE")
            deadline = time.monotonic() + min(self.command_timeout_s, 2.0)
            while self.active is not None and time.monotonic() < deadline:
                self._update_move(operation, self._last_world or {"sequence": operation["started_sequence"]})
                if self.active is not None:
                    await asyncio.sleep(0.01)
        if self.active is not None:
            operation = self.active
            task = operation.get("task")
            if task is not None and task.done():
                try:
                    result = task.result()
                except Exception as error:
                    result = {"status": "unknown", "reason": f"ROS operation outcome uncertain: {error}"}
                if result["status"] != "unknown":
                    self._finish(result["status"], result["reason"], {"sequence": operation["started_sequence"]}, result.get("message", ""))
            if self.active is not None:
                self.active["status"] = "unknown"
                self.active["reason"] = "runner stopped; remote operation may still be active"
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=1.0)
            self._executor = None
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
            self._spin_thread = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_rclpy and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()


async def _wait_ros_future(future: Any, timeout_s: float) -> Any:
    started = time.monotonic()
    while not future.done():
        if time.monotonic() - started > timeout_s:
            raise asyncio.TimeoutError
        await asyncio.sleep(0.01)
    return future.result()


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default
