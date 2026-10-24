"""Check explicitly enabled server telemetry without changing robot parameters."""
import time


def require_command_events(node, remote_nodes, timeout_s=8.0):
    import rclpy
    from rcl_interfaces.msg import ParameterType
    from rcl_interfaces.srv import GetParameters

    deadline = time.monotonic() + timeout_s
    for remote in remote_nodes:
        client = node.create_client(GetParameters, f'{remote.rstrip("/")}/get_parameters')
        instruction = (f'{remote}: publish_command_events must be enabled at startup. '
                       'Restart the simulation with command_events:=true, or start this server '
                       'with --ros-args -p publish_command_events:=true.')
        try:
            if not client.wait_for_service(timeout_sec=max(0., deadline - time.monotonic())):
                raise RuntimeError(f'{instruction} Parameter service unavailable.')
            request = GetParameters.Request()
            request.names = ['publish_command_events']
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=max(0., deadline - time.monotonic()))
            if not future.done():
                raise RuntimeError(f'{instruction} Parameter query timed out.')
            response = future.result()
            values = response.values if response is not None else []
            if (len(values) != 1 or values[0].type != ParameterType.PARAMETER_BOOL
                    or not values[0].bool_value):
                raise RuntimeError(instruction)
        finally:
            node.destroy_client(client)
