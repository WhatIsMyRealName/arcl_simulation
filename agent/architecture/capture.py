"""Incremental source-independent sessions and causal supervised preparation."""
from __future__ import annotations
import bisect
import copy
import json
import math
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from .actions import ACTIONS, action_mask, validate_speed
from .demonstrations import operator_choice_to_decision
from .targets import TARGET_DATASET_FORMAT, one_hot_action, append_target_samples

SESSION_FORMAT = 'arcl-ros-session'


def operation_event(event, active, last):
    command, status = event.get('command'), event.get('status')
    if command == 'cancel' and event.get('status') == 'accepted':
        if active and active.get('type') == 'MOVE':
            active = {**active, 'status': 'canceling', 'cancelable': False}
        return active, last
    if command not in ('move', 'suction', 'wrist', 'speed'):
        return active, last
    parameters = event.get('parameters', {})
    action = ('MOVE' if command == 'move' else 'SPEED' if command == 'speed' else
              'TOGGLE_WRIST' if command == 'wrist' else
              f"SUCTION_{str(parameters.get('group', 'all')).upper()}_{'ON' if parameters.get('enabled') else 'OFF'}")
    operation = {'id': event.get('id'), 'type': action, 'status': status,
                 'cancelable': action == 'MOVE', 'failed': status in ('failed', 'rejected')}
    if command == 'move' and all(k in parameters for k in ('x', 'y', 'theta')):
        operation['target_pose'] = {'x': parameters['x'], 'y': parameters['y'],
                                   'sin': math.sin(parameters['theta']), 'cos': math.cos(parameters['theta'])}
    if status == 'accepted':
        operation['status'] = 'running'
        return (operation, last) if command != 'speed' else (active, last)
    if status in ('succeeded', 'failed', 'canceled', 'rejected', 'expired'):
        if active and active.get('id') == event.get('id'):
            operation['type'] = active['type']
            operation['target_pose'] = active.get('target_pose')
            active = None
        return active, operation
    return active, last


class SessionWriter:
    def __init__(self, directory, configuration):
        self.path = Path(directory).resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / 'images').mkdir()
        self.manifest = {'format': SESSION_FORMAT, 'session_id': str(uuid.uuid4()),
                         'complete': False, 'configuration': configuration, 'started_unix_s': time.time()}
        self._manifest()
        self.streams = {name: (self.path / f'{name}.jsonl').open('a', buffering=1)
                        for name in ('events', 'observations')}

    def _manifest(self):
        temporary = self.path / 'manifest.tmp'
        temporary.write_text(json.dumps(self.manifest, indent=2, allow_nan=False) + '\n')
        temporary.replace(self.path / 'manifest.json')

    def write(self, stream, payload, received=None):
        row = {'received_monotonic_s': time.monotonic() if received is None else received,
               'payload': payload}
        self.streams[stream].write(json.dumps(row, allow_nan=False) + '\n')

    def close(self, complete=False):
        for stream in self.streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        self.manifest.update(complete=complete, ended_unix_s=time.time())
        self._manifest()


def _rows(path):
    # A killed writer may leave an unfinished last line; preserve every complete row.
    content = Path(path).read_text()
    lines = content.splitlines()
    for index, line in enumerate(lines):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1 or content.endswith('\n'):
                raise ValueError(f'Corrupt record {path}:{index + 1}')


def prepare_session(directory, output, *, speed_linear_limit=0.25, speed_angular_limit=1.0):
    if any(not math.isfinite(v) or v <= 0 for v in (speed_linear_limit, speed_angular_limit)):
        raise ValueError('speed limits must be finite and positive')
    directory, output = Path(directory).resolve(), Path(output).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('format') != SESSION_FORMAT:
        raise ValueError('unsupported session format')
    observations = sorted(_rows(directory / 'observations.jsonl'),
                          key=lambda r: r['payload']['observer_monotonic_s'])
    times = [r['payload']['observer_monotonic_s'] for r in observations]
    events = sorted(_rows(directory / 'events.jsonl'), key=lambda r: r['received_monotonic_s'])
    results = {r['payload'].get('id'): r['payload'] for r in events
               if r['payload'].get('status') in ('succeeded', 'failed', 'canceled')}
    counts = Counter()
    samples = []
    active = last = None
    # Keep the last executed velocity in each 100 ms window, including terminal stops.
    speed_windows = {}
    origin = times[0] if times else 0.
    for row in events:
        event = row['payload']
        if event.get('command') == 'speed' and event.get('status') == 'accepted':
            bucket = math.floor((row['received_monotonic_s'] - origin + 1e-8) / 0.1)
            speed_windows[bucket] = id(row)
    selected_speeds = set(speed_windows.values())
    # Wait examples use the operation state reconstructed from the same event stream.
    timeline = [(r['received_monotonic_s'], 'event', r) for r in events]
    timeline += [(r['payload']['observer_monotonic_s'], 'observation', r) for r in observations]
    for at, kind, row in sorted(timeline, key=lambda item: (item[0], item[1])):
        event = row['payload']
        if kind == 'observation':
            if not active:
                continue
            observation = copy.deepcopy(event)
            decision = {'action': 'IDLE'}
            source_id = f"idle:{event['sequence']}"
        else:
            if event.get('format') != 'arcl-command-event':
                continue  # raw requested speed is retained; labels require server acceptance
            if event.get('status') != 'accepted':
                active, last = operation_event(event, active, last)
                counts[event.get('status', 'unknown')] += 1
                continue
            index = bisect.bisect_left(times, at) - 1
            if index < 0 or at - times[index] > 0.2:
                counts['no_prior_observation'] += 1
                active, last = operation_event(event, active, last)
                continue
            observation = copy.deepcopy(observations[index]['payload'])
            source_id = event['id']
            if event['command'] == 'speed':
                if id(row) not in selected_speeds:
                    counts['speed_rate_limited'] += 1
                    continue
                decision = {'action': 'SPEED', 'target_speed': event['parameters']}
            else:
                decision = operator_choice_to_decision(event['command'], event['parameters'], observation)
        observation['active_operation'] = copy.deepcopy(active)
        observation['last_action_result'] = copy.deepcopy(last)
        image = observation.get('front_camera') or {}
        # Account for time between observation assembly and the demonstrated command.
        if image.get('path'):
            path = Path(image['path'])
            if not path.is_absolute():
                path = directory / path
            image['path'] = str(path.resolve())
            image['age_s'] = float(image.get('age_s', 0)) + max(0., at - observation['observer_monotonic_s'])
        action = decision['action']
        try:
            if image.get('path') and (image.get('timestamp', -math.inf) > observation['timestamp'] + 1e-6
                                       or image.get('received_monotonic_s', -math.inf) > at):
                raise ValueError('future image')
            if action not in ACTIONS or not action_mask(observation)[ACTIONS.index(action)]:
                raise ValueError('action masked or unsupported')
            if action == 'SPEED':
                validate_speed(observation, decision['target_speed'], speed_linear_limit, speed_angular_limit)
            if image.get('path'):
                from .vision import validate_image_reference
                validate_image_reference(observation)
            if image.get('path'):
                image['path'] = os.path.relpath(image['path'], output.parent)
            samples.append({'format': TARGET_DATASET_FORMAT, 'source': 'ros',
                'group_id': manifest['session_id'], 'sample_id': f"{manifest['session_id']}:{source_id}",
                'observation': observation, 'chosen_action': action,
                'target_action_probs': one_hot_action(action),
                'move_target_valid': action == 'MOVE', 'target_move_pose': decision.get('target_pose'),
                'speed_target_valid': action == 'SPEED', 'target_speed': decision.get('target_speed'),
                'metadata': {'session': str(directory), 'event_id': source_id,
                             'session_complete': manifest['complete'], 'result': results.get(source_id)}})
            counts[action] += 1
        except (ValueError, OSError, KeyError) as error:
            counts[f'excluded:{error}'] += 1
        if kind == 'event':
            active, last = operation_event(event, active, last)
    count = append_target_samples(output, samples)
    output.with_suffix('.preparation.json').write_text(json.dumps({
        'session': str(directory), 'output': str(output), 'counts': counts,
        'samples_appended': count, 'configuration': manifest['configuration']}, indent=2) + '\n')
    return count
