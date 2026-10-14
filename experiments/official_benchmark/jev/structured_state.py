"""Causal, observation-only decision-oriented state (not a lossless log codec)."""
import copy
import json


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)


def measurement(event):
    """Preserve the actual tool measurement, including every numeric data field."""
    result = event['result']
    action = result.get('info', {}).get('action_result')
    if action is not None:
        return copy.deepcopy(action)
    # list_actions is supplied as candidates at selection time, not repeated here.
    return copy.deepcopy({k: v for k, v in result.items()
                          if k not in ('observation', 'reward', 'info', 'valid_actions')})


class StructuredState:
    def __init__(self):
        self.consumed = 0
        self.step = 0
        self.topology = {'matrix_index_adjacency': {}, 'known_devices': [],
                         'semantics': 'Matrix indices are not device IDs; absent cells mean unknown (0). Values: 1 connected, -1 disconnected, 2 degraded.',
                         'changes': []}
        self.devices = {}
        self.memory = {}
        self.latest = None
        self.notices = []

    def update(self, event):
        self.consumed += 1
        if 'result' not in event:
            self.notices.append(copy.deepcopy(event))
            return
        raw = event['result']
        info = raw.get('info', {})
        self.step = info.get('step_count', self.step)
        value = measurement(event)
        stamp = value.get('timestamp')
        point = {'event': self.consumed, 'step': self.step, 'timestamp': stamp}
        self.latest = dict(tool=event['tool'], arguments=copy.deepcopy(event['arguments']),
                           observation=value, **point)
        # Equality compares every measurement field except the observation clock.
        semantic = {k: v for k, v in value.items() if k != 'timestamp'}
        key = dumps([event['tool'], event['arguments']])
        record = self.memory.setdefault(key, {'tool': event['tool'],
            'arguments': copy.deepcopy(event['arguments']), 'count': 0,
            'first_step': self.step, 'last_step': self.step, 'last_timestamp': stamp,
            'result_changed': False, 'measurements': []})
        record['count'] += 1
        record['last_step'], record['last_timestamp'] = self.step, stamp
        versions = record['measurements']
        if not versions or versions[-1]['result'] != semantic:
            record['result_changed'] = bool(versions)
            versions.append({'result': semantic, 'observed_at': []})
        versions[-1]['observed_at'].append(point)
        payload = value.get('result', {})
        data = payload.get('data', {})
        if event['tool'] == 'check_status':
            device = event['arguments'].get('device')
            self.devices[device] = {'last_checked': self.step, 'timestamp': stamp,
                                    'result': copy.deepcopy(payload)}
        discovered = data.get('discovered_devices', [])
        self.topology['known_devices'] = sorted(set(self.topology['known_devices']) | set(discovered))
        matrix = raw.get('observation', {}).get('discovery_matrix')
        if matrix is not None:
            adjacency = {str(i): {str(j): v for j, v in enumerate(row) if v != 0}
                         for i, row in enumerate(matrix) if any(row)}
            previous = self.topology['matrix_index_adjacency']
            if previous != adjacency:
                changes = []
                for i in sorted(set(previous) | set(adjacency)):
                    for j in sorted(set(previous.get(i, {})) | set(adjacency.get(i, {}))):
                        old, new = previous.get(i, {}).get(j, 0), adjacency.get(i, {}).get(j, 0)
                        if old != new:
                            changes.append([i, j, old, new])
                self.topology['changes'].append(dict(point, cells=changes))
            self.topology['matrix_index_adjacency'] = adjacency
            self.topology['matrix_shape'] = [len(matrix), len(matrix[0]) if matrix else 0]

    def render(self, raw):
        assert len(raw['history']) >= self.consumed
        for event in raw['history'][self.consumed:]:
            self.update(event)
        return copy.deepcopy({
            'task': {'objective': raw['objective'], 'hint': raw.get('hint'),
                     'network_size': raw['network_size'], 'step': self.step,
                     'remaining_steps': raw['max_steps'] - self.step,
                     'solver_turn': raw.get('solver_turn', 0),
                     'remaining_turns': 100 - raw.get('solver_turn', 0)},
            'topology': self.topology, 'device_state': self.devices,
            'diagnostic_evidence_and_action_memory': list(self.memory.values()),
            'latest_observation': self.latest, 'adapter_notices': self.notices,
            'state_semantics': 'Observed facts only. Measurement versions preserve all data, successes, failures and observation times. Unchecked device status is unknown. No fault inference. Repeated actions remain allowed.'})
