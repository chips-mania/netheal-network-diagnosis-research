"""Frozen public diagnosis inputs shared by local LAYA and a future Jev run.

Targets are deliberately absent from every function in this module.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / 'netheal-topology-generalization/scripts'
sys.path.insert(0, str(LEGACY))
from jev_context_codec_v4 import encode_checked, dumps
from jev_tree_partition_v4 import restore

INSTRUCTIONS = ('You are an AI agent that completes tasks using available tools. '
                'Read the task and objective. Use the tool descriptions and observed results '
                'to choose the next tool or its arguments. Complete the task before the budget '
                'runs out. Return a choice; no explanation text is required.')
MAX_LEN = 4096
HEAD_LEN = 1024
BYTE_LIMIT = 24000


def digest(value):
    return hashlib.sha256(dumps(value).encode('utf-8')).hexdigest()


def question(purpose, options):
    assert 1 <= len(options) <= 255
    return {'decision': {'type': 'choice', 'instructions': INSTRUCTIONS + ' ' + purpose,
            'criteria': {f'c{i:03d}': v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                         for i, v in enumerate(options)}}}


def diagnosis_options(catalog, fault=None):
    actions = [a for a in catalog if a['category'] == 'diagnosis']
    if fault is None:
        return list(dict.fromkeys(a['action_type'] for a in actions))
    return list(dict.fromkeys(a['parameters']['location'] for a in actions if a['action_type'] == fault))


def public_audit(value):
    forbidden = {'ground_truth_fault', 'ground_truth', 'injected_fault', 'fault_info',
                 'correct_diagnosis', 'expected_solution', 'truth', 'target', 'expected', 'teacher_result'}
    if isinstance(value, dict):
        assert not set(value) & forbidden, set(value) & forbidden
        for v in value.values():
            public_audit(v)
    elif isinstance(value, list):
        for v in value:
            public_audit(v)


def prepare(tok, state, purpose, options):
    """One immutable partition plan, checked against both model limits.

    Always use the same JSON-path envelope when splitting. Weight subtrees by
    their UTF-8 JSON size. Stable argmax uses the first candidate on ties.
    """
    from laya.common import build_sequence
    from laya.train import to_internal
    public_audit(state)
    qs = question(purpose, options)
    internal = to_internal('decision', qs['decision'])
    cache = {}

    def inspect(s):
        key = digest(s)
        if key not in cache:
            text = encode_checked(s)
            byte_ok = len(text.encode()) + len(json.dumps(qs, ensure_ascii=False).encode()) + 512 <= BYTE_LIMIT
            if byte_ok:
                ids, markers, stats, cut = build_sequence(tok, text, internal, MAX_LEN, HEAD_LEN,
                    return_stats=True, return_truncation_stats=True)
                ok = not cut['truncated'] and len(markers) == stats['options_distinct'] == len(options)
                cache[key] = (ok, text, len(ids))
            else:
                cache[key] = (False, text, 0)
        return cache[key]

    original_hash = digest(state)
    common = {k: state[k] for k in ('task_description', 'objective', 'hint', 'network_size', 'max_steps', 'solver_turn') if k in state}

    def wrap(items):
        return dict(common, partition_notice='Lossless evidence partition: path is the location in the original JSON state; value is the exact subtree. All partitions together preserve all evidence. Evaluate the same options from this evidence.',
                    original_state_sha256=original_hash, evidence_fragments=items,
                    partition_weight=sum(len(dumps(x['value']).encode()) for x in items))

    if inspect(state)[0]:
        chunks = [(state, 1)]
    else:
        assert inspect(wrap([]))[0], 'Question cannot fit the shared budget'
        fragments = []

        def split(value, path):
            item = {'path': path, 'value': value}
            if inspect(wrap([item]))[0]:
                fragments.append(item)
            elif isinstance(value, dict) and value:
                for k, v in value.items():
                    split(v, path + [k])
            elif isinstance(value, list) and value:
                for i, v in enumerate(value):
                    split(v, path + [i])
            else:
                raise ValueError('Indivisible scalar exceeds shared budget')
        split(state, [])
        assert dumps(restore(fragments)) == dumps(state)
        groups, current = [], []
        for f in fragments:
            if current and not inspect(wrap(current + [f]))[0]:
                groups.append(wrap(current))
                current = []
            current.append(f)
        if current:
            groups.append(wrap(current))
        chunks = [(g, g['partition_weight']) for g in groups]
    result = []
    for chunk, weight in chunks:
        ok, text, tokens = inspect(chunk)
        assert ok and weight > 0
        result.append({'input': {'state': text, 'questions': qs}, 'weight': weight, 'tokens': tokens})
    return {'state_sha256': original_hash, 'purpose': purpose, 'options': options, 'chunks': result,
            'roundtrip_verified': True}


def combine(plan, probabilities):
    keys = list(plan['chunks'][0]['input']['questions']['decision']['criteria'])
    assert len(probabilities) == len(plan['chunks'])
    total = sum(c['weight'] for c in plan['chunks'])
    scores = {k: sum(c['weight'] * p[k] for c, p in zip(plan['chunks'], probabilities)) / total for k in keys}
    return {'answers': {'decision': {'choice': max(keys, key=scores.get), 'probabilities': scores}}}
