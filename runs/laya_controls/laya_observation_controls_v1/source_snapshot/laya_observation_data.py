"""Fixed public-probe collection and independently reversible matched renderings."""
import sys, json, hashlib, re, time
from pathlib import Path
from collections import Counter
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.run_research_suite import make_env, step_env, FAULTS, strict_location
from src.public_observation import PublicView, public_step_info
from src.graph_obs import build_graph_obs
from src.rule_agent_v2 import RuleAgentV2, _tool_payload
from src.pointer_actions import pointer_scan, pointer_check_status, pointer_check_interfaces, pointer_ping, resolve_pointer
from src.laya_schema import questions
from src.laya_schema_compact import state_text
from src.external_policy_adapter import NODE_FEATURES, EDGE_FEATURES, GLOBAL_FEATURES

OUT = PROJECT / 'results/laya_observation_controls_v1'
SPEC = PROJECT.parent / 'docs/specs/laya_observation_controls_v1.md'
HEADER = ('Observed network only. Unchecked zero is unknown, not healthy. '
          'Status -1 down, 0 unknown, 0.5 degraded, 1 up. Latency is divided by 100 ms.')

def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def facts(g):
    groups = [('Network', 'network', GLOBAL_FEATURES, g.globals)]
    groups += [('Device', n, NODE_FEATURES, v) for n, v in zip(g.node_ids, g.node_feat)]
    groups += [('Link', '->'.join(e), EDGE_FEATURES, v) for e, v in zip(g.edge_keys, g.edge_feat)]
    return [[kind, name, list(keys), [f'{float(v):.5g}' for v in values]] for kind, name, keys, values in groups]

def render(fs, arm):
    lines = [HEADER]
    if arm == 'table':
        for kind in ['Network', 'Device', 'Link']:
            group = [x for x in fs if x[0] == kind]
            lines.append(kind + ' columns: id ' + ' '.join(group[0][2]))
            lines += [x[1] + ' ' + ' '.join(x[3]) for x in group]
    else:
        for kind, name, keys, values in fs:
            lines.append(f'For {kind.lower()} {name}, ' + '; '.join(
                f'{k.replace("_", " ")} is {v}' for k, v in zip(keys, values)) + '.')
    return '\n'.join(lines)

def parse(text, arm):
    lines = text.splitlines(); assert lines.pop(0) == HEADER
    result = []
    if arm == 'table':
        for line in lines:
            if ' columns: id ' in line:
                kind, names = line.split(' columns: id '); keys = names.split()
            else:
                name, *values = line.split(); assert len(values) == len(keys)
                result.append([kind, name, keys, values])
    else:
        for line in lines:
            match = re.fullmatch(r'For (network|device|link) ([^,]+), (.+)\.', line)
            assert match
            keys, values = [], []
            for clause in match[3].split('; '):
                k, v = clause.rsplit(' is ', 1); keys.append(k.replace(' ', '_')); values.append(v)
            result.append([match[1].title(), match[2], keys, values])
    return result

def core(fs):
    # Ignore action count: it must not make an otherwise identical observation novel.
    return [[k, n, keys[1:] if k == 'Network' else keys,
             vs[1:] if k == 'Network' else vs] for k, n, keys, vs in fs]

def old_facts(text):
    if 'Device columns: id ' not in text:
        return None
    lines = text.splitlines()
    gi = next(i for i, x in enumerate(lines) if x.startswith('Global columns: '))
    ni = next(i for i, x in enumerate(lines) if x.startswith('Device columns: '))
    ei = next(i for i, x in enumerate(lines) if x.startswith('Link columns: '))
    out = [['Network', 'network', lines[gi].split(': ', 1)[1].split(), lines[gi+1].split()]]
    for kind, keys, block in [('Device', NODE_FEATURES, lines[ni+1:ei]), ('Link', EDGE_FEATURES, lines[ei+1:])]:
        for line in block:
            n, *v = line.split(); out.append([kind, n, keys, v])
    return out

def collect(topology, ft, seed):
    env = make_env(topology, 4, ft, 40)
    _, info = env.reset(seed=seed); gt = dict(info['ground_truth_fault'])
    view = PublicView(env); memory = RuleAgentV2(); occ = Counter(); trace = []
    actions = [pointer_scan()]
    actions += [pointer_check_status(n) for n in sorted(view.network.nodes)]
    actions += [pointer_check_interfaces(n) for n in sorted(view.network.nodes)]
    actions += [pointer_ping(a,b) for a,b in sorted(view.network.edges)]
    for ptr in actions:
        aid = resolve_pointer(env, ptr)
        _, _, term, trunc, info = step_env(env, aid, seed, occ)
        trace.append(dict(action=env.action_space_manager.get_action_spec(aid).to_dict(), public_result=_tool_payload(info)))
        memory.observe_step(public_step_info(info)); view.update(env)
        assert not term and not trunc, 'Premature fixed-sweep termination'
    memory.refresh_plan(view)
    g = build_graph_obs(view, mem=memory.mem)
    # Ground truth and this expert conclusion stay in evaluator metadata, never rendering.
    conclusion = memory._try_conclude()
    expert_ok = bool(conclusion and conclusion[0] == gt['type'] and strict_location(gt['location'], conclusion[1], gt['type']))
    fs = facts(g); q = questions(g)
    q = {k:q[k] for k in ['fault','diagnosis_node','diagnosis_edge']}
    location_key = 'diagnosis_node' if gt['type'] in ['device_failure','misconfiguration'] else 'diagnosis_edge'
    location = gt['location']
    if location not in q[location_key]['criteria'] and location_key == 'diagnosis_edge':
        location = '->'.join(reversed(location.split('->')))
    assert location in q[location_key]['criteria']
    result = dict(episode_seed=seed, topology=topology, n=4, facts=fs, questions=q,
                  expected={'fault':gt['type'], location_key:location}, ground_truth=gt,
                  expert_conclusion=conclusion, expert_success=expert_ok, trace=trace,
                  compact_legacy=state_text(g), observation_sha256=digest(fs), core_sha256=digest(core(fs)))
    env.close()
    return result

def main():
    OUT.mkdir(exist_ok=False)
    (OUT/'frozen_spec.md').write_bytes(SPEC.read_bytes())
    snapshot = OUT/'source_snapshot'; snapshot.mkdir()
    files = list((PROJECT/'src').glob('*.py')) + list((PROJECT/'scripts').glob('laya_observation*.py'))
    for p in files:
        (snapshot/p.name).write_bytes(p.read_bytes())
    protocol = dict(status='preregistered', created_unix=time.time(), spec_sha256=sha(SPEC),
                    sources={str(p.relative_to(PROJECT)):sha(p) for p in files},
                    split_counts={'train':32,'test':64}, train_seeds=[41,52,63], updates=1024,
                    train_questions=64, epochs=16, arms=['table','prose'], max_len=4096, head_max_len=1024)
    dump(OUT/'protocol.json',protocol)
    historical_seeds = set(); historical_core = set(); historical_text = set(); sources = {}
    # Complete old compact dataset contains every step used by previous LAYA controls.
    for p in sorted((PROJECT/'results/laya_data_compact_v2').glob('*.jsonl')):
        sources[str(p.relative_to(PROJECT))] = sha(p)
        for line in p.read_text(encoding='utf8').splitlines():
            row=json.loads(line); historical_seeds.add(row['episode_seed'])
            historical_text.add(row['state']); fs=old_facts(row['state'])
            if fs: historical_core.add(digest(core(fs)))
    seen = set(historical_core); ids = set(historical_seeds); audit=[]; accepted=[]
    with (OUT/'candidates.jsonl').open('w', encoding='utf8') as candidate_file:
        for split, base, count in [('train',91000000,4), ('test',92000000,8)]:
            for ti, topology in enumerate(['star','linear']):
                for fi, ft in enumerate(FAULTS):
                    kept=0
                    for attempt in range(1000):
                        seed=base+ti*100000+fi*10000+attempt
                        row=collect(topology,ft,seed); row['split']=split
                        reasons=[]
                        if seed in ids: reasons.append('episode_id_duplicate')
                        if row['compact_legacy'] in historical_text: reasons.append('historical_exact_text')
                        if row['core_sha256'] in seen: reasons.append('observation_core_duplicate')
                        row['exclusion_reasons']=reasons
                        candidate_file.write(json.dumps(row)+'\n');candidate_file.flush()
                        audit.append({k:row[k] for k in ['episode_seed','split','topology','ground_truth','expert_success','observation_sha256','core_sha256','exclusion_reasons']})
                        if not reasons:
                            ids.add(seed);seen.add(row['core_sha256']); accepted.append(row);kept+=1
                        if kept==count:break
                    assert kept==count, (split,topology,ft,kept)
                    print(split,topology,ft,kept,attempt+1,flush=True)
    for split in ['train','test']:
        rows=[r for r in accepted if r['split']==split]
        (OUT/f'{split}_cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
        for arm in ['table','prose']:
            rr=[]
            for r in rows:
                state=render(r['facts'],arm)
                assert parse(state,arm)==r['facts']
                assert parse(render(r['facts'],'table'),'table')==parse(render(r['facts'],'prose'),'prose')
                rr.append(dict(state=state, questions=r['questions'], expected=r['expected'],episode_seed=r['episode_seed']))
            (OUT/f'{arm}_{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr),encoding='utf8')
    dump(OUT/'data_audit.json',dict(status='completed',historical_sources=sources,
        historical_episode_count=len(historical_seeds),historical_core_count=len(historical_core),
        accepted=len(accepted),attempts=len(audit),excluded=len(audit)-len(accepted),
        exclusions=dict(Counter(x for r in audit for x in r['exclusion_reasons'])),
        expert_success_by_split={s:sum(r['expert_success'] for r in accepted if r['split']==s) for s in ['train','test']},
        expert_success_selection=False,roundtrip_checked=2*len(accepted),records=audit,
        data_sha256={p.name:sha(p) for p in OUT.glob('*.jsonl')}))
    print('DATA COMPLETE', len(accepted), flush=True)

if __name__=='__main__':main()
