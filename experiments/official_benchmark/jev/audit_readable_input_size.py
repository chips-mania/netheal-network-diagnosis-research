"""Offline byte attribution for one failed raw-history request."""
import collections
import json
from compare_first_case_input import ROOT, save

p = ROOT / 'runs/official_benchmark/jev/2026-10-13_readable_baseline_full_r01/episodes/0283628d-f76d-4f49-87ae-41925030db96/requests.jsonl'
records = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]
r = records[-1]
s = json.loads(r['request']['state'])
def size(x):
    return len(json.dumps(x, ensure_ascii=False, separators=(',', ':')).encode())
c, obs, info = collections.Counter(), collections.Counter(), collections.Counter()
hist = s['history']
for event in hist:
    result = event['result']
    c.update({k:size(v) for k,v in result.items()})
    obs.update({k:size(v) for k,v in result.get('observation', {}).items()})
    info.update({k:size(v) for k,v in result.get('info', {}).items()})
report = {'case':p.parent.name, 'status':r['status'], 'state_bytes':len(r['request']['state'].encode()),
          'history_events':len(hist), 'top_field_value_bytes':{k:size(v) for k,v in s.items()},
          'result_value_bytes_sum':dict(c), 'observation_value_bytes_sum':dict(obs), 'info_value_bytes_sum':dict(info),
          'tools':[e['tool'] for e in hist], 'previous_request_usage':records[-2]['response']['usage'],
          'unique_observation_values':{k:len({json.dumps(e['result']['observation'].get(k),sort_keys=True) for e in hist}) for k in hist[0]['result']['observation']}}
save(p.parents[2] / 'input_size_audit.json', report)
print(json.dumps(report, ensure_ascii=False, indent=2))
