"""Audit old requests without choosing new settings from model accuracy."""
import asyncio
import json
from pathlib import Path
from jev_context_codec import encode_checked, decode, dumps
import jev_official_eval_v3 as run

async def main():
    prior=run.OUT.parent/'netheal_official_jev_20261011_v2'
    checked=0; failures=[]; sizes=[]
    for f in prior.glob('episodes/*/attempt_00/requests.jsonl'):
        for line in f.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            state=r['request']['state']; compact=encode_checked(state)
            assert json.dumps(decode(json.loads(compact)),sort_keys=True)==json.dumps(state,sort_keys=True)
            checked+=1
            size={'old_bytes':len(dumps(state).encode()),'new_bytes':len(compact.encode())}
            sizes.append(size)
            if r['status']==400:
                failures.append(dict(size,snapshot_id=f.parents[1].name))
    assert checked==1571 and len(failures)==39
    # Numeric type, repeated values, temporal changes and late evidence must survive.
    edge_cases=[{'a':[0,0.0,0,0.0], 'b':[[0],[0.0],[0]]},
                {'history':[{'ping':1.1234567890123},{'ping':float('1.1234567890124')}]},
                {'history':[{'status':'up'}]*99+[{'status':'down'}]}]
    for state in edge_cases: encode_checked(state)
    # A new 15-node development episode, full 100-action history; no paid API.
    env=run.NetworkTroubleshootingEnv(min_devices=15,max_devices=15,max_episode_steps=100,
                 topology_types=['star'],enable_user_hints=True,hint_provider_mode='heuristic')
    env.reset(seed=93001102)
    snap=run.export_snapshot(env,seed=93001102); env.close()
    e,w,m=run.runtime(snap)
    history=[]
    state={'task_description':run.TASK,'objective':run.OBJECTIVE,'hint':e.user_hint,
      'network_size':15,'max_steps':100,'tool_schemas':json.loads((run.OUT/'tool_schemas.json').read_text()),
      'history':history,'solver_turn':99}
    for i in range(100):
        if i==0: tool,args='scan_network',{}
        else:
            public=await run.call(m,'list_actions',{})
            tool=['check_status','check_interfaces','ping','traceroute','discover_neighbors'][(i-1)%5]
            options=[a for a in public['valid_actions'] if run.action_tool(a)==tool]
            args=options[(i//5)%len(options)]['parameters']
        result=await run.call(m,tool,args)
        history.append({'tool':tool,'arguments':args,'result':result})
    compact=encode_checked(state)
    run.save(run.OUT/'development_100step_state.json',state)
    audit={'status':'passed','old_requests_exact_roundtrip':checked,'failed_request_count':len(failures),
      'old_max_bytes':max(x['old_bytes'] for x in sizes),'new_max_bytes':max(x['new_bytes'] for x in sizes),
      'failures':failures,'numeric_and_temporal_edge_cases':len(edge_cases),
      'development_100step_raw_bytes':len(dumps(state).encode()),
      'development_100step_compact_bytes':len(compact.encode())}
    run.save(run.OUT/'codec_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='failures'},indent=2))
    e.close()

if __name__=='__main__': asyncio.run(main())
