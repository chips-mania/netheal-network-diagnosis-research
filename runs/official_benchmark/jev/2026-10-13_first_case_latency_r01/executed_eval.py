"""Jev adapter over the unmodified tools/metrics extracted from the pinned green image.

No model gets snapshots, private runtime.info, evaluator metrics or ground truth.
Only the public task, hint, MCP schemas and results enter its request.
"""
import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from jev_context_codec_v4 import encode_checked, decode, dumps
from jev_tree_partition_v4 import tree_partitions

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'netheal-topology-generalization/results/netheal_official_jev_20261011_v4'
UP = ROOT / 'netheal-topology-generalization/results/netheal_official_jev_20261011_v1/upstream'
sys.path.insert(0, str(UP / 'green_source/app'))
from netheal.scenario import load_snapshot_episodes, create_env_from_snapshot
from netheal.environment.env import NetworkTroubleshootingEnv
from netheal.scenario.snapshot import export_snapshot
from netheal.aaa.mcp_server import NetHealMCPServer, EpisodeRuntime
from netheal.evaluation.wrapper import MetricsCollectorWrapper
from netheal.evaluation.metrics import CompetitionEvaluator, EpisodeMetrics
import httpx
from dotenv import dotenv_values

TASK = ('Diagnose the network fault. The simulated network has an injected fault '
        '(device failure, link failure, misconfiguration, or performance degradation). '
        'Use the diagnostic tools to explore the network, identify the fault, '
        'and submit your diagnosis using the submit_diagnosis tool.')
OBJECTIVE = ('Identify the fault type and location within 100 tool calls. '
             'Call submit_diagnosis(fault_type, location) with your answer. '
             'Each tool call (except list_actions) consumes one step from your budget.')
INSTRUCTIONS = ('You are an AI agent that completes tasks using available tools. '
                'Read the task and objective. Use the tool descriptions and observed results '
                'to choose the next tool or its arguments. Complete the task before the budget '
                'runs out. Return a choice; no explanation text is required.')
TOOLS = ['list_actions','scan_network','discover_neighbors','ping','traceroute',
         'check_status','check_interfaces','submit_diagnosis']
API_LOCK = asyncio.Lock()
LAST_REQUEST = 0.0
MAX_REQUEST_TEXT_BYTES = 24000

class ContextBudgetError(RuntimeError): pass

def question(purpose,options):
    criteria={f'c{i:03d}':(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False)) for i,v in enumerate(options)}
    return {'decision':{'type':'choice','instructions':INSTRUCTIONS+' '+purpose,'criteria':criteria}}

def partition_state(state,purpose,options):
    try: return history_partitions(state,purpose,options)
    except ContextBudgetError:
        overhead=len(json.dumps(question(purpose,options),ensure_ascii=False).encode())+512
        return tree_partitions(state,MAX_REQUEST_TEXT_BYTES-overhead)

def history_partitions(state,purpose,options):
    overhead=len(json.dumps(question(purpose,options),ensure_ascii=False).encode())+512
    def fits(s): return len(encode_checked(s).encode())+overhead<=MAX_REQUEST_TEXT_BYTES
    if fits(state): return [state]
    history=state.get('history',[])
    base={k:v for k,v in state.items() if k!='history'}
    base['latest_public_result']=next((e['result'] for e in reversed(history) if 'result' in e),None)
    chunks=[]; current=[]; start=0
    def envelope(events,first):
        return dict(base,history=events,history_partition={'first':first,'count':len(events),'total':len(history),
          'rule':'This is a contiguous evidence partition. Other partitions are evaluated separately with the same options.'})
    for event in history:
        candidate=envelope(current+[event],start)
        if fits(candidate): current.append(event); continue
        if not current: raise ContextBudgetError('single event exceeds safe request budget')
        chunks.append(envelope(current,start)); start+=len(current); current=[event]
        if not fits(envelope(current,start)): raise ContextBudgetError('single event exceeds safe request budget')
    if current: chunks.append(envelope(current,start))
    if not chunks: raise ContextBudgetError('fixed context exceeds safe request budget')
    assert [e for c in chunks for e in c['history']]==history
    return chunks

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def append(path, value):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, default=str)+'\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshots():
    return load_snapshot_episodes(next((UP/'leaderboard').iterdir())/'snapshots/v1')

def runtime(snapshot):
    env = create_env_from_snapshot(snapshot)
    wrapped = MetricsCollectorWrapper(env)
    obs, info = env.observation.to_dict(), env._get_info()
    wrapped._start_new_trace(obs, info, seed=snapshot.get('seed'), snapshot_id=snapshot['snapshot_id'])
    server = NetHealMCPServer(EpisodeRuntime(wrapped, obs, info))
    return env, wrapped, server

async def call(server, name, args):
    result = await server._mcp.call_tool(name, args)
    # FastMCP structured tools return (content, structuredContent) in MCP 1.x.
    if isinstance(result, tuple):
        if isinstance(result[1], dict): return result[1].get('result', result[1])
        result = result[0]
    if isinstance(result, dict): return result
    return json.loads(''.join(x.text for x in result if hasattr(x, 'text')))

def action_tool(a):
    return 'submit_diagnosis' if a['category']=='diagnosis' else a['action_type']

class Jev:
    def __init__(self, key, folder, deadline):
        self.client = httpx.AsyncClient(headers={'Authorization':'Bearer '+key}, timeout=120,
                                        follow_redirects=False)
        self.folder, self.deadline, self.calls = folder, deadline, 0

    async def choose(self, state, purpose, options):
        chunks=partition_state(state,purpose,options)
        answers=[]
        for chunk in chunks: answers.append(await self._choose_one(chunk,purpose,options))
        if len(chunks)==1: return answers[0][0]
        weights=[c.get('partition_weight',len(c.get('history',[]))) for c in chunks]
        scores=[sum(weight*ans[1][f'c{i:03d}'] for weight,ans in zip(weights,answers))/sum(weights) for i in range(len(options))]
        selected=max(range(len(options)),key=lambda i:scores[i])
        append(self.folder/'partition_decisions.jsonl',{'purpose':purpose,'partitions':len(chunks),
          'weights':weights,'scores':scores,'selected':selected,'original_events':len(state.get('history',[])),
          'mode':'tree' if 'evidence_fragments' in chunks[0] else 'history',
          'original_state_sha256':hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()})
        return options[selected]

    async def _choose_one(self, state, purpose, options):
        global LAST_REQUEST
        assert 1 <= len(options) <= 255
        criteria={f'c{i:03d}': (v if isinstance(v,str) else json.dumps(v,ensure_ascii=False))
                  for i,v in enumerate(options)}
        compact_state=encode_checked(state)
        body={'model':'jev-1.13.0','state':compact_state,'questions':{'decision':{
            'type':'choice','instructions':INSTRUCTIONS+' '+purpose,'criteria':criteria}}}
        assert len(compact_state.encode())+len(json.dumps(body['questions'],ensure_ascii=False).encode())+512<=MAX_REQUEST_TEXT_BYTES
        if time.monotonic() >= self.deadline: raise TimeoutError('episode deadline')
        async with API_LOCK:
            await asyncio.sleep(0)
            LAST_REQUEST=time.monotonic()
        started=time.perf_counter()
        self.calls+=1
        append(self.folder/'request_started.jsonl',{'call':self.calls,'unix':time.time(),
                    'request_sha256':hashlib.sha256(json.dumps(body).encode()).hexdigest()})
        response=await self.client.post('https://api.typesafe.ai/v1/systemone',json=body,
                    timeout=min(120, max(1,self.deadline-time.monotonic())))
        record={'request':body,'status':response.status_code,'seconds':time.perf_counter()-started,
                'codec_roundtrip_passed':True,'raw_state_sha256':hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest(),
                'raw_state_bytes':len(dumps(state).encode()),'compact_state_bytes':len(compact_state.encode())}
        if response.status_code!=200:
            try:
                detail=response.json().get('detail',{})
                if isinstance(detail,dict):
                    code=detail.get('error_type')
                    if isinstance(code,str) and code.replace('_','').isalnum(): record['error_type']=code
            except ValueError:
                pass
            append(self.folder/'requests.jsonl',record)
            raise RuntimeError('Jev HTTP '+str(response.status_code))
        data=response.json()
        record['response']={k:data[k] for k in ('model','answers','usage','id') if k in data}
        append(self.folder/'requests.jsonl',record)
        choice=data['answers']['decision']['choice']
        if choice not in criteria: raise RuntimeError('invalid choice')
        probabilities=data['answers']['decision']['probabilities']
        assert set(probabilities)==set(criteria)
        assert all(isinstance(v,(int,float)) and 0<=v<=1 for v in probabilities.values())
        return options[list(criteria).index(choice)],probabilities

async def preflight():
    ss=snapshots()
    upstream=json.loads((UP/'manikyabard-20260201-100940.json').read_text())['results'][0]
    rows=next(a['data']['results'] for a in upstream['artifacts'] if a['label']=='aaa_metrics')
    reference={r['snapshot_id']:r for r in rows}
    checks=[]
    for s in ss:
        e,w,m=runtime(s)
        checks.append({'id':s['snapshot_id'], 'fingerprint':w._trace.scenario_fingerprint,
          'fingerprint_matches':w._trace.scenario_fingerprint==reference[s['snapshot_id']]['scenario_fingerprint'],
          'max_steps':e.max_episode_steps, 'has_hint':bool(e.user_hint)})
        e.close()
    assert len(checks)==100 and all(c['fingerprint_matches'] for c in checks)
    assert all(c['max_steps']==100 for c in checks)
    weighted=sum(r['network_size']*r['pass'] for r in rows)/sum(r['network_size'] for r in rows)
    assert abs(weighted-upstream['summary']['episodes']['diagnosis_success_rate'])<1e-12
    # Development case is newly generated and is not an official evaluation snapshot.
    env=NetworkTroubleshootingEnv(min_devices=4,max_devices=4,max_episode_steps=100,
                    topology_types=['star'],enable_user_hints=True,hint_provider_mode='heuristic')
    env.reset(seed=93001101)
    dev=export_snapshot(env,seed=93001101)
    save(OUT/'development_snapshot.json',dev)
    env.close()
    e,w,m=runtime(dev)
    schema=[t.model_dump(mode='json') for t in await m._mcp.list_tools()]
    save(OUT/'tool_schemas.json',schema)
    assert set(t['name'] for t in schema)==set(TOOLS)
    listed=await call(m,'list_actions',{})
    scanned=await call(m,'scan_network',{})
    assert listed['count']>0 and 'observation' in scanned
    assert 'ground_truth_fault' not in scanned.get('info',{})
    # Verify all public valid action mappings without consuming evaluation cases.
    actions=(await call(m,'list_actions',{}))['valid_actions']
    assert all(action_tool(a) in TOOLS for a in actions)
    for tool in ('discover_neighbors','check_status','check_interfaces','ping','traceroute'):
        a=next(a for a in actions if action_tool(a)==tool)
        response=await call(m,tool,a['parameters'])
        assert 'observation' in response, tool
        assert 'ground_truth_fault' not in response.get('info',{})
    # Deterministic development-only oracle is a plumbing test, never model input.
    truth=e._get_info()['ground_truth_fault']
    response=await call(m,'submit_diagnosis',{'fault_type':truth['type'],'location':truth['location']})
    assert response['terminated'] and w.last_episode_metrics.diagnosis_success
    e.close()
    save(OUT/'preflight.json',{'status':'passed','snapshots':checks,'python':sys.version,
        'platform':platform.platform(),'claude_weighted_success':weighted,
        'claude_unweighted_success':sum(r['pass'] for r in rows)/len(rows),
        'development_smoke_no_api':True,'source_sha256':sha(Path(__file__))})
    print('PREFLIGHT PASSED: 100 snapshot fingerprints; original weighted metric verified',flush=True)

async def episode(s,key,folder,max_turns=100):
    folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    e,w,m=runtime(s)
    j=Jev(key,folder,started+1200)
    history=[]
    state={'task_description':TASK,'objective':OBJECTIVE,'hint':e.user_hint,
           'network_size':len(e.network.get_all_devices()),'max_steps':100,
           'tool_schemas':json.loads((OUT/'tool_schemas.json').read_text()),'history':history}
    status='turn_limit'
    try:
        # One selected tool per solver turn; list_actions does not consume env budget.
        for turn in range(max_turns):
            state['solver_turn']=turn
            tool=await j.choose(state,'Select the next tool.',TOOLS)
            args={}
            if tool not in ('scan_network','list_actions'):
                # Adapter helper supplies only the chosen tool's argument candidates.
                # The full public list is logged, not redundantly inserted into model history.
                public=await call(m,'list_actions',{})
                append(folder/'helpers.jsonl',{'tool':'list_actions','arguments':{},'result':public,'turn':turn})
                options=[a for a in public['valid_actions'] if action_tool(a)==tool]
                if not options:
                    history.append({'adapter_notice':'No currently valid arguments for '+tool})
                    continue
                if tool=='submit_diagnosis':
                    types=list(dict.fromkeys(a['action_type'] for a in options))
                    ft=await j.choose(state,'Select fault_type for submit_diagnosis.',types)
                    locations=list(dict.fromkeys(a['parameters']['location'] for a in options if a['action_type']==ft))
                    loc=await j.choose(state,'Select location for predicted fault_type '+ft+'.',locations)
                    args={'fault_type':ft,'location':loc}
                else:
                    param_options=[a['parameters'] for a in options]
                    if len(param_options)>255:
                        sources=list(dict.fromkeys(x['source'] for x in param_options))
                        src=await j.choose(state,'Select source for '+tool+'.',sources)
                        param_options=[x for x in param_options if x['source']==src]
                    args=await j.choose(state,'Select arguments for '+tool+'.',param_options)
            result=await call(m,tool,args)
            history.append({'tool':tool,'arguments':args,'result':result})
            append(folder/'tools.jsonl',history[-1])
            if result.get('terminated') or result.get('truncated'):
                status='completed' if result.get('terminated') else 'step_limit'
                break
    except Exception as ex:
        # Do not persist exception messages, which may contain request credentials.
        status='error_'+type(ex).__name__
        if isinstance(ex,RuntimeError) and str(ex).startswith('Jev HTTP '): status=str(ex)
    finally:
        await j.client.aclose()
    if w.last_episode_metrics is None:
        w._finalize_trace(final_observation=m.runtime.observation,final_info=m.runtime.info,
                          terminated=False,truncated=True)
    metrics=dataclasses.asdict(w.last_episode_metrics)
    save(folder/'result.json',{'snapshot_id':s['snapshot_id'],'status':status,
           'seconds':time.monotonic()-started,'model_requests':j.calls,'metrics':metrics})
    e.close()
    print(f"{s['snapshot_id']} {status} calls={j.calls}",flush=True)
    return status

async def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['preflight','smoke','evaluate'])
    args=parser.parse_args()
    if args.mode=='preflight': return await preflight()
    assert json.loads((OUT/'preflight.json').read_text())['status']=='passed'
    key=os.environ.get('TYPESAFE_API_KEY') or dotenv_values(ROOT/'.env').get('TYPESAFE_API_KEY')
    if not key: raise RuntimeError('TYPESAFE_API_KEY is not configured')
    if args.mode=='smoke':
        return await episode(json.loads((OUT/'development_snapshot.json').read_text()),key,OUT/'smoke',max_turns=3)
    assert (OUT/'frozen_protocol.json').exists(), 'Freeze protocol before evaluation'
    assert sha(Path(__file__))==json.loads((OUT/'frozen_protocol.json').read_text())['script_sha256']
    assert sha(Path(__file__).with_name('jev_context_codec_v4.py'))==json.loads((OUT/'frozen_protocol.json').read_text())['codec_sha256']
    assert sha(Path(__file__).with_name('jev_tree_partition_v4.py'))==json.loads((OUT/'frozen_protocol.json').read_text())['partition_sha256']
    semaphore=asyncio.Semaphore(5)
    async def one(s):
        folder=OUT/'episodes'/s['snapshot_id']
        if (folder/'result.json').exists(): return
        async with semaphore:
            folder.mkdir(parents=True,exist_ok=True)
            for attempt in range(11):
                attempt_folder=folder/f'attempt_{attempt:02d}'
                if (attempt_folder/'result.json').exists():
                    status=json.loads((attempt_folder/'result.json').read_text())['status']
                elif attempt_folder.exists():
                    raise RuntimeError('Interrupted attempt requires explicit resume audit')
                else:
                    status=await episode(s,key,attempt_folder)
                retry=(status.startswith('error_') or status in
                       ('Jev HTTP 429','Jev HTTP 500','Jev HTTP 502','Jev HTTP 503','Jev HTTP 504'))
                if status in ('Jev HTTP 400','Jev HTTP 413','Jev HTTP 422'):
                    retry=False # deterministic invalid input: do not repeat paid identical trajectory
                if status=='error_ContextBudgetError': retry=False
                if not retry or attempt==10:
                    result=json.loads((attempt_folder/'result.json').read_text())
                    result['attempts']=attempt+1
                    save(folder/'result.json',result)
                    break
                await asyncio.sleep(min(30,2**attempt))
    await asyncio.gather(*(one(s) for s in snapshots()))
    evaluator=CompetitionEvaluator()
    results=[]
    for s in snapshots():
        r=json.loads((OUT/'episodes'/s['snapshot_id']/'result.json').read_text())
        results.append(r)
        evaluator.add_episode_metrics(EpisodeMetrics(**r['metrics']))
    save(OUT/'summary.json',{'official_summary':evaluator.compute_summary(),'episodes':results})
    print('EVALUATION COMPLETE',flush=True)

if __name__=='__main__': asyncio.run(main())
