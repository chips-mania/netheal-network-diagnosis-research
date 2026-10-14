"""Training traces in the pinned official environment; never expose labels as state."""
import asyncio,json,hashlib,copy
from pathlib import Path
import jev_official_eval_v4 as envrun
from netheal.faults.injector import FaultType

ROOT=envrun.ROOT;OUT=ROOT/'netheal-topology-generalization/results/laya_official_20261012_v1'
TOPOLOGIES=['star','linear','mesh','hierarchical','random']
FAULTS=['device_failure','link_failure','misconfiguration','performance_degradation']
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def state(case,turn):
    return {'task_description':envrun.TASK,'objective':envrun.OBJECTIVE,'hint':case['hint'],'network_size':case['trace'][0]['result']['info']['network_size'],
            'max_steps':100,'tool_schemas':read(OUT/'tool_schemas.json'),'history':case['trace'][:turn],'solver_turn':turn}
def node_options(catalog,tool):return [a['parameters'] for a in catalog if envrun.action_tool(a)==tool]
def option_questions(catalog,tool,args):
    if tool in ('scan_network','list_actions'):return []
    opts=node_options(catalog,tool)
    if tool=='submit_diagnosis':
        ft=args['fault_type'];types=list(dict.fromkeys(a['action_type'] for a in catalog if envrun.action_tool(a)==tool))
        result=[('fault','Select fault_type for submit_diagnosis.',types,ft)]
        locs=list(dict.fromkeys(a['parameters']['location'] for a in catalog if envrun.action_tool(a)==tool and a['action_type']==ft))
        location=args['location']
        if '->' in location:
            sources=list(dict.fromkeys(x.split('->')[0] for x in locs));source=location.split('->')[0]
            result.append(('location','Select location source for predicted fault_type '+ft+'.',sources,source))
            locs=[x for x in locs if x.split('->')[0]==source]
        result.append(('location','Select location for predicted fault_type '+ft+'.',locs,location));return result
    result=[]
    if 'source' in args and len({o['source'] for o in opts})>1:
        sources=list(dict.fromkeys(o['source'] for o in opts))
        result.append(('argument','Select source for '+tool+'.',sources,args['source']))
        opts=[o for o in opts if o['source']==args['source']]
    result.append(('argument','Select arguments for '+tool+'.',opts,args))
    return result
def core(x):
    if isinstance(x,dict):return {k:core(v) for k,v in x.items() if k not in {'timestamp','step_count','episode_progress','episode_metadata','reward','reward_breakdown','recent_diagnostics'}}
    if isinstance(x,list):return [core(v) for v in x]
    return x
def public_audit(x):
    if isinstance(x,dict):
        assert not set(x)&{'ground_truth_fault','ground_truth','injected_fault','fault_info','correct_diagnosis','expected_solution'}
        for v in x.values():public_audit(v)
    elif isinstance(x,list):
        for v in x:public_audit(v)

async def main():
    OUT.mkdir(exist_ok=False)
    save(OUT/'tool_schemas.json',read(envrun.OUT/'tool_schemas.json'))
    spec=ROOT/'docs/specs/laya_official_training_20261012.md';(OUT/'frozen_spec.md').write_bytes(spec.read_bytes())
    (OUT/'data_source.py').write_bytes(Path(__file__).read_bytes())
    save(OUT/'data_protocol.json',{'spec_sha256':envrun.sha(spec),'source_sha256':envrun.sha(Path(__file__)),'train':40,'development':20})
    official=envrun.snapshots();official_ids={s['snapshot_id'] for s in official};official_seeds={s['seed'] for s in official}
    reference=read(envrun.UP/'manikyabard-20260201-100940.json')['results'][0]
    official_fingerprints={r['scenario_fingerprint'] for a in reference['artifacts'] if a['label']=='aaa_metrics' for r in a['data']['results']}
    historical_seeds=set();history_files=0
    for p in (ROOT/'netheal-topology-generalization/results').glob('laya*/**/*.jsonl'):
        if OUT in p.parents:continue
        for line in p.read_text(encoding='utf-8').splitlines():
            try:r=json.loads(line)
            except ValueError:continue
            if 'episode_seed' in r:historical_seeds.add(r['episode_seed'])
        history_files+=1
    # Only check equality against stored public tool observations; do not use Jev decisions as targets.
    prior_exact=set();prior_core=set();prior_files=0
    for p in envrun.OUT.glob('episodes/*/attempt_*/tools.jsonl'):
        for line in p.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)['result'];prior_exact.add(digest(r));prior_core.add(digest(core(r)))
        prior_files+=1
    seen_fingerprints=set(official_fingerprints);seen_core=set();cases=[];duplicates=[]
    for split,ns,seedbase in [('train',[8,12],94010000),('development',[10],94020000)]:
        index=0
        for top in TOPOLOGIES:
            for n in ns:
                for ft in FAULTS:
                    seed=seedbase+index;index+=1
                    assert seed not in official_seeds|historical_seeds
                    e=envrun.NetworkTroubleshootingEnv(min_devices=n,max_devices=n,max_episode_steps=100,topology_types=[top],fault_types=[FaultType(ft)],enable_user_hints=True,hint_provider_mode='heuristic')
                    e.reset(seed=seed);snapshot=envrun.export_snapshot(e,seed=seed);e.close()
                    assert snapshot['snapshot_id'] not in official_ids
                    e,w,m=envrun.runtime(snapshot);fingerprint=w._trace.scenario_fingerprint
                    case={'split':split,'seed':seed,'n':n,'topology':top,'snapshot_id':snapshot['snapshot_id'],'fingerprint':fingerprint,'hint':e.user_hint,'trace':[],'questions':[],'truth':dict(e._get_info()['ground_truth_fault'])}
                    catalog=(await envrun.call(m,'list_actions',{}))['valid_actions'];case['catalog']=catalog
                    nodes=[a['device'] for a in node_options(catalog,'check_status')]
                    plan=[('scan_network',{})]+[('check_status',{'device':x}) for x in nodes]+[('check_interfaces',{'device':x}) for x in nodes]
                    step=0
                    while step<len(plan):
                        tool,args=plan[step]
                        questions=[('tool','Select the next tool.',envrun.TOOLS,tool)]+option_questions(catalog,tool,args)
                        for category,purpose,options,answer in questions:
                            assert answer in options
                            case['questions'].append({'turn':step,'category':category,'purpose':purpose,'options':options,'answer':options.index(answer)})
                        result=await envrun.call(m,tool,args);public_audit(result)
                        case['trace'].append({'tool':tool,'arguments':args,'result':result})
                        if step==0:
                            ids=result['info']['action_result']['result']['data']['discovered_devices'];matrix=result['observation']['discovery_matrix']
                            pairs=[(a,b) for i,a in enumerate(ids) for j,b in enumerate(ids) if i<j and (matrix[i][j]!=0 or matrix[j][i]!=0)]
                            plan += [('ping',{'source':a,'destination':b}) for a,b in pairs]
                            truth=case['truth'];plan.append(('submit_diagnosis',{'fault_type':truth['type'],'location':truth['location']}))
                        if result.get('terminated') or result.get('truncated'):break
                        step+=1
                    case['teacher_result']=__import__('dataclasses').asdict(w.last_episode_metrics)
                    evidence=[ev['result'] for ev in case['trace'][:-1]]
                    signature=digest([core(x) for x in evidence]);reasons=[]
                    if fingerprint in seen_fingerprints:reasons.append('scenario_fingerprint_duplicate')
                    if signature in seen_core:reasons.append('complete_public_trace_core_duplicate')
                    # Individual repeated observations are reported, not used for success-based selection.
                    case['duplicate_observations']={'against_official_exact':sum(digest(x) in prior_exact for x in evidence),'against_official_core':sum(digest(core(x)) in prior_core for x in evidence)}
                    case['exclusion_reasons']=reasons
                    save(OUT/'snapshots'/f'{seed}.json',snapshot);save(OUT/'cases'/f'{seed}.json',case)
                    seen_fingerprints.add(fingerprint);seen_core.add(signature)
                    if reasons:duplicates.append({'seed':seed,'reasons':reasons})
                    else:cases.append({k:case[k] for k in ['split','seed','n','topology','snapshot_id','fingerprint','duplicate_observations']})
                    e.close();print(split,seed,'steps',len(case['trace']),'questions',len(case['questions']),'excluded',reasons,flush=True)
    assert len(cases)==60 and not duplicates,'Preserve exclusions; investigate before training, do not silently replace cases'
    save(OUT/'data_audit.json',{'status':'completed','cases':cases,'excluded':duplicates,'official_snapshots':100,'historical_seed_files_checked':history_files,'historical_seeds':len(historical_seeds),'official_tool_files_checked':prior_files,'expert_success_selection':False,'case_hashes':{p.name:envrun.sha(p) for p in (OUT/'cases').glob('*.json')}})

if __name__=='__main__':asyncio.run(main())
