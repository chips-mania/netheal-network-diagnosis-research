"""Held-out teacher-collected evidence; predictions do NOT use teacher fault branches."""
import sys,json,time
from pathlib import Path
from types import SimpleNamespace
import torch,laya
from laya.common import build_sequence
from laya.train import to_internal
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.laya_schema import questions
from src.run_research_suite import strict_location,sha,dump

def main():
    torch.set_num_threads(4)
    data=PROJECT/'results/laya_data_compact_v2/validation.jsonl'
    rows=[json.loads(s) for s in data.read_text().splitlines()]
    rows=[r for r in rows if r['expected']['kind']=='diagnose']
    assert len(rows)==288 and len({r['episode_seed'] for r in rows})==288
    checkpoints={'base':str(PROJECT.parent/'.runtime/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148'),
        'head':str(PROJECT/'results/laya_head_compact_48ep_v2/checkpoint')}
    out=PROJECT/'results/laya_diagnosis_only_v2';out.mkdir(exist_ok=False)
    manifest=dict(status='running',episodes=288,task='Diagnosis only from held-out expert-collected public evidence; not end-to-end.',
        data_sha256=sha(data),source_sha256=sha(Path(__file__)),precision='fp32',teacher_fault_branch_used=False)
    dump(out/'manifest.json',manifest);summary={}
    try:
        for label,checkpoint in checkpoints.items():
            agent=laya.load(checkpoint,device='cuda');agent.amp_enabled=False;agent.dtype=torch.float32
            scores=[]
            for i,row in enumerate(rows):
                state=row['state'];parts=state.splitlines()
                ni=next(j for j,s in enumerate(parts) if s.startswith('Device columns:'))
                ei=next(j for j,s in enumerate(parts) if s.startswith('Link columns:'))
                g=SimpleNamespace(node_ids=[s.split()[0] for s in parts[ni+1:ei]],
                    edge_keys=[s.split()[0].split('->') for s in parts[ei+1:]])
                qs=questions(g);usage=[]
                def ask(name):
                    q=qs[name]
                    _,markers,stats,trunc=build_sequence(agent.tok,state,to_internal(name,q),4096,1024,
                        return_stats=True,return_truncation_stats=True)
                    assert not trunc['truncated'] and len(markers)==len(q['criteria']) and stats['options_distinct']==stats['options']
                    r=agent.predict(state,{name:q},max_len=4096,head_max_len=1024)
                    usage.append(r['usage']);return r['answers'][name]['choice']
                start=time.perf_counter();ft=ask('fault')
                loc=ask('diagnosis_node' if ft in ('device_failure','misconfiguration') else 'diagnosis_edge')
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                expected=row['expected'];gt=expected['fault'];gtloc=expected.get('diagnosis_node',expected.get('diagnosis_edge'))
                type_ok=ft==gt;loc_ok=strict_location(gtloc,loc,gt)
                result=dict(model=label,episode_seed=row['episode_seed'],fault_type=gt,fault_location=gtloc,
                    predicted_type=ft,predicted_location=loc,type_correct=type_ok,location_correct=loc_ok,
                    exact=type_ok and loc_ok,seconds=seconds,usage=usage)
                scores.append(result)
                with (out/'predictions.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(result)+'\n')
                if (i+1)%48==0:print(json.dumps(dict(model=label,done=i+1,total=len(rows))),flush=True)
            summary[label]=dict(n=len(scores),exact=sum(r['exact'] for r in scores)/len(scores),
                type_accuracy=sum(r['type_correct'] for r in scores)/len(scores),
                location_accuracy=sum(r['location_correct'] for r in scores)/len(scores),
                mean_seconds=sum(r['seconds'] for r in scores)/len(scores),cpu_fallback_count=agent.cpu_fallback_count,
                fault_accuracy={f:sum(r['exact'] for r in scores if r['fault_type']==f)/sum(r['fault_type']==f for r in scores)
                    for f in sorted({r['fault_type'] for r in scores})})
            dump(out/'summary.json',summary)
            del agent
            import gc
            gc.collect();torch.cuda.empty_cache()
        manifest['status']='completed'
    except BaseException as e:
        manifest['status']='interrupted';manifest['error_type']=type(e).__name__;raise
    finally:dump(out/'manifest.json',manifest)
if __name__=='__main__':main()
