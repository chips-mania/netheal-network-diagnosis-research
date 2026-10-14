"""Train all fixed arms before opening evaluation; no APIs or policy rollouts."""
import argparse, gc, itertools, json, time, threading, subprocess, platform
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from laya.train import load_checkpoint, items_from_rows, TrainConfig, train_model, encode_item, collate_items, _forward, to_internal, optimizer_updates
from laya.common import build_sequence
from laya_observation_data import OUT, PROJECT, dump, sha, SPEC

BASE=PROJECT.parent/'.runtime/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148'
DEVICE=torch.device('cuda')
NODE_FAULTS={'device_failure','misconfiguration'}
FAULTS=['device_failure','performance_degradation','misconfiguration','link_failure']
RUNS=[(arm,seed) for seed in [41,52,63] for arm in (['table','prose'] if seed!=52 else ['prose','table'])]

def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(arm,split):return [json.loads(l) for l in (OUT/f'{arm}_{split}.jsonl').read_text(encoding='utf8').splitlines()]

class Monitor:
    def __enter__(self):
        self.samples=[];self.stop=threading.Event()
        def poll():
            while not self.stop.is_set():
                try:
                    value=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,creationflags=0x08000000)
                    util,mem=map(float,value.strip().split(','));self.samples.append(dict(unix=time.time(),utilization_percent=util,device_memory_MiB=mem))
                except Exception as e:self.samples.append({'error':str(e)})
                self.stop.wait(2)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()

def checked_items(tok,rr,all_questions=False):
    checked=[];lengths=[]
    for r in rr:
        # Dummy targets on unlabelled auxiliary branches only build inference items.
        # _forward never reads targets; main branch selection uses predicted type.
        expected=dict(r['expected'])
        if all_questions:
            for k,q in r['questions'].items():expected.setdefault(k,next(iter(q['criteria'])))
        for k in expected:
            q=r['questions'][k]
            ids,markers,stats,cut=build_sequence(tok,r['state'],to_internal(k,q),4096,1024,return_stats=True,return_truncation_stats=True)
            assert not cut['truncated'] and len(markers)==len(q['criteria'])==stats['options_distinct']
            lengths.append(len(ids))
        item,skip=items_from_rows(tok,[dict(r,expected=expected)],4096,1024)
        assert not skip and len(item)==len(expected)
        checked.append(dict(zip([k for k in r['questions'] if k in expected],item)))
    return checked,lengths

def load():
    model,tok,_=load_checkpoint(str(BASE));model.to(DEVICE)
    return model,tok

def train_all():
    audit=read(OUT/'data_audit.json')
    for name,h in audit['data_sha256'].items():assert sha(OUT/name)==h
    assert sha(SPEC)==read(OUT/'protocol.json')['spec_sha256']
    for arm,seed in RUNS:
        target=OUT/f'{arm}_{seed}';target.mkdir(exist_ok=True)
        if (target/'training.json').exists():continue
        model,tok=load()
        for p in model.encoder.parameters():p.requires_grad_(False)
        for p in model.encoder.layers[-1].parameters():p.requires_grad_(True)
        rr=rows(arm,'train');grouped,lengths=checked_items(tok,rr)
        items=[v for r in grouped for v in r.values()];assert len(items)==64
        initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        cfg=TrainConfig(epochs=16,head_lr=1e-4,encoder_lr=2.5e-5,micro_batch=1,grad_accum=1,
                        freeze_encoder=False,loss='soft-ce',seed=seed,amp=False,gradient_checkpointing=False,
                        max_len=4096,head_max_len=1024,shuffle_options=('choice',),calib_frac=0,calib_max=0,log_every=0)
        assert optimizer_updates(len(items),1,1,16)==1024
        dump(target/'config.json',asdict(cfg));dump(target/'status.json',{'status':'training','started_unix':time.time()})
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();update_count=[0]
        original_step=torch.optim.AdamW.step
        def counted_step(opt,*a,**kw):
            result=original_step(opt,*a,**kw);update_count[0]+=1;return result
        torch.optim.AdamW.step=counted_step
        try:
            with Monitor() as monitor:
                history=train_model(model,tok,items,cfg,DEVICE,4096,1024)
            torch.cuda.synchronize()
        finally:torch.optim.AdamW.step=original_step
        seconds=time.perf_counter()-start;assert update_count[0]==1024
        learned={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        changes={n:float((v-initial[n]).abs().max()) for n,v in learned.items()}
        assert any(v>0 for n,v in changes.items() if n.startswith('encoder.'))
        assert all(n.startswith('encoder.layers.27.') for n in learned if n.startswith('encoder.'))
        torch.save(learned,target/'trainable.pt')
        dump(target/'training.json',dict(status='completed',arm=arm,seed=seed,optimizer_updates=update_count[0],
            questions=64,type_updates=512,location_updates=512,epochs=16,seconds=seconds,loss_history=history,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_samples=monitor.samples,sequence_tokens={'min':min(lengths),'max':max(lengths)},
            parameter_changes=changes,checkpoint_sha256=sha(target/'trainable.pt')))
        dump(target/'status.json',{'status':'training_completed_evaluation_unopened'})
        print('TRAIN COMPLETE',arm,seed,seconds,flush=True)
        del model,tok,initial,learned,items,grouped;gc.collect();torch.cuda.empty_cache()
    dump(OUT/'training_complete.json',dict(status='completed',completed_unix=time.time(),
        base_sha256=sha(BASE/'model.safetensors'),checkpoints={f'{a}_{s}':sha(OUT/f'{a}_{s}/trainable.pt') for a,s in RUNS}))

@torch.no_grad()
def predict(model,tok,item,keys,order=None):
    model.eval()
    if order is None:order=list(range(len(keys)))
    b=collate_items([[encode_item(tok,item,4096,1024,list(order))]],tok.pad_token_id)
    z=_forward(model,b,DEVICE,False,False)[0].cpu().numpy()
    assert np.isfinite(z).all()
    return dict(prediction=keys[order[int(np.argmax(z))]],logits=z.tolist(),order=list(order))

def loc_ok(actual,predicted,ft):
    if ft in NODE_FAULTS:return actual==predicted
    return sorted(actual.split('->'))==sorted(predicted.split('->'))

def metrics(records):
    n=len(records);conf=np.zeros((4,4),dtype=int)
    for r in records:conf[FAULTS.index(r['truth_type']),FAULTS.index(r['predicted_type'])]+=1
    f1=[]
    for i in range(4):
        denom=conf[i,:].sum()+conf[:,i].sum();f1.append(2*conf[i,i]/denom if denom else 0.)
    return dict(n=n,type_accuracy=sum(r['type_correct'] for r in records)/n,macro_f1=float(np.mean(f1)),
        location_accuracy=sum(r['location_correct'] for r in records)/n,exact=sum(r['exact'] for r in records)/n,
        oracle_location_accuracy=sum(r['oracle_location_correct'] for r in records)/n,confusion_matrix=conf.tolist())

def record(r,fp,lp,oracle):
    truth=r['expected']['fault'];key='diagnosis_node' if truth in NODE_FAULTS else 'diagnosis_edge';loc=r['expected'][key]
    type_correct=fp==truth;location_correct=loc_ok(loc,lp,truth)
    return dict(episode_seed=r['episode_seed'],truth_type=truth,truth_location=loc,predicted_type=fp,predicted_location=lp,
                type_correct=type_correct,location_correct=location_correct,exact=type_correct and location_correct,
                oracle_location_correct=loc_ok(loc,oracle,truth))

def evaluate(model,tok,rr,permutations):
    items,lengths=checked_items(tok,rr,True);canonical=[];raw=[];timings=[]
    type_records=[[] for _ in range(24)];location_records=[[],[],[]]
    orders=list(itertools.permutations(range(4)))
    start=time.perf_counter()
    for index,(r,its) in enumerate(zip(rr,items)):
        t=time.perf_counter();keys={q:list(v['criteria']) for q,v in r['questions'].items()}
        preds={k:predict(model,tok,it,keys[k]) for k,it in its.items()}
        fp=preds['fault']['prediction'];branch='diagnosis_node' if fp in NODE_FAULTS else 'diagnosis_edge'
        truth_branch='diagnosis_node' if r['expected']['fault'] in NODE_FAULTS else 'diagnosis_edge'
        oracle=preds[truth_branch]['prediction']
        rec=record(r,fp,preds[branch]['prediction'],oracle);canonical.append(rec)
        timings.append(time.perf_counter()-t)
        raw_case=dict(episode_seed=r['episode_seed'],canonical=preds,type_orders=[],location_orders={})
        if permutations:
            for j,order in enumerate(orders):
                pr=preds['fault'] if j==0 else predict(model,tok,its['fault'],keys['fault'],order)
                ft=pr['prediction'];bk='diagnosis_node' if ft in NODE_FAULTS else 'diagnosis_edge'
                type_records[j].append(record(r,ft,preds[bk]['prediction'],oracle));raw_case['type_orders'].append(pr)
            for k in ['diagnosis_node','diagnosis_edge']:
                n=len(keys[k]);ll=[list(range(n)),list(reversed(range(n))),list(range(1,n))+[0]]
                raw_case['location_orders'][k]=[preds[k]]+[predict(model,tok,its[k],keys[k],o) for o in ll[1:]]
            for j in range(3):
                lp=raw_case['location_orders'][branch][j]['prediction']
                op=raw_case['location_orders'][truth_branch][j]['prediction']
                location_records[j].append(record(r,fp,lp,op))
        raw.append(raw_case)
        if index%16==15:print('EVAL CASES',index+1,len(rr),flush=True)
    output=dict(metrics=metrics(canonical),records=canonical,raw=raw,seconds=time.perf_counter()-start,
                canonical_three_question_mean_seconds=float(np.mean(timings)),
                canonical_three_question_p95_seconds=float(np.percentile(timings,95)),
                sequence_tokens={'min':min(lengths),'max':max(lengths)},type_orders=[],location_orders=[])
    for kind,collection in [('type_orders',type_records),('location_orders',location_records)]:
        if not permutations:break
        for j,recs in enumerate(collection):
            output[kind].append(dict(order=list(orders[j]) if kind=='type_orders' else ['canonical','reverse','rotate1'][j],
                metrics=metrics(recs),prediction_disagreement=sum((r['predicted_type'],r['predicted_location'])!=(c['predicted_type'],c['predicted_location']) for r,c in zip(recs,canonical))/len(rr)))
    return output

def evaluate_all():
    frozen=read(OUT/'training_complete.json')
    for run,h in frozen['checkpoints'].items():assert sha(OUT/run/'trainable.pt')==h
    if not (OUT/'evaluation_opened.json').exists():dump(OUT/'evaluation_opened.json',{'opened_unix':time.time(),'frozen_training_manifest_sha256':sha(OUT/'training_complete.json')})
    for arm,seed in [('table','base'),('prose','base')]+RUNS:
        target=OUT/f'{arm}_{seed}';target.mkdir(exist_ok=True)
        if (target/'evaluation.json').exists():continue
        model,tok=load()
        if seed!='base':
            weights=torch.load(target/'trainable.pt',weights_only=True,map_location='cpu')
            incompatible=model.load_state_dict(weights,strict=False);assert not incompatible.unexpected_keys
            del weights
        torch.cuda.reset_peak_memory_stats()
        with Monitor() as monitor:
            train_result=evaluate(model,tok,rows(arm,'train'),False)
            test_result=evaluate(model,tok,rows(arm,'test'),True)
        result=dict(status='completed',arm=arm,seed=seed,train=train_result,test=test_result,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_samples=monitor.samples)
        dump(target/'evaluation.json',result)
        print('EVAL COMPLETE',arm,seed,test_result['metrics'],flush=True)
        del model,tok;gc.collect();torch.cuda.empty_cache()
    dump(OUT/'evaluation_complete.json',{'status':'completed','completed_unix':time.time()})

def restore():
    checks=[]
    for arm,seed in RUNS:
        target=OUT/f'{arm}_{seed}';saved=read(target/'evaluation.json')['test']
        model,tok=load();weights=torch.load(target/'trainable.pt',weights_only=True,map_location='cpu')
        result=model.load_state_dict(weights,strict=False);assert not result.unexpected_keys
        actual=evaluate(model,tok,rows(arm,'test'),False);errors=[]
        for a,b in zip(actual['raw'],saved['raw']):
            for q in a['canonical']:
                errors.append(float(np.max(np.abs(np.asarray(a['canonical'][q]['logits'])-np.asarray(b['canonical'][q]['logits'])))))
        assert max(errors)<1e-4 and actual['records']==saved['records']
        checks.append(dict(arm=arm,seed=seed,max_logit_error=max(errors),metrics=actual['metrics']))
        del model,tok,weights;gc.collect();torch.cuda.empty_cache()
    dump(OUT/'restoration_audit.json',{'status':'completed_verified','checks':checks})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train','evaluate','restore']);args=ap.parse_args()
    torch.set_num_threads(4)
    {'train':train_all,'evaluate':evaluate_all,'restore':restore}[args.stage]()

if __name__=='__main__':main()
