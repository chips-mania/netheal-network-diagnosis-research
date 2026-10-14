"""Predeclared small-set learning controls, not a network benchmark."""
import sys,json,time,random,hashlib,gc,math
from pathlib import Path
from collections import Counter
from dataclasses import asdict
import numpy as np
import torch
from laya.train import load_checkpoint,items_from_rows,TrainConfig,train_model,calibration_records,to_internal
from laya.common import build_sequence

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parent
OUT=PROJECT/'results/laya_learning_controls_20261010'
BASE=ROOT/'.runtime/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148'
DATA=PROJECT/'results/laya_data_compact_v2'
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def select(split,per_fault):
    rows=[json.loads(l) for l in (DATA/(split+'.jsonl')).read_text().splitlines()]
    rows=[r for r in rows if r['expected']['kind']=='diagnose']
    selected=[]
    for ft in sorted({r['expected']['fault'] for r in rows}):
        group=sorted([r for r in rows if r['expected']['fault']==ft],key=lambda r:r['episode_seed'])
        random.Random(710).shuffle(group);selected.extend(group[:per_fault])
    return selected
def convert(rows,arm):
    return [dict(state=(('CONTROL ONLY: correct fault type = '+r['expected']['fault']+'.\n') if arm=='label_tag' else '')+r['state'],
        questions={'fault':r['questions']['fault']},expected={'fault':r['expected']['fault']},episode_seed=r['episode_seed']) for r in rows]
def score(model,tok,items):
    records=calibration_records(model,tok,items,torch.device('cuda'),4096,1024,batch_size=1)
    pred=[int(np.argmax(r[1])) for r in records];truth=[int(np.argmax(r[2])) for r in records]
    confusion=[[sum(t==i and p==j for t,p in zip(truth,pred)) for j in range(4)] for i in range(4)]
    return dict(accuracy=sum(a==b for a,b in zip(pred,truth))/len(pred),predicted_indices=pred,
        target_indices=truth,confusion_matrix=confusion,prediction_counts=dict(Counter(pred)))
def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4)
    train=select('train',4);valid=select('validation',8)
    assert len(train)==16 and len(valid)==32
    assert not {r['episode_seed'] for r in train}&{r['episode_seed'] for r in valid}
    protocol=dict(created_on='2026-10-10',arms=['label_tag','real_table'],seeds=[41,52,63],epochs=32,
        train_episodes=16,validation_episodes=32,task='fault type only on final expert-collected evidence',
        optimizer_updates=512,evaluation_epochs=[0,4,8,16,32],freeze_encoder=True,loss='rlcd',
        thresholds={'positive_control_train_and_validation':0.95,'memorization_train':0.95},
        warning='Label-tag arm deliberately reveals truth in BOTH train and validation. Never count it as diagnosis performance.',
        training_seeds=[r['episode_seed'] for r in train],validation_seeds=[r['episode_seed'] for r in valid],
        data_hashes={s:sha(DATA/(s+'.jsonl')) for s in ['train','validation']},script_sha256=sha(Path(__file__)))
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text(encoding='utf8'))==protocol
    else:dump(OUT/'protocol.json',protocol)
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    for arm in protocol['arms']:
        tr=convert(train,arm);va=convert(valid,arm)
        for split,rows in [('train',tr),('validation',va)]:
            (OUT/(arm+'_'+split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
        for seed in protocol['seeds']:
            target=OUT/f'{arm}_{seed}'
            if (target/'summary.json').exists():continue
            target.mkdir(exist_ok=True)
            model,tok,cfg=load_checkpoint(str(BASE));model.to('cuda')
            for r in tr+va:
                q=r['questions']['fault']
                _,markers,st,cut=build_sequence(tok,r['state'],to_internal('fault',q),4096,1024,
                    return_stats=True,return_truncation_stats=True)
                assert not cut['truncated'] and len(markers)==4 and st['options_distinct']==4
            ti,skip=items_from_rows(tok,tr,4096,1024);vi,vskip=items_from_rows(tok,va,4096,1024)
            assert not any(skip.values()) and not any(vskip.values())
            curves=[]
            def evaluate(epoch,loss=None):
                if epoch not in protocol['evaluation_epochs']:return
                row=dict(epoch=epoch,optimizer_updates=epoch*len(ti),loss=loss,train=score(model,tok,ti),validation=score(model,tok,vi))
                curves.append(row);dump(target/'curve.json',curves)
                print(json.dumps(dict(arm=arm,seed=seed,epoch=epoch,train=row['train']['accuracy'],validation=row['validation']['accuracy'])),flush=True)
                model.train();model.encoder.eval()
            evaluate(0)
            tc=TrainConfig(epochs=32,micro_batch=1,grad_accum=1,freeze_encoder=True,loss='rlcd',
                seed=seed,amp=False,gradient_checkpointing=False,max_len=4096,head_max_len=1024,
                shuffle_options=('choice',),calib_frac=0,calib_max=0,log_every=0)
            dump(target/'config.json',asdict(tc));torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            history=train_model(model,tok,ti,tc,torch.device('cuda'),4096,1024,on_epoch_end=lambda epoch,loss:evaluate(epoch+1,loss))
            torch.cuda.synchronize()
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('encoder.')},target/'head.pt')
            final=curves[-1]
            dump(target/'summary.json',dict(status='completed',arm=arm,seed=seed,seconds=time.perf_counter()-start,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),final=final,loss_history=history,
                passed=final['train']['accuracy']>=.95 and (arm!='label_tag' or final['validation']['accuracy']>=.95)))
            del model,tok,ti,vi;gc.collect();torch.cuda.empty_cache()
    summaries=[json.loads(p.read_text()) for p in sorted(OUT.glob('*/summary.json'))]
    assert len(summaries)==6
    dump(OUT/'summary.json',dict(status='completed',runs=summaries,api_cost_usd=0))
if __name__=='__main__':main()
