"""Predeclared one-factor learning diagnostics; synthetic signals, not diagnosis."""
import gc, json, time, itertools
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from laya.train import (load_checkpoint,items_from_rows,TrainConfig,train_model,
    calibration_records,evaluate_records,encode_item,collate_items,_forward)
from laya_learning_controls import BASE, PROJECT, dump, sha

OUT=PROJECT/'results/laya_optimization_controls_v1'
SOURCE=PROJECT/'results/laya_symbol_controls_v1'
ARMS={
    'reference':dict(epochs=16,head_lr=1e-4,shuffle_options=('choice',)),
    'head_lr_1e3':dict(epochs=16,head_lr=1e-3,shuffle_options=('choice',)),
    'updates_1024':dict(epochs=64,head_lr=1e-4,shuffle_options=('choice',)),
    'fixed_order':dict(epochs=16,head_lr=1e-4,shuffle_options=()),
}

def score(model,tok,items):
    rec=calibration_records(model,tok,items,torch.device('cuda'),512,512,batch_size=1)
    result=evaluate_records(rec)
    result['logits']=[r[1].tolist() for r in rec]
    result['targets']=[r[2] for r in rec]
    return result

@torch.no_grad()
def permutation_score(model,tok,items):
    model.eval(); results=[]; canonical=[]
    # All 24 permutations; map slot predictions back to semantic option IDs.
    for order in itertools.permutations(range(4)):
        predictions=[]; gold=[]
        for item in items:
            b=collate_items([[encode_item(tok,item,512,512,list(order))]],tok.pad_token_id)
            z=_forward(model,b,torch.device('cuda'),False,False)[0].cpu().numpy()
            predictions.append(order[int(np.argmax(z))]); gold.append(int(np.argmax(item['target'])))
        if order==(0,1,2,3): canonical=predictions
        results.append(dict(order=order,accuracy=sum(a==b for a,b in zip(predictions,gold))/len(gold),
            disagreement_with_canonical=sum(a!=b for a,b in zip(predictions,canonical))/len(gold)))
    return results

def main():
    OUT.mkdir(exist_ok=True); torch.set_num_threads(4)
    protocol=dict(arms=ARMS,seeds=[41,52,63],train_n=16,validation_n=32,
        encoder='last block only',encoder_lr=2.5e-5,loss='soft-ce',threshold=.95,
        comparison='one factor at a time against reference; no interaction inference',
        budget_caveat='Cosine scheduler stretches with epochs; longer run changes the LR trajectory too.',
        tests=['canonical accuracy/CE/Brier/ECE','all 24 option orders','signal removal with same case IDs'],
        data_hashes={s:sha(SOURCE/f'{s}.jsonl') for s in ['train','validation']},
        base_sha256=sha(BASE/'model.safetensors'),script_sha256=sha(Path(__file__)),
        warning='Synthetic association and reused development split; NOT network diagnosis or generalization performance.')
    # Normalize tuples to JSON arrays for stable resumability.
    protocol=json.loads(json.dumps(protocol))
    if (OUT/'protocol.json').exists(): assert json.loads((OUT/'protocol.json').read_text())==protocol
    else: dump(OUT/'protocol.json',protocol)
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    rows={}
    for split in ['train','validation']:
        raw=(SOURCE/f'{split}.jsonl').read_bytes(); (OUT/f'{split}.jsonl').write_bytes(raw)
        rows[split]=[json.loads(l) for l in raw.decode().splitlines()]
    ablated=[dict(r,state=f"Case {r['episode_seed']}. Signal = unknown.") for r in rows['validation']]
    for arm,settings in ARMS.items():
        for seed in protocol['seeds']:
            target=OUT/f'{arm}_{seed}';target.mkdir(exist_ok=True)
            if (target/'summary.json').exists():continue
            model,tok,_=load_checkpoint(str(BASE));model.to('cuda')
            for p in model.encoder.parameters():p.requires_grad_(False)
            for p in model.encoder.layers[-1].parameters():p.requires_grad_(True)
            data={}
            for split,rr in dict(rows,ablated=ablated).items():
                data[split],skips=items_from_rows(tok,rr,512,512)
                assert not any(skips.values())
            trainable={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
            dump(target/'trainable.json',dict(names=list(trainable),count=sum(v.numel() for v in trainable.values())))
            curves=[]
            def evaluate(epoch):
                if epoch not in [0,4,8,16,32,64]:return
                row=dict(epoch=epoch,updates=epoch*16,train=score(model,tok,data['train']),validation=score(model,tok,data['validation']))
                curves.append(row);dump(target/'curve.json',curves)
                print(json.dumps(dict(arm=arm,seed=seed,epoch=epoch,train=row['train']['accuracy'],validation=row['validation']['accuracy'])),flush=True)
                model.train()
            evaluate(0)
            cfg=TrainConfig(**settings,micro_batch=1,grad_accum=1,freeze_encoder=False,
                loss='soft-ce',seed=seed,amp=False,gradient_checkpointing=False,
                max_len=512,head_max_len=512,calib_frac=0,calib_max=0,log_every=0)
            dump(target/'config.json',asdict(cfg));torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            history=train_model(model,tok,data['train'],cfg,torch.device('cuda'),512,512,on_epoch_end=lambda e,l:evaluate(e+1))
            torch.cuda.synchronize();training_seconds=time.perf_counter()-start
            perm=permutation_score(model,tok,data['validation']); removal=score(model,tok,data['ablated'])
            learned={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
            changes={n:float((v-trainable[n]).abs().max()) for n,v in learned.items()}
            torch.save(learned,target/'trainable.pt')
            final=curves[-1]
            dump(target/'summary.json',dict(status='completed',arm=arm,seed=seed,final=final,
                passed=all(final[s]['accuracy']>=.95 for s in ['train','validation']),
                training_and_eval_seconds=training_seconds,total_seconds=time.perf_counter()-start,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),loss_history=history,
                parameter_changes=changes,permutations=perm,signal_removed=removal,
                checkpoint_sha256=sha(target/'trainable.pt')))
            del model,tok,data,trainable,learned;gc.collect();torch.cuda.empty_cache()
    runs=[json.loads(p.read_text()) for p in sorted(OUT.glob('*/summary.json'))]
    assert len(runs)==12
    dump(OUT/'summary.json',dict(status='completed',runs=runs,api_cost_usd=0))

if __name__=='__main__':main()
