"""Short synthetic signal, matched frozen/full encoder learning diagnostic."""
import json, time, gc
from pathlib import Path
from dataclasses import asdict
import torch
from laya.train import load_checkpoint, items_from_rows, TrainConfig, train_model, calibration_records, evaluate_records
from laya.common import build_sequence
from laya.train import to_internal
from laya_learning_controls import BASE, PROJECT, select, dump, sha

OUT = PROJECT / 'results/laya_short_controls_v1'

def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    protocol = dict(arms=['frozen', 'full'], seeds=[41,52,63], epochs=16,
        updates=256, loss='soft-ce', max_len=512, head_max_len=512,
        train_n=16, validation_n=32, threshold=.95, script_sha256=sha(Path(__file__)),
        warning='Synthetic answer-exposed diagnostic, NOT network diagnosis accuracy. Different length and budget from prior controls.')
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text()) == protocol
    else:
        dump(OUT/'protocol.json', protocol)
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    splits = {}
    for split, count in [('train',4),('validation',8)]:
        rows = [dict(state=f"Case {r['episode_seed']}. Correct fault type = {r['expected']['fault']}.",
            questions={'fault':r['questions']['fault']}, expected={'fault':r['expected']['fault']},
            episode_seed=r['episode_seed']) for r in select(split,count)]
        splits[split]=rows
        (OUT/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            target=OUT/f'{arm}_{seed}'; target.mkdir(exist_ok=True)
            if (target/'summary.json').exists(): continue
            model,tok,_=load_checkpoint(str(BASE)); model.to('cuda')
            data={}
            for split,rows in splits.items():
                for r in rows:
                    _,markers,stats,cut=build_sequence(tok,r['state'],to_internal('fault',r['questions']['fault']),512,512,return_stats=True,return_truncation_stats=True)
                    assert not cut['truncated'] and len(markers)==4
                data[split],skips=items_from_rows(tok,rows,512,512)
                assert not any(skips.values())
            frozen=arm=='frozen'; curves=[]
            def evaluate(epoch):
                if epoch not in [0,4,8,16]: return
                row={'epoch':epoch}
                for split,items in data.items():
                    records=calibration_records(model,tok,items,torch.device('cuda'),512,512,batch_size=1)
                    row[split]=evaluate_records(records)
                    row[split]['logits']=[r[1].tolist() for r in records]
                curves.append(row); dump(target/'curve.json',curves)
                print(json.dumps(dict(arm=arm,seed=seed,epoch=epoch,train=row['train']['accuracy'],validation=row['validation']['accuracy'])),flush=True)
                model.train()
                if frozen: model.encoder.eval()
            evaluate(0)
            cfg=TrainConfig(epochs=16,micro_batch=1,grad_accum=1,freeze_encoder=frozen,
                loss='soft-ce',seed=seed,amp=False,gradient_checkpointing=False,
                max_len=512,head_max_len=512,shuffle_options=('choice',),calib_frac=0,calib_max=0,log_every=0)
            dump(target/'config.json',asdict(cfg)); torch.cuda.reset_peak_memory_stats()
            enc_name,enc_param=next(model.encoder.named_parameters())
            before=enc_param.detach().cpu().clone()
            start=time.perf_counter()
            history=train_model(model,tok,data['train'],cfg,torch.device('cuda'),512,512,on_epoch_end=lambda e,l:evaluate(e+1))
            torch.cuda.synchronize()
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('encoder.')},target/'head.pt')
            dump(target/'summary.json',dict(status='completed',arm=arm,seed=seed,final=curves[-1],
                seconds=time.perf_counter()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                encoder_parameter=enc_name,encoder_parameter_max_change=float((enc_param.detach().cpu()-before).abs().max()),
                loss_history=history,passed=all(curves[-1][s]['accuracy']>=.95 for s in splits)))
            del model,tok,data,before; gc.collect(); torch.cuda.empty_cache()
    runs=[json.loads(p.read_text()) for p in sorted(OUT.glob('*/summary.json'))]
    assert len(runs)==6
    dump(OUT/'summary.json',dict(status='completed',runs=runs,api_cost_usd=0))

if __name__=='__main__': main()
