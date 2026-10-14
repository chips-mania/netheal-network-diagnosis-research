"""Reload every saved parameter delta and reproduce held-out canonical logits."""
import json,gc,sys,importlib.metadata
import numpy as np
import torch
from laya.train import load_checkpoint,items_from_rows
from laya_learning_controls import BASE,dump
from laya_optimization_controls import OUT,score

def main():
    summary=json.loads((OUT/'summary.json').read_text());assert len(summary['runs'])==12
    rows=[json.loads(l) for l in (OUT/'validation.jsonl').read_text().splitlines()]
    torch.set_num_threads(4)
    model,tok,_=load_checkpoint(str(BASE));model.to('cuda').eval()
    items,skips=items_from_rows(tok,rows,512,512);assert not skips
    expected_keys=None;checks=[]
    for run in summary['runs']:
        target=OUT/f'{run["arm"]}_{run["seed"]}'
        delta=torch.load(target/'trainable.pt',weights_only=True,map_location='cpu')
        if expected_keys is None:expected_keys=set(delta)
        assert set(delta)==expected_keys
        incompatible=model.load_state_dict(delta,strict=False)
        assert not incompatible.unexpected_keys
        assert all(n.startswith('encoder.') or n=='temperature' for n in incompatible.missing_keys)
        actual=score(model,tok,items);expected=run['final']['validation']
        error=float(np.max(np.abs(np.asarray(actual['logits'])-np.asarray(expected['logits']))))
        assert actual['accuracy']==expected['accuracy'] and error<1e-4
        checks.append(dict(arm=run['arm'],seed=run['seed'],accuracy=actual['accuracy'],max_logit_difference=error))
        del delta;gc.collect()
    dump(OUT/'restoration_audit.json',dict(status='completed_verified',runs=checks,
        environment=dict(python=sys.version,packages={n:importlib.metadata.version(n) for n in ['laya','torch','transformers','numpy']},
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))))
    print(json.dumps(dict(status='completed_verified',restored_runs=len(checks))))

if __name__=='__main__':main()
