"""Check that head training changed head weights and kept the encoder fixed."""
import json,hashlib
import numpy as np
from pathlib import Path
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'.runtime/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148/model.safetensors'
OUT=ROOT/'netheal-topology-generalization/results/laya_head_compact_48ep_v2'

def main():
    changed=[];encoder_keys=0;head_keys=0
    with safe_open(BASE,framework='numpy') as a,safe_open(OUT/'checkpoint/model.safetensors',framework='numpy') as b:
        assert set(a.keys())==set(b.keys())
        for k in a.keys():
            x=a.get_tensor(k);y=b.get_tensor(k)
            assert x.shape==y.shape
            assert np.isfinite(x).all() and np.isfinite(y).all(),k
            if not (x==y).all():changed.append(k)
            if k.startswith('encoder.'):encoder_keys+=1
            else:head_keys+=1
    assert changed and not any(k.startswith('encoder.') for k in changed)
    result=dict(status='passed',encoder_tensors=encoder_keys,head_tensors=head_keys,changed_head_tensors=len(changed),
        encoder_unchanged=True,warning='Verifies the saved fp16 checkpoint tensors, not calibrated accuracy.')
    (OUT/'checkpoint_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))
if __name__=='__main__':main()
