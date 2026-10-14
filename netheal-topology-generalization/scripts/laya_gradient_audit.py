"""Gradient/update and train/eval execution-path checks, using synthetic tagged training data."""
import json,time
from pathlib import Path
import torch
from laya.train import load_checkpoint,items_from_rows,encode_item,collate_items,_forward,soft_ce_loss
from laya_learning_controls import BASE,OUT,dump

def main():
    torch.set_num_threads(4)
    model,tok,cfg=load_checkpoint(str(BASE));model.to('cuda').eval()
    for p in model.encoder.parameters():p.requires_grad_(False)
    rows=[json.loads(l) for l in (OUT/'label_tag_train.jsonl').read_text().splitlines()]
    items,skip=items_from_rows(tok,rows,4096,1024)
    item=items[4]
    batch=collate_items([[encode_item(tok,item,4096,1024)]],tok.pad_token_id)
    def infer(fast):
        torch.backends.mha.set_fastpath_enabled(fast)
        with torch.no_grad():return _forward(model,batch,torch.device('cuda'),False,True).detach().cpu()
    before_fast=infer(True);before_slow=infer(False)
    torch.backends.mha.set_fastpath_enabled(False)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([p for n,p in named],lr=1e-4)
    optimizer.zero_grad()
    logits=_forward(model,batch,torch.device('cuda'),False,True)
    target=batch['target'].to('cuda');mask=batch['marker_mask'].to('cuda')
    loss=soft_ce_loss(logits,target,mask);loss.backward()
    norms={n:float(p.grad.norm()) for n,p in named if p.grad is not None}
    assert norms and all(torch.isfinite(p.grad).all() for n,p in named if p.grad is not None)
    assert all(p.grad is None for p in model.encoder.parameters())
    chosen=max(norms,key=norms.get);parameter=dict(named)[chosen];old=parameter.detach().clone()
    torch.nn.utils.clip_grad_norm_([p for n,p in named],1.0);optimizer.step()
    changed=float((parameter-old).abs().max());after=infer(False)
    result=dict(loss=float(loss),nonzero_gradient_tensors=sum(v>0 for v in norms.values()),
        encoder_gradients_absent=True,gradient_norms=norms,checked_parameter=chosen,parameter_max_change=changed,
        logits_before=before_slow.tolist(),logits_after=after.tolist(),logit_max_change=float((after-before_slow).abs().max()),
        eval_fastpath_max_difference=float((before_fast-before_slow).abs().max()),target=batch['target'].tolist(),
        warning='One controlled soft-CE update checks mechanics, not useful task learning or RLCD convergence.')
    assert changed>0 and result['logit_max_change']>0
    dump(OUT/'gradient_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='gradient_norms'}),flush=True)
    torch.backends.mha.set_fastpath_enabled(True)
if __name__=='__main__':main()
