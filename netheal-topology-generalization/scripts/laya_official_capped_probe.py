import json,time
import torch,bitsandbytes as bnb
from laya.train import load_checkpoint,encode_item,collate_items,_forward,soft_ce_loss
from laya_official_data import ROOT,OUT,save
from laya_official_prepare import BASE
from laya_official_resource_probe import groups

def main():
    target=OUT/'capped_probe';target.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(1012);torch.cuda.set_per_process_memory_fraction(.88)
    records=torch.load(OUT/'prepared/items.pt',map_location='cpu',weights_only=False)
    candidates=sorted((p for r in records.values() for p in r['parts']),key=lambda p:p['tokens'],reverse=True)[:3]
    del records
    model,tok,_=load_checkpoint(str(BASE));model.cuda().train();model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.head_checkpointing=True
    optimizer=bnb.optim.AdamW8bit(groups(model),weight_decay=.01);log=[]
    torch.cuda.reset_peak_memory_stats();start=time.time()
    try:
        for i,p in enumerate(candidates):
            optimizer.zero_grad(set_to_none=True);t=time.time()
            b=collate_items([[encode_item(tok,p['item'],4096,1024)]],tok.pad_token_id)
            z=_forward(model,b,torch.device('cuda'),False,False);loss=soft_ce_loss(z,b['target'].cuda(),b['marker_mask'].cuda());assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False);assert torch.isfinite(norm)
            optimizer.step();torch.cuda.synchronize();log.append({'update':i+1,'tokens':p['tokens'],'loss':float(loss.detach()),'seconds':time.time()-t});print(log[-1],flush=True)
        result={'status':'passed','updates':log}
    except Exception as ex:result={'status':'failed','updates':log,'error_type':type(ex).__name__,'error':str(ex)}
    result.update(seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),allocator_fraction=.88)
    save(target/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
