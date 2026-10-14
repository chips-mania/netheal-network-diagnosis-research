import gc,hashlib,json,time,threading,subprocess,random
from pathlib import Path
import torch,psutil,bitsandbytes as bnb
from laya.train import load_checkpoint,items_from_rows,encode_item,collate_items,_forward,soft_ce_loss,to_internal
from laya.common import build_sequence

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'netheal-topology-generalization/results/laya_official_resource_long_20261012_v1'
BASE=ROOT/'.runtime/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148'
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def groups(m):return [{'params':[p for n,p in m.named_parameters() if n.startswith('encoder.')],'lr':2.5e-5},{'params':[p for n,p in m.named_parameters() if not n.startswith('encoder.')],'lr':1e-4}]
def part(n):return '.'.join(n.split('.')[:3]) if n.startswith('encoder.layers.') else ('encoder.embeddings' if n.startswith('encoder.embeddings.') else 'head' if not n.startswith('encoder.') else 'encoder.other')
def cpu_copy(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu_copy(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu_copy(x) for x in v]
    return v
def same(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def main():
    OUT.mkdir(exist_ok=False)
    spec=ROOT/'docs/specs/laya_official_long_resource_probe_20261012.md'
    (OUT/'spec.md').write_bytes(spec.read_bytes());(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    save('protocol.json',{'script_sha256':sha(Path(__file__)),'spec_sha256':sha(spec),'updates':8,'seed':1012,'dtype':'float32','optimizer':'AdamW8bit','max_len':4096,'head_max_len':1024,'bnb_version':bnb.__version__,'torch_version':torch.__version__})
    torch.set_num_threads(4);torch.manual_seed(1012);random.seed(1012)
    log=[];samples=[];stop=threading.Event();start=time.perf_counter()
    def poll():
        while not stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,creationflags=0x08000000)
                util,mem=map(float,raw.strip().split(','));samples.append({'unix':time.time(),'gpu_util':util,'device_MiB':mem,'ram_rss':psutil.Process().memory_info().rss})
            except Exception:pass
            stop.wait(1)
    thread=threading.Thread(target=poll,daemon=True);thread.start()
    stage='load';completed=0
    try:
        model,tok,_=load_checkpoint(str(BASE));model.to('cuda').train()
        for p in model.parameters():p.requires_grad_(True)
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.head_checkpointing=True
        source=ROOT/'netheal-topology-generalization/results/laya_observation_controls_v1/prose_train.jsonl'
        rows=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()][:4]
        for row in rows: row['state']='\nRepeated public observation:\n'.join([row['state']]*4)
        lengths=[]
        for row in rows:
            for q in row['expected']:
                ids,markers,stats,cut=build_sequence(tok,row['state'],to_internal(q,row['questions'][q]),4096,1024,return_stats=True,return_truncation_stats=True)
                assert not cut['truncated'] and len(markers)==stats['options_distinct']==len(row['questions'][q]['criteria'])
                lengths.append(len(ids))
        items,skip=items_from_rows(tok,rows,4096,1024);assert len(items)==8 and not any(skip.values())
        before={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        optimizer=bnb.optim.AdamW8bit(groups(model),weight_decay=.01)
        save('model.json',{'parameters':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'sequence_lengths':lengths,'data_sha256':sha(source),'block_count':len(model.encoder.layers)})
        torch.cuda.reset_peak_memory_stats();gradient_parts={}
        for i,item in enumerate(items):
            stage='backward';t=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            order=list(range(item['k']));random.shuffle(order)
            batch=collate_items([[encode_item(tok,item,4096,1024,order)]],tok.pad_token_id)
            logits=_forward(model,batch,torch.device('cuda'),False,False)
            loss=soft_ce_loss(logits,batch['target'].cuda(),batch['marker_mask'].cuda());assert torch.isfinite(loss)
            loss.backward()
            for n,p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(),n
                    gradient_parts[part(n)]=max(gradient_parts.get(part(n),0),float(p.grad.abs().max()))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False);assert torch.isfinite(norm)
            stage='optimizer_step';optimizer.step();completed+=1;torch.cuda.synchronize()
            log.append({'update':completed,'loss':float(loss.detach()),'grad_norm':float(norm),'seconds':time.perf_counter()-t})
            save('updates.json',log);print(json.dumps(log[-1]),flush=True)
        optimizer.zero_grad(set_to_none=True)
        changes={}
        for n,p in model.named_parameters(): changes[part(n)]=max(changes.get(part(n),0),float((p.detach().cpu()-before[n]).abs().max()))
        for i in range(len(model.encoder.layers)):
            assert gradient_parts[f'encoder.layers.{i}']>0 and changes[f'encoder.layers.{i}']>0
        assert changes['head']>0 and gradient_parts['head']>0
        stage='save';model.eval()
        test=collate_items([[encode_item(tok,items[0],4096,1024)]],tok.pad_token_id)
        with torch.no_grad():expected=_forward(model,test,torch.device('cuda'),False,False).cpu()
        weights=cpu_copy(model.state_dict());opt_state=cpu_copy(optimizer.state_dict())
        torch.save(weights,OUT/'model.pt');torch.save(opt_state,OUT/'optimizer.pt')
        peak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
        del before,model,optimizer,weights,opt_state,batch,logits,loss;gc.collect();torch.cuda.empty_cache()
        stage='restore';model,tok,_=load_checkpoint(str(BASE));model.load_state_dict(torch.load(OUT/'model.pt',map_location='cpu',weights_only=True),strict=True);model.to('cuda').eval()
        optimizer=bnb.optim.AdamW8bit(groups(model),weight_decay=.01)
        saved_opt=torch.load(OUT/'optimizer.pt',map_location='cpu',weights_only=True);optimizer.load_state_dict(saved_opt)
        assert same(saved_opt,optimizer.state_dict())
        with torch.no_grad():actual=_forward(model,test,torch.device('cuda'),False,False).cpu()
        error=float((actual-expected).abs().max());assert error<1e-4
        save('result.json',{'status':'completed_verified','updates':completed,'seconds':time.perf_counter()-start,'peak_gpu_bytes':peak,'gradient_max_by_part':gradient_parts,'weight_change_by_part':changes,'restore_logit_error':error,'optimizer_state_equal':True,'model_sha256':sha(OUT/'model.pt'),'optimizer_sha256':sha(OUT/'optimizer.pt')})
        print('RESOURCE PROBE PASSED',flush=True)
    except Exception as e:
        save('result.json',{'status':'failed','stage':stage,'updates_completed':completed,'error_type':type(e).__name__,'error':str(e),'seconds':time.perf_counter()-start,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
        raise
    finally:
        stop.set();thread.join();save('resource_samples.json',samples)

if __name__=='__main__':main()
