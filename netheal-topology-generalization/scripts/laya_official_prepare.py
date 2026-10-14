import json,random,time,collections
from pathlib import Path
import torch
from transformers import AutoTokenizer
from laya_official_data import OUT,ROOT,read,state,save,public_audit
from laya_official_input import prepared
import jev_official_eval_v4 as envrun

BASE=ROOT/'.runtime/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148'
def main():
    target=OUT/'prepared';target.mkdir(exist_ok=False)
    audit=read(OUT/'data_audit.json');assert audit['status']=='completed'
    cases={}
    for p in sorted((OUT/'cases').glob('*.json')):
        assert envrun.sha(p)==audit['case_hashes'][p.name]
        c=read(p)
        if c['split']=='train':cases[c['seed']]=c
    buckets={k:[] for k in ['tool','argument','fault','location']}
    for seed,c in cases.items():
        for qi,q in enumerate(c['questions']):buckets[q['category']].append((seed,qi))
    rng=random.Random(1012);queues={k:[] for k in buckets};schedule=[]
    for i in range(2048):
        category=list(buckets)[i%4]
        if not queues[category]:queues[category]=buckets[category][:];rng.shuffle(queues[category])
        seed,qi=queues[category].pop();schedule.append({'seed':seed,'question':qi,'category':category})
    tok=AutoTokenizer.from_pretrained(str(BASE/'tokenizer'),local_files_only=True)
    unique=list(dict.fromkeys((s['seed'],s['question']) for s in schedule));records={};lengths=[];counts=[];start=time.time()
    for i,(seed,qi) in enumerate(unique):
        c=cases[seed];q=c['questions'][qi];s=state(c,q['turn']);public_audit(s)
        parts=prepared(tok,s,q['purpose'],q['options'],q['answer']);key=f'{seed}_{qi}'
        records[key]={'seed':seed,'question':qi,'category':q['category'],'parts':parts}
        lengths.extend(p['tokens'] for p in parts);counts.append(len(parts))
        if (i+1)%25==0:print('PREPARED',i+1,len(unique),'parts',sum(counts),'seconds',round(time.time()-start),flush=True)
    torch.save(records,target/'items.pt')
    save(target/'schedule.json',schedule)
    actual=[{'seed':c['seed'],'requested_n':c['n'],'actual_n':c['trace'][0]['result']['info']['network_size'],'teacher_success':c['teacher_result']['diagnosis_success']} for c in [read(p) for p in sorted((OUT/'cases').glob('*.json'))]]
    save(target/'audit.json',{'status':'completed','updates':len(schedule),'category_updates':dict(collections.Counter(s['category'] for s in schedule)),
       'unique_questions':len(unique),'available_questions':{k:len(v) for k,v in buckets.items()},'parts':sum(counts),'max_parts':max(counts),'tokens_min':min(lengths),'tokens_max':max(lengths),
       'items_sha256':envrun.sha(target/'items.pt'),'schedule_sha256':envrun.sha(target/'schedule.json'),'seconds':time.time()-start,'actual_network_sizes':actual})
    print('PREPARATION COMPLETE',flush=True)

if __name__=='__main__':main()
