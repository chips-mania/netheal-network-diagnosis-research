"""Batch only option-order probes; canonical inference and routing are unchanged."""
import itertools, json, sys
import numpy as np
import torch
import laya_observation_controls as controls
from laya_observation_data import OUT, dump, sha

original_predict=controls.predict
original_evaluate=controls.evaluate
cache={}
errors=[]

@torch.no_grad()
def batch_orders(model,tok,item,keys,orders):
    results=[];model.eval()
    for start in range(0,len(orders),4):
        chunk=orders[start:start+4]
        batch=controls.collate_items([[controls.encode_item(tok,item,4096,1024,list(o)) for o in chunk]],tok.pad_token_id)
        logits=controls._forward(model,batch,controls.DEVICE,False,False).cpu().numpy()
        for order,z in zip(chunk,logits):
            z=z[:len(keys)];assert np.isfinite(z).all()
            results.append(dict(prediction=keys[order[int(np.argmax(z))]],logits=z.tolist(),order=list(order)))
    return results

def predict(model,tok,item,keys,order=None):
    key=id(item)
    if order is None:
        result=original_predict(model,tok,item,keys)
        cache[key]={'canonical':result}
        return result
    if 'orders' not in cache[key]:
        n=len(keys)
        orders=(list(itertools.permutations(range(4))) if set(keys)==set(controls.FAULTS)
                else [list(range(n)),list(reversed(range(n))),list(range(1,n))+[0]])
        results=batch_orders(model,tok,item,keys,orders)
        difference=float(np.max(np.abs(np.asarray(results[0]['logits'])-np.asarray(cache[key]['canonical']['logits']))))
        assert difference<1e-4 and results[0]['prediction']==cache[key]['canonical']['prediction']
        errors.append(difference)
        cache[key]['orders']={tuple(r['order']):r for r in results}
    return cache[key]['orders'][tuple(order)]

def evaluate(model,tok,rr,permutations):
    cache.clear();errors.clear()
    result=original_evaluate(model,tok,rr,permutations)
    result['order_probe_batch_size']=4
    result['batched_canonical_max_logit_difference']=max(errors,default=0.)
    result['batched_canonical_comparisons']=len(errors)
    return result

def conformance():
    assert (OUT/'training_complete.json').exists()
    assert not (OUT/'evaluation_opened.json').exists()
    model,tok=controls.load();checks=[]
    for arm in ['table','prose']:
        rr=controls.rows(arm,'train')[:1]
        items,_=controls.checked_items(tok,rr,True)
        for q,item in items[0].items():
            keys=list(rr[0]['questions'][q]['criteria']);n=len(keys)
            orders=list(itertools.permutations(range(4))) if q=='fault' else [list(range(n)),list(reversed(range(n))),list(range(1,n))+[0]]
            batched=batch_orders(model,tok,item,keys,orders)
            differences=[]
            for order,actual in zip(orders,batched):
                expected=original_predict(model,tok,item,keys,order)
                error=float(np.max(np.abs(np.asarray(actual['logits'])-np.asarray(expected['logits']))))
                assert error<1e-4 and expected['prediction']==actual['prediction']
                differences.append(error)
            checks.append(dict(arm=arm,question=q,orders=len(orders),max_logit_difference=max(differences)))
    dump(OUT/'batch_conformance.json',{'status':'completed_verified','data_split':'train_only','batch_size':4,
        'checks':checks,'source_sha256':sha(__import__('pathlib').Path(__file__))})
    print(json.dumps(checks),flush=True)

def main():
    torch.set_num_threads(4)
    if sys.argv[1]=='conformance':return conformance()
    assert json.loads((OUT/'batch_conformance.json').read_text())['status']=='completed_verified'
    controls.predict=predict;controls.evaluate=evaluate
    controls.main()

if __name__=='__main__':main()
