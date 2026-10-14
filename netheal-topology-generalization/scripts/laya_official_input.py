import json,hashlib
from laya.train import to_internal,items_from_rows
from laya.common import build_sequence
from jev_context_codec_v4 import encode_checked,dumps
from jev_tree_partition_v4 import restore
from jev_official_eval_v4 import INSTRUCTIONS

MAX_LEN=4096;HEAD_LEN=1024
def question(purpose,options):
    return {'type':'choice','instructions':INSTRUCTIONS+' '+purpose,'criteria':{f'c{i:03d}':v if isinstance(v,str) else json.dumps(v,ensure_ascii=False) for i,v in enumerate(options)}}
def checked(tok,text,q):
    ids,markers,stats,cut=build_sequence(tok,text,to_internal('decision',q),MAX_LEN,HEAD_LEN,return_stats=True,return_truncation_stats=True)
    return not cut['truncated'] and len(markers)==len(q['criteria'])==stats['options_distinct'],len(ids)

def chunks(tok,state,purpose,options):
    q=question(purpose,options)
    def fits(s):return checked(tok,encode_checked(s),q)[0]
    if fits(state):return [(state,1)]
    history=state.get('history',[]);base={k:v for k,v in state.items() if k!='history'}
    base['latest_public_result']=next((e['result'] for e in reversed(history) if 'result' in e),None)
    def env(events,first):return dict(base,history=events,history_partition={'first':first,'count':len(events),'total':len(history)})
    result=[];current=[];first=0;possible=True
    for event in history:
        if fits(env(current+[event],first)):current.append(event);continue
        if current:result.append((env(current,first),len(current)));first+=len(current)
        current=[event]
        if not fits(env(current,first)):possible=False;break
    if possible and current:
        result.append((env(current,first),len(current)))
        assert json.dumps([e for c,w in result for e in c['history']],sort_keys=True)==json.dumps(history,sort_keys=True)
        return result
    common={k:state[k] for k in ['task_description','objective','hint','network_size','max_steps','solver_turn'] if k in state}
    def wrap(items):return dict(common,partition_notice='Lossless evidence partition. path locates the exact value in the original state. All partitions together preserve the complete evidence.',evidence_fragments=items)
    assert fits(wrap([])),'Question alone exceeds model head budget'
    fragments=[]
    def split(value,path):
        item={'path':path,'value':value}
        if fits(wrap([item])):fragments.append(item);return
        if isinstance(value,dict) and value:
            for k,v in value.items():split(v,path+[k])
        elif isinstance(value,list) and value:
            for i,v in enumerate(value):split(v,path+[i])
        else:raise ValueError('Indivisible public scalar exceeds context')
    split(state,[])
    assert json.dumps(restore(fragments),sort_keys=True)==json.dumps(state,sort_keys=True)
    result=[];current=[]
    def weight(xs):return sum(len(dumps(x['value']).encode()) for x in xs)
    for x in fragments:
        if current and not fits(wrap(current+[x])):result.append((wrap(current),weight(current)));current=[]
        current.append(x)
    if current:result.append((wrap(current),weight(current)))
    return result

def prepared(tok,state,purpose,options,answer=0):
    q=question(purpose,options);result=[]
    for c,w in chunks(tok,state,purpose,options):
        text=encode_checked(c);ok,n=checked(tok,text,q);assert ok
        row={'state':text,'questions':{'decision':q},'expected':{'decision':f'c{answer:03d}'}}
        items,skips=items_from_rows(tok,[row],MAX_LEN,HEAD_LEN);assert len(items)==1 and not any(skips.values())
        result.append({'item':items[0],'weight':w,'tokens':n,'state_sha256':hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()})
    return result
