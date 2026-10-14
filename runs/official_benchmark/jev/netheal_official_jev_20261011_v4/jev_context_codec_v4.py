"""Lossless, human-readable JSON compaction; never summarizes or rounds evidence."""
import collections
import copy
import json

def dumps(x):
    return json.dumps(x,ensure_ascii=False,separators=(',',':'))

def delta(before,after,path=None):
    path=[] if path is None else path
    if dumps(before)==dumps(after): return []
    replacement=[['set',path,after]]
    operations=[]
    if isinstance(before,dict) and isinstance(after,dict) and list(before)==list(after):
        for k in before: operations+=delta(before[k],after[k],path+[k])
    elif isinstance(before,list) and isinstance(after,list) and len(before)==len(after):
        n=len(before)
        for k in range(1,min(4,n)):
            if dumps(before[k:])==dumps(after[:-k]):
                return [['shift_left',path,k,after[-k:]]]
            if dumps(before[:-k])==dumps(after[k:]):
                return [['shift_right',path,k,after[:k]]]
        for i in range(n): operations+=delta(before[i],after[i],path+[i])
    else: return replacement
    return min([operations,replacement],key=lambda v:len(dumps(v)))

def apply_delta(value,operations):
    value=copy.deepcopy(value)
    for op,path,*args in operations:
        parent=value
        for part in path[:-1]: parent=parent[part]
        old=value if not path else parent[path[-1]]
        if op=='set': new=copy.deepcopy(args[0])
        elif op=='shift_left': new=old[args[0]:]+copy.deepcopy(args[1])
        elif op=='shift_right': new=copy.deepcopy(args[1])+old[:-args[0]]
        else: raise ValueError(op)
        if not path: value=new
        else: parent[path[-1]]=new
    return value

def temporal_pack(state):
    state=copy.deepcopy(state)
    if 'history' not in state: return {'base':state,'events':None}
    events=[]; snapshots=[]
    for event in state.pop('history'):
        if 'result' not in event:
            events.append({'notice':event}); continue
        result=copy.deepcopy(event['result'])
        # Explicit per-tool evidence remains easy to read; snapshot arrays are delta encoded.
        info=result.get('info',{})
        had_evidence='action_result' in info
        evidence=info.pop('action_result',None)
        events.append({'tool':event['tool'],'arguments':event['arguments'],
          'evidence':evidence,'had_evidence':had_evidence,'snapshot':len(snapshots),'event_extra':{k:v for k,v in event.items() if k not in ('tool','arguments','result')}})
        snapshots.append(result)
    reverse=[]
    for i in range(len(snapshots)-1,0,-1): reverse.append(delta(snapshots[i],snapshots[i-1]))
    return {'base':state,'events':events,'latest_snapshot':snapshots[-1] if snapshots else None,
            'reverse_snapshot_changes':reverse}

def temporal_unpack(packed):
    state=copy.deepcopy(packed['base'])
    if packed['events'] is None: return state
    reverse=[]
    if packed['latest_snapshot'] is not None:
        reverse=[packed['latest_snapshot']]
        for patch in packed['reverse_snapshot_changes']: reverse.append(apply_delta(reverse[-1],patch))
    snapshots=list(reversed(reverse)); events=[]
    for event in packed['events']:
        if 'notice' in event: events.append(event['notice']); continue
        result=copy.deepcopy(snapshots[event['snapshot']])
        if event['had_evidence']:
            result['info']['action_result']=event['evidence']
        events.append(dict(tool=event['tool'],arguments=event['arguments'],result=result,**event['event_extra']))
    state['history']=events
    return state

def pack_arrays(x):
    if isinstance(x,dict):
        assert not any(k.startswith('$') for k in x), 'reserved codec key in input'
        return {k:pack_arrays(v) for k,v in x.items()}
    if not isinstance(x,list): return x
    values=[pack_arrays(v) for v in x]
    candidates=[values]
    if len(x)>=3 and all(isinstance(v,dict) for v in x) and all(list(v)==list(x[0]) for v in x):
        columns=list(x[0])
        candidates.append({'$table':[columns,[[v[k] for k in columns] for v in values]]})
    if len(x)>=4 and all(type(v) is int for v in x):
        step=x[1]-x[0]
        if all(v==x[0]+i*step for i,v in enumerate(x)):
            candidates.append({'$range':[x[0],step,len(x)]})
    if len(x)>=3 and all(dumps(v)==dumps(x[0]) for v in x):
        candidates.append({'$repeat':[values[0],len(x)]})
    return min(candidates,key=lambda v:len(dumps(v)))

def unpack_arrays(x):
    if isinstance(x,list): return [unpack_arrays(v) for v in x]
    if not isinstance(x,dict): return x
    if '$range' in x:
        first,step,n=x['$range']; return [first+i*step for i in range(n)]
    if '$repeat' in x:
        value,n=x['$repeat']; return [copy.deepcopy(unpack_arrays(value)) for _ in range(n)]
    if '$table' in x:
        columns,rows=x['$table']
        return [{k:unpack_arrays(v) for k,v in zip(columns,row)} for row in rows]
    return {k:unpack_arrays(v) for k,v in x.items()}

def encode(state):
    packed=pack_arrays(temporal_pack(state))
    counts=collections.Counter()
    def count(x):
        if isinstance(x,(dict,list)):
            key=dumps(x)
            if len(key)>=100: counts[key]+=1
            for v in (x.values() if isinstance(x,dict) else x): count(v)
    count(packed)
    pool={}; ids={}
    def intern(x):
        if not isinstance(x,(dict,list)): return x
        key=dumps(x)
        if counts[key]>1:
            if key not in ids:
                name='v'+str(len(ids)); ids[key]=name
                pool[name]=children(x)
            return {'$ref':ids[key]}
        return children(x)
    def children(x):
        return {k:intern(v) for k,v in x.items()} if isinstance(x,dict) else [intern(v) for v in x]
    payload=intern(packed)
    return {'encoding':'lossless-json-v3',
      'reading_guide':'$ref names a value in shared_values. $range=[first,step,count] is the exact integer list. '
       '$repeat=[value,count] is an exact repeated list. All tool results and their chronological order are retained; '
       '$table=[columns,rows] represents a list of objects with those named fields. '
       'events lists every tool and its unmodified evidence in order. latest_snapshot holds the latest public state. '
       'reverse_snapshot_changes restores earlier public snapshots, newest to oldest, with set/shift_left/shift_right path operations. '
       'These are storage abbreviations, not observations or diagnostic conclusions.',
      'shared_values':pool,'state':payload}

def decode(encoded):
    pool=encoded['shared_values']
    def expand(x):
        if isinstance(x,list): return [expand(v) for v in x]
        if not isinstance(x,dict): return x
        if '$ref' in x: return expand(pool[x['$ref']])
        return {k:expand(v) for k,v in x.items()}
    return temporal_unpack(unpack_arrays(expand(encoded['state'])))

def encode_checked(state):
    encoded=encode(state)
    assert json.dumps(decode(encoded),sort_keys=True)==json.dumps(state,sort_keys=True)
    return dumps(encoded)
