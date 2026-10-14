"""Lossless JSON-path partition fallback for individually oversized tool results."""
import hashlib, json
from jev_context_codec_v4 import encode_checked, dumps

def restore(fragments):
    root=None
    for item in fragments:
        path=item['path']; value=item['value']
        if not path: root=value; continue
        if root is None: root=[] if isinstance(path[0],int) else {}
        node=root
        for i,key in enumerate(path):
            last=i==len(path)-1
            if isinstance(node,list):
                while len(node)<=key: node.append(None)
            if last: node[key]=value
            else:
                child=node[key] if isinstance(node,list) else node.get(key)
                if child is None:
                    child=[] if isinstance(path[i+1],int) else {}; node[key]=child
                node=child
    return root

def tree_partitions(state,budget):
    identity=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()
    common={k:state[k] for k in ('task_description','objective','hint','network_size','max_steps','solver_turn') if k in state}
    def envelope(items):
        return dict(common,partition_notice='Lossless evidence partition: path is the location in the original JSON state; value is the exact subtree. All partitions together preserve all evidence. Evaluate the same options from this evidence.',
                    original_state_sha256=identity,evidence_fragments=items,
                    partition_weight=sum(len(dumps(x['value']).encode()) for x in items))
    def fits(items): return len(encode_checked(envelope(items)).encode())<=budget
    assert fits([]),'fixed question/context cannot fit'
    fragments=[]
    def split(value,path):
        item={'path':path,'value':value}
        if fits([item]): fragments.append(item); return
        if isinstance(value,dict) and value:
            for k,v in value.items(): split(v,path+[k])
        elif isinstance(value,list) and value:
            for i,v in enumerate(value): split(v,path+[i])
        else: raise ValueError('indivisible scalar exceeds request budget')
    split(state,[])
    # Serialization compares numeric types and array order as well as values.
    assert json.dumps(restore(fragments),sort_keys=True)==json.dumps(state,sort_keys=True)
    groups=[]; current=[]
    for item in fragments:
        if current and not fits(current+[item]): groups.append(envelope(current)); current=[]
        current.append(item)
    if current: groups.append(envelope(current))
    assert all(len(encode_checked(g).encode())<=budget for g in groups)
    return groups
