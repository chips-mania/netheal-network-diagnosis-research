"""Consistent public-graph relabeling, ordering and pointer inverse mapping."""
from dataclasses import replace
import numpy as np
from .graph_obs import GraphObs
from .graph_policy import pointer_to_labels
from .pointer_actions import PointerAction

def transform_graph(g, seed=None, remove_rank=False):
    n,e=g.n_nodes,g.n_edges
    if seed is None:
        mapping={x:x for x in g.node_ids}
        order=np.arange(n);edge_order=np.arange(e)
    else:
        rng=np.random.default_rng(seed)
        ranks=rng.permutation(n)
        mapping={x:f"node_{int(ranks[i]):03d}" for i,x in enumerate(g.node_ids)}
        order=np.array(sorted(range(n),key=lambda i:mapping[g.node_ids[i]]))
        edge_order=np.array(sorted(range(e),key=lambda i:tuple(mapping[x] for x in g.edge_keys[i])),dtype=int)
    ids=[mapping[g.node_ids[i]] for i in order]
    index={x:i for i,x in enumerate(ids)}
    edges=[tuple(mapping[x] for x in g.edge_keys[i]) for i in edge_order]
    nf=g.node_feat[order].copy();ef=g.edge_feat[edge_order].copy()
    ei=np.array([[index[a] for a,b in edges],[index[b] for a,b in edges]],dtype=np.int64)
    if seed is not None:
        rank={x:i/max(n-1,1) for i,x in enumerate(sorted(ids))}
        nf[:,9]=[rank[x] for x in ids]
        if e: ef[:,8:10]=[[rank[a],rank[b]] for a,b in edges]
    if remove_rank:
        nf[:,9]=0
        ef[:,8:10]=0
    result=GraphObs(ids,index,nf,ei,ef,edges,g.globals.copy(),n,e)
    return result,mapping

def map_pointer(p,mapping):
    def node(x): return mapping[x] if x is not None else None
    loc=p.location
    if loc is not None:
        loc="->".join(mapping[x] for x in loc.split("->"))
    return replace(p,node=node(p.node),src=node(p.src),dst=node(p.dst),location=loc)

def map_labels(labels,old,new,mapping):
    result=dict(labels)
    if labels["node"]>=0:
        result["node"]=new.node_index[mapping[old.node_ids[labels["node"]]]]
    if labels["edge"]>=0:
        key=tuple(mapping[x] for x in old.edge_keys[labels["edge"]])
        result["edge"]=new.edge_keys.index(key)
    return result

class RelabeledPolicy:
    def __init__(self,model,rank_mode,renamed=False):
        self.model=model
        self.rank_mode=rank_mode
        self.renamed=renamed
        self.episode_seed=0
    def eval(self):
        self.model.eval()
        return self
    def act(self,g,device):
        from .run_research_suite import stable_seed
        transformed,mapping=transform_graph(
            g,stable_seed(self.episode_seed,"final_relabel") if self.renamed else None,
            remove_rank=self.rank_mode=="removed")
        p=self.model.act(transformed,device)
        return map_pointer(p,{v:k for k,v in mapping.items()})
