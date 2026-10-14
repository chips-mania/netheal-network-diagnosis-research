import numpy as np
import torch
from src.run_research_suite import make_env,stable_seed
from src.graph_obs import build_graph_obs
from src.graph_policy import pointer_to_labels
from src.pointer_actions import pointer_diagnose,pointer_ping,pointer_check_status
from src.public_observation import PublicView
from src.relabel_graph import transform_graph,map_pointer,map_labels,RelabeledPolicy
from src.run_factorial_v3 import select_episodes

def graph():
    env=make_env("star",6,"link_failure");env.reset(seed=51)
    return build_graph_obs(PublicView(env))

def test_relabel_is_bijection_preserves_evidence_and_connectivity():
    g=graph();new,mapping=transform_graph(g,123)
    assert len(set(mapping.values()))==g.n_nodes
    assert set(new.edge_keys)=={tuple(mapping[x] for x in e) for e in g.edge_keys}
    for old_id,new_id in mapping.items():
        np.testing.assert_array_equal(g.node_feat[g.node_index[old_id],:9],
                                      new.node_feat[new.node_index[new_id],:9])
    for i,e in enumerate(g.edge_keys):
        j=new.edge_keys.index(tuple(mapping[x] for x in e))
        np.testing.assert_array_equal(g.edge_feat[i,:8],new.edge_feat[j,:8])
        assert tuple(new.node_ids[k] for k in new.edge_index[:,j])==new.edge_keys[j]
    np.testing.assert_array_equal(g.globals,new.globals)

def test_pointer_and_supervised_labels_round_trip():
    g=graph();new,mapping=transform_graph(g,125)
    inverse={v:k for k,v in mapping.items()}
    a,b=g.edge_keys[0]
    for p in [pointer_ping(a,b),pointer_check_status(a),pointer_diagnose("link_failure",f"{a}->{b}"),
              pointer_diagnose("device_failure",a)]:
        nptr=map_pointer(p,mapping)
        assert map_pointer(nptr,inverse)==p
        assert map_labels(pointer_to_labels(g,p),g,new,mapping)==pointer_to_labels(new,nptr)

def test_remove_rank_zeroes_only_identifier_features():
    g=graph();new,_=transform_graph(g,remove_rank=True)
    assert np.all(new.node_feat[:,9]==0)
    assert np.all(new.edge_feat[:,8:10]==0)
    np.testing.assert_array_equal(new.node_feat[:,:9],g.node_feat[:,:9])
    np.testing.assert_array_equal(new.edge_feat[:,:8],g.edge_feat[:,:8])

def test_curriculum_preserves_budget_and_shares_star_subset():
    pool=[]
    for top,count in [("star",4),("linear",2)]:
        pool.extend({"topology":top,"within_group":i} for i in range(count))
    star=select_episodes(pool,"star",16)
    mixed=select_episodes(pool,"mixed",16)
    assert len(star)==len(mixed)==4
    assert sum(e["topology"]=="star" for e in mixed)==2

def test_adapter_maps_model_targets_back_to_original_ids():
    g=graph()
    class Pick:
        def eval(self):return self
        def act(self,x,device):return pointer_check_status(x.node_ids[-1])
    p=RelabeledPolicy(Pick(),"augmented",True);p.episode_seed=8123
    actual=p.act(g,torch.device("cpu"))
    ng,mapping=transform_graph(g,stable_seed(8123,"final_relabel"))
    assert mapping[actual.node]==ng.node_ids[-1]
