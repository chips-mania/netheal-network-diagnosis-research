import copy
import json
from collections import Counter
import numpy as np
import torch
from src.run_research_suite import make_env, step_env, strict_location
from src.graph_obs import build_graph_obs
from src.public_observation import PublicView, policy_packet
from src.rule_agent_v2 import RuleAgentV2
from src.pointer_actions import resolve_pointer, pointer_scan
from src.graph_policy import collate_graphs
from src.flat_pointer_policy import FlatPointerPolicy
from src.research_metrics import paired_bootstrap, network_pair, wilson

def test_hidden_state_cannot_change_public_input():
    env=make_env("star",6,"link_failure")
    env.reset(seed=123)
    view=PublicView(env); memory=RuleAgentV2()
    before=policy_packet(view,memory.mem)
    for a,b in env.network.get_all_connections():
        env.network.graph.edges[a,b]["latency"]=9999.
        env.network.graph.edges[a,b]["status"]="up"
    env.ground_truth_fault.location="DO_NOT_EXPOSE"
    after=policy_packet(view,memory.mem)
    assert before==after
    assert "DO_NOT_EXPOSE" not in json.dumps(after)
    assert not hasattr(view,"ground_truth_fault")
    assert not hasattr(view.network,"graph")
    assert not hasattr(view.network,"is_connection_up")
    assert np.all(build_graph_obs(view).edge_feat[:,3]==0)
    assert np.all(build_graph_obs(view).node_feat[:,1]==0)

def test_clock_discovery_and_common_probe_randomness():
    outputs=[]
    for _ in range(2):
        env=make_env("star",6,"performance_degradation")
        env.reset(seed=78);view=PublicView(env)
        aid=resolve_pointer(env,pointer_scan())
        step_env(env,aid,78,Counter());view.update(env)
        g=build_graph_obs(view)
        assert np.isclose(g.globals[0],1/40)
        assert np.all(g.node_feat[:,1]==1)
        outputs.append(g.edge_feat)
    np.testing.assert_array_equal(*outputs)

def test_offline_probes_do_not_mutate_episode():
    env=make_env("star",4,"performance_degradation")
    env.reset(seed=73)
    before=copy.deepcopy(dict(env.network.graph.edges))
    rng_state=env._rng.getstate()
    result=network_pair(env,19)
    assert dict(env.network.graph.edges)==before
    assert env._rng.getstate()==rng_state
    assert result["faulted"]["probe_attempts"]==4*3*5
    assert result["faulted"]["throughput_mbps"] is None

def test_mlp_handles_mesh_and_unseen_size():
    model=FlatPointerPolicy()
    for family,n in [("star",12),("mesh",6)]:
        env=make_env(family,n,"device_failure");env.reset(seed=72)
        g=build_graph_obs(PublicView(env))
        act=model.act(g,torch.device("cpu"))
        assert resolve_pointer(env,act)>=0

def test_statistics_and_strict_location():
    assert not strict_location("a","a->b","device_failure")
    assert strict_location("a->b","b->a","link_failure")
    lo,hi=wilson(100,100)
    assert lo<1 and hi==1
    a=[{"train_seed":s,"case_id":str(i),"exact":False} for s in [1,2,3] for i in range(8)]
    b=[dict(r,exact=True) for r in a]
    result=paired_bootstrap(a,b,repeats=100)
    assert result["delta_exact"]==1
    assert result["paired_hierarchical_bootstrap95"]==[1,1]


def test_external_adapter_rejects_unknown_actions():
    from src.external_policy_adapter import ExternalPolicyAdapter, graph_packet, parse_action
    import pytest
    env=make_env("mesh",6,"link_failure");env.reset(seed=12)
    g=build_graph_obs(PublicView(env))
    assert "ground_truth" not in json.dumps(graph_packet(g))
    adapter=ExternalPolicyAdapter(lambda p: {"action":{"kind":"scan"}},"mock","hash",0)
    assert adapter.act(g).kind.value=="scan"
    with pytest.raises(ValueError):
        parse_action({"kind":"check_status","node":"missing"},g)


def test_external_policy_runs_in_common_evaluator():
    from src.external_policy_adapter import ExternalPolicyAdapter
    from src.run_research_suite import rollout, CONDITIONS
    adapter=ExternalPolicyAdapter(lambda p: {"action":{"kind":"scan"}},"mock","hash",0)
    row=rollout(adapter,CONDITIONS[0],0,900000,3,0,"mock")
    assert row["timeout"] and row["steps"]==3
    assert row["tool_calls"]==3 and len(adapter.calls)==3
