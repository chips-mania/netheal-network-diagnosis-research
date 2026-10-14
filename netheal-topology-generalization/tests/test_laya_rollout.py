"""Branch selection must follow model outputs, never teacher/ground truth."""
import time
from types import SimpleNamespace
import pytest
from src import run_laya_comparison as runner
from src.external_policy_adapter import graph_packet,parse_action
from src.run_research_suite import make_env
from src.public_observation import PublicView
from src.graph_obs import build_graph_obs
from src.laya_schema_compact import state_text

@pytest.mark.parametrize("kind,branch",[("scan",[]),("check_status",["node"]),
    ("check_interfaces",["node"]),("ping",["edge"]),("diagnose",["fault","diagnosis_edge"])])
def test_model_driven_branch(monkeypatch,kind,branch):
    env=make_env("star",6,"device_failure");env.reset(seed=12)
    g=build_graph_obs(PublicView(env));called=[]
    def predict(state,qs,**kwargs):
        name=next(iter(qs));called.append(name)
        value=kind if name=="kind" else "link_failure" if name=="fault" else next(iter(qs[name]["criteria"]))
        return {"answers":{name:{"choice":value}},"usage":{}}
    monkeypatch.setattr(runner,"to_internal",lambda name,q:q)
    def build(tok,state,q,*args,**kwargs):
        k=len(q["criteria"])
        return [],list(range(k)),{"options":k,"options_distinct":k},{"truncated":False}
    monkeypatch.setattr(runner,"build_sequence",build)
    monkeypatch.setattr(runner.torch.cuda,"synchronize",lambda:None)
    client=runner.LayaClient(SimpleNamespace(tok=None,predict=predict))
    action=client.choose(graph_packet(g),[],time.monotonic()+10,{"step":1})
    assert parse_action(action,g).kind.value==kind
    assert called==["kind"]+branch
    env.close()

def test_compact_table_preserves_numeric_features():
    env=make_env("star",6,"device_failure");env.reset(seed=12)
    g=build_graph_obs(PublicView(env));lines=state_text(g).splitlines()
    start=next(i for i,s in enumerate(lines) if s.startswith("Device columns:"))+1
    for line,expected in zip(lines[start:start+g.n_nodes],g.node_feat):
        got=list(map(float,line.split()[1:]));assert got==pytest.approx(expected,rel=1e-4,abs=1e-8)
    start=next(i for i,s in enumerate(lines) if s.startswith("Link columns:"))+1
    for line,expected in zip(lines[start:],g.edge_feat):
        got=list(map(float,line.split()[1:]));assert got==pytest.approx(expected,rel=1e-4,abs=1e-8)
    env.close()
