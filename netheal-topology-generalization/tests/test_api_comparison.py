import json,time
import httpx
import pytest
from src.api_policy_client import Client,APIError,candidates,estimate_cost
from src.run_research_suite import make_env,FAULTS
from src.run_factorial_v3 import CONDITIONS
from src.public_observation import PublicView
from src.graph_obs import build_graph_obs
from src.external_policy_adapter import parse_action
from src.pointer_actions import resolve_pointer
from src.relabel_graph import transform_graph,map_pointer
from pathlib import Path

def config():
    return json.loads((Path(__file__).parents[1]/"config/api_comparison_v1.json").read_text())
def test_all_candidates_resolve_and_rename():
    for _,top,n,sev in CONDITIONS:
        env=make_env(top,n,FAULTS[0]);env.reset(seed=123)
        g=build_graph_obs(PublicView(env))
        for seed in (None,42):
            ng,m=transform_graph(g,seed);a=candidates(ng)
            assert 0<len(a)<=255
            for action in a.values():
                ptr=map_pointer(parse_action(action,ng),{v:k for k,v in m.items()})
                assert resolve_pointer(env,ptr)>=0
        env.close()
def test_cost_cached_and_reasoning_not_double_counted():
    cfg=config()
    usage={"input_tokens":1000,"input_tokens_details":{"cached_tokens":400},
           "output_tokens":100,"output_tokens_details":{"reasoning_tokens":80}}
    assert estimate_cost("gpt",usage,cfg["pricing"])==pytest.approx(.00224)
    assert estimate_cost("jev",usage,cfg["pricing"])==pytest.approx(.000042)
    assert estimate_cost("gpt",None,cfg["pricing"]) is None
def mock_client(provider,handler,rows):
    c=Client(provider,config(),"test-secret",rows.append)
    c.client.close();c.client=httpx.Client(transport=httpx.MockTransport(handler))
    return c
def test_invalid_output_is_logged_not_repaired():
    rows=[]
    c=mock_client("gpt",lambda r:httpx.Response(200,json={"status":"completed","usage":{"input_tokens":1,"output_tokens":1},
       "output":[{"content":[{"type":"output_text","text":'{"action_id":"invalid"}'}]}]}),rows)
    with pytest.raises(APIError,match="invalid_output"):
        c.choose({},{"a000":{"kind":"scan"}},time.monotonic()+10,{})
    assert len(rows)==1 and rows[0]["estimated_usd"]>0
    assert "test-secret" not in json.dumps(rows)
    c.close()
def test_auth_failure_no_retry_and_no_error_body_saved():
    rows=[]
    c=mock_client("jev",lambda r:httpx.Response(401,json={"error":"test-secret"}),rows)
    with pytest.raises(APIError) as ex:c.choose({},{"a000":{"kind":"scan"}},time.monotonic()+10,{})
    assert ex.value.status==401 and len(rows)==1
    assert rows[0]["estimated_usd"] is None
    assert "test-secret" not in json.dumps(rows)
    c.close()
