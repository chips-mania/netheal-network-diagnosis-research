import json
from pathlib import Path
from src.laya_schema import expected,pointer_from_answers,questions,state_text
from src.run_research_suite import make_env
from src.public_observation import PublicView
from src.graph_obs import build_graph_obs
from src.graph_policy import pointer_to_labels
from src.pointer_actions import encode_flat_as_pointer,resolve_pointer
from src.rule_agent_v2 import RuleAgentV2
def test_teacher_roundtrip():
    env=make_env("star",6,"device_failure");env.reset(seed=12)
    view=PublicView(env);g=build_graph_obs(view);teacher=RuleAgentV2()
    aid=teacher.choose(view,None,1,[])
    ptr=encode_flat_as_pointer(env,aid)
    ans=expected(g,pointer_to_labels(g,ptr))
    assert resolve_pointer(env,pointer_from_answers(ans))==aid
    assert all(v in questions(g)[k]["criteria"] for k,v in ans.items())
    env.close()
def test_episode_split_and_evaluation_seed_exclusion():
    p=Path(__file__).parents[1]/"results/laya_data_v1"
    split=json.loads((p/"splits.json").read_text())
    seeds=[r["seed"] for r in split]
    assert len(seeds)==len(set(seeds))==2112
    for name in ("train","calibration","validation"):
        allowed={r["seed"] for r in split if r["split"]==name}
        rows=[json.loads(l) for l in (p/(name+".jsonl")).read_text().splitlines()]
        assert {r["episode_seed"] for r in rows}==allowed
        assert all(r["episode_seed"]<9000000 for r in rows)
        assert all("fault_location" not in r["state"] and "expected" not in r["state"] for r in rows)
