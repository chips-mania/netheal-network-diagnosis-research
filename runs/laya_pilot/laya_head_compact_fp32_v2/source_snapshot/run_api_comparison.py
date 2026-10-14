"""Paired live API pilot. Fresh output folders; durable request/episode logs."""
import argparse,json,time,random,hashlib,sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from dotenv import dotenv_values
from .api_policy_client import Client,APIError,packet,INSTRUCTIONS
from .external_policy_adapter import parse_action
from .run_research_suite import make_env,step_env,strict_location,stable_seed,dump,sha,FAULTS
from .run_factorial_v3 import CONDITIONS
from .graph_obs import build_graph_obs
from .public_observation import PublicView,public_step_info
from .rule_agent_v2 import RuleAgentV2,_tool_payload
from .pointer_actions import resolve_pointer
from .relabel_graph import transform_graph,map_pointer

def append(path,row):
    with path.open("a",encoding="utf8") as f:
        f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush()

def episode(client,condition,index,seed,variant,repeat,cfg,out):
    key,topology,n,severity=condition
    env=make_env(topology,n,FAULTS[index%4],cfg["common"]["max_decisions"],severity)
    _,info=env.reset(seed=seed);gt=dict(info["ground_truth_fault"])
    view=PublicView(env);memory=RuleAgentV2();occ=Counter();seen=set()
    row={"case_id":f"{key}:{index}","condition":key,"episode_seed":seed,"variant":variant,
         "repeat":repeat,"provider":client.provider,"fault_type":gt["type"],"fault_location":gt["location"],
         "predicted_type":None,"predicted_location":None,"tool_calls":0,"repeated_tools":0,
         "tool_cost":0.,"trace":[],"requests":[]}
    started=time.monotonic();deadline=started+cfg["common"]["episode_watchdog_seconds"]
    row["termination"]="step_limit"
    old_sink=client.sink
    def sink(record):
        record=dict(record,case_id=row["case_id"],repeat=repeat,variant=variant)
        append(out/"requests.jsonl",record)
        row["requests"].append({k:record.get(k) for k in ("seconds","usage","estimated_usd","error_kind","http_status","response_model","request_id")})
    client.sink=sink
    fatal=None
    try:
        for step in range(1,cfg["common"]["max_decisions"]+1):
            row["steps"]=step
            if time.monotonic()>=deadline:row["termination"]="watchdog";break
            g=build_graph_obs(view,mem=memory.mem)
            g,mapping=transform_graph(g,stable_seed(seed,"api_relabel") if variant=="renamed" else None)
            state,options=packet(g)
            try:action=client.choose(state,options,deadline,{"step":step})
            except APIError as e:
                row["termination"]=e.kind;row["http_status"]=e.status
                if e.status in (400,401,403,404,422):fatal=e
                break
            ptr=map_pointer(parse_action(action,g),{v:k for k,v in mapping.items()})
            aid=resolve_pointer(env,ptr)
            action_key=json.dumps(action,sort_keys=True)
            t=time.perf_counter()
            _,_,term,trunc,info=step_env(env,aid,seed,occ)
            tool_seconds=time.perf_counter()-t
            row["trace"].append({"step":step,"action":action,"tool_result":_tool_payload(info),"tool_seconds":tool_seconds})
            if ptr.kind.value=="diagnose":
                row["predicted_type"],row["predicted_location"]=ptr.fault_type,ptr.location
                row["termination"]="diagnosed"
            else:
                row["tool_calls"]+=1;row["repeated_tools"]+=int(action_key in seen);seen.add(action_key)
                ar=info.get("action_result")
                if ar is not None and ar.result is not None:row["tool_cost"]+=float(ar.result.cost)
            memory.observe_step(public_step_info(info));view.update(env);memory.refresh_plan(view)
            if term or trunc:break
    finally:
        env.close();client.sink=old_sink
    row["wall_seconds"]=time.monotonic()-started
    row["type_correct"]=row["predicted_type"]==gt["type"]
    row["location_correct"]=strict_location(gt["location"],row["predicted_location"],gt["type"])
    row["exact"]=row["type_correct"] and row["location_correct"]
    row["known_estimated_usd"]=sum(r["estimated_usd"] or 0 for r in row["requests"])
    row["unknown_cost_requests"]=sum(r["estimated_usd"] is None for r in row["requests"])
    row["estimated_usd"]=None if row["unknown_cost_requests"] else row["known_estimated_usd"]
    append(out/"episodes.jsonl",row)
    if fatal:raise fatal
    return row

def summary(rows):
    result={}
    for provider in ("gpt","jev"):
        rs=[r for r in rows if r["provider"]==provider]
        if not rs:continue
        correct=sum(r["exact"] for r in rs);known=sum(r["known_estimated_usd"] for r in rs)
        unknown=sum(r["unknown_cost_requests"] for r in rs)
        result[provider]={"episodes":len(rs),"exact":correct/len(rs),"correct":correct,
            "termination_counts":dict(Counter(r["termination"] for r in rs)),
            "mean_seconds":float(np.mean([r["wall_seconds"] for r in rs])),
            "p95_seconds":float(np.percentile([r["wall_seconds"] for r in rs],95)),
            "mean_tool_calls":float(np.mean([r["tool_calls"] for r in rs])),
            "known_estimated_usd":known,"unknown_cost_requests":unknown,
            "mean_estimated_usd":None if unknown else known/len(rs),
            "usd_per_correct":known/correct if correct and not unknown else None,
            "conditions":{c:{"n":len(cr),"exact":sum(r["exact"] for r in cr)/len(cr)}
              for c in sorted({r["condition"] for r in rs}) if (cr:=[r for r in rs if r["condition"]==c])}}
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--phase",choices=["smoke","pilot"],default="pilot")
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];project=root/"netheal-topology-generalization"
    cfg=json.loads((project/"config/api_comparison_v1.json").read_text())
    keys=dotenv_values(root/".env")
    for p in ("gpt","jev"):
        if not keys.get(cfg["providers"][p]["key_env"]):raise ValueError(f"Missing {p} key")
    torch.set_num_threads(1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    count=1 if args.phase=="smoke" else cfg["evaluation"]["pilot_cases_per_condition"]
    conds=CONDITIONS[:1] if args.phase=="smoke" else CONDITIONS
    base=9000000 if args.phase=="smoke" else cfg["evaluation"]["pilot_seed_base"]
    cases=[(c,i,base+CONDITIONS.index(c)*10000+i) for c in conds for i in range(count)]
    random.Random(20261009).shuffle(cases)
    schedule=[]
    for c,i,s in cases:
        order=["gpt","jev"]
        random.Random(stable_seed(s,"provider_order")).shuffle(order)
        schedule.extend((p,c,i,s) for p in order)
    dump(out/"config.json",cfg)
    dump(out/"schedule.json",schedule)
    (out/"instructions.txt").write_text(INSTRUCTIONS,encoding="utf8")
    manifest={"status":"running","phase":args.phase,"started_unix":time.time(),"expected_episodes":len(schedule),
        "source_sha256":{p.name:sha(p) for p in (project/"src").glob("*.py")},
        "config_sha256":sha(out/"config.json"),"prompt_sha256":sha(out/"instructions.txt")}
    dump(out/"manifest.json",manifest)
    clients={p:Client(p,cfg,keys[cfg["providers"][p]["key_env"]],lambda r:None) for p in ("gpt","jev")}
    rows=[]
    limit_text=keys.get("NETHEAL_RUN_SPEND_LIMIT_USD")
    limit=float(limit_text) if limit_text else None
    try:
        for j,(p,c,i,s) in enumerate(schedule):
            if limit is not None and sum(r["known_estimated_usd"] for r in rows)>=limit:
                manifest["status"]="spend_guard_stopped";break
            row=episode(clients[p],c,i,s,"canonical",0,cfg,out);rows.append(row)
            dump(out/"summary.json",summary(rows))
            print(json.dumps({"done":j+1,"total":len(schedule),"provider":p,"case":row["case_id"],
                "exact":row["exact"],"termination":row["termination"],"seconds":round(row["wall_seconds"],2),
                "estimated_usd":row["estimated_usd"]}),flush=True)
        else:manifest["status"]="completed"
    except APIError as e:
        manifest["status"]="api_configuration_blocked";manifest["error"]={"kind":e.kind,"http_status":e.status}
    except Exception as e:
        manifest["status"]="implementation_error";manifest["error_type"]=type(e).__name__
        raise
    finally:
        for c in clients.values():c.close()
        manifest["finished_unix"]=time.time()
        persisted=[json.loads(l) for l in (out/"episodes.jsonl").read_text(encoding="utf8").splitlines()] if (out/"episodes.jsonl").exists() else []
        manifest["completed_episodes"]=len(persisted)
        dump(out/"summary.json",summary(persisted));dump(out/"manifest.json",manifest)
    print(json.dumps({"status":manifest["status"],"path":str(out)}),flush=True)
if __name__=="__main__":main()
