"""Prospective 3x2 factorial experiment; no LAYA, Jev or LLM calls."""
from __future__ import annotations
import argparse
import copy
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from .run_research_suite import (FAULTS,dump,sha,stable_seed,seed_all,make_env,
    strict_location,step_env,tensor_dataset,fit)
from .graph_obs import build_graph_obs
from .graph_policy import GraphPointerPolicy,pointer_to_labels
from .pointer_actions import encode_flat_as_pointer,resolve_pointer
from .public_observation import PublicView,public_step_info
from .rule_agent_v2 import RuleAgentV2,_tool_payload
from .research_metrics import summarize,network_pair
from .relabel_graph import transform_graph,map_labels,RelabeledPolicy

CONDITIONS=[
 ("star_6","star",6,(10.,20.)),("star_10","star",10,(10.,20.)),
 ("star_12","star",12,(10.,20.)),("linear_6","linear",6,(10.,20.)),
 ("mesh_6","mesh",6,(10.,20.)),("mesh_10","mesh",10,(10.,20.)),
 ("random_6","random",6,(10.,20.)),("random_10","random",10,(10.,20.))]
MODES=["original","removed","augmented"]
CURRICULA=["star","mixed"]
REPLICAS=[41,52,63]

def collect_pool(replica,per_n,budget):
    pool=[];audit=[]
    for ti,topology in enumerate(["star","linear"]):
        for n in [4,5,6]:
            for fi,ft in enumerate(FAULTS):
                target=per_n//4 if topology=="star" else per_n//8
                kept=attempt=0
                while kept<target and attempt<target*10:
                    seed=2000000+REPLICAS.index(replica)*1000000+ti*200000+n*10000+fi*2000+attempt
                    attempt+=1
                    env=make_env(topology,n,ft,budget)
                    _,info=env.reset(seed=seed)
                    view=PublicView(env);agent=RuleAgentV2();occ=Counter()
                    graphs=[];labels=[];actions=[];success=False
                    for step in range(1,budget+1):
                        g=build_graph_obs(view,mem=agent.mem)
                        aid=agent.choose(view,None,step,[])
                        ptr=encode_flat_as_pointer(env,aid)
                        lab=pointer_to_labels(g,ptr)
                        graphs.append(g);labels.append(lab)
                        actions.append(dict(kind=ptr.kind.value,node=ptr.node,src=ptr.src,dst=ptr.dst,
                                            fault_type=ptr.fault_type,location=ptr.location))
                        _,_,term,trunc,info=step_env(env,aid,seed,occ)
                        agent.observe_step(public_step_info(info));view.update(env)
                        if ptr.kind.value=="diagnose":
                            gt=info["ground_truth_fault"]
                            success=ptr.fault_type==gt["type"] and strict_location(gt["location"],ptr.location,gt["type"])
                        if term or trunc:break
                    audit.append({"replica":replica,"topology":topology,"n":n,"fault":ft,
                                  "seed":seed,"success":success,"steps":step})
                    if success:
                        pool.append({"topology":topology,"n":n,"fault":ft,"seed":seed,
                                     "within_group":kept,"graphs":graphs,"labels":labels,"actions":actions})
                        kept+=1
                    env.close()
                print("collect",replica,topology,n,ft,kept,attempt,flush=True)
                if kept!=target:raise RuntimeError("Insufficient demos")
    return pool,audit

def select_episodes(pool,curriculum,per_n):
    if curriculum=="star":
        return [e for e in pool if e["topology"]=="star"]
    return [e for e in pool if e["topology"]=="linear" or e["within_group"]<per_n//8]

def prepare(episodes,mode):
    graphs=[];labels=[]
    for ep in episodes:
        variants=range(4) if mode=="augmented" else range(1)
        for variant in variants:
            seed=stable_seed(ep["seed"],"training_relabel",variant) if mode=="augmented" else None
            for g,label in zip(ep["graphs"],ep["labels"]):
                ng,mapping=transform_graph(g,seed,remove_rank=mode=="removed")
                graphs.append(ng);labels.append(map_labels(label,g,ng,mapping))
    return tensor_dataset(graphs,labels),len(labels)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--updates",type=int,default=6000)
    ap.add_argument("--demos-per-n",type=int,default=704)
    ap.add_argument("--episodes",type=int,default=100)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    if args.demos_per_n%8 or args.episodes%4:raise ValueError("Balanced counts required")
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);(out/"checkpoints").mkdir()
    project=Path(__file__).resolve().parents[1]
    spec=project.parent/"docs/specs/reliability_v3_hypotheses.md"
    replicas=REPLICAS[:1] if args.smoke else REPLICAS
    conditions=CONDITIONS[:1]+CONDITIONS[4:5] if args.smoke else CONDITIONS
    cfg=vars(args)|{"replicas":replicas,"conditions":conditions,"modes":MODES,"curricula":CURRICULA,
         "test_seed_base":8000000,"budget":40,"learning_rate":3e-4,"batch_size":64,
         "weight":1,"oversample":1,"augmentation_variants":4,"primary_interval":.975,
         "excluded":["LAYA","Jev","general LLM"],"spec_sha256":sha(spec)}
    dump(out/"protocol.json",cfg);(out/"frozen_spec.md").write_bytes(spec.read_bytes())
    meta={"status":"training","sources":{p.name:sha(p) for p in (project/"src").glob("*.py")},
          "python":platform.python_version(),"platform":platform.platform(),
          "packages":{x:importlib.metadata.version(x) for x in ["torch","numpy","gymnasium","networkx"]},
          "started_unix":time.time(),"spec_sha256":sha(spec),
          "project_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=project,text=True).strip(),
          "simulator_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=project.parent/"netheal-ai-agent-benchmark",text=True).strip()}
    dump(out/"manifest.json",meta)
    training={}
    for replica in replicas:
        pool,audit=collect_pool(replica,args.demos_per_n,40)
        torch.save(pool,out/f"demonstrations_{replica}.pt")
        dump(out/f"demo_audit_{replica}.json",audit)
        for curriculum in CURRICULA:
            episodes=select_episodes(pool,curriculum,args.demos_per_n)
            dump(out/f"selected_demos_{curriculum}_{replica}.json",[
                {k:e[k] for k in ["topology","n","fault","seed"]}|{"steps":len(e["labels"])} for e in episodes])
            for mode in MODES:
                dataset,aug_transitions=prepare(episodes,mode)
                seed_all(replica);model=GraphPointerPolicy()
                name=f"{curriculum}_{mode}_{replica}"
                print("TRAIN",name,args.updates,flush=True)
                result=fit(model,dataset,updates=args.updates,seed=replica,batch=64)
                training[name]=result|{"demos":len(episodes),"raw_transitions":sum(len(e["labels"]) for e in episodes),
                                      "augmented_transitions":aug_transitions,"replica":replica}
                torch.save(model.state_dict(),out/"checkpoints"/f"{name}.pt")
                dump(out/"training.json",training)
                del dataset,model
    meta["status"]="training_complete"
    meta["training_completed_unix"]=time.time()
    meta["checkpoint_sha256"]={p.name:sha(p) for p in (out/"checkpoints").glob("*.pt")}
    meta["data_sha256"]={p.name:sha(p) for p in out.glob("demonstrations_*.pt")}
    dump(out/"training_complete_manifest.json",meta)
    dump(out/"manifest.json",meta)
    print("ALL TRAINING COMPLETE; OPENING FINAL EVALUATION",flush=True)
    network_file=out/"network_probes.jsonl"
    with network_file.open("w",encoding="utf8") as f:
        for condition in conditions:
            key,topology,n,severity=condition
            for i in range(args.episodes):
                seed=8000000+CONDITIONS.index(condition)*10000+i
                env=make_env(topology,n,FAULTS[i%4],40,severity);env.reset(seed=seed)
                row={"case_id":f"{key}:{i}","condition":key,"fault_type":FAULTS[i%4],
                     "metrics":network_pair(env,stable_seed(seed,"network"))}
                f.write(json.dumps(row)+"\n");env.close()
            f.flush();print("PROBES",key,flush=True)
    summaries={}
    def evaluate(policy,name,replica,renamed):
        variant="renamed" if renamed else "canonical"
        path=out/f"episodes_{name}_{replica}_{variant}.jsonl"
        with path.open("w",encoding="utf8") as f:
            for condition in conditions:
                rows=[]
                for i in range(args.episodes):
                    row=rollout_case(policy,condition,i,8000000,40,replica,name,save_trace=i<4)
                    row["variant"]=variant
                    rows.append(row);f.write(json.dumps(row)+"\n")
                f.flush()
                block=summarize(rows)
                summaries.setdefault(name,{}).setdefault(str(replica),{}).setdefault(variant,{})[condition[0]]=block
                print("EVAL",name,replica,variant,condition[0],block["exact"],flush=True)
        dump(out/"summaries.json",summaries)
    evaluate(None,"rule_public",0,False)
    for replica in replicas:
        for curriculum in CURRICULA:
            for mode in MODES:
                name=f"{curriculum}_{mode}"
                model=GraphPointerPolicy()
                model.load_state_dict(torch.load(out/"checkpoints"/f"{name}_{replica}.pt",weights_only=True))
                for renamed in [False,True]:
                    evaluate(RelabeledPolicy(model,mode,renamed),name,replica,renamed)
    meta["status"]="completed";meta["completed_unix"]=time.time()
    dump(out/"manifest.json",meta)
    print("COMPLETE",out,flush=True)


def rollout_case(policy, condition, index, eval_base, budget, train_seed, name, save_trace=False,
            rank_shuffle=False, legacy_privileged=False):
    key,topology,n,severity=condition
    seed=eval_base+CONDITIONS.index(condition)*10000+index
    ft=FAULTS[index%4]
    env=make_env(topology,n,ft,budget,severity)
    obs,info=env.reset(seed=seed)
    gt=dict(info["ground_truth_fault"])
    view=PublicView(env);memory=RuleAgentV2()
    decisions=[];trace=[];occ=Counter();seen=set()
    calls=repeated=invalid=0;cost=0.;pred_type=pred_loc=None
    wall=time.perf_counter()
    ranks=np.random.default_rng(stable_seed(seed,"ranks")).permutation(n)/max(n-1,1)
    if policy is not None:
        policy.episode_seed=seed
        policy.eval()
    for step in range(1,budget+1):
        t=time.perf_counter()
        if policy is None:
            aid=memory.choose(view,None,step,[])
            ptr=encode_flat_as_pointer(env,aid)
        else:
            g=build_graph_obs(env if legacy_privileged else view,mem=memory.mem,
                              privileged=legacy_privileged)
            if rank_shuffle:
                g.node_feat[:,9]=ranks
                for ei,(a,b) in enumerate(g.edge_keys):
                    g.edge_feat[ei,8]=ranks[g.node_index[a]]
                    g.edge_feat[ei,9]=ranks[g.node_index[b]]
            ptr=policy.act(g,torch.device("cpu"))
            aid=resolve_pointer(env,ptr)
        decisions.append((time.perf_counter()-t)*1000)
        spec=env.action_space_manager.get_action_spec(aid)
        invalid+=int(aid not in env.get_valid_actions())
        obs,reward,term,trunc,info=step_env(env,aid,seed,occ)
        payload=_tool_payload(info)
        if ptr.kind.value=="diagnose":
            pred_type,pred_loc=ptr.fault_type,ptr.location
        else:
            calls+=1
            action_key=json.dumps(spec.to_dict(),sort_keys=True)
            repeated+=int(action_key in seen);seen.add(action_key)
            ar=info.get("action_result")
            if ar is not None and ar.result is not None:
                cost+=float(ar.result.cost)
        if save_trace:
            trace.append({"step":step,"action":spec.to_dict(),"tool_result":payload,
                          "decision_ms":decisions[-1]})
        memory.observe_step(public_step_info(info));view.update(env)
        if policy is not None: memory.refresh_plan(view)
        if term or trunc: break
    elapsed=(time.perf_counter()-wall)*1000
    type_ok=gt["type"]==pred_type
    loc_ok=strict_location(gt["location"],pred_loc,gt["type"])
    row={"case_id":f"{key}:{index}", "condition":key,"episode_seed":seed,
         "train_seed":train_seed,"model":name, "fault_type":gt["type"],
         "fault_location":gt["location"],"predicted_type":pred_type,"predicted_location":pred_loc,
         "exact":type_ok and loc_ok,"type_correct":type_ok,"location_correct":loc_ok,
         "submitted":pred_type is not None,"timeout":bool(trunc and pred_type is None),
         "steps":step,"tool_calls":calls,"tool_cost":cost,"repeated_tools":repeated,
         "invalid_actions":invalid,"decision_ms":decisions,"wall_ms":elapsed,
         "coverage":len(view.observation.discovery_matrix.get_discovered_devices())/n}
    if save_trace:row["trace"]=trace
    env.close()
    return row

if __name__=="__main__":main()
