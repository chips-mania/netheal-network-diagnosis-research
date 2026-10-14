"""Reproducible, paired, observation-only diagnosis benchmark.

Run python -m src.run_research_suite --out results/reliability_v2
Defaults are frozen before test evaluation. No existing report/checkpoint overwritten.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from collections import Counter, defaultdict
import numpy as np
import torch
from netheal import NetworkTroubleshootingEnv
from netheal.faults.injector import FaultType
from netheal.environment.actions import ActionCategory
from .graph_obs import build_graph_obs
from .graph_policy import GraphPointerPolicy, collate_graphs, pointer_to_labels, bc_loss
from .flat_pointer_policy import FlatPointerPolicy
from .pointer_actions import encode_flat_as_pointer, resolve_pointer
from .public_observation import PublicView, public_step_info, policy_packet
from .rule_agent_v2 import RuleAgentV2, _tool_payload
from .research_metrics import summarize, paired_bootstrap, network_pair

FAULTS = ["device_failure","performance_degradation","misconfiguration","link_failure"]
CONDITIONS = [
    ("star_6","star",6,(10.,20.)), ("star_8","star",8,(10.,20.)),
    ("star_10","star",10,(10.,20.)), ("star_12","star",12,(10.,20.)),
    ("linear_6","linear",6,(10.,20.)), ("mesh_6","mesh",6,(10.,20.)),
    ("random_6","random",6,(10.,20.)), ("star_10_mild","star",10,(2.,5.)),
]
def dump(path, data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf8")
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stable_seed(*parts):
    return int(hashlib.sha256(json.dumps(parts,sort_keys=True).encode()).hexdigest()[:8],16)
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def make_env(topology,n,fault,budget=40,severity=(10.,20.)):
    return NetworkTroubleshootingEnv(min_devices=n,max_devices=n,max_episode_steps=budget,
           topology_types=[topology],fault_types=[FaultType(fault)],enable_user_hints=False,
           latency_multiplier_range=severity)
def strict_location(actual,predicted,fault):
    if not predicted:
        return False
    if fault in ("link_failure","performance_degradation"):
        return sorted(actual.split("->"))==sorted(predicted.split("->"))
    return actual==predicted

def step_env(env, action, episode_seed, occurrences):
    spec = env.action_space_manager.get_action_spec(action)
    key = json.dumps(spec.to_dict(),sort_keys=True)
    occurrence = occurrences[key]
    occurrences[key] += 1
    # Common random numbers for the same semantic probe and occurrence across agents.
    env.tool_simulator.rng = random.Random(stable_seed(episode_seed,key,occurrence))
    return env.step(action)

def collect(demos_per_n, seed_base, budget):
    graphs, labels, audit = [], [], []
    for n in [4,5,6]:
        for fi, ft in enumerate(FAULTS):
            target = demos_per_n//4
            kept = tries = 0
            lengths = []
            while kept < target and tries < target*10:
                episode_seed = seed_base+n*10000+fi*1000+tries
                tries += 1
                env = make_env("star",n,ft,budget)
                obs,info = env.reset(seed=episode_seed)
                view = PublicView(env); agent = RuleAgentV2()
                eg,el = [],[]
                occurrences = Counter()
                success = False
                for step in range(1,budget+1):
                    g = build_graph_obs(view,mem=agent.mem)
                    aid = agent.choose(view,None,step,[])
                    p = encode_flat_as_pointer(env,aid)
                    lab = pointer_to_labels(g,p)
                    if lab["kind"] < 0:
                        raise RuntimeError("Unsupported expert action")
                    eg.append(g);el.append(lab)
                    obs,reward,term,trunc,info = step_env(env,aid,episode_seed,occurrences)
                    agent.observe_step(public_step_info(info));view.update(env)
                    if p.kind.value=="diagnose":
                        gt=info["ground_truth_fault"]
                        success=p.fault_type==gt["type"] and strict_location(gt["location"],p.location,gt["type"])
                    if term or trunc:
                        break
                if success:
                    kept += 1;graphs.extend(eg);labels.extend(el);lengths.append(len(el))
                env.close()
            audit.append({"n":n,"fault":ft,"attempts":tries,"accepted":kept,
                          "expert_success_rate":kept/tries,"transitions":sum(lengths)})
            print("collect",audit[-1],flush=True)
            if kept != target:
                raise RuntimeError("Insufficient successful demonstrations")
    return graphs,labels,audit

def tensor_dataset(graphs, labels):
    pn=max(g.n_nodes for g in graphs);pe=max(g.n_edges for g in graphs)
    inputs=collate_graphs(graphs,pad_nodes=pn,pad_edges=pe)
    targets={k:torch.tensor([x[k] for x in labels]) for k in ("kind","node","edge","fault")}
    return inputs,targets

def fit(policy, dataset, *, updates, seed, weight=1., oversample=1., batch=64):
    seed_all(seed);policy.train()
    x,y=dataset
    opt=torch.optim.Adam(policy.parameters(),lr=3e-4)
    probabilities=np.where(y["kind"].numpy()==4,oversample,1.).astype(float)
    probabilities/=probabilities.sum()
    rng=np.random.default_rng(seed)
    losses=[]
    start=time.perf_counter()
    for u in range(updates):
        ix=torch.from_numpy(rng.choice(len(probabilities),batch,replace=True,p=probabilities))
        a={k:v[ix] for k,v in x.items()};b={k:v[ix] for k,v in y.items()}
        logits=policy.forward_logits(a["node_feat"],a["edge_index"],a["edge_feat"],
                                      a["node_mask"],a["edge_mask"],a["globals"])
        w=torch.where(b["kind"]==4,weight,1.)
        loss=bc_loss(logits,b,kind_weight=w)
        opt.zero_grad();loss.backward();opt.step()
        losses.append(float(loss.detach()))
    return {"updates":updates,"seconds":time.perf_counter()-start,
            "last_100_loss":float(np.mean(losses[-100:])),
            "weight":weight,"sampling_multiplier":oversample,
            "parameters":sum(p.numel() for p in policy.parameters() if p.requires_grad)}

def rollout(policy, condition, index, eval_base, budget, train_seed, name, save_trace=False,
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
    if policy is not None: policy.eval()
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--seeds",type=int,nargs="+",default=[11,22,33])
    ap.add_argument("--demos-per-n",type=int,default=700)
    ap.add_argument("--epochs",type=int,default=80)
    ap.add_argument("--continuation-epochs",type=int,default=40)
    ap.add_argument("--eval-episodes",type=int,default=200)
    ap.add_argument("--budget",type=int,default=40)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    if args.demos_per_n%4 or args.eval_episodes%4:
        raise ValueError("balanced counts must be multiples of 4")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    (out/"checkpoints").mkdir()
    project=Path(__file__).resolve().parents[1]
    config=vars(args)|{"train_seed_base":300000,"test_seed_base":900000,
       "conditions":CONDITIONS,"known_inventory":True,"tool_rng":"semantic-action/occurrence keyed",
       "input":"observed-only; slot3 raw link-down disabled; step_count fixed",
       "training":"same balanced successful expert trajectories across all seeds/models",
       "updates":"fixed optimizer updates per arm; oversampling changes sampling probabilities only",
       "network_probe_repeats":5,"clock":"host wall-clock stored separately from simulated network delay"}
    dump(out/"protocol.json",config)
    source={str(p.relative_to(project)):sha(p) for p in (project/"src").glob("*.py")}
    meta={"source_sha256":source,"python":platform.python_version(),"platform":platform.platform(),
          "packages":{x:importlib.metadata.version(x) for x in ["torch","numpy","gymnasium","networkx","stable-baselines3","sb3-contrib"]}}
    for k,p in [("project_commit",project),("simulator_commit",project.parent/"netheal-ai-agent-benchmark")]:
        meta[k]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=p,text=True).strip()
    dump(out/"manifest.json",meta)
    graphs,labels,audit=collect(args.demos_per_n,300000,args.budget)
    dump(out/"demo_audit.json",audit)
    torch.save({"graphs":graphs,"labels":labels},out/"demonstrations.pt")
    dataset=tensor_dataset(graphs,labels)
    updates=int(np.ceil(len(labels)/64))*args.epochs
    more=int(np.ceil(len(labels)/64))*args.continuation_epochs
    conditions=CONDITIONS[:2] if args.smoke else CONDITIONS
    # Fixed policy-independent probes. Never feed these data to policy inputs.
    with (out/"network_probes.jsonl").open("w",encoding="utf8") as f:
        for condition in conditions:
            key,topology,n,severity=condition
            for i in range(args.eval_episodes):
                case_seed=900000+CONDITIONS.index(condition)*10000+i
                env=make_env(topology,n,FAULTS[i%4],args.budget,severity)
                env.reset(seed=case_seed)
                result={"case_id":f"{key}:{i}","condition":key,"fault_type":FAULTS[i%4],
                        "metrics":network_pair(env,stable_seed(case_seed,"network"))}
                f.write(json.dumps(result)+"\n");env.close()
            f.flush();print("network probes",key,flush=True)
    summaries={};all_rows=[];training={}
    def evaluate(model,name,train_seed,shuffle=False):
        rows=[]
        with (out/f"episodes_{name}_{train_seed}.jsonl").open("w",encoding="utf8") as f:
            for condition in conditions:
                block=[]
                for i in range(args.eval_episodes):
                    row=rollout(model,condition,i,900000,args.budget,train_seed,name,
                                save_trace=i<4,rank_shuffle=shuffle)
                    block.append(row);rows.append(row);f.write(json.dumps(row)+"\n")
                f.flush()
                summ=summarize(block)
                summaries.setdefault(name,{}).setdefault(str(train_seed),{})[condition[0]]=summ
                print("eval",name,train_seed,condition[0],round(summ["exact"],3),flush=True)
        all_rows.extend(rows)
        dump(out/"summaries.json",summaries)
    evaluate(None,"rule_public",0)
    for seed in args.seeds:
        seed_all(seed)
        baseline=GraphPointerPolicy()
        training[f"gnn_base_{seed}"]=fit(baseline,dataset,updates=updates,seed=seed)
        base_state=copy.deepcopy(baseline.state_dict())
        torch.save(base_state,out/"checkpoints"/f"gnn_base_{seed}.pt")
        evaluate(baseline,"gnn_base",seed)
        for name,weight,over in [("gnn_continue",1.,1.),("gnn_weight",4.,1.),
                                  ("gnn_oversample",1.,3.),("gnn_generic",4.,3.)]:
            model=GraphPointerPolicy();model.load_state_dict(base_state)
            training[f"{name}_{seed}"]=fit(model,dataset,updates=more,seed=seed+1000,weight=weight,oversample=over)
            torch.save(model.state_dict(),out/"checkpoints"/f"{name}_{seed}.pt")
            evaluate(model,name,seed)
            if name=="gnn_generic" and not args.smoke:
                evaluate(model,"gnn_generic_rank_shuffle",seed,shuffle=True)
        seed_all(seed)
        mlp=FlatPointerPolicy()
        training[f"mlp_base_{seed}"]=fit(mlp,dataset,updates=updates,seed=seed)
        torch.save(mlp.state_dict(),out/"checkpoints"/f"mlp_base_{seed}.pt")
        evaluate(mlp,"mlp_base",seed)
        dump(out/"training.json",training)
    comparisons={}
    for a,b in [("mlp_base","gnn_base"),("gnn_base","gnn_continue"),
                ("gnn_continue","gnn_weight"),("gnn_continue","gnn_oversample"),
                ("gnn_continue","gnn_generic"),("gnn_weight","gnn_generic"),
                ("gnn_generic","gnn_generic_rank_shuffle")]:
        comparisons[f"{b}_minus_{a}"]={}
        for condition in conditions:
            ar=[r for r in all_rows if r["model"]==a and r["condition"]==condition[0]]
            br=[r for r in all_rows if r["model"]==b and r["condition"]==condition[0]]
            comparisons[f"{b}_minus_{a}"][condition[0]]=paired_bootstrap(ar,br)
    dump(out/"comparisons.json",comparisons)
    meta["checkpoint_sha256"]={p.name:sha(p) for p in (out/"checkpoints").glob("*.pt")}
    meta["demonstrations_sha256"]=sha(out/"demonstrations.pt")
    meta["status"]="completed"
    dump(out/"manifest.json",meta)
    print("COMPLETE",out,flush=True)

if __name__=="__main__":
    main()
