"""Predefined primary contrasts and artifact integrity checks for factorial v3."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from .research_metrics import summarize
from .run_factorial_v3 import MODES,CURRICULA

def load(path):return json.loads(path.read_text(encoding="utf8"))
def lines(path):return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def bootstrap(values,case_ids,coverage=.975,repeats=4000,seed=940):
    # columns are scenarios; each scenario's model and variant contrasts stay paired.
    groups=defaultdict(list)
    for i,c in enumerate(case_ids):groups[c.split(":")[0]].append(i)
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(repeats):
        rs=rng.integers(values.shape[0],size=values.shape[0])
        cs=np.concatenate([rng.choice(ix,size=len(ix),replace=True) for ix in groups.values()])
        boot.append(values[rs][:,cs].mean())
    lo,hi=np.quantile(boot,[(1-coverage)/2,1-(1-coverage)/2])
    return {"delta":float(values.mean()),"interval":[float(lo),float(hi)],
        "coverage":coverage,"replicas":values.shape[0],"paired_scenarios":len(case_ids),
        "per_replica_delta":values.mean(axis=1).tolist(),
        "verdict":"improvement" if lo>0 else ("deterioration" if hi<0 else "inconclusive")}

def analyze(run):
    cfg=load(run/"protocol.json");meta=load(run/"manifest.json")
    assert meta["status"]=="completed","Run not completed"
    project=Path(__file__).resolve().parents[1]
    assert sha(run/"frozen_spec.md")==meta["spec_sha256"]
    for name,h in meta["sources"].items():
        assert sha(project/"src"/name)==h,f"source drift {name}"
    for name,h in meta["checkpoint_sha256"].items():
        assert sha(run/"checkpoints"/name)==h,f"checkpoint drift {name}"
    for name,h in meta["data_sha256"].items():
        assert sha(run/name)==h,f"data drift {name}"
    training_manifest=load(run/"training_complete_manifest.json")
    assert training_manifest["training_completed_unix"]<=meta["completed_unix"]
    summaries=load(run/"summaries.json")
    lookup={};gt={};total=0;case_sets={};row_groups=defaultdict(list)
    for path in sorted(run.glob("episodes_*.jsonl")):
        assert path.stat().st_mtime >= training_manifest["training_completed_unix"], "Evaluation artifact predates training completion"
        rows=lines(path);ids=[r["case_id"] for r in rows]
        assert len(ids)==len(set(ids)),path
        case_sets[path.name]=set(ids)
        groups=defaultdict(list)
        for r in rows:
            k=r["case_id"];truth=(r["episode_seed"],r["fault_type"],r["fault_location"])
            if k in gt:assert gt[k]==truth,f"unpaired scenario {k}"
            gt[k]=truth
            assert r["exact"]==(r["type_correct"] and r["location_correct"])
            assert not (r["timeout"] and r["submitted"])
            assert r["steps"]==len(r["decision_ms"]) and 1<=r["steps"]<=40
            lk=(r["model"],r["train_seed"],r["variant"],k)
            assert lk not in lookup
            lookup[lk]=r
            groups[r["condition"]].append(r)
            row_groups[(r["model"],r["variant"],r["condition"])].append(r)
        for c,block in groups.items():
            assert len(block)==cfg["episodes"]
            assert all(sum(r["fault_type"]==ft for r in block)==len(block)//4 for ft in
                  ["device_failure","performance_degradation","misconfiguration","link_failure"])
            r=block[0];recorded=summaries[r["model"]][str(r["train_seed"])][r["variant"]][c]
            assert summarize(block)==recorded,f"summary mismatch {path} {c}"
        total+=len(rows)
    ids=sorted(next(iter(case_sets.values())))
    assert all(set(ids)==s for s in case_sets.values()),"Different case sets"
    net=lines(run/"network_probes.jsonl")
    assert {r["case_id"] for r in net}==set(ids)
    evaluation_seeds={v[0] for v in gt.values()}
    train=load(run/"training.json")
    assert len(train)==len(cfg["replicas"])*6
    assert all(v["updates"]==cfg["updates"] for v in train.values())
    assert all(v["demos"]==3*cfg["demos_per_n"] for v in train.values())
    for rep in cfg["replicas"]:
        audit=load(run/f"demo_audit_{rep}.json")
        assert not evaluation_seeds & {e["seed"] for e in audit}
        star=load(run/f"selected_demos_star_{rep}.json")
        mixed=load(run/f"selected_demos_mixed_{rep}.json")
        assert len(star)==len(mixed)==3*cfg["demos_per_n"]
        for selected in [star,mixed]:
            for n in [4,5,6]:
                for fault in ["device_failure","performance_degradation","misconfiguration","link_failure"]:
                    assert sum(e["n"]==n and e["fault"]==fault for e in selected)==cfg["demos_per_n"]//4
        assert sum(e["topology"]=="star" for e in mixed)==len(mixed)//2
        assert {e["seed"] for e in mixed if e["topology"]=="star"} <= {e["seed"] for e in star}
    def value(model,rep,variant,case):
        return float(lookup[model,rep,variant,case]["exact"])
    def contrast(case_ids,func,coverage=.975):
        matrix=np.array([[func(rep,case) for case in case_ids] for rep in cfg["replicas"]])
        return bootstrap(matrix,case_ids,coverage)
    h1=contrast(ids,lambda r,c:np.mean([
        value(f"{cur}_augmented",r,"renamed",c)-value(f"{cur}_original",r,"renamed",c)
        for cur in CURRICULA]))
    h1gap=contrast(ids,lambda r,c:np.mean([
        value(f"{cur}_augmented",r,"renamed",c)-value(f"{cur}_augmented",r,"canonical",c)
       -value(f"{cur}_original",r,"renamed",c)+value(f"{cur}_original",r,"canonical",c)
        for cur in CURRICULA]),.95)
    heldout=[c for c in ids if c.startswith(("mesh_","random_"))]
    h2=contrast(heldout,lambda r,c:np.mean([
        value(f"mixed_{mode}",r,var,c)-value(f"star_{mode}",r,var,c)
        for mode in MODES for var in ["canonical","renamed"]]))
    h3=contrast(heldout,lambda r,c:np.mean([
        value("mixed_augmented",r,var,c)-value("mixed_original",r,var,c)
       -value("star_augmented",r,var,c)+value("star_original",r,var,c)
        for var in ["canonical","renamed"]]),.95)
    effects={"H1_augmentation_renamed":h1,"H1_gap_reduction_secondary":h1gap,
             "H2_diversity_heldout":h2,"H3_interaction_exploratory":h3}
    secondary={}
    for condition in [c[0] for c in cfg["conditions"]]:
        ci=[c for c in ids if c.split(":")[0]==condition]
        secondary[condition]={}
        for cur in CURRICULA:
            secondary[condition][f"{cur}_augmentation_renamed"]=contrast(ci,
                lambda r,c: value(f"{cur}_augmented",r,"renamed",c)-value(f"{cur}_original",r,"renamed",c),.95)
        for mode in MODES:
            secondary[condition][f"diversity_{mode}"]=contrast(ci,
                lambda r,c:np.mean([value(f"mixed_{mode}",r,v,c)-value(f"star_{mode}",r,v,c) for v in ["canonical","renamed"]]),.95)
    verification={"status":"passed","episodes":total,"paired_cases":len(ids),
        "trained_models":len(train),"source_files":len(meta["sources"]),
        "training_completed_before_evaluation":True,
        "checks":["paired ground truth","source/data/model hashes","summary recalculation",
                  "balanced faults","matched demo budgets","matched update budgets","disjoint train/eval seeds"]}
    (run/"verification.json").write_text(json.dumps(verification,indent=2),encoding="utf8")
    (run/"hypothesis_results.json").write_text(json.dumps(effects,indent=2),encoding="utf8")
    (run/"secondary_comparisons.json").write_text(json.dumps(secondary,indent=2),encoding="utf8")
    print(json.dumps({"verification":verification,"primary":effects},indent=2))
    return cfg,summaries,effects,net,train

def main():
    ap=argparse.ArgumentParser();ap.add_argument("run");args=ap.parse_args()
    analyze(Path(args.run))
if __name__=="__main__":main()
