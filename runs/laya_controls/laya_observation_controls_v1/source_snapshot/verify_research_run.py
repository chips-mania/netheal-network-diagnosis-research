"""Verify completed run integrity without retraining or changing artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from .research_metrics import summarize

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("run");args=ap.parse_args()
    run=Path(args.run);project=Path(__file__).resolve().parents[1]
    m=json.loads((run/"manifest.json").read_text(encoding="utf8"))
    p=json.loads((run/"protocol.json").read_text(encoding="utf8"))
    assert m["status"]=="completed"
    for name,h in m["source_sha256"].items():
        assert sha(project/name)==h, f"Source drift: {name}"
    for name,h in m["checkpoint_sha256"].items():
        assert sha(run/"checkpoints"/name)==h,f"Checkpoint drift: {name}"
    assert sha(run/"demonstrations.pt")==m["demonstrations_sha256"]
    summaries=json.loads((run/"summaries.json").read_text(encoding="utf8"))
    gt={};total=0;case_sets={}
    for path in run.glob("episodes_*.jsonl"):
        rows=[json.loads(x) for x in path.read_text(encoding="utf8").splitlines()]
        ids=[r["case_id"] for r in rows];assert len(ids)==len(set(ids)),path
        case_sets[path.name]=set(ids)
        groups=defaultdict(list)
        for r in rows:
            key=r["case_id"];value=(r["episode_seed"],r["fault_type"],r["fault_location"])
            if key in gt: assert gt[key]==value,f"Unpaired scenario: {key}"
            gt[key]=value
            assert r["exact"]==(r["type_correct"] and r["location_correct"])
            assert not (r["timeout"] and r["submitted"])
            assert 1<=r["steps"]<=p["budget"]
            assert len(r["decision_ms"])==r["steps"]
            groups[r["condition"]].append(r)
        for condition,block in groups.items():
            assert len(block)==p["eval_episodes"]
            assert len(set(r["episode_seed"] for r in block))==len(block)
            assert all(sum(r["fault_type"]==ft for r in block)==len(block)//4 for ft in
                       ["device_failure","performance_degradation","misconfiguration","link_failure"])
            reconstructed=summarize(block)
            recorded=summaries[block[0]["model"]][str(block[0]["train_seed"])][condition]
            assert reconstructed==recorded, f"Summary mismatch: {path} {condition}"
        total+=len(rows)
    first=next(iter(case_sets.values()))
    assert all(v==first for v in case_sets.values()),"Different evaluation sets"
    network=[json.loads(x) for x in (run/"network_probes.jsonl").read_text(encoding="utf8").splitlines()]
    assert {x["case_id"] for x in network}==first
    training=json.loads((run/"training.json").read_text(encoding="utf8"))
    for s in p["seeds"]:
        assert training[f"gnn_base_{s}"]["updates"]==training[f"mlp_base_{s}"]["updates"]
        updates=[training[f"{name}_{s}"]["updates"] for name in
                 ["gnn_continue","gnn_weight","gnn_oversample","gnn_generic"]]
        assert len(set(updates))==1
    print(json.dumps({"status":"passed","episode_rows":total,"paired_cases":len(first),
                      "checkpoints":len(m["checkpoint_sha256"]),"source_files":len(m["source_sha256"])},indent=2))
if __name__=="__main__":main()
