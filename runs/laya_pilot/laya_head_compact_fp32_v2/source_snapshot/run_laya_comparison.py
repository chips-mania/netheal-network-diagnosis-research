"""Local LAYA rollout on frozen API pilot cases, with model-selected branches."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import numpy as np
import torch,laya
from laya.train import to_internal
from laya.common import build_sequence
from .laya_schema import state_text,questions,pointer_from_answers
from .api_policy_client import APIError
from .run_api_comparison import episode
from .run_factorial_v3 import CONDITIONS
from .run_research_suite import dump,sha

class LayaClient:
    provider="laya"
    def __init__(self,agent):self.agent=agent;self.sink=lambda r:None
    def choose(self,packet,options,deadline,context):
        g=SimpleNamespace(node_ids=packet["nodes"],edge_keys=packet["edges"],
            node_feat=packet["node_features"],edge_feat=packet["edge_features"],globals=packet["global_features"])
        state=state_text(g);qs=questions(g);answers={}
        def ask(name):
            if time.monotonic()>=deadline:raise APIError("watchdog")
            q=qs[name]
            _,markers,stats,trunc=build_sequence(self.agent.tok,state,to_internal(name,q),4096,1024,
                return_stats=True,return_truncation_stats=True)
            if trunc["truncated"] or len(markers)!=len(q["criteria"]) or stats["options_distinct"]!=stats["options"]:
                raise APIError("input_capacity_exceeded")
            started=time.perf_counter()
            result=self.agent.predict(state,{name:q},max_len=4096,head_max_len=1024)
            torch.cuda.synchronize()
            value=result["answers"][name]["choice"];answers[name]=value
            self.sink(dict(seconds=time.perf_counter()-started,usage=result["usage"],estimated_usd=0,
                question=name,choice=value,step=context["step"],response_model="laya-local"))
            return value
        kind=ask("kind")
        if kind in ("check_status","check_interfaces"):ask("node")
        elif kind=="ping":ask("edge")
        elif kind=="diagnose":
            ft=ask("fault");ask("diagnosis_node" if ft in ("device_failure","misconfiguration") else "diagnosis_edge")
        p=pointer_from_answers(answers)
        action={"kind":p.kind.value}
        if kind in ("check_status","check_interfaces"):action["node"]=p.node
        elif kind=="ping":action.update(src=p.src,dst=p.dst)
        elif kind=="diagnose":action.update(fault_type=p.fault_type,location=p.location)
        return action

def summarize(rows):
    n=len(rows)
    if not n:return {}
    faults=sorted({r["fault_type"] for r in rows});f1=[]
    for f in faults:
        tp=sum(r["fault_type"]==f and r["predicted_type"]==f for r in rows)
        fp=sum(r["fault_type"]!=f and r["predicted_type"]==f for r in rows)
        fn=sum(r["fault_type"]==f and r["predicted_type"]!=f for r in rows)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return dict(episodes=n,exact=sum(r["exact"] for r in rows)/n,
        type_accuracy=sum(r["type_correct"] for r in rows)/n,
        location_accuracy=sum(r["location_correct"] for r in rows)/n,fault_macro_f1=float(np.mean(f1)),
        termination_counts=dict(Counter(r["termination"] for r in rows)),
        wrong_diagnosis_rate=sum(r["termination"]=="diagnosed" and not r["exact"] for r in rows)/n,
        mean_seconds=float(np.mean([r["wall_seconds"] for r in rows])),
        p95_seconds=float(np.percentile([r["wall_seconds"] for r in rows],95)),
        mean_tool_calls=float(np.mean([r["tool_calls"] for r in rows])),
        mean_tool_cost=float(np.mean([r["tool_cost"] for r in rows])),
        repeated_tool_rate=sum(r["repeated_tools"] for r in rows)/max(1,sum(r["tool_calls"] for r in rows)),
        api_cost_usd=0,local_compute_cost_usd=None,
        conditions={c:dict(n=len(rs),exact=sum(r["exact"] for r in rs)/len(rs))
            for c in sorted({r["condition"] for r in rows}) if (rs:=[r for r in rows if r["condition"]==c])})

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--checkpoint")
    ap.add_argument("--cases-per-condition",type=int,default=8)
    ap.add_argument("--compact",action="store_true");ap.add_argument("--resume",action="store_true");args=ap.parse_args()
    global state_text
    if args.compact:
        from .laya_schema_compact import state_text
    if args.cases_per_condition%4:raise ValueError("Use balanced multiples of four")
    root=Path(__file__).resolve().parents[2]
    checkpoint=args.checkpoint or str(root/".runtime/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=args.resume);torch.set_num_threads(4)
    cfg=json.loads(Path("config/api_comparison_v1.json").read_text())
    started=time.perf_counter();agent=laya.load(checkpoint,device="cuda");load_seconds=time.perf_counter()-started
    # GTX 16xx lacks Tensor Cores: default CUDA fp16 is substantially slower here.
    agent.amp_enabled=False;agent.dtype=torch.float32
    client=LayaClient(agent);rows=[]
    if args.resume and (out/"episodes.jsonl").exists():
        rows=[json.loads(l) for l in (out/"episodes.jsonl").read_text(encoding="utf8").splitlines()]
    manifest=dict(status="running",checkpoint=checkpoint,device=torch.cuda.get_device_name(),torch=torch.__version__,precision="fp32, autocast disabled",
        load_seconds=load_seconds,source_sha256={p.name:sha(p) for p in Path("src").glob("*.py")},
        expected_episodes=8*args.cases_per_condition,representation="compact-table-v2" if args.compact else "named-features-v1",
        comparable="same physical cases and 40 decisions as API pilot; different question interface and local hardware",
        checkpoint_sha256=sha(Path(checkpoint)/"model.safetensors"))
    if args.resume:
        previous=json.loads((out/"manifest.json").read_text(encoding="utf8"))
        for key in ("checkpoint_sha256","representation","expected_episodes","source_sha256"):
            if previous[key]!=manifest[key]:raise ValueError("Resume protocol mismatch: "+key)
    completed={r["case_id"] for r in rows}
    dump(out/"manifest.json",manifest)
    try:
        for ci,c in enumerate(CONDITIONS):
            for i in range(args.cases_per_condition):
                if f"{c[0]}:{i}" in completed:continue
                row=episode(client,c,i,cfg["evaluation"]["pilot_seed_base"]+ci*10000+i,"canonical",0,cfg,out)
                rows.append(row);dump(out/"summary.json",summarize(rows))
                print(json.dumps(dict(done=len(rows),case=row["case_id"],exact=row["exact"],termination=row["termination"])),flush=True)
        manifest["status"]="completed"
    except BaseException as exc:
        manifest["status"]="interrupted";manifest["error_type"]=type(exc).__name__
        raise
    finally:
        manifest["completed_episodes"]=len(rows);manifest["peak_gpu_allocated_bytes"]=torch.cuda.max_memory_allocated()
        manifest["cpu_fallback_count"]=agent.cpu_fallback_count
        dump(out/"manifest.json",manifest)
if __name__=="__main__":main()
