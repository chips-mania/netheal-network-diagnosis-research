"""Explicit episode-split LAYA training. Smoke is not a completed research fine-tune."""
import argparse,json,time,random
from pathlib import Path
from dataclasses import asdict
import torch
from laya.train import load_checkpoint,items_from_rows,TrainConfig,train_model,save_checkpoint,to_internal
from laya.common import build_sequence
from .run_research_suite import dump,sha
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--device",default="cpu");ap.add_argument("--base");ap.add_argument("--seed",type=int,default=41)
    ap.add_argument("--train-episodes",type=int);ap.add_argument("--epochs",type=int,default=4)
    ap.add_argument("--full-smoke",action="store_true")
    ap.add_argument("--freeze-encoder",action="store_true")
    ap.add_argument("--data-dir",default="results/laya_data_v1")
    args=ap.parse_args()
    if not args.smoke and args.device=="cpu":raise ValueError("Full run requires explicit GPU device; use --smoke for CPU feasibility.")
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[2]
    base=args.base or str(root/".runtime/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    model,tok,cfg=load_checkpoint(base)
    data=Path(args.data_dir)/"train.jsonl"
    rows=[json.loads(l) for l in data.read_text().splitlines()]
    if args.train_episodes:
        groups={}
        splits=json.loads((Path(args.data_dir)/"splits.json").read_text())
        for r in splits:
            if r["split"]=="train":groups.setdefault(tuple(r["group"]),set()).add(r["seed"])
        ordered=[]
        for key in sorted(groups):
            seeds=sorted(groups[key]);random.Random(args.seed).shuffle(seeds)
            groups[key]=seeds
        for i in range(max(map(len,groups.values()))):
            ordered.extend(groups[k][i] for k in sorted(groups) if i<len(groups[k]))
        selected=set(ordered[:args.train_episodes]);rows=[r for r in rows if r["episode_seed"] in selected]
    if args.smoke:rows=[rows[i*len(rows)//8] for i in range(8)]
    for row in rows:
        for name,q in row["questions"].items():
            _,markers,stats,trunc=build_sequence(tok,row["state"],to_internal(name,q),4096,1024,
                return_stats=True,return_truncation_stats=True)
            if trunc["truncated"] or stats["options_distinct"]!=stats["options"] or len(markers)!=len(q["criteria"]):
                raise ValueError("Refusing truncated training inputs")
    items,skipped=items_from_rows(tok,rows,4096,1024)
    if any(skipped.values()):raise ValueError("Skipped training questions")
    if args.smoke:items=items[:8]
    frozen=args.freeze_encoder or (args.smoke and not args.full_smoke)
    tc=TrainConfig(epochs=1 if args.smoke else args.epochs,micro_batch=1,grad_accum=1 if args.smoke else 16,
        freeze_encoder=frozen,max_len=4096,head_max_len=1024,seed=args.seed,
        shuffle_options=("choice",),calib_max=0,calib_frac=0,amp=False,
        gradient_checkpointing=not frozen,log_every=1 if args.smoke else 50)
    dump(out/"protocol.json",{"smoke":args.smoke,"config":asdict(tc),"training_items":len(items),
        "data_sha256":sha(data),"base":base,"selected_episode_seeds":sorted({r["episode_seed"] for r in rows}),
        "calibration":"not fitted; separate calibration.jsonl reserved"})
    head_before={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith("encoder.")}
    t=time.perf_counter()
    history=train_model(model,tok,items,tc,torch.device(args.device),4096,1024)
    seconds=time.perf_counter()-t
    # A smoke run saves head weights only to avoid implying a deployment-ready model.
    if args.smoke:torch.save({k:v for k,v in model.state_dict().items() if not k.startswith("encoder.")},out/"smoke_head.pt")
    else:save_checkpoint(model,tok,dict(cfg,max_len=4096,head_max_len=1024),str(out/"checkpoint"))
    dump(out/"summary.json",{"status":"smoke_completed" if args.smoke else "training_completed_uncalibrated",
        "seconds":seconds,"items":len(items),"history":history,"freeze_encoder":frozen,
        "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated() if args.device=="cuda" else None,
        "warning":"Smoke demonstrates gradients and runtime only, not useful fine-tuned accuracy.",
        "head_weights_changed":any(not torch.equal(head_before[k],v.detach().cpu()) for k,v in model.state_dict().items() if k in head_before),
        "full_training_completed":not args.smoke})
    print(json.dumps({"seconds":seconds,"items":len(items),"smoke":args.smoke}),flush=True)
if __name__=="__main__":main()
