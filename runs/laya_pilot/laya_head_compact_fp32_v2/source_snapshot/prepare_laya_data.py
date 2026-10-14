"""Export existing expert trajectories, split at physical episode level."""
import json,random,argparse
from pathlib import Path
from collections import defaultdict,Counter
import torch
from .laya_schema import state_text,questions,expected
from .run_research_suite import dump,sha
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--compact",action="store_true");args=ap.parse_args()
    global state_text
    if args.compact:
        from .laya_schema_compact import state_text
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    source=Path("results/factorial_v3_20261008")
    chosen=json.loads((source/"selected_demos_mixed_41.json").read_text())
    ids={r["seed"] for r in chosen}
    pool=torch.load(source/"demonstrations_41.pt",weights_only=False)
    groups=defaultdict(list)
    for ep in pool:
        if ep["seed"] in ids:groups[ep["topology"],ep["n"],ep["fault"]].append(ep)
    counts=Counter();assignments=[];step_counts=Counter()
    files={s:(out/(s+".jsonl")).open("w",encoding="utf8") for s in ("train","calibration","validation")}
    try:
        for group,eps in sorted(groups.items()):
            eps=sorted(eps,key=lambda e:e["seed"]);random.Random(712).shuffle(eps)
            assert len(eps)==88
            for i,ep in enumerate(eps):
                split="train" if i<64 else "calibration" if i<76 else "validation"
                counts[split]+=1;assignments.append(dict(seed=ep["seed"],split=split,group=group))
                for step,(g,label) in enumerate(zip(ep["graphs"],ep["labels"])):
                    q=questions(g);answer=expected(g,label)
                    row={"state":state_text(g),"questions":{k:q[k] for k in answer},"expected":answer,
                      "episode_seed":ep["seed"],"step":step,"topology":ep["topology"],"n":ep["n"]}
                    files[split].write(json.dumps(row)+"\n");step_counts[split]+=1
    finally:
        for f in files.values():f.close()
    dump(out/"splits.json",assignments)
    dump(out/"manifest.json",{"episodes":dict(counts),"steps":dict(step_counts),
      "source_sha256":{p.name:sha(p) for p in (source/"selected_demos_mixed_41.json",source/"demonstrations_41.pt")},
      "output_sha256":{p.name:sha(p) for p in out.glob("*.jsonl")},
      "schema_sha256":sha(Path(__file__).with_name("laya_schema_compact.py" if args.compact else "laya_schema.py")),
      "split_unit":"episode","seed":712,"training_data_replica":41,
      "warning":"Shared initial observations across distinct episodes are expected; never random-split individual steps."})
    print(json.dumps({"episodes":dict(counts),"steps":dict(step_counts)}))
if __name__=="__main__":main()
