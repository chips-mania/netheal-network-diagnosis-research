"""CPU feasibility and teacher-action baseline; no paid API calls."""
import argparse,json,time
from pathlib import Path
import torch,laya
from laya.common import build_sequence
from laya.train import to_internal
from .run_research_suite import dump,sha
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--limit",type=int,default=32);args=ap.parse_args()
    torch.set_num_threads(4)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    checkpoint=root/".runtime/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148"
    start=time.perf_counter();agent=laya.load(str(checkpoint),device="cpu")
    load_seconds=time.perf_counter()-start
    data=Path("results/laya_data_v1/validation.jsonl")
    allrows=[json.loads(l) for l in data.read_text().splitlines()]
    # Deterministic evenly spaced examples, fixed before inspecting predictions.
    indexes=[i*len(allrows)//args.limit for i in range(args.limit)]
    results=[]
    for j,i in enumerate(indexes):
        row=allrows[i];qs=row["questions"];audit={}
        for name,q in qs.items():
            ids,markers,stats,trunc=build_sequence(agent.tok,row["state"],to_internal(name,q),4096,1024,
                return_stats=True,return_truncation_stats=True)
            if trunc["truncated"] or stats["options_distinct"]!=stats["options"] or len(markers)!=len(q["criteria"]):
                raise ValueError("Input truncation/collapsed options")
            audit[name]={"tokens":len(ids),"state_tokens_dropped":trunc["state_tokens_dropped"]}
        t=time.perf_counter();r=agent.predict(row["state"],qs,max_len=4096,head_max_len=1024)
        seconds=time.perf_counter()-t
        ans={k:v["choice"] for k,v in r["answers"].items()}
        result={"episode_seed":row["episode_seed"],"step":row["step"],"expected":row["expected"],
            "predicted":ans,"all_teacher_answers_correct":ans==row["expected"],"seconds":seconds,
            "usage":r["usage"],"audit":audit}
        results.append(result)
        with (out/"decisions.jsonl").open("a") as f:f.write(json.dumps(result)+"\n")
        print(json.dumps({"done":j+1,"total":args.limit,"seconds":seconds,"correct":ans==row["expected"]}),flush=True)
    dump(out/"summary.json",{"status":"completed","model_revision":checkpoint.name,"package":"0.4.1",
        "device":"cpu","threads":4,"load_seconds":load_seconds,"n":len(results),
        "teacher_action_match":sum(r["all_teacher_answers_correct"] for r in results)/len(results),
        "mean_seconds":sum(r["seconds"] for r in results)/len(results),
        "input_truncation":False,"data_sha256":sha(data),
        "interpretation":"Teacher-forced action matching, NOT end-to-end diagnosis accuracy. One arbitrary teacher action may not be uniquely optimal.",
        "api_cost_usd":0,"compute_cost_usd":None})
if __name__=="__main__":main()
