"""Offline pilot audit and metrics. No API calls."""
import argparse,json,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from .run_research_suite import sha,dump,strict_location,FAULTS
from .research_metrics import dist,wilson

def metrics(rows):
    if not rows:return {}
    n=len(rows);correct=sum(r["exact"] for r in rows)
    req=[q for r in rows for q in r["requests"]]
    unknown=sum(q["estimated_usd"] is None for q in req)
    cost=sum(q["estimated_usd"] or 0 for q in req)
    confusion=Counter((r["fault_type"],r["predicted_type"] or "no_diagnosis") for r in rows)
    per={}
    for ft in FAULTS:
        sub=[r for r in rows if r["fault_type"]==ft];tp=confusion[ft,ft]
        fp=sum(v for (t,p),v in confusion.items() if t!=ft and p==ft);fn=len(sub)-tp
        per[ft]={"n":len(sub),"exact":sum(r["exact"] for r in sub)/len(sub) if sub else None,
            "precision":tp/(tp+fp) if tp+fp else 0,"recall":tp/len(sub) if sub else None,
            "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
    return {"n":n,"exact":correct/n,"exact_wilson95":wilson(correct,n),
        "type_accuracy":sum(r["type_correct"] for r in rows)/n,
        "location_accuracy":sum(r["location_correct"] for r in rows)/n,
        "macro_type_f1":sum(p["f1"] for p in per.values())/4,
        "wrong_diagnosis_rate":sum(r["predicted_type"] is not None and not r["exact"] for r in rows)/n,
        "termination_counts":dict(Counter(r["termination"] for r in rows)),
        "episode_seconds":dist([r["wall_seconds"] for r in rows]),
        "successful_episode_seconds":dist([r["wall_seconds"] for r in rows if r["exact"]]),
        "request_seconds":dist([q["seconds"] for q in req]),
        "tool_seconds":dist([sum(t["tool_seconds"] for t in r["trace"]) for r in rows]),
        "tool_calls":dist([r["tool_calls"] for r in rows]),"weighted_tool_cost":dist([r["tool_cost"] for r in rows]),
        "repeat_tool_rate":sum(r["repeated_tools"] for r in rows)/max(1,sum(r["tool_calls"] for r in rows)),
        "known_estimated_usd":cost,"unknown_cost_requests":unknown,
        "episode_usd":dist([r["estimated_usd"] for r in rows if r["estimated_usd"] is not None]),
        "usd_per_correct":cost/correct if correct and not unknown else None,
        "failed_episode_known_usd":sum(r["known_estimated_usd"] for r in rows if not r["exact"]),
        "api_requests":len(req),"input_tokens":sum((q["usage"] or {}).get("input_tokens",0) for q in req),
        "output_tokens":sum((q["usage"] or {}).get("output_tokens",0) for q in req),
        "cached_tokens":sum((q["usage"] or {}).get("input_tokens_details",{}).get("cached_tokens",0) for q in req),
        "reasoning_tokens":sum((q["usage"] or {}).get("output_tokens_details",{}).get("reasoning_tokens",0) for q in req),
        "per_fault":per,"confusion":[{"true":t,"predicted":p,"n":v} for (t,p),v in sorted(confusion.items())]}

def paired_analysis(rows,repeats=10000):
    by=defaultdict(dict)
    for r in rows:by[r["condition"]][(r["case_id"],r["provider"])]=r
    rng=np.random.default_rng(20261009);answer={}
    for metric in ("exact","wall_seconds","estimated_usd"):
        strata=[]
        for c,d in sorted(by.items()):
            ids=sorted({k[0] for k in d})
            values=[]
            for i in ids:
                if (i,"gpt") not in d or (i,"jev") not in d:continue
                a,b=d[i,"gpt"][metric],d[i,"jev"][metric]
                if a is None or b is None:continue
                values.append(float(a)-float(b))
            if values:strata.append(np.array(values))
        if not strata:continue
        boot=np.zeros(repeats)
        for v in strata:boot+=v[rng.integers(len(v),size=(repeats,len(v)))].mean(axis=1)/len(strata)
        answer[metric]={"direction":"GPT minus Jev","delta":float(np.mean([v.mean() for v in strata])),
            "exploratory_98_33_ci":np.quantile(boot,[.05/6,1-.05/6]).tolist(),
            "paired_cases":sum(len(v) for v in strata),"bootstrap":"paired within-condition case resampling"}
    return answer

def main():
    ap=argparse.ArgumentParser();ap.add_argument("run");args=ap.parse_args();out=Path(args.run)
    manifest=json.loads((out/"manifest.json").read_text())
    rows=[json.loads(l) for l in (out/"episodes.jsonl").read_text(encoding="utf8").splitlines()]
    schedule=json.loads((out/"schedule.json").read_text())
    errors=[]
    src=Path(__file__).parent
    for name,digest in manifest["source_sha256"].items():
        if sha(src/name)!=digest:errors.append("source hash: "+name)
    if sha(out/"config.json")!=manifest["config_sha256"]:errors.append("config hash")
    if sha(out/"instructions.txt")!=manifest["prompt_sha256"]:errors.append("prompt hash")
    identities=[(r["provider"],r["condition"],r["episode_seed"]) for r in rows]
    if len(set(identities))!=len(rows):errors.append("duplicate episodes")
    expected={(p,c[0],s) for p,c,i,s in schedule}
    if set(identities)!=expected:errors.append("incomplete schedule")
    by_case=defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)
        exact=r["predicted_type"]==r["fault_type"] and strict_location(r["fault_location"],r["predicted_location"],r["fault_type"])
        if r["exact"]!=exact:errors.append("exact recomputation")
        if not math.isclose(r["known_estimated_usd"],sum(q["estimated_usd"] or 0 for q in r["requests"]),abs_tol=1e-12):
            errors.append("cost sum")
    for cr in by_case.values():
        if len(cr)!=2 or len({(r["fault_type"],r["fault_location"],r["episode_seed"]) for r in cr})!=1:
            errors.append("pair mismatch")
    aggregate={p:metrics([r for r in rows if r["provider"]==p]) for p in ("gpt","jev")}
    conditions={c:{p:metrics([r for r in rows if r["provider"]==p and r["condition"]==c]) for p in ("gpt","jev")} for c in sorted({r["condition"] for r in rows})}
    dump(out/"metrics.json",{"overall":aggregate,"conditions":conditions,"paired_exploratory":paired_analysis(rows)})
    verification={"status":"passed" if not errors and manifest["status"]=="completed" else "incomplete_or_failed",
       "errors":errors,"episodes":len(rows),"paired_cases":len(by_case),"phase":manifest["phase"],
       "invoice_reconciled":False}
    dump(out/"verification.json",verification)
    lines=["# GPT-6.1 Sol vs Jev API pilot","","This is a development pilot, not the final confirmatory evaluation.",
      "Costs are usage-based USD estimates; invoices have not been reconciled.",
      f"Run status: {manifest['status']}; verification: {verification['status']}.","",
      "| Provider | Episodes | Exact | Wrong diagnosis | Mean seconds | p95 seconds | Mean USD | Total USD | Tools |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p,m in aggregate.items():
        if not m:continue
        mean_cost="unknown" if m["unknown_cost_requests"] else f"{m['episode_usd']['mean']:.6f}"
        lines.append(f"| {p} | {m['n']} | {m['exact']:.1%} | {m['wrong_diagnosis_rate']:.1%} | {m['episode_seconds']['mean']:.2f} | {m['episode_seconds']['p95']:.2f} | {mean_cost} | {m['known_estimated_usd']:.6f} | {m['tool_calls']['mean']:.2f} |")
    lines+=["","## Per condition","","| Condition | GPT Exact | Jev Exact | GPT seconds | Jev seconds |","|---|---:|---:|---:|---:|"]
    for c,ps in conditions.items():
        a,b=ps["gpt"],ps["jev"]
        if a and b:lines.append(f"| {c} | {a['exact']:.1%} | {b['exact']:.1%} | {a['episode_seconds']['mean']:.2f} | {b['episode_seconds']['mean']:.2f} |")
    lines+=["","Detailed fault metrics, failure reasons, token counts and timing distributions: [metrics.json](metrics.json).",
      "Requests and observations: requests.jsonl. Actions and ground truth: episodes.jsonl.",
      "One canonical run per physical case: repeat consistency and renamed robustness are not measured here.",
      "The policy consumes current public graph features, not raw full tool transcripts; results apply to this representation.",
      "Fast wrong diagnoses must not be interpreted as an efficiency win without reporting accuracy.",
      "Confidence intervals on this small development sample do not establish broad model superiority."]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    dump(out/"analysis_manifest.json",{"source_sha256":sha(Path(__file__))})
    print(json.dumps({"verification":verification,"overall":aggregate},ensure_ascii=False))
if __name__=="__main__":main()
