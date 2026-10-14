"""Credential and model/schema preflight; two small paid requests, no benchmark claims."""
import json,time
from pathlib import Path
from dotenv import dotenv_values
from .api_policy_client import Client,APIError
from .run_research_suite import dump
def main():
    root=Path(__file__).resolve().parents[2]
    cfg=json.loads((root/"netheal-topology-generalization/config/api_comparison_v1.json").read_text())
    env=dotenv_values(root/".env")
    out=root/"netheal-topology-generalization/results"/("api_preflight_"+time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir()
    results={}
    for name in ("gpt","jev"):
        rows=[]
        key=env.get(cfg["providers"][name]["key_env"])
        if not key:results[name]={"status":"missing_key"};continue
        c=Client(name,cfg,key,rows.append)
        try:
            action=c.choose({"test":"Connectivity check. Only scan is allowed."},{"a000":{"kind":"scan"}},
                time.monotonic()+120,{"phase":"preflight"})
            results[name]={"status":"passed","action":action}
        except APIError as e:results[name]={"status":e.kind,"http_status":e.status}
        finally:c.close()
        dump(out/(name+".json"),rows)
        results[name]["estimated_usd"]=sum(r["estimated_usd"] or 0 for r in rows)
        results[name]["cost_unknown_requests"]=sum(r["estimated_usd"] is None for r in rows)
    dump(out/"status.json",results)
    print(json.dumps({"path":str(out),"results":results}),flush=True)
if __name__=="__main__":main()
