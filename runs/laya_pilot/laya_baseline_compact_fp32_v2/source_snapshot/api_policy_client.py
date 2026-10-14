"""API IO and common candidate policy. Credentials never enter persisted records."""
import json, time, hashlib
import httpx
from .external_policy_adapter import graph_packet
from .graph_policy import FAULT_ORDER, NODE_FAULTS

INSTRUCTIONS = """Choose the next action to diagnose exactly one network fault, including its type and location.
Use only the observed state. Nominal inventory is not live health. Unknown observations are not healthy evidence.
Faults: device_failure is a failed node; misconfiguration is a node configuration fault;
link_failure is a broken link; performance_degradation is an abnormally slow link.
Gather evidence with tools when needed, then diagnose. Minimize unnecessary or repeated tools.
Select one complete action from the supplied candidates. Do not assume hidden state."""

def candidates(g):
    a=[dict(kind="scan")]
    for n in g.node_ids:
        a += [dict(kind="check_status",node=n),dict(kind="check_interfaces",node=n)]
    a += [dict(kind="ping",src=s,dst=d) for s,d in g.edge_keys]
    for ft in FAULT_ORDER:
        locations=g.node_ids if ft in NODE_FAULTS else list(dict.fromkeys("->".join(sorted(e)) for e in g.edge_keys))
        a += [dict(kind="diagnose",fault_type=ft,location=l) for l in locations]
    if len(a)>255: raise ValueError("Jev Choice candidate limit exceeded")
    return {f"a{i:03d}":v for i,v in enumerate(a)}

def packet(g):
    p=graph_packet(g)
    return p,candidates(g)

def estimate_cost(provider, usage, pricing):
    if not isinstance(usage,dict) or "input_tokens" not in usage: return None
    if provider=="jev": return usage["input_tokens"]*pricing["jev_usd_per_million"]["input"]/1e6
    if "output_tokens" not in usage:return None
    rates=pricing["gpt_usd_per_million"]
    cached=usage.get("input_tokens_details",{}).get("cached_tokens",0)
    # Cache-write accounting is not inferred from absent fields.
    return ((usage["input_tokens"]-cached)*rates["input_uncached"]+cached*rates["input_cached"]+
            usage["output_tokens"]*rates["output"])/1e6

class APIError(RuntimeError):
    def __init__(self,kind,status=None):
        self.kind,self.status=kind,status
        super().__init__(f"{kind} (HTTP {status})")

class Client:
    def __init__(self,provider,cfg,key,sink):
        self.provider,self.cfg,self.sink=provider,cfg,sink
        self.client=httpx.Client(headers={"Authorization":"Bearer "+key},follow_redirects=False)
    def close(self):self.client.close()
    def choose(self,state,options,deadline,context):
        p=self.cfg["providers"][self.provider]
        if self.provider=="gpt":
            url=p["base_url"]+"/responses"
            body={"model":p["model"],"instructions":INSTRUCTIONS,
              "input":json.dumps({"state":state,"candidates":options},separators=(",",":")),
              "reasoning":{"effort":p["reasoning_effort"]},"max_output_tokens":p["max_output_tokens"],
              "service_tier":p["service_tier"],"store":False,
              "text":{"format":{"type":"json_schema","name":"action_choice","strict":True,
                  "schema":{"type":"object","properties":{"action_id":{"type":"string","enum":list(options)}},
                    "required":["action_id"],"additionalProperties":False}}}}
        else:
            url=p["endpoint"]
            body={"model":p["model"],"state":state,"questions":{"action":{"type":"choice",
                   "instructions":INSTRUCTIONS,"criteria":{k:json.dumps(v,separators=(",",":")) for k,v in options.items()}}}}
        raw=json.dumps(body,separators=(",",":")).encode()
        for attempt in range(self.cfg["common"]["max_transport_retries"]+1):
            remaining=deadline-time.monotonic()
            if remaining<=0:raise APIError("watchdog")
            record=dict(context,provider=self.provider,attempt=attempt,model_requested=p["model"],
                utc_unix=time.time(),request_sha256=hashlib.sha256(raw).hexdigest(),request=body)
            started=time.perf_counter();response=None;data={};error=None
            try:
                response=self.client.post(url,content=raw,headers={"Content-Type":"application/json"},
                    timeout=min(remaining,self.cfg["common"]["request_timeout_seconds"]))
                record["http_status"]=response.status_code
                record["request_id"]=response.headers.get("x-request-id")
                try:data=response.json()
                except ValueError:pass
                if response.status_code>=400:error=APIError("http_error",response.status_code)
            except httpx.HTTPError:
                error=APIError("transport_error")
            record["seconds"]=time.perf_counter()-started
            record["usage"]=data.get("usage")
            record["estimated_usd"]=estimate_cost(self.provider,data.get("usage"),self.cfg["pricing"])
            record["response_model"]=data.get("model")
            # Persist only allowlisted successful response fields; never raw errors/headers.
            if error is None:record["response"]={k:data[k] for k in ("id","model","status","output","answers","usage","incomplete_details") if k in data}
            else:record["error_kind"]=error.kind
            self.sink(record)
            if error:
                retry=error.status in (429,500,502,503,504,529) or error.kind=="transport_error"
                if retry and attempt<self.cfg["common"]["max_transport_retries"]:
                    delay=self.cfg["common"]["retry_backoff_seconds"]
                    if response is not None:
                        try:delay=max(delay,float(response.headers.get("retry-after","0")))
                        except ValueError:pass
                    if time.monotonic()+delay>=deadline:raise APIError("watchdog")
                    time.sleep(delay);continue
                raise error
            if time.monotonic()>deadline:raise APIError("watchdog")
            try:
                if self.provider=="gpt":
                    if data.get("status")!="completed":raise ValueError()
                    text="".join(c.get("text","") for o in data.get("output",[]) for c in o.get("content",[]) if c.get("type")=="output_text")
                    choice=json.loads(text)["action_id"]
                else:choice=data["answers"]["action"]["choice"]
                if choice not in options:raise ValueError()
                return options[choice]
            except (KeyError,ValueError,TypeError):raise APIError("invalid_output")
