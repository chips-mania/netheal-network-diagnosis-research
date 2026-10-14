"""Provider-neutral adapter; no API calls or credentials are implemented here.

client(packet) -> {"action": {...}, "usage": {...}}
client must enforce its own network timeout. Malformed actions raise ValueError;
the external experiment runner must count them, not silently drop the episode.
"""
from .pointer_actions import PointerAction, PointerKind
from .graph_policy import FAULT_ORDER, NODE_FAULTS

NODE_FEATURES = ["observed_status","discovered","nominal_degree_normalized",
    "status_checked","observed_down","ping_failure_endpoint","high_latency_ping_endpoint",
    "interfaces_checked","observed_interface_down","id_lex_rank"]
EDGE_FEATURES = ["observed_connected","observed_latency_div_100ms","nominal",
    "reserved_zero","missing_after_scan","ping_done","ping_failed","ping_high_latency",
    "source_id_rank","destination_id_rank"]
GLOBAL_FEATURES = ["step_fraction","scanned","node_count","nominal_density",
    "missing_edge_fraction","ping_edge_fraction"]

def graph_packet(g):
    return {"schema_version":"netheal.graph-policy.v1",
        "nodes":g.node_ids,"edges":[list(x) for x in g.edge_keys],
        "node_feature_names":NODE_FEATURES,"node_features":g.node_feat.tolist(),
        "edge_feature_names":EDGE_FEATURES,"edge_features":g.edge_feat.tolist(),
        "global_feature_names":GLOBAL_FEATURES,"global_features":g.globals.tolist(),
        "allowed_kinds":[x.value for x in PointerKind],
        "fault_types":FAULT_ORDER,
        "action_schema":{
            "scan":{"kind":"scan"},
            "check_status":{"kind":"check_status","node":"node ID"},
            "check_interfaces":{"kind":"check_interfaces","node":"node ID"},
            "ping":{"kind":"ping","src":"edge source","dst":"edge destination"},
            "diagnose":{"kind":"diagnose","fault_type":"fault type",
                        "location":"node ID or src->dst edge"}},
        "constraints":"Use only supplied nodes/edges; link locations may reverse an existing edge. Inventory is known but live status is observed only. Reserved edge feature is always zero."}

def parse_action(data,g):
    if not isinstance(data,dict):
        raise ValueError("action must be a JSON object")
    kind=PointerKind(data.get("kind"))
    fields={"scan":{"kind"},"check_status":{"kind","node"},
            "check_interfaces":{"kind","node"},"ping":{"kind","src","dst"},
            "diagnose":{"kind","fault_type","location"}}[kind.value]
    if set(data)!=fields:
        raise ValueError("missing or unexpected action keys")
    if kind in (PointerKind.CHECK_STATUS,PointerKind.CHECK_INTERFACES):
        if data["node"] not in g.node_ids:
            raise ValueError("unknown node")
    elif kind==PointerKind.PING:
        if (data["src"],data["dst"]) not in g.edge_keys:
            raise ValueError("ping not in common edge action space")
    elif kind==PointerKind.DIAGNOSE:
        ft=data["fault_type"];loc=data["location"]
        if ft not in FAULT_ORDER:
            raise ValueError("unknown fault type")
        if ft in NODE_FAULTS:
            if loc not in g.node_ids: raise ValueError("unknown diagnosis node")
        else:
            edge=tuple(loc.split("->"))
            if edge not in g.edge_keys and edge[::-1] not in g.edge_keys:
                raise ValueError("unknown diagnosis edge")
    return PointerAction(kind=kind,**{k:v for k,v in data.items() if k!="kind"})

class ExternalPolicyAdapter:
    """Drop-in policy for run_research_suite.rollout; client owns provider IO."""
    def __init__(self,client,model_id,prompt_hash,temperature):
        self.client=client
        self.metadata={"model_id":model_id,"prompt_hash":prompt_hash,"temperature":temperature}
        self.calls=[]
    def eval(self):
        return self
    def act(self,g,device=None):
        response=self.client(graph_packet(g))
        if not isinstance(response,dict) or "action" not in response:
            raise ValueError("response requires action object")
        self.calls.append({"usage":response.get("usage",{}),"metadata":self.metadata})
        return parse_action(response["action"],g)
