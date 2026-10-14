"""Lossless named rendering of public graph features; no hidden state or teacher hints."""
from .external_policy_adapter import NODE_FEATURES,EDGE_FEATURES,GLOBAL_FEATURES
from .graph_policy import FAULT_ORDER,NODE_FAULTS,KIND_ORDER
from .pointer_actions import PointerAction,PointerKind

def state_text(g):
    def render(names,values):
        return ", ".join(f"{k}={float(v):.5g}" for k,v in zip(names,values))
    return ("Only observed facts; zero observation flags are not proof of health. "
        "observed_status: -1 down, 0 unknown, 1 up. Latency is divided by 100 ms.\n"+
        "Network: "+render(GLOBAL_FEATURES,g.globals)+"\n"+
        "\n".join(n+": "+render(NODE_FEATURES,f) for n,f in zip(g.node_ids,g.node_feat))+"\n"+
        "\n".join(a+"->"+b+": "+render(EDGE_FEATURES,f) for (a,b),f in zip(g.edge_keys,g.edge_feat)))

def questions(g):
    def choice(instructions,criteria):
        return dict(type="choice",instructions=instructions,criteria=criteria)
    nodes={n:"device "+n for n in g.node_ids}
    edges={"->".join(e):"link from "+e[0]+" to "+e[1] for e in g.edge_keys}
    return {
      "kind":choice("Choose the next operation to identify the single fault. Investigate when evidence is insufficient; diagnose when sufficient. Avoid repeating known observations.",
          {"scan":"Discover live connectivity.","check_status":"Read one device's operational status.",
           "check_interfaces":"Read one device's interface status.","ping":"Test connectivity and latency on one directed link.",
           "diagnose":"Submit the fault type and location based on observed evidence."}),
      "node":choice("If inspecting a device next, which device should be inspected to resolve the fault?",nodes),
      "edge":choice("If running ping next, which directed link should be tested to resolve the fault?",edges),
      "fault":choice("If submitting a diagnosis now, which fault type is supported by the observations?",
          {"device_failure":"A device is down.","performance_degradation":"A link has excessive latency.",
           "misconfiguration":"A device has a configuration fault.","link_failure":"A link is broken."}),
      "diagnosis_node":choice("If the fault is device_failure or misconfiguration, which device is the cause?",nodes),
      "diagnosis_edge":choice("If the fault is link_failure or performance_degradation, which link is the cause?",edges)}

def expected(g,label):
    kind=KIND_ORDER[label["kind"]].value;ans={"kind":kind}
    if kind in ("check_status","check_interfaces"):ans["node"]=g.node_ids[label["node"]]
    elif kind=="ping":ans["edge"]="->".join(g.edge_keys[label["edge"]])
    elif kind=="diagnose":
        ft=FAULT_ORDER[label["fault"]];ans["fault"]=ft
        ans["diagnosis_node" if ft in NODE_FAULTS else "diagnosis_edge"]=(
          g.node_ids[label["node"]] if ft in NODE_FAULTS else "->".join(g.edge_keys[label["edge"]]))
    return ans

def pointer_from_answers(ans):
    kind=ans["kind"]
    if kind=="scan":return PointerAction(PointerKind.SCAN)
    if kind in ("check_status","check_interfaces"):return PointerAction(PointerKind(kind),node=ans["node"])
    if kind=="ping":
        a,b=ans["edge"].split("->");return PointerAction(PointerKind.PING,src=a,dst=b)
    ft=ans["fault"]
    return PointerAction(PointerKind.DIAGNOSE,fault_type=ft,
        location=ans["diagnosis_node" if ft in NODE_FAULTS else "diagnosis_edge"])
