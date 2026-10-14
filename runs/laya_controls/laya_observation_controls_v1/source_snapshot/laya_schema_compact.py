"""Compact public feature table: column names once, all values retained."""
from .laya_schema import questions,expected,pointer_from_answers
from .external_policy_adapter import NODE_FEATURES,EDGE_FEATURES,GLOBAL_FEATURES

def state_text(g):
    def values(xs):return " ".join(f"{float(x):.5g}" for x in xs)
    return ("Observed network only. Unchecked zero is unknown, not healthy. Status -1 down, 0 unknown, 0.5 degraded, 1 up.\n"
        +"Global columns: "+" ".join(GLOBAL_FEATURES)+"\n"+values(g.globals)+"\n"
        +"Device columns: id "+" ".join(NODE_FEATURES)+"\n"
        +"\n".join(n+" "+values(v) for n,v in zip(g.node_ids,g.node_feat))+"\n"
        +"Link columns: source->destination "+" ".join(EDGE_FEATURES)+"\n"
        +"\n".join(a+"->"+b+" "+values(v) for (a,b),v in zip(g.edge_keys,g.edge_feat)))
