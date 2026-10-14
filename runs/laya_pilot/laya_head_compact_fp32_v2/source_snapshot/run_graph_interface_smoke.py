"""Phase 2 smoke: graph_obs shapes + pointer stability across N."""
from __future__ import annotations

import json

from netheal.faults.injector import FaultType

from .env_utils import make_masked_eval_env
from .graph_obs import NODE_DIM, EDGE_DIM, build_graph_obs, pad_graph_obs
from .pointer_actions import smoke_pointer_stable_across_n
from .protocol import PROJECT_ROOT, load_protocol
from .rule_agent_v2 import RuleAgentV2


def _graph_smoke(protocol) -> dict:
    rows = {}
    for n, topo in [(4, "star"), (6, "star"), (8, "star"), (6, "mesh"), (6, "linear")]:
        env = make_masked_eval_env(
            [topo],
            fault_types=[FaultType.LINK_FAILURE],
            protocol=protocol,
            evidence_obs=True,
            min_devices=n,
            max_devices=n,
            max_episode_steps=25,
        )
        env.reset(seed=100 + n)
        # one scan so discovery fills
        agent = RuleAgentV2()
        obs, info = env.reset(seed=100 + n)
        a = agent.choose(env, obs, 1, [])
        obs, _, _, _, info = env.step(a)
        agent.observe_step(info)
        g = build_graph_obs(env)
        padded = pad_graph_obs(g, pad_nodes=12, pad_edges=12 * 11)
        rows[f"{topo}_{n}"] = {
            "n_nodes": g.n_nodes,
            "n_edges": g.n_edges,
            "node_feat_shape": list(g.node_feat.shape),
            "edge_feat_shape": list(g.edge_feat.shape),
            "edge_index_shape": list(g.edge_index.shape),
            "globals_shape": list(g.globals.shape),
            "pad_node_sum": float(padded["node_mask"].sum()),
            "node_dim_ok": g.node_feat.shape[1] == NODE_DIM,
            "edge_dim_ok": g.edge_feat.shape[1] == EDGE_DIM or g.n_edges == 0,
            "nodes_match": g.n_nodes == n,
        }
        env.close()
    ok = all(r["node_dim_ok"] and r["edge_dim_ok"] and r["nodes_match"] for r in rows.values())
    return {"ok": ok, "rows": rows}


def main() -> None:
    protocol = load_protocol()
    graph = _graph_smoke(protocol)
    print("graph", graph["ok"], {k: (v["n_nodes"], v["n_edges"]) for k, v in graph["rows"].items()})
    ptr = smoke_pointer_stable_across_n(protocol, sizes=(4, 6, 8))
    print(
        "pointer",
        ptr["ok"],
        "spaces",
        ptr["action_space_ns"],
        "ping_ids",
        ptr["ping_ids"],
    )
    ok = bool(graph["ok"] and ptr["ok"])
    decision = "go" if ok else "no_go"
    lines = [
        "# Graph interface smoke (Phase 2)",
        "",
        f"> Decision: **{decision}**",
        "",
        "## graph_obs",
        "",
        f"- ok: {graph['ok']}",
        "| Setting | N nodes | N edges | node_dim_ok |",
        "|---|---:|---:|:---:|",
    ]
    for k, v in graph["rows"].items():
        lines.append(f"| {k} | {v['n_nodes']} | {v['n_edges']} | {v['node_dim_ok']} |")
    lines.extend(
        [
            "",
            "## pointer_actions",
            "",
            f"- ok: {ptr['ok']} (action_space sizes vary across N: {ptr['spaces_vary']})",
            f"- action_space_n by N: {ptr['action_space_ns']}",
            f"- ping flat ids by N: {ptr['ping_ids']}",
            "",
            "Run: `python -m src.run_graph_interface_smoke`",
            "",
        ]
    )
    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "graph_interface_smoke.md").write_text("\n".join(lines), encoding="utf-8")
    (results / "graph_interface_smoke.json").write_text(
        json.dumps({"decision": decision, "graph": graph, "pointer": ptr}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"decision={decision}")
    print("Wrote", reports / "graph_interface_smoke.md")


if __name__ == "__main__":
    main()
