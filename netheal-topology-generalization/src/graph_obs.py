"""Build graph-structured tensors from NetHeal env (no package edits).

Includes tool-evidence history when `mem` (RuleMemory-like) is provided.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from netheal.environment.observation import ConnectionStatus

from .action_mask import base_env
from .evidence_obs import LATENCY_SCALE, _status_feat, structured_obs

# node: status, discovered, degree, status_checked, is_down, ping_fail_ep, ping_hi_ep,
#       iface_checked, iface_has_down, id_rank (breaks star spoke isomorphism; Rule uses lex order)
NODE_DIM = 10
# edge: connected, lat_norm, nominal, link_down, missing, ping_done, ping_fail, ping_hi_lat,
#       src_rank, dst_rank
EDGE_DIM = 10


@dataclass
class GraphObs:
    node_ids: List[str]
    node_index: Dict[str, int]
    node_feat: np.ndarray
    edge_index: np.ndarray
    edge_feat: np.ndarray
    edge_keys: List[Tuple[str, str]]
    globals: np.ndarray
    n_nodes: int
    n_edges: int


def _device_ids(root) -> List[str]:
    if root.network is None:
        return []
    return [str(d) for d in root.network.get_all_devices()]


def _nominal_edges(root) -> List[Tuple[str, str]]:
    if root.network is None:
        return []
    return [(str(a), str(b)) for a, b in root.network.get_all_connections()]


def build_graph_obs(env, *, mem: Any = None) -> GraphObs:
    """Construct GraphObs. Pass RuleAgentV2.mem (or compatible) for evidence history."""
    root = base_env(env)
    struct = structured_obs(env)
    node_ids = _device_ids(root)
    node_index = {d: i for i, d in enumerate(node_ids)}
    n = len(node_ids)
    node_feat = np.zeros((n, NODE_DIM), dtype=np.float32)

    discovered = set()
    connected: set = set()
    lat_edge: Dict[Tuple[str, str], float] = {}
    if struct is not None:
        discovered = set(str(x) for x in struct.discovery_matrix.get_discovered_devices())
        matrix = struct.discovery_matrix
        for i in range(matrix.adjacency.shape[0]):
            for j in range(matrix.adjacency.shape[1]):
                src = matrix.reverse_map.get(i)
                dst = matrix.reverse_map.get(j)
                if not src or not dst:
                    continue
                src, dst = str(src), str(dst)
                if int(matrix.adjacency[i, j]) == ConnectionStatus.CONNECTED.value:
                    connected.add((src, dst))
        for (si, di), props in (matrix.connection_properties or {}).items():
            src = matrix.reverse_map.get(si)
            dst = matrix.reverse_map.get(di)
            if not src or not dst:
                continue
            lat = props.get("latency")
            if lat is not None:
                lat_edge[(str(src), str(dst))] = float(lat)
        for did, info in struct.device_status.devices.items():
            if str(did) not in node_index:
                continue
            i = node_index[str(did)]
            node_feat[i, 0] = _status_feat(info.get("status"))

    # Stable lex ranks (matches Rule v2 sort keys; breaks star spoke isomorphism)
    sorted_ids = sorted(node_ids)
    denom_r = max(len(sorted_ids) - 1, 1)
    id_rank = {d: (i / denom_r) for i, d in enumerate(sorted_ids)}
    if len(sorted_ids) == 1:
        id_rank[sorted_ids[0]] = 0.0

    for i, did in enumerate(node_ids):
        node_feat[i, 1] = 1.0 if (not discovered) or (did in discovered) else 0.0
        node_feat[i, 9] = float(id_rank.get(did, 0.0))

    deg = {d: 0 for d in node_ids}
    nominal = _nominal_edges(root)
    seen_u = set()
    for a, b in nominal:
        key = tuple(sorted((a, b)))
        if key in seen_u:
            continue
        seen_u.add(key)
        if a in deg:
            deg[a] += 1
        if b in deg:
            deg[b] += 1
    max_deg = max(deg.values()) if deg else 1
    for i, did in enumerate(node_ids):
        node_feat[i, 2] = float(deg[did]) / max(max_deg, 1)

    # --- evidence history from mem ---
    ping_map: Dict[Tuple[str, str], dict] = {}
    status_map: Dict[str, str] = {}
    iface_map: Dict[str, dict] = {}
    missing_mem = set()
    scanned = False
    if mem is not None:
        ping_map = dict(getattr(mem, "ping", {}) or {})
        status_map = dict(getattr(mem, "status", {}) or {})
        iface_map = dict(getattr(mem, "ifaces", {}) or {})
        missing_mem = set(getattr(mem, "missing", set()) or set())
        scanned = bool(getattr(mem, "scanned", False))

    for did, st in status_map.items():
        if did not in node_index:
            continue
        i = node_index[did]
        node_feat[i, 3] = 1.0  # status checked
        if str(st).lower() == "down":
            node_feat[i, 0] = -1.0
            node_feat[i, 4] = 1.0

    for (src, dst), pr in ping_map.items():
        if src in node_index and not pr.get("success", True):
            node_feat[node_index[src], 5] = 1.0
        if dst in node_index and not pr.get("success", True):
            node_feat[node_index[dst], 5] = 1.0
        if pr.get("success") and pr.get("latency_ms") is not None and pr["latency_ms"] >= 10.0:
            if src in node_index:
                node_feat[node_index[src], 6] = 1.0
            if dst in node_index:
                node_feat[node_index[dst], 6] = 1.0

    for did, iface in iface_map.items():
        if did not in node_index:
            continue
        i = node_index[did]
        node_feat[i, 7] = 1.0
        downs = [
            1
            for x in (iface.get("interfaces") or [])
            if str(x.get("status", "")).lower() == "down"
        ]
        if downs:
            node_feat[i, 8] = 1.0

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_feat_list: List[List[float]] = []
    edge_keys: List[Tuple[str, str]] = []
    for a, b in nominal:
        if a not in node_index or b not in node_index:
            continue
        ia, ib = node_index[a], node_index[b]
        is_conn = 1.0 if (a, b) in connected else 0.0
        lat = lat_edge.get((a, b))
        if lat is None and root.network is not None:
            try:
                lat = float(root.network.graph.edges[a, b].get("latency", 0.0) or 0.0)
            except Exception:
                lat = 0.0
        lat_n = float(lat or 0.0) / LATENCY_SCALE
        is_down = 0.0
        if root.network is not None and not root.network.is_connection_up(a, b):
            is_down = 1.0
        missing = 1.0 if ((a, b) in missing_mem) or ((a, b) not in connected and scanned) else 0.0
        # if mem.missing empty but scanned, missing := nominal not connected
        if scanned and (a, b) not in connected:
            missing = 1.0
        ping_done = ping_fail = ping_hi = 0.0
        pr = ping_map.get((a, b))
        if pr is not None:
            ping_done = 1.0
            if not pr.get("success", True):
                ping_fail = 1.0
            elif pr.get("latency_ms") is not None and pr["latency_ms"] >= 10.0:
                ping_hi = 1.0
                lat_n = max(lat_n, float(pr["latency_ms"]) / LATENCY_SCALE)
        edge_src.append(ia)
        edge_dst.append(ib)
        edge_feat_list.append(
            [
                is_conn,
                lat_n,
                1.0,
                is_down,
                missing,
                ping_done,
                ping_fail,
                ping_hi,
                float(id_rank.get(a, 0.0)),
                float(id_rank.get(b, 0.0)),
            ]
        )
        edge_keys.append((a, b))

    e = len(edge_keys)
    edge_index = np.array([edge_src, edge_dst], dtype=np.int64) if e else np.zeros((2, 0), dtype=np.int64)
    edge_feat = np.array(edge_feat_list, dtype=np.float32) if e else np.zeros((0, EDGE_DIM), dtype=np.float32)

    step = float(getattr(root, "current_step", 0) or 0)
    max_steps = float(getattr(root, "max_episode_steps", 15) or 15)
    n_miss = float(sum(1 for row in edge_feat_list if row[4] > 0.5)) if edge_feat_list else 0.0
    n_ping = float(len(ping_map))
    globals_ = np.array(
        [
            step / max(max_steps, 1.0),
            1.0 if scanned else 0.0,
            float(n),
            float(e) / max(n * max(n - 1, 1), 1),
            n_miss / max(e, 1),
            n_ping / max(e, 1),
        ],
        dtype=np.float32,
    )

    return GraphObs(
        node_ids=node_ids,
        node_index=node_index,
        node_feat=node_feat,
        edge_index=edge_index,
        edge_feat=edge_feat,
        edge_keys=edge_keys,
        globals=globals_,
        n_nodes=n,
        n_edges=e,
    )


def pad_graph_obs(g: GraphObs, pad_nodes: int, pad_edges: int) -> Dict[str, np.ndarray]:
    pn, pe = int(pad_nodes), int(pad_edges)
    nf = np.zeros((pn, NODE_DIM), dtype=np.float32)
    ef = np.zeros((pe, EDGE_DIM), dtype=np.float32)
    ei = np.zeros((2, pe), dtype=np.int64)
    mask_n = np.zeros((pn,), dtype=np.float32)
    mask_e = np.zeros((pe,), dtype=np.float32)
    n = min(g.n_nodes, pn)
    e = min(g.n_edges, pe)
    nf[:n] = g.node_feat[:n]
    mask_n[:n] = 1.0
    if e > 0:
        ef[:e] = g.edge_feat[:e]
        ei[:, :e] = g.edge_index[:, :e]
        mask_e[:e] = 1.0
    gl = np.zeros(6, dtype=np.float32)
    gl[: min(6, len(g.globals))] = g.globals[:6]
    return {
        "node_feat": nf,
        "edge_feat": ef,
        "edge_index": ei,
        "node_mask": mask_n,
        "edge_mask": mask_e,
        "globals": gl,
    }
