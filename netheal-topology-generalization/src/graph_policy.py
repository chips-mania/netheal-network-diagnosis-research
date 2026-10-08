"""Pure-PyTorch GNN encoder + pointer action heads (no torch-geometric)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_obs import EDGE_DIM, NODE_DIM, GraphObs
from .pointer_actions import (
    PointerAction,
    PointerKind,
    pointer_check_interfaces,
    pointer_check_status,
    pointer_diagnose,
    pointer_ping,
    pointer_scan,
)

KIND_ORDER = [
    PointerKind.SCAN,
    PointerKind.CHECK_STATUS,
    PointerKind.CHECK_INTERFACES,
    PointerKind.PING,
    PointerKind.DIAGNOSE,
]
KIND_TO_I = {k: i for i, k in enumerate(KIND_ORDER)}

FAULT_ORDER = [
    "device_failure",
    "performance_degradation",
    "misconfiguration",
    "link_failure",
]
FAULT_TO_I = {f: i for i, f in enumerate(FAULT_ORDER)}
NODE_FAULTS = {"device_failure", "misconfiguration"}


class GraphPointerPolicy(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 2, global_dim: int = 6):
        super().__init__()
        self.hidden = hidden
        self.node_in = nn.Linear(NODE_DIM, hidden)
        self.edge_in = nn.Linear(EDGE_DIM, hidden)
        self.msg = nn.ModuleList(
            [nn.Linear(hidden * 3, hidden) for _ in range(layers)]
        )
        self.upd = nn.ModuleList([nn.GRUCell(hidden, hidden) for _ in range(layers)])
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden + global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.kind_head = nn.Linear(hidden, len(KIND_ORDER))
        self.node_score = nn.Linear(hidden * 2, 1)
        self.edge_score = nn.Linear(hidden * 3, 1)
        self.fault_head = nn.Linear(hidden, len(FAULT_ORDER))

    def encode(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        globals_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        node_feat: [B,N,Dn], edge_index [B,2,E], edge_feat [B,E,De]
        masks: [B,N], [B,E]; globals [B,G]
        returns node_h [B,N,H], edge_h [B,E,H], g_h [B,H]
        """
        b, n, _ = node_feat.shape
        e = edge_feat.shape[1]
        h = F.relu(self.node_in(node_feat))
        he = F.relu(self.edge_in(edge_feat))
        for msg_l, upd_l in zip(self.msg, self.upd):
            agg = torch.zeros_like(h)
            if e > 0:
                src = edge_index[:, 0, :].clamp(0, n - 1)  # [B,E]
                dst = edge_index[:, 1, :].clamp(0, n - 1)
                h_src = torch.gather(h, 1, src.unsqueeze(-1).expand(-1, -1, self.hidden))
                h_dst = torch.gather(h, 1, dst.unsqueeze(-1).expand(-1, -1, self.hidden))
                m = F.relu(msg_l(torch.cat([h_src, h_dst, he], dim=-1)))  # [B,E,H]
                m = m * edge_mask.unsqueeze(-1)
                # scatter-add to dst
                agg = agg.scatter_add(1, dst.unsqueeze(-1).expand_as(m), m)
            h_flat = h.reshape(b * n, self.hidden)
            agg_flat = agg.reshape(b * n, self.hidden)
            h = upd_l(agg_flat, h_flat).reshape(b, n, self.hidden)
            h = h * node_mask.unsqueeze(-1)
        # global
        denom = node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * node_mask.unsqueeze(-1)).sum(dim=1) / denom
        g_h = self.global_mlp(torch.cat([pooled, globals_], dim=-1))
        return h, he, g_h

    def forward_logits(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        globals_: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h, he, g = self.encode(node_feat, edge_index, edge_feat, node_mask, edge_mask, globals_)
        b, n, _ = h.shape
        e = he.shape[1]
        kind_logits = self.kind_head(g)
        # node scores
        g_exp = g.unsqueeze(1).expand(-1, n, -1)
        node_logits = self.node_score(torch.cat([h, g_exp], dim=-1)).squeeze(-1)
        node_logits = node_logits.masked_fill(node_mask < 0.5, -1e9)
        # edge scores
        if e > 0:
            src = edge_index[:, 0, :].clamp(0, n - 1)
            dst = edge_index[:, 1, :].clamp(0, n - 1)
            h_src = torch.gather(h, 1, src.unsqueeze(-1).expand(-1, -1, self.hidden))
            h_dst = torch.gather(h, 1, dst.unsqueeze(-1).expand(-1, -1, self.hidden))
            g_e = g.unsqueeze(1).expand(-1, e, -1)
            edge_logits = self.edge_score(torch.cat([h_src, h_dst, he], dim=-1)).squeeze(-1)
            # blend global context lightly via residual bias from g_e mean
            edge_logits = edge_logits + 0.0 * g_e.sum(dim=-1)
            edge_logits = edge_logits.masked_fill(edge_mask < 0.5, -1e9)
        else:
            edge_logits = torch.zeros(b, 1, device=h.device)
        fault_logits = self.fault_head(g)
        return {
            "kind": kind_logits,
            "node": node_logits,
            "edge": edge_logits,
            "fault": fault_logits,
        }

    @torch.no_grad()
    def act(self, g: GraphObs, device: torch.device) -> PointerAction:
        batch = collate_graphs([g], pad_nodes=max(g.n_nodes, 1), pad_edges=max(g.n_edges, 1))
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = self.forward_logits(
            batch["node_feat"],
            batch["edge_index"],
            batch["edge_feat"],
            batch["node_mask"],
            batch["edge_mask"],
            batch["globals"],
        )
        kind_i = int(logits["kind"][0].argmax().item())
        kind = KIND_ORDER[kind_i]
        if kind == PointerKind.SCAN:
            return pointer_scan()
        if kind == PointerKind.CHECK_STATUS:
            ni = int(logits["node"][0].argmax().item())
            return pointer_check_status(g.node_ids[ni])
        if kind == PointerKind.CHECK_INTERFACES:
            ni = int(logits["node"][0].argmax().item())
            return pointer_check_interfaces(g.node_ids[ni])
        if kind == PointerKind.PING:
            if g.n_edges == 0:
                return pointer_scan()
            ei = int(logits["edge"][0, : g.n_edges].argmax().item())
            src, dst = g.edge_keys[ei]
            return pointer_ping(src, dst)
        # diagnose
        fi = int(logits["fault"][0].argmax().item())
        ft = FAULT_ORDER[fi]
        if ft in NODE_FAULTS:
            ni = int(logits["node"][0].argmax().item())
            return pointer_diagnose(ft, g.node_ids[ni])
        if g.n_edges == 0:
            return pointer_diagnose(ft, g.node_ids[0] if g.node_ids else "center")
        ei = int(logits["edge"][0, : g.n_edges].argmax().item())
        src, dst = g.edge_keys[ei]
        return pointer_diagnose(ft, f"{src}->{dst}")


def collate_graphs(
    graphs: List[GraphObs], *, pad_nodes: int, pad_edges: int
) -> Dict[str, torch.Tensor]:
    b = len(graphs)
    nf = torch.zeros(b, pad_nodes, NODE_DIM)
    ef = torch.zeros(b, pad_edges, EDGE_DIM)
    ei = torch.zeros(b, 2, pad_edges, dtype=torch.long)
    nm = torch.zeros(b, pad_nodes)
    em = torch.zeros(b, pad_edges)
    gl = torch.zeros(b, 6)
    for i, g in enumerate(graphs):
        n = min(g.n_nodes, pad_nodes)
        e = min(g.n_edges, pad_edges)
        nf[i, :n] = torch.from_numpy(g.node_feat[:n])
        nm[i, :n] = 1.0
        if e > 0:
            ef[i, :e] = torch.from_numpy(g.edge_feat[:e])
            ei[i, :, :e] = torch.from_numpy(g.edge_index[:, :e])
            em[i, :e] = 1.0
        gdim = min(6, len(g.globals))
        gl[i, :gdim] = torch.from_numpy(g.globals[:gdim])
    return {
        "node_feat": nf,
        "edge_feat": ef,
        "edge_index": ei,
        "node_mask": nm,
        "edge_mask": em,
        "globals": gl,
    }


def pointer_to_labels(g: GraphObs, p: PointerAction) -> Dict[str, int]:
    kind = KIND_TO_I[p.kind]
    node_i = -1
    edge_i = -1
    fault_i = -1
    if p.kind in (PointerKind.CHECK_STATUS, PointerKind.CHECK_INTERFACES):
        node_i = g.node_index.get(p.node or "", -1)
    elif p.kind == PointerKind.PING:
        key = (p.src or "", p.dst or "")
        edge_i = g.edge_keys.index(key) if key in g.edge_keys else -1
    elif p.kind == PointerKind.DIAGNOSE:
        fault_i = FAULT_TO_I.get(p.fault_type or "", -1)
        loc = p.location or ""
        if (p.fault_type or "") in NODE_FAULTS:
            node_i = g.node_index.get(loc, -1)
        else:
            if "->" in loc:
                a, b = loc.split("->", 1)
                key = (a.strip(), b.strip())
                edge_i = g.edge_keys.index(key) if key in g.edge_keys else -1
    return {"kind": kind, "node": node_i, "edge": edge_i, "fault": fault_i}


def bc_loss(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    *,
    kind_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BC loss. Optional per-row `kind_weight` [B] upweights kind CE (e.g. diagnose commits)."""
    if kind_weight is None:
        loss = F.cross_entropy(logits["kind"], labels["kind"])
    else:
        per = F.cross_entropy(logits["kind"], labels["kind"], reduction="none")
        w = kind_weight.to(per.dtype)
        loss = (per * w).sum() / w.sum().clamp_min(1.0)
    # node-supervised rows
    node_rows = labels["node"] >= 0
    if node_rows.any():
        loss = loss + F.cross_entropy(logits["node"][node_rows], labels["node"][node_rows])
    edge_rows = labels["edge"] >= 0
    if edge_rows.any():
        loss = loss + F.cross_entropy(logits["edge"][edge_rows], labels["edge"][edge_rows])
    fault_rows = labels["fault"] >= 0
    if fault_rows.any():
        loss = loss + F.cross_entropy(logits["fault"][fault_rows], labels["fault"][fault_rows])
    return loss
