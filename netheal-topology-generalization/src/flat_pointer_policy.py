"""Fixed-capacity MLP with the SAME graph tensors, labels and actions as GNN."""
import torch
from torch import nn
import torch.nn.functional as F
from .graph_policy import GraphPointerPolicy
from .graph_obs import NODE_DIM, EDGE_DIM

class FlatPointerPolicy(GraphPointerPolicy):
    def __init__(self, max_nodes=12, max_edges=132, hidden=64):
        super().__init__(hidden=hidden, layers=0)
        self.max_nodes, self.max_edges = max_nodes, max_edges
        width = max_nodes * (NODE_DIM + 1) + max_edges * (EDGE_DIM + 3) + 6
        self.flat = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.node_slots = nn.Linear(hidden, max_nodes * hidden)
        self.edge_slots = nn.Linear(hidden, max_edges * hidden)
    def encode(self, nf, ei, ef, nm, em, gl):
        b, n, _ = nf.shape
        e = ef.shape[1]
        if n > self.max_nodes or e > self.max_edges:
            raise ValueError("MLP capacity exceeded; report N/A, never zero")
        parts = [F.pad(nf, (0, 0, 0, self.max_nodes-n)).flatten(1),
                 F.pad(ef, (0, 0, 0, self.max_edges-e)).flatten(1),
                 F.pad(ei.float()/self.max_nodes, (0, self.max_edges-e)).flatten(1),
                 F.pad(nm, (0, self.max_nodes-n)), F.pad(em, (0, self.max_edges-e)), gl]
        g = self.flat(torch.cat(parts, dim=1))
        h = self.node_slots(g).reshape(b,self.max_nodes,self.hidden)[:,:n]
        he = self.edge_slots(g).reshape(b,self.max_edges,self.hidden)[:,:e]
        return h * nm.unsqueeze(-1), he * em.unsqueeze(-1), g
