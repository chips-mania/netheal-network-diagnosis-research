"""Explicit known-inventory protocol. Never pass the simulator to a policy."""
from types import SimpleNamespace
from .graph_obs import build_graph_obs

class Inventory:
    def __init__(self, nodes, edges):
        self.nodes = tuple(nodes)
        self.edges = tuple(tuple(e) for e in edges)
    def get_all_devices(self):
        return list(self.nodes)
    def get_all_connections(self):
        return list(self.edges)

class PublicView:
    """Allowlist view: inventory, observed evidence, action catalogue and budget only."""
    def __init__(self, env):
        self.network = Inventory(env.network.get_all_devices(), env.network.get_all_connections())
        self.action_space_manager = SimpleNamespace(action_map=dict(env.action_space_manager.action_map))
        self.max_episode_steps = env.max_episode_steps
        self.update(env)
    @property
    def unwrapped(self):
        return self
    def update(self, env):
        self.observation = env.observation
        self.step_count = env.step_count

def policy_packet(view, memory):
    """JSON-safe input for future adapters. No GT, reward or evaluator info."""
    g = build_graph_obs(view, mem=memory)
    return {
        "schema_version": "netheal.public.v1",
        "inventory_assumption": "known node IDs and nominal directed edges; no live status",
        "nodes": g.node_ids,
        "edges": [list(x) for x in g.edge_keys],
        "node_features": g.node_feat.tolist(),
        "edge_features": g.edge_feat.tolist(),
        "global_features": g.globals.tolist(),
        "steps_remaining": view.max_episode_steps - view.step_count,
        "actions": ["scan", "check_status(node)", "check_interfaces(node)",
                    "ping(src,dst)", "diagnose(fault_type,location)"],
    }

def public_step_info(info):
    return {"action_result": info.get("action_result")}
