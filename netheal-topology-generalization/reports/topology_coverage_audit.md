# Topology Coverage Metric Audit (Step 9.7-A)

## Official definition

```
node_coverage     = discovered_nodes / network_size
edge_coverage     = discovered_edges / total_edges
topology_coverage = 0.5 * (node_coverage + edge_coverage)
```

Populated from `observation.discovery_matrix` at **episode end** in `MetricsCollectorWrapper._finalize_trace`.

## Root cause of Coverage=0 in Step 9.6 masked eval

When MetricsCollectorWrapper wraps ActionMasker, self.env has no .observation; discovered_nodes/edges remain 0 at finalize → topology_coverage=0 even after scan.

**Fix:** ActionMasker(MetricsCollectorWrapper(base)) — see make_masked_eval_env().

## Empirical comparison (scan → diagnose, seed=42)

| Wrapper order | After scan (direct) | Episode-end metrics | Nodes |
|---|---:|---:|---|
| Unmasked MetricsCollector(base) | 0.700 (direct after scan) | 0.700 (metrics at episode end) | 6/6 nodes |
| Broken: MetricsCollector(ActionMasker) | 0.700 (direct after scan) | 0.000 (metrics at episode end) | 6/6 nodes |
| Fixed: ActionMasker(MetricsCollector) | 0.700 (direct after scan) | 0.700 (metrics at episode end) | 6/6 nodes |

## Takeaway

- `scan_network` **does** increase discovery → direct coverage >0 after scan.
- Broken wrapper order zeroes out metrics at episode end.
- Step 9.7+ uses fixed wrapper; re-evaluate Coverage before Mesh generalization claims.
