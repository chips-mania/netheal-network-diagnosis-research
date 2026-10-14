"""Offline network probes and diagnosis statistics. Never exposed to policies."""
import copy
import random
from collections import Counter
import numpy as np
from netheal.tools.simulator import ToolSimulator

def dist(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {"n": len(a), "mean": float(a.mean()), "p50": float(np.median(a)),
            "p95": float(np.quantile(a, .95))}

def wilson(k, n):
    if n == 0:
        return [None, None]
    z = 1.959963984540054
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half), float(center+half)]

def probe_network(network, seed, repeats=5):
    simulator = ToolSimulator(network, rng=random.Random(seed))
    nodes = sorted(network.get_all_devices())
    latencies, variation, capacities = [], [], []
    attempts = failures = tx = rx = counted_calls = 0
    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue
            previous = None
            for _ in range(repeats):
                r = simulator.ping(src, dst)
                attempts += 1
                failures += int(not r.success)
                if "packets_transmitted" in r.data:
                    tx += r.data["packets_transmitted"]
                    rx += r.data["packets_received"]
                    counted_calls += 1
                lat = r.data.get("latency_ms")
                if r.success and lat is not None:
                    latencies.append(lat)
                    if previous is not None:
                        variation.append(abs(lat-previous))
                    previous = lat
                else:
                    previous = None
            path = network.find_path(src, dst)
            if path and network.is_device_up(src) and network.is_device_up(dst):
                capacities.append(min(network.graph.edges[a,b].get("bandwidth",0)
                                      for a,b in zip(path,path[1:])))
    return {
        "probe_attempts": attempts,
        "probe_failure_rate": failures / attempts if attempts else None,
        "simulated_latency_ms": dist(latencies),
        "same_pair_abs_latency_delta_ms": dist(variation),
        "packet_counter_loss_rate": (tx-rx)/tx if tx else None,
        "packet_counter_call_coverage": counted_calls/attempts if attempts else None,
        "configured_path_bottleneck_mbps": dist(capacities),
        "throughput_mbps": None, "rfc_ipdv_ms": None, "recovery_time_ms": None,
    }

def network_pair(env, seed):
    faulty = probe_network(env.network, seed)
    healthy_injector = copy.deepcopy(env.fault_injector)
    healthy_injector.clear_all_faults()
    healthy = probe_network(healthy_injector.network, seed)
    return {"faulted": faulty, "healthy_counterfactual": healthy}

def summarize(rows):
    n = len(rows)
    confusion = Counter((r["fault_type"], r["predicted_type"] or "abstain") for r in rows)
    faults = sorted(set(r["fault_type"] for r in rows))
    per_fault = {}
    f1s = []
    for ft in faults:
        sub = [r for r in rows if r["fault_type"] == ft]
        tp = confusion[ft,ft]
        fp = sum(v for (t,p),v in confusion.items() if p==ft and t!=ft)
        fn = len(sub)-tp
        precision = tp/(tp+fp) if tp+fp else 0
        recall = tp/len(sub)
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        f1s.append(f1)
        exact = sum(r["exact"] for r in sub)
        per_fault[ft] = {"n":len(sub), "exact":exact/len(sub), "wilson95":wilson(exact,len(sub)),
                         "type_precision":precision, "type_recall":recall, "type_f1":f1}
    exact = sum(r["exact"] for r in rows)
    submitted = sum(r["submitted"] for r in rows)
    return {
        "n":n, "exact":exact/n, "exact_wilson95":wilson(exact,n),
        "type_accuracy":sum(r["type_correct"] for r in rows)/n,
        "location_accuracy":sum(r["location_correct"] for r in rows)/n,
        "macro_type_f1":float(np.mean(f1s)),
        "submission_rate":submitted/n,
        "wrong_submission_rate":sum(r["submitted"] and not r["exact"] for r in rows)/n,
        "timeout_rate":sum(r["timeout"] for r in rows)/n,
        "selective_accuracy":exact/submitted if submitted else None,
        "invalid_action_rate":sum(r["invalid_actions"] for r in rows)/max(1,sum(r["steps"] for r in rows)),
        "repeat_tool_rate":sum(r["repeated_tools"] for r in rows)/max(1,sum(r["tool_calls"] for r in rows)),
        "steps":dist([r["steps"] for r in rows]),
        "successful_diagnosis_steps":dist([r["steps"] for r in rows if r["exact"]]),
        "tool_calls":dist([r["tool_calls"] for r in rows]),
        "simulator_tool_cost":dist([r["tool_cost"] for r in rows]),
        "policy_decision_ms":dist([x for r in rows for x in r["decision_ms"]]),
        "episode_wall_ms":dist([r["wall_ms"] for r in rows]),
        "discovered_node_fraction":dist([r["coverage"] for r in rows]),
        "success_by_step":{str(k):sum(r["exact"] and r["steps"]<=k for r in rows)/n for k in [5,10,15,20,30,40]},
        "per_fault":per_fault,
        "confusion":[{"true":t,"predicted":p,"n":v} for (t,p),v in sorted(confusion.items())],
        "healthy_false_positive_rate":None,
    }

def paired_bootstrap(a, b, seed=818, repeats=2000):
    """Mean (b-a); resample training seeds then matched cases, not iid pooled runs."""
    seeds = sorted(set(r["train_seed"] for r in a) & set(r["train_seed"] for r in b))
    aa = {(r["train_seed"],r["case_id"]):r for r in a}
    bb = {(r["train_seed"],r["case_id"]):r for r in b}
    ids = sorted(set(k[1] for k in aa) & set(k[1] for k in bb))
    if not seeds or not ids:
        return None
    differences = np.array([[float(bb[s,i]["exact"])-float(aa[s,i]["exact"]) for i in ids] for s in seeds])
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats)
    for j in range(repeats):
        si = rng.integers(len(seeds),size=len(seeds))
        ci = rng.integers(len(ids),size=len(ids))
        boot[j] = differences[si][:,ci].mean()
    return {"delta_exact":float(differences.mean()),
            "paired_hierarchical_bootstrap95":np.quantile(boot,[.025,.975]).tolist(),
            "training_seeds":len(seeds),"matched_cases":len(ids),
            "per_seed_delta":differences.mean(axis=1).tolist()}
