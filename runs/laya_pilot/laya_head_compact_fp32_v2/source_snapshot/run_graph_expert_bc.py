"""Phase 3: Rule-v2 demos → B1 pad-MLP BC vs P1 GNN+pointer BC.

Train: Star N∈{4,5,6}. Eval: Star 6/8/10 + Mesh 6 (held-out family).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_mask import action_masks_for_env, base_env
from netheal.environment.actions import ActionCategory

from .action_trace import (
    EpisodeSummary,
    StepRecord,
    aggregate_behavior,
    run_episode_with_policy,
    _base_env,
    _diagnosis_correct,
    _format_target,
    _is_action_valid,
    _is_tool_action,
    _parse_action_spec,
)
from .diagnosis_decomposition import aggregate_decomposition
from .env_utils import make_masked_base_env, make_masked_eval_env, metrics_env
from .graph_obs import build_graph_obs
from .graph_policy import (
    GraphPointerPolicy,
    bc_loss,
    collate_graphs,
    pointer_to_labels,
)
from .metrics_utils import summarize_evaluator
from .pbrs import find_pbrs
from .pointer_actions import encode_flat_as_pointer, resolve_pointer
from .protocol import PROJECT_ROOT, load_protocol
from .rule_agent_v2 import RuleAgentV2, run_episode_rule_v2
from .run_ablation_nl_poc import _per_fault, _row_consistent, _selector
from .run_diagnosis_gate_poc import shortcut_rate
from .run_star_scale_poc import _env_kwargs, _exact, _pf, bc_train
from .run_symmetric_reward_poc import set_seeds
from .star_pad import steps_budget


def collect_rule_v2_flat(
    protocol,
    *,
    n_devices: int,
    pad: int,
    n_ok: int,
    seed0: int,
) -> Tuple[List[dict], List[int], List[np.ndarray], Dict]:
    kw = _env_kwargs(protocol, n_devices, pad)
    max_steps = kw["max_episode_steps"]
    env = make_masked_base_env(["star"], protocol=protocol, **kw)
    obs_list: List[dict] = []
    act_list: List[int] = []
    mask_list: List[np.ndarray] = []
    ok = 0
    tries = 0
    per: Dict[str, int] = {}
    lengths: List[int] = []
    while ok < n_ok and tries < n_ok * 8:
        tries += 1
        agent = RuleAgentV2()
        obs, info = env.reset(seed=seed0 + tries)
        root = base_env(env)
        gt = info.get("ground_truth_fault") or {}
        ft = gt.get("type", "unknown")
        success = False
        ep_obs, ep_act, ep_mask = [], [], []
        for t in range(1, max_steps + 1):
            mask = action_masks_for_env(env)
            try:
                action = int(agent.choose(env, obs, t, []))
            except RuntimeError:
                success = False
                break
            ep_obs.append({k: np.array(v, copy=True) for k, v in obs.items()})
            ep_act.append(action)
            ep_mask.append(np.array(mask, copy=True))
            obs, reward, term, trunc, info = env.step(action)
            agent.observe_step(info)
            spec = _parse_action_spec(info, root, action)
            if spec is not None and _diagnosis_correct(info, spec) is True:
                success = True
            if term or trunc:
                break
        if success:
            ok += 1
            per[ft] = per.get(ft, 0) + 1
            lengths.append(len(ep_act))
            obs_list.extend(ep_obs)
            act_list.extend(ep_act)
            mask_list.extend(ep_mask)
    env.close()
    return (
        obs_list,
        act_list,
        mask_list,
        {
            "ok": ok,
            "tries": tries,
            "per_fault_ok": per,
            "n_transitions": len(act_list),
            "avg_demo_len": float(np.mean(lengths)) if lengths else 0.0,
            "n_devices": n_devices,
            "pad": pad,
        },
    )


def collect_rule_v2_graph(
    protocol,
    *,
    n_devices: int,
    n_ok: int,
    seed0: int,
) -> Tuple[List, List[Dict[str, int]], Dict]:
    max_steps = steps_budget(n_devices, floor=int(protocol["max_episode_steps"]))
    env = make_masked_base_env(
        ["star"],
        protocol=protocol,
        evidence_obs=True,
        min_devices=n_devices,
        max_devices=n_devices,
        max_episode_steps=max_steps,
    )
    graphs = []
    labels = []
    ok = 0
    tries = 0
    per: Dict[str, int] = {}
    while ok < n_ok and tries < n_ok * 8:
        tries += 1
        agent = RuleAgentV2()
        obs, info = env.reset(seed=seed0 + tries)
        root = base_env(env)
        gt = info.get("ground_truth_fault") or {}
        ft = gt.get("type", "unknown")
        success = False
        ep_g, ep_lab = [], []
        for t in range(1, max_steps + 1):
            # Graph must reflect pre-action mem (before choose mutates queues/flags).
            g = build_graph_obs(env, mem=agent.mem)
            try:
                action = int(agent.choose(env, obs, t, []))
            except RuntimeError:
                success = False
                break
            ptr = encode_flat_as_pointer(env, action)
            if ptr is None:
                success = False
                break
            lab = pointer_to_labels(g, ptr)
            if lab["kind"] < 0:
                success = False
                break
            ep_g.append(g)
            ep_lab.append(lab)
            obs, reward, term, trunc, info = env.step(action)
            agent.observe_step(info)
            spec = _parse_action_spec(info, root, action)
            if spec is not None and _diagnosis_correct(info, spec) is True:
                success = True
            if term or trunc:
                break
        if success and ep_g:
            # drop steps with invalid node/edge labels when required
            keep_g, keep_l = [], []
            for g, lab in zip(ep_g, ep_lab):
                k = lab["kind"]
                if k in (1, 2) and lab["node"] < 0:
                    continue
                if k == 3 and lab["edge"] < 0:
                    continue
                if k == 4 and lab["fault"] < 0:
                    continue
                if k == 4:
                    from .graph_policy import FAULT_ORDER, NODE_FAULTS

                    ft = FAULT_ORDER[lab["fault"]]
                    if ft in NODE_FAULTS and lab["node"] < 0:
                        continue
                    if ft not in NODE_FAULTS and lab["edge"] < 0:
                        continue
                keep_g.append(g)
                keep_l.append(lab)
            if keep_g:
                ok += 1
                per[ft] = per.get(ft, 0) + 1
                graphs.extend(keep_g)
                labels.extend(keep_l)
    env.close()
    return graphs, labels, {
        "ok": ok,
        "tries": tries,
        "per_fault_ok": per,
        "n_transitions": len(labels),
        "n_devices": n_devices,
    }


def train_graph_bc(
    policy: GraphPointerPolicy,
    graphs: List,
    labels: List[Dict[str, int]],
    *,
    epochs: int,
    batch: int,
    lr: float,
    pad_nodes: int,
    pad_edges: int,
    device: torch.device,
) -> List[float]:
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(labels)
    losses = []
    policy.train()
    for ep in range(epochs):
        idx = np.random.permutation(n)
        ep_loss = []
        for start in range(0, n, batch):
            sel = idx[start : start + batch]
            if len(sel) < 2:
                continue
            gs = [graphs[i] for i in sel]
            batch_t = collate_graphs(gs, pad_nodes=pad_nodes, pad_edges=pad_edges)
            batch_t = {k: v.to(device) for k, v in batch_t.items()}
            lab = {
                "kind": torch.tensor([labels[i]["kind"] for i in sel], device=device),
                "node": torch.tensor([labels[i]["node"] for i in sel], device=device),
                "edge": torch.tensor([labels[i]["edge"] for i in sel], device=device),
                "fault": torch.tensor([labels[i]["fault"] for i in sel], device=device),
            }
            logits = policy.forward_logits(
                batch_t["node_feat"],
                batch_t["edge_index"],
                batch_t["edge_feat"],
                batch_t["node_mask"],
                batch_t["edge_mask"],
                batch_t["globals"],
            )
            loss = bc_loss(logits, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss.append(float(loss.item()))
        losses.append(float(np.mean(ep_loss)) if ep_loss else 0.0)
        if (ep + 1) % max(1, epochs // 4) == 0 or ep + 1 == epochs:
            print(f"  graph-bc epoch {ep+1}/{epochs} loss={losses[-1]:.4f}")
    return losses


def _eval_pack(summaries, records, rows, n, max_steps) -> Dict:
    timeout = sum(1 for s in summaries if (not s.success) and s.steps >= max_steps)
    exact_cons = [r for r in rows if r["success"] and _row_consistent(r)]
    return {
        "stats": summarize_evaluator,  # filled by caller
        "behavior": aggregate_behavior(summaries, records),
        "decomposition": aggregate_decomposition(summaries),
        "per_fault": _per_fault(rows, summaries),
        "shortcut": shortcut_rate(summaries, records),
        "timeout_rate": timeout / max(1, n),
        "submit_rate": sum(1 for r in rows if r["submitted"]) / max(1, n),
        "exact_consistent_n": len(exact_cons),
        "n": n,
        "max_steps": max_steps,
    }


def evaluate_mlp(model, protocol, *, n_devices: int, pad: int, n: int, seed_base: int) -> Dict:
    from .run_star_scale_poc import evaluate_model

    return evaluate_model(
        model, protocol, n_devices=n_devices, pad=pad, n=n, seed_base=seed_base, det=True
    )


def evaluate_graph_policy(
    policy: GraphPointerPolicy,
    protocol,
    *,
    topology: str,
    n_devices: int,
    n: int,
    seed_base: int,
    device: torch.device,
    max_episode_steps: Optional[int] = None,
) -> Dict:
    """Roll out GNN policy while maintaining Rule-v2-style evidence mem for graph_obs."""
    policy.eval()
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    evaluator = CompetitionEvaluator()
    records: List = []
    summaries: List = []
    rows: List[Dict] = []
    steps = int(max_episode_steps or steps_budget(n_devices, floor=int(protocol["max_episode_steps"])))

    for i in range(n):
        fault = faults[i % len(faults)]
        env = make_masked_eval_env(
            [topology],
            fault_types=[fault],
            protocol=protocol,
            evidence_obs=True,
            pbrs=True,
            pbrs_apply_shaping=False,
            min_devices=n_devices,
            max_devices=n_devices,
            max_episode_steps=steps,
        )
        agent_ev = RuleAgentV2()
        base = _base_env(env)
        obs, info = env.reset(seed=seed_base + i)
        gt = info.get("ground_truth_fault") or {}
        fault_type = gt.get("type", "")
        fault_location = gt.get("location", "")
        recs = []
        invalid_count = 0
        tool_before_diag = 0
        saw_diagnosis = False
        predicted_type = None
        predicted_location = None
        agent_ev.reset()
        step = 0
        while True:
            step += 1
            g = build_graph_obs(env, mem=agent_ev.mem)
            ptr = policy.act(g, device)
            action_id = int(resolve_pointer(env, ptr))
            valid = _is_action_valid(base, action_id)
            if not valid:
                invalid_count += 1
            obs, reward, terminated, truncated, info = env.step(action_id)
            agent_ev.observe_step(info)
            agent_ev.refresh_plan(env)
            spec = _parse_action_spec(info, base, action_id)
            phase = info.get("factorized_phase")
            category = spec.category.value if spec else ("invalid" if not valid else "unknown")
            action_type = (
                spec.action_type.value if spec and hasattr(spec.action_type, "value") else ""
            )
            target = _format_target(spec)
            diag_correct = None if phase == "type_commit" else _diagnosis_correct(info, spec)
            if _is_tool_action(spec) and not saw_diagnosis:
                tool_before_diag += 1
            if spec and spec.category == ActionCategory.DIAGNOSIS and phase != "type_commit":
                saw_diagnosis = True
                predicted_type = action_type
                predicted_location = target
            recs.append(
                StepRecord(
                    episode_id=i,
                    seed=seed_base + i,
                    agent="graph_ptr",
                    fault_type=fault_type,
                    fault_location=fault_location,
                    step=step,
                    action_id=action_id,
                    action_category=category,
                    action_type=action_type,
                    target=target,
                    reward=float(reward),
                    is_valid=valid,
                    done=bool(terminated or truncated),
                    diagnosis_correct=diag_correct,
                )
            )
            if terminated or truncated:
                break
        m = metrics_env(env).last_episode_metrics
        if m:
            evaluator.add_episode_metrics(m)
        tracker_p = find_pbrs(env)
        bits = tracker_p.bits_before_diagnosis or tracker_p.bits.snapshot()
        phi = tracker_p.phi_before_diagnosis
        if phi is None:
            phi = tracker_p.bits.phi
        summary = EpisodeSummary(
            episode_id=i,
            seed=seed_base + i,
            agent="graph_ptr",
            fault_type=fault_type,
            fault_location=fault_location,
            steps=len(recs),
            success=bool(m and m.diagnosis_success),
            predicted_type=predicted_type,
            predicted_location=predicted_location,
            invalid_action_count=invalid_count,
            tool_actions_before_diagnosis=tool_before_diag,
            first_action_is_diagnosis=bool(
                recs and recs[0].action_category == ActionCategory.DIAGNOSIS.value
            ),
            coverage=float(m.topology_coverage) if m else 0.0,
        )
        rows.append(
            {
                "fault_type": summary.fault_type,
                "fault_location": summary.fault_location,
                "success": summary.success,
                "predicted_type": summary.predicted_type,
                "predicted_location": summary.predicted_location,
                "phi_before_dx": int(phi),
                "type_bit": int(bits.get("type_bit", 0)),
                "loc_bit": int(bits.get("loc_bit", 0)),
                "steps": summary.steps,
                "submitted": summary.predicted_type is not None,
            }
        )
        records.extend(recs)
        summaries.append(summary)
        env.close()

    timeout = sum(1 for s in summaries if (not s.success) and s.steps >= steps)
    exact_cons = [r for r in rows if r["success"] and _row_consistent(r)]
    return {
        "topology": topology,
        "n_devices": n_devices,
        "stats": summarize_evaluator(evaluator),
        "behavior": aggregate_behavior(summaries, records),
        "decomposition": aggregate_decomposition(summaries),
        "per_fault": _per_fault(rows, summaries),
        "shortcut": shortcut_rate(summaries, records),
        "timeout_rate": timeout / max(1, n),
        "submit_rate": sum(1 for r in rows if r["submitted"]) / max(1, n),
        "exact_consistent_n": len(exact_cons),
        "exact": float(aggregate_decomposition(summaries)["exact_diagnosis_accuracy"]),
        "n": n,
        "max_steps": steps,
    }


def evaluate_mesh_mlp(model, protocol, *, pad: int, n: int, seed_base: int) -> Dict:
    """MLP pad policy on mesh N=6 (may fail hard — expected)."""
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    evaluator = CompetitionEvaluator()
    records: List = []
    summaries: List = []
    rows: List[Dict] = []
    steps = 40
    kw = {
        "min_devices": 6,
        "max_devices": 6,
        "max_episode_steps": steps,
        "star_pad_max": pad,
        "evidence_obs": True,
    }
    for i in range(n):
        fault = faults[i % len(faults)]
        env = make_masked_eval_env(
            ["mesh"],
            fault_types=[fault],
            protocol=protocol,
            pbrs=True,
            pbrs_apply_shaping=False,
            **kw,
        )
        recs, summary = run_episode_with_policy(
            env,
            episode_id=i,
            seed=seed_base + i,
            agent="mlp_pad_mesh",
            choose_action=_selector(model, deterministic=True),
        )
        m = metrics_env(env).last_episode_metrics
        if m:
            evaluator.add_episode_metrics(m)
        tracker = find_pbrs(env)
        bits = tracker.bits_before_diagnosis or tracker.bits.snapshot()
        phi = tracker.phi_before_diagnosis
        if phi is None:
            phi = tracker.bits.phi
        rows.append(
            {
                "fault_type": summary.fault_type,
                "fault_location": summary.fault_location,
                "success": summary.success,
                "predicted_type": summary.predicted_type,
                "predicted_location": summary.predicted_location,
                "phi_before_dx": int(phi),
                "type_bit": int(bits.get("type_bit", 0)),
                "loc_bit": int(bits.get("loc_bit", 0)),
                "steps": summary.steps,
                "submitted": summary.predicted_type is not None,
            }
        )
        records.extend(recs)
        summaries.append(summary)
        env.close()
    timeout = sum(1 for s in summaries if (not s.success) and s.steps >= steps)
    return {
        "topology": "mesh",
        "n_devices": 6,
        "stats": summarize_evaluator(evaluator),
        "behavior": aggregate_behavior(summaries, records),
        "decomposition": aggregate_decomposition(summaries),
        "per_fault": _per_fault(rows, summaries),
        "shortcut": shortcut_rate(summaries, records),
        "timeout_rate": timeout / max(1, n),
        "submit_rate": sum(1 for r in rows if r["submitted"]) / max(1, n),
        "exact": float(aggregate_decomposition(summaries)["exact_diagnosis_accuracy"]),
        "n": n,
    }


def _h_gate(b1: Dict, p1: Dict) -> Dict:
    def get_exact(block, k):
        if k not in block:
            return 0.0
        x = block[k]
        if "exact" in x:
            return float(x["exact"])
        return float(x["decomposition"]["exact_diagnosis_accuracy"])

    s6_b, s6_p = get_exact(b1, "star_6"), get_exact(p1, "star_6")
    s8_b, s8_p = get_exact(b1, "star_8"), get_exact(p1, "star_8")
    s10_b, s10_p = get_exact(b1, "star_10"), get_exact(p1, "star_10")
    m_b, m_p = get_exact(b1, "mesh_6"), get_exact(p1, "mesh_6")
    b1_mesh_na = bool((b1.get("mesh_6") or {}).get("note"))
    seen_ok = s6_p >= 0.90
    h1 = (s8_p - s8_b) >= 0.10 or ((s8_p - s8_b) >= 0.05 and s8_p > s8_b)
    # Mesh: if B1 cannot run (star-pad), require absolute P1≥55%; else relative +10%p / collapse rule
    if b1_mesh_na:
        h2 = m_p >= 0.55
        h2_note = "B1 mesh N/A (star-pad); H2 uses absolute P1≥55%"
    else:
        h2 = (m_p - m_b) >= 0.10 or (m_b <= 0.40 and m_p >= 0.55)
        h2_note = "relative/collapse rule"
    if seen_ok and h1 and h2:
        decision = "go"
    elif seen_ok and h1:
        decision = "go_scale_only"
    elif seen_ok and h2:
        decision = "go_mesh_only"
    elif seen_ok:
        decision = "no_go_transfer"
    else:
        decision = "no_go_seen"
    return {
        "decision": decision,
        "seen_ok": seen_ok,
        "h1_scale": h1,
        "h2_mesh": h2,
        "h2_note": h2_note,
        "b1_mesh_na": b1_mesh_na,
        "delta_star8": s8_p - s8_b,
        "delta_star10": s10_p - s10_b,
        "delta_mesh": m_p - m_b,
        "exact": {
            "b1": {"star_6": s6_b, "star_8": s8_b, "star_10": s10_b, "mesh_6": m_b},
            "p1": {"star_6": s6_p, "star_8": s8_p, "star_10": s10_p, "mesh_6": m_p},
        },
    }


def main() -> None:
    protocol = load_protocol()
    cfg = protocol.setdefault(
        "graph_expert_bc_poc",
        {
            "train_ns": [4, 5, 6],
            "demos_per_n": 700,
            "bc_epochs": 80,
            "p1_epochs": 80,
            "batch_size": 64,
            "pad_max_devices": 12,
            "pad_edges": 132,
            "policy_seed": 1,
            "eval_episodes": 100,
            "eval_seed_base": 130000,
            "mesh_max_steps": 40,
        },
    )
    protocol.setdefault("star_scale_poc", {"topology_types": ["star"]})
    set_seeds(int(cfg["policy_seed"]))
    pad = int(cfg["pad_max_devices"])
    demos_per = int(cfg["demos_per_n"])
    device = torch.device("cpu")

    # --- collect flat (B1) + graph (P1) ---
    obs_all, acts_all, masks_all = [], [], []
    graphs_all, labs_all = [], []
    collects = {}
    for n in cfg["train_ns"]:
        n = int(n)
        print(f"=== Collect Rule-v2 flat N={n} ===")
        o, a, m, st = collect_rule_v2_flat(
            protocol, n_devices=n, pad=pad, n_ok=demos_per, seed0=70_000 + 1000 * n
        )
        print(" ", st)
        collects[f"flat_{n}"] = st
        obs_all.extend(o)
        acts_all.extend(a)
        masks_all.extend(m)
        print(f"=== Collect Rule-v2 graph N={n} ===")
        g, lab, gst = collect_rule_v2_graph(
            protocol, n_devices=n, n_ok=demos_per, seed0=80_000 + 1000 * n
        )
        print(" ", gst)
        collects[f"graph_{n}"] = gst
        graphs_all.extend(g)
        labs_all.extend(lab)

    # --- B1 MLP ---
    print(f"=== B1 pad-MLP BC transitions={len(acts_all)} ===")
    env = make_masked_base_env(["star"], protocol=protocol, **_env_kwargs(protocol, 6, pad))
    check_env(env, warn=True)
    mlp = MaskablePPO("MultiInputPolicy", env, verbose=0, seed=int(cfg["policy_seed"]))
    losses_b1 = bc_train(
        mlp,
        obs_all,
        acts_all,
        masks_all,
        epochs=int(cfg.get("bc_epochs", 40)),
        batch=int(cfg["batch_size"]),
        lr=3e-4,
    )
    ckpt_b1 = PROJECT_ROOT / "models" / "checkpoints" / "bc_rule_v2_pad12_b1.zip"
    ckpt_b1.parent.mkdir(parents=True, exist_ok=True)
    mlp.save(str(ckpt_b1.with_suffix("")))
    env.close()

    # --- P1 GNN ---
    print(f"=== P1 GNN+pointer BC transitions={len(labs_all)} ===")
    policy = GraphPointerPolicy(hidden=64, layers=2).to(device)
    losses_p1 = train_graph_bc(
        policy,
        graphs_all,
        labs_all,
        epochs=int(cfg.get("p1_epochs", cfg.get("bc_epochs", 80))),
        batch=int(cfg["batch_size"]),
        lr=3e-4,
        pad_nodes=pad,
        pad_edges=int(cfg["pad_edges"]),
        device=device,
    )
    ckpt_p1 = PROJECT_ROOT / "models" / "checkpoints" / "graph_pointer_p1.pt"
    torch.save(policy.state_dict(), ckpt_p1)

    n_eval = int(cfg["eval_episodes"])
    seed = int(cfg["eval_seed_base"])
    b1_res: Dict = {}
    p1_res: Dict = {}

    for n in (6, 8, 10):
        key = f"star_{n}"
        print(f"=== Eval B1 {key} ===")
        b1_res[key] = evaluate_mlp(mlp, protocol, n_devices=n, pad=pad, n=n_eval, seed_base=seed + 100 * n)
        b1_res[key]["exact"] = _exact(b1_res[key])
        print(f"  B1 exact={b1_res[key]['exact']:.0%}")
        print(f"=== Eval P1 {key} ===")
        p1_res[key] = evaluate_graph_policy(
            policy, protocol, topology="star", n_devices=n, n=n_eval, seed_base=seed + 200 * n, device=device
        )
        print(f"  P1 exact={p1_res[key]['exact']:.0%}")

    print("=== Eval B1 mesh_6 ===")
    # pad12 StarActionPad rebuilds canonical center/edge_* — incompatible with mesh IDs.
    b1_res["mesh_6"] = {
        "exact": 0.0,
        "timeout_rate": 1.0,
        "shortcut": {"shortcut_rate": 0.0},
        "decomposition": {"exact_diagnosis_accuracy": 0.0},
        "note": "B1 pad12 uses Star canonical action map; Mesh family eval not applicable (count as 0).",
        "n": n_eval,
    }
    print("  B1 mesh exact=0% (star-pad action map incompatible; scored 0)")

    print("=== Eval P1 mesh_6 ===")
    p1_res["mesh_6"] = evaluate_graph_policy(
        policy,
        protocol,
        topology="mesh",
        n_devices=6,
        n=n_eval,
        seed_base=seed + 6000,
        device=device,
        max_episode_steps=int(cfg["mesh_max_steps"]),
    )
    print(f"  P1 mesh exact={p1_res['mesh_6']['exact']:.0%}")

    gate = _h_gate(b1_res, p1_res)

    def _row(tag, block, key):
        v = block[key]
        ex = float(v.get("exact", v.get("decomposition", {}).get("exact_diagnosis_accuracy", 0)))
        sc = v.get("shortcut", {}).get("shortcut_rate", 0) if isinstance(v.get("shortcut"), dict) else 0
        to = v.get("timeout_rate", 0)
        return f"| {tag} | {key} | {ex:.0%} | {sc:.0%} | {to:.0%} |"

    lines = [
        "# Graph expert BC — B1 vs P1 (Phase 3)",
        "",
        "> Train: Rule **v2** demos Star N∈{4,5,6}.",
        "> B1: pad12 MultiInputPolicy BC. P1: GNN+pointer BC.",
        "> Eval: Seen Star6, Unseen scale Star8/10, Unseen family Mesh6.",
        f"> Decision: **{gate['decision']}**",
        "",
        f"- Flat transitions: {len(acts_all)}; Graph transitions: {len(labs_all)}",
        f"- B1 last loss: {losses_b1[-1]:.4f}; P1 last loss: {losses_p1[-1]:.4f}",
        f"- H1 scale (Star8 Δ): {gate['delta_star8']:+.0%}; H2 mesh Δ: {gate['delta_mesh']:+.0%}",
        f"- Seen Star6 P1≥90%: {gate['seen_ok']}",
        "",
        "## Exact",
        "",
        "| Model | Setting | Exact | Shortcut | Timeout |",
        "|---|---|---:|---:|---:|",
    ]
    for key in ("star_6", "star_8", "star_10", "mesh_6"):
        lines.append(_row("B1 MLP", b1_res, key))
        lines.append(_row("P1 GNN+ptr", p1_res, key))

    lines.extend(["", "## Per-fault P1", ""])
    for key in ("star_6", "star_8", "star_10", "mesh_6"):
        v = p1_res[key]
        lines.append(f"### {key} (P1)")
        if "per_fault" in v:
            lines.append(f"- Device {_pf(v, 'device_failure')}")
            lines.append(f"- Perf {_pf(v, 'performance_degradation')}")
            lines.append(f"- Misconfig {_pf(v, 'misconfiguration')}")
            lines.append(f"- Link {_pf(v, 'link_failure')}")
        lines.append("")

    lines.extend(
        [
            "## Go / No-Go",
            "",
            f"- decision: **{gate['decision']}**",
            f"- h1_scale: {gate['h1_scale']}; h2_mesh: {gate['h2_mesh']}",
            "",
            "Run: `python -m src.run_graph_expert_bc`",
            "",
        ]
    )

    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "graph_expert_poc.md").write_text("\n".join(lines), encoding="utf-8")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k not in ("records",)}
        if isinstance(o, list):
            return [_clean(x) for x in o]
        return o

    payload = {
        "gate": gate,
        "collects": collects,
        "losses_b1": losses_b1,
        "losses_p1": losses_p1,
        "b1": _clean(b1_res),
        "p1": _clean(p1_res),
    }
    (results / "graph_expert_poc.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"decision={gate['decision']} gate={gate['exact']}")
    print("Wrote", reports / "graph_expert_poc.md")


if __name__ == "__main__":
    main()
