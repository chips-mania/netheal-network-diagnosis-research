"""Generate tables and a scientific figure from completed experiment artifacts."""
from pathlib import Path
import argparse
import json
import numpy as np

def loadlines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("run");args=ap.parse_args()
    run=Path(args.run)
    manifest=json.loads((run/"manifest.json").read_text(encoding="utf8"))
    if manifest.get("status")!="completed": raise RuntimeError("Run has not completed")
    protocol=json.loads((run/"protocol.json").read_text(encoding="utf8"))
    summaries=json.loads((run/"summaries.json").read_text(encoding="utf8"))
    comp=json.loads((run/"comparisons.json").read_text(encoding="utf8"))
    train=json.loads((run/"training.json").read_text(encoding="utf8"))
    rows=[]
    for p in sorted(run.glob("episodes_*.jsonl")): rows+=loadlines(p)
    network=loadlines(run/"network_probes.jsonl")
    conds=[c[0] for c in protocol["conditions"] if c[0] in summaries["rule_public"]["0"]]
    names=["rule_public","mlp_base","gnn_base","gnn_continue","gnn_weight","gnn_oversample","gnn_generic","gnn_generic_rank_shuffle"]
    def values(model,condition,metric):
        return [v[condition][metric] for v in summaries[model].values()]
    def meanstd(model,condition,metric):
        vals=values(model,condition,metric)
        avg=np.mean(vals)*100
        sd=np.std(vals,ddof=1)*100 if len(vals)>1 else 0
        return f"{avg:.1f} ± {sd:.1f}" if len(vals)>1 else f"{avg:.1f}"
    lines=["# 신뢰도 보완 재실험 결과 (2026-10-08)","",
      "저장된 에피소드와 체크포인트를 가진 실제 재학습·재평가 결과다. "
      "원본 PoC와 입력·예산·데이터·난수 프로토콜이 달라 과거 수치와의 차이를 단일 수정 효과로 해석하지 않는다.","",
      f"- 학습 seed: {protocol['seeds']}; 조건별 독립 사례: {protocol['eval_episodes']}개.",
      f"- 총 진단 평가 에피소드: {len(rows):,}개. 동일 사례의 모델/seed별 반복을 포함한다.",
      f"- 정책과 독립된 네트워크 측정 사례: {len(network):,}개; 노드 쌍당 ping 5회.",
      "- 표의 ±는 학습 seed 간 표본 표준편차다. Rule은 학습 seed가 없다.",
      "- 자세한 조건: [프로토콜](../../../docs/specs/reliability_v2_protocol.md).",
      "- 원자료: protocol.json, manifest.json, summaries.json, comparisons.json, episodes_*.jsonl, network_probes.jsonl.","",
      "## 1. 정확도와 일반화","",
      "| 모델 | "+" | ".join(conds)+" |",
      "|---|"+"---:|"*len(conds)]
    for name in names:
        if name in summaries:
            lines.append("| "+name+" | "+" | ".join(meanstd(name,c,"exact") for c in conds)+" |")
    lines+=["","단위: %. mlp_base는 GNN과 같은 입력 및 pointer 행동을 사용하는 새 MLP다. "
       "continue는 진단 가중치·샘플링 없이 같은 수의 추가 update를 수행한 대조군이다. "
       "rank_shuffle은 Generic의 순위 특징 교란 평가이며 별도 재학습이 아니다.","",
       "## 2. 개선 효과의 불확실성","",
       "값은 뒤 모델−앞 모델의 Exact 차이(%p)와 paired hierarchical bootstrap 95% 구간이다. "
       "학습 seed와 같은 평가 사례의 의존성을 반영한다. seed 3개와 다중 비교의 한계가 있다.","",
       "| 비교 | "+" | ".join(conds)+" |","|---|"+"---:|"*len(conds)]
    for key,cv in comp.items():
        vals=[]
        for c in conds:
            v=cv.get(c)
            if v:
                lo,hi=v["paired_hierarchical_bootstrap95"]
                vals.append(f"{100*v['delta_exact']:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]")
            else: vals.append("N/A")
        lines.append("| "+key+" | "+" | ".join(vals)+" |")
    lines+=["","## 3. 운영 관점의 진단 비용·실패","",
       "시간은 CPU 정책 결정시간이다. 네트워크 ping 지연과 다른 값이며 API·전송 지연을 포함하지 않는다. "
       "각 seed에서 구한 평균/분위수를 다시 평균한다. 측정 머신은 manifest에 기록했다.","",
       "| 조건 | 모델 | Exact % | 오진 제출 % | 미제출 timeout % | 도구 호출 평균 | 도구 상대비용 평균 | 반복 호출 % | 정책 결정 p95 ms | 성공 진단 단계 평균 |",
       "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    detailed=[c for c in ["star_6","star_10","mesh_6","star_10_mild"] if c in conds]
    for c in detailed:
        for name in ["rule_public","mlp_base","gnn_base","gnn_continue","gnn_oversample","gnn_generic"]:
            blocks=[v[c] for v in summaries[name].values()]
            def avg(key,sub=None):
                vs=[v[key][sub] if sub else v[key] for v in blocks]
                vs=[x for x in vs if x is not None]
                return np.mean(vs) if vs else None
            fields=[f"{avg('exact')*100:.1f}",f"{avg('wrong_submission_rate')*100:.1f}",
                    f"{avg('timeout_rate')*100:.1f}",f"{avg('tool_calls','mean'):.2f}",
                    f"{avg('simulator_tool_cost','mean'):.2f}",f"{avg('repeat_tool_rate')*100:.1f}",
                    f"{avg('policy_decision_ms','p95'):.3f}"]
            succ=avg("successful_diagnosis_steps","mean")
            fields.append(f"{succ:.2f}" if succ is not None else "N/A")
            lines.append("| "+c+" | "+name+" | "+" | ".join(fields)+" |")
    lines+=["","성공 진단 단계는 성공 사례에 조건부인 값이다. 낮은 성공률의 모델이 여기서 빠르다고 해서 "
       "전체 효율이 높은 것은 아니다. 전체 단계·성공 누적 단계 분포·선택적 정확도는 summaries.json에 있다.","",
       "## 4. 장애 유형별 진단","",
       "| 조건 | 모델 | Device % | Performance % | Misconfiguration % | Link % | 유형 macro F1 |",
       "|---|---|---:|---:|---:|---:|---:|"]
    for c in detailed:
        for name in ["rule_public","mlp_base","gnn_base","gnn_oversample","gnn_generic"]:
            blocks=[v[c] for v in summaries[name].values()]
            vals=[f"{100*np.mean([v['per_fault'][ft]['exact'] for v in blocks]):.1f}" for ft in
                  ["device_failure","performance_degradation","misconfiguration","link_failure"]]
            vals.append(f"{np.mean([v['macro_type_f1'] for v in blocks]):.3f}")
            lines.append("| "+c+" | "+name+" | "+" | ".join(vals)+" |")
    lines+=["","## 5. 진단 전 네트워크 상태","",
       "모델 성능이 아니라 같은 장애 사례의 공통 환경 측정이다. 각 사례의 측정치를 평균했다. "
       "지연은 성공한 ping에 조건부이며 실패율과 함께 읽어야 한다. "
       "p95 열은 사례별 p95의 평균으로 전체 패킷의 pooled p95가 아니다.","",
       "| 조건 | ping 실패 % | 정상 반사실 실패 % | 지연 평균 ms | 사례별 지연 p95 평균 ms | 정상 p95 평균 ms | 동일 pair 지연차 절댓값 평균 ms | 명시 카운터 손실 % | 카운터 제공 % |",
       "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    network_summary={}
    for c in conds:
        group=[r for r in network if r["condition"]==c]
        def avg_net(state,key,sub=None):
            vals=[r["metrics"][state][key][sub] if sub else r["metrics"][state][key] for r in group]
            vals=[v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None
        d={"probe_failure":avg_net("faulted","probe_failure_rate"),
           "healthy_probe_failure":avg_net("healthy_counterfactual","probe_failure_rate"),
           "latency_mean_ms":avg_net("faulted","simulated_latency_ms","mean"),
           "mean_episode_p95_ms":avg_net("faulted","simulated_latency_ms","p95"),
           "healthy_mean_episode_p95_ms":avg_net("healthy_counterfactual","simulated_latency_ms","p95"),
           "same_pair_abs_delta_ms":avg_net("faulted","same_pair_abs_latency_delta_ms","mean"),
           "conditional_packet_loss":avg_net("faulted","packet_counter_loss_rate"),
           "packet_counter_coverage":avg_net("faulted","packet_counter_call_coverage")}
        network_summary[c]=d
        vals=[f"{100*d['probe_failure']:.1f}",f"{100*d['healthy_probe_failure']:.1f}",
          f"{d['latency_mean_ms']:.2f}",f"{d['mean_episode_p95_ms']:.2f}",
          f"{d['healthy_mean_episode_p95_ms']:.2f}",f"{d['same_pair_abs_delta_ms']:.2f}",
          f"{100*d['conditional_packet_loss']:.1f}",f"{100*d['packet_counter_coverage']:.1f}"]
        lines.append("| "+c+" | "+" | ".join(vals)+" |")
    (run/"network_summary.json").write_text(json.dumps(network_summary,indent=2),encoding="utf8")
    lines+=["","원본 ping은 경로 지연 합에 ±10% 잡음을 넣는다. 카운터가 없는 장비-down 실패가 있어 "
       "조건부 패킷 손실률을 전체 손실률로 쓰면 과소평가할 수 있다. "
       "동일 pair 지연 변동은 합성 관측값으로 RFC IPDV와 동일하지 않다.","",
       "## 6. 재현성과 적용 범위","",
       "- 사전에 알려진 장비/명목 연결 목록을 사용한다. 무지식 토폴로지 발견 문제로 일반화하지 않는다.",
       "- 모델 간 데이터·입력·행동·평가 사례·optimizer update 예산을 통일했다. 파라미터 수는 다르다.",
       "- 3개 학습 seed는 같은 데이터셋을 사용한다. 데이터 수집 seed를 바꾼 재현은 남아 있다.",
       "- 성공 데모 선별률은 demo_audit.json에 기록했다. 실패 데모를 학습한 결과는 아니다.",
       "- 정상 무장애, 복합 장애, 실제 트래픽·실장비, API 기반 LLM은 이번 평가에 없다.",
       "- 처리량/굿풋, 실제 packet-level jitter, MTTR, SLA 복구율은 측정하지 않았다.",
       "- 후속 모델용 JSON 어댑터를 제공했으나 실제 모델 접속·추론은 하지 않았다.","",
       "## 7. 학습 자원","",
       "| 모델/seed | optimizer updates | 학습 초 | 파라미터 수 | 마지막 100 update 평균 loss |",
       "|---|---:|---:|---:|---:|"]
    for name,v in train.items():
        lines.append(f"| {name} | {v['updates']} | {v['seconds']:.1f} | {v['parameters']:,} | {v['last_100_loss']:.5f} |")
    (run/"report.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5),layout="constrained")
    xs=np.arange(len(conds))
    for name in ["rule_public","mlp_base","gnn_base","gnn_continue","gnn_oversample","gnn_generic"]:
        means=[100*np.mean(values(name,c,"exact")) for c in conds]
        errors=[100*np.std(values(name,c,"exact"),ddof=1) if len(values(name,c,"exact"))>1 else 0 for c in conds]
        ax.errorbar(xs,means,yerr=errors,marker="o",capsize=3,label=name)
    ax.set_xticks(xs,conds,rotation=25,ha="right")
    ax.set_ylabel("Exact diagnosis (%)")
    ax.set_title("Observation-only benchmark: mean +/- training-seed SD")
    ax.set_ylim(-3,103);ax.grid(alpha=.2);ax.legend(ncol=3)
    fig.savefig(run/"exact_by_condition.png",dpi=180);plt.close(fig)
    print(run/"report.md")
if __name__=="__main__":main()
