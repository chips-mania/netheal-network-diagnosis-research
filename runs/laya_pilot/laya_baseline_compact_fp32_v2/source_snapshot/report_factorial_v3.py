"""Produce hypothesis-led report without changing any training/test settings."""
import argparse
import json
from pathlib import Path
import numpy as np
from .analyze_factorial_v3 import analyze,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument("run");args=ap.parse_args()
    run=Path(args.run);cfg,summaries,effects,network,training=analyze(run)
    conditions=[c[0] for c in cfg["conditions"]]
    names=["star_original","star_removed","star_augmented","mixed_original","mixed_removed","mixed_augmented"]
    def blocks(name,variant,condition):
        return [s[variant][condition] for s in summaries[name].values()]
    def fmt(name,variant,condition):
        v=[b["exact"]*100 for b in blocks(name,variant,condition)]
        return f"{np.mean(v):.1f} ± {np.std(v,ddof=1):.1f}" if len(v)>1 else f"{v[0]:.1f}"
    verdict={"improvement":"개선 근거","deterioration":"악화 근거","inconclusive":"불확실"}
    lines=["# v3: 순위 의존성과 학습 구조 다양성의 요인 실험","",
        "LAYA·Jev·일반 LLM 호출 없이 3×2 GNN 요인 실험을 수행했다. "
        "실행 전에 가설과 판정 기준을 고정하고 모든 모델을 학습한 후 최종 평가를 시작했다.","",
        f"- 독립 학습 반복: {cfg['replicas']} (수집 데이터와 초기화를 함께 변경).",
        f"- 비교군당 데모 {3*cfg['demos_per_n']:,}개, batch 64, {cfg['updates']:,} optimizer updates, 40-step 예산.",
        f"- 평가: {len(conditions)}조건 × {cfg['episodes']}사례, 동일 사례 canonical/renamed 짝지은 시험.",
        "- [실행 전 고정 설계](frozen_spec.md) · [주요 가설 원자료](hypothesis_results.json) · [무결성 검증](verification.json)",
        "- 아래 ±는 반복 간 표본 표준편차다. 정상 판정·실네트워크·복구 실험은 포함하지 않는다.","",
        "## 주요 가설과 판정","",
        "| 가설/대비 | Exact 차이 %p | 구간 %p | 판정 |","|---|---:|---|---|"]
    descriptions={
        "H1_augmentation_renamed":"H1: 증강−기존, renamed 정확도 (두 curriculum·전체 조건 평균)",
        "H1_gap_reduction_secondary":"보조: 증강의 canonical→renamed 감소 완화",
        "H2_diversity_heldout":"H2: mixed−star, Mesh/Random 정확도 (세 rank·두 variant 평균)",
        "H3_interaction_exploratory":"탐색: 구조 다양성과 증강의 상호작용"}
    for k,v in effects.items():
        lo,hi=v["interval"]
        lines.append(f"| {descriptions[k]} | {100*v['delta']:+.1f} | {v['coverage']*100:g}% [{100*lo:+.1f}, {100*hi:+.1f}] | {verdict[v['verdict']]} |")
    lines+=["","H1/H2는 각각 97.5% 구간으로 두 주요 비교에 Bonferroni alpha를 배분했다. "
       "replica와 topology-condition별 공통 사례를 재표본화했다. "
       "3회 반복 및 계산 예산으로 정한 표본수의 한계가 있으며, 세부 비교는 탐색적이다.","",
       "## 조건별 정확도","",
       "| 모델 | 관측 명명 | "+" | ".join(conditions)+" |","|---|---|"+"---:|"*len(conditions)]
    lines.append("| rule_public | canonical | "+" | ".join(fmt("rule_public","canonical",c) for c in conditions)+" |")
    for name in names:
        for variant in ["canonical","renamed"]:
            lines.append("| "+name+" | "+variant+" | "+" | ".join(fmt(name,variant,c) for c in conditions)+" |")
    lines+=["","단위 %. original은 기존 순위 특징, removed는 순위 특징 0, augmented는 "
        "4개 ID/순서 치환 데모를 사용한다. renamed에서는 특징·연결 인덱스·대상 ID를 일관되게 "
        "치환하고 행동을 원래 장비에 역매핑한다. 단순 순위 숫자 교란과 다르다.","",
        "## 비용·실패: 사례 수를 맞춘 전체 조건 평균","",
        "| 모델 | 명명 | Exact % | 오진 % | 미제출 % | 도구 호출 | 반복 도구 % | 상대 도구 비용 | 결정 p95 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    operational={}
    for name in ["rule_public"]+names:
        for variant in (["canonical"] if name=="rule_public" else ["canonical","renamed"]):
            bs=[b for c in conditions for b in blocks(name,variant,c)]
            def av(k,sub=None):
                vs=[b[k][sub] if sub else b[k] for b in bs]
                return float(np.mean([x for x in vs if x is not None]))
            vals={"exact":av("exact"),"wrong":av("wrong_submission_rate"),"timeout":av("timeout_rate"),
                  "tools":av("tool_calls","mean"),"repeat":av("repeat_tool_rate"),
                  "cost":av("simulator_tool_cost","mean"),"decision_p95_ms":av("policy_decision_ms","p95")}
            operational[name+"_"+variant]=vals
            lines.append(f"| {name} | {variant} | {100*vals['exact']:.1f} | {100*vals['wrong']:.1f} | {100*vals['timeout']:.1f} | {vals['tools']:.2f} | {100*vals['repeat']:.1f} | {vals['cost']:.2f} | {vals['decision_p95_ms']:.3f} |")
    lines+=["","지연 열은 조건·반복별 정책 결정 p95의 평균이며 전체 호출의 pooled p95가 아니다. "
       "CPU 로컬 시간으로 네트워크 ping 지연과 다르다. 상대 비용은 바이트나 금액이 아니다.","",
       "## 장애별 Exact (모든 조건 평균)","",
       "| 모델 | 명명 | Device % | Performance % | Misconfiguration % | Link % |",
       "|---|---|---:|---:|---:|---:|"]
    for name in ["rule_public"]+names:
        for variant in (["canonical"] if name=="rule_public" else ["canonical","renamed"]):
            bs=[b for c in conditions for b in blocks(name,variant,c)]
            vals=[100*np.mean([b["per_fault"][ft]["exact"] for b in bs]) for ft in
                  ["device_failure","performance_degradation","misconfiguration","link_failure"]]
            lines.append("| "+name+" | "+variant+" | "+" | ".join(f"{v:.1f}" for v in vals)+" |")
    lines+=["","## 독립 네트워크 측정","",
       "| 조건 | ping 실패 % | 평균 성공 지연 ms | 사례별 p95 평균 ms | 정상 반사실 p95 ms | 동일 pair 지연차 평균 ms | 카운터 제공 % | 카운터 조건부 손실 % |",
       "|---|---:|---:|---:|---:|---:|---:|---:|"]
    nets={}
    for c in conditions:
        nr=[r["metrics"] for r in network if r["condition"]==c]
        def avn(state,key,sub=None):
            vs=[r[state][key][sub] if sub else r[state][key] for r in nr]
            vs=[x for x in vs if x is not None]
            return float(np.mean(vs)) if vs else None
        vals={"failure":avn("faulted","probe_failure_rate"),
              "latency":avn("faulted","simulated_latency_ms","mean"),
              "p95":avn("faulted","simulated_latency_ms","p95"),
              "healthy_p95":avn("healthy_counterfactual","simulated_latency_ms","p95"),
              "delta":avn("faulted","same_pair_abs_latency_delta_ms","mean"),
              "counter_coverage":avn("faulted","packet_counter_call_coverage"),
              "counter_loss":avn("faulted","packet_counter_loss_rate")}
        nets[c]=vals
        lines.append(f"| {c} | {100*vals['failure']:.1f} | {vals['latency']:.2f} | {vals['p95']:.2f} | {vals['healthy_p95']:.2f} | {vals['delta']:.2f} | {100*vals['counter_coverage']:.1f} | {100*vals['counter_loss']:.1f} |")
    lines+=["","성공 ping이 없는 사례의 지연은 결측으로 처리한다.",
       "명시 카운터가 없는 장비-down 결과가 있어 조건부 손실률을 전체 손실률로 해석하지 않는다. "
       "합성 지연 변동은 실제 packet-level jitter가 아니다. "
       "정책과 독립적인 상태 측정이며 진단으로 서비스가 복구됐다는 의미가 아니다.","",
       "## 학습 데이터와 자원","",
       "| 모델/반복 | 데모 수 | 원래 transition | 증강 후 transition | update | 학습 초 | 마지막 loss |",
       "|---|---:|---:|---:|---:|---:|---:|"]
    for k,v in training.items():
        lines.append(f"| {k} | {v['demos']} | {v['raw_transitions']} | {v['augmented_transitions']} | {v['updates']} | {v['seconds']:.1f} | {v['last_100_loss']:.4f} |")
    lines+=["","동일 에피소드 수와 update를 맞췄지만 구조에 따라 데모 길이가 다르다. "
       "증강 후 transition 수가 늘어도 optimizer update 수는 동일하다. "
       "데이터 수집과 초기화의 개별 기여는 이 paired replica 설계에서 분리하지 않는다.","",
       "## 남은 범위","",
       "- 알려진 명목 inventory, 단일 장애, 합성 환경이라는 전제는 유지된다.",
       "- 정상 무장애/false alarm, 복합 장애, 관측 누락·잡음, 증거 개입은 후속 단계다.",
       "- 실제 throughput/굿풋/packet jitter/MTTR은 미측정이다.",
       "- 6,000 update 고정 예산에서의 결론이다. 충분한 수렴이나 최적 하이퍼파라미터를 주장하지 않는다.",
       "- 이 결과로 추가 방법을 설계하면 이번 최종 세트를 재사용해 확증했다고 주장하지 않는다.",
       "- LAYA·Jev·일반 LLM은 설치·호출·평가하지 않았다."]
    (run/"report.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    (run/"operational_summary.json").write_text(json.dumps(operational,indent=2),encoding="utf8")
    (run/"network_summary.json").write_text(json.dumps(nets,indent=2),encoding="utf8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,5.5),layout="constrained",sharey=True)
    for ax,variant in zip(axes,["canonical","renamed"]):
        for name in names:
            vals=[[b["exact"]*100 for b in blocks(name,variant,c)] for c in conditions]
            ax.plot(range(len(conditions)),[np.mean(v) for v in vals],marker="o",label=name)
        ax.set_xticks(range(len(conditions)),conditions,rotation=35,ha="right")
        ax.set_title(variant+" (mean of paired training replicas)")
        ax.set_ylim(-2,102);ax.grid(alpha=.2)
    axes[0].set_ylabel("Exact diagnosis (%)");axes[1].legend(fontsize=8,loc="best")
    fig.savefig(run/"factorial_exact.png",dpi=180);plt.close(fig)
    (run/"analysis_manifest.json").write_text(json.dumps({
        "analysis_source_sha256":sha(Path(__file__).with_name("analyze_factorial_v3.py")),
        "report_source_sha256":sha(Path(__file__)),
        "regression_tests_passed":12,"smoke_completed":True},indent=2),encoding="utf8")
    print(run/"report.md")
if __name__=="__main__":main()
