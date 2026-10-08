# Stochastic evidence + observation conditioning audit

> 재학습 없음. E3 Control zip + Ablation-NL zip. NL-only **stochastic** 400 episode.
> det에서 Evidence-Consistent 0%는 tool-use=0이라 동어반복. 이번이 그 분모 보정.

## A — Tool-use 조건부 (갈래 1 vs 2)

| | Control | Ablation-NL |
|---|---:|---:|
| n | 400 | 400 |
| Tool-use | 144 (36%) | 148 (37%) |
| Φ≥1 | 41 (10%) | 37 (9%) |
| **P(Φ≥1 \| tool-use)** | **28.5%** | **25.0%** |
| **P(exact ∧ consistent \| tool-use)** | **0.7% (1/144)** | **0.7% (1/148)** |
| **P(exact ∧ consistent \| Φ≥1)** | **2.4% (1/41)** | **2.7% (1/37)** |

Fault별 exact-consistent: Device **0**, Misconfig **0**, Perf **1** (양쪽 모델).

도구를 쓰는 궤적의 ~1/4에서 Φ는 켜진다. 그 위에서 informed exact는 **거의 없다**. 갈래 1(“informed diagnosis가 가끔 있는데 greedy가 버린다”)은 1/144로는 지지되지 않는다.

## B — Evidence가 Gym obs / diagnose 확률에 들어가는가

시뮬레이터 내부 상태는 갱신된다. **PPO가 보는 벡터는 일부를 버린다.**

| 사실 | 내부 상태 | Gym 벡터 |
|---|---|---|
| Device DOWN (`check_status` / failed ping) | `DeviceStatus.DOWN` | `device_status[:,0] = 1` iff `status == 'active'` (문자열). enum과 불일치 → **feat0는 항상 0** |
| Perf latency 20.65ms (Φ=2) | `connection_properties['latency']` | `discovery_matrix`는 int8 adjacency만. CONNECTED 유지 → **latency 없음** |
| Ping fail | error 문자열 | `recent_diagnostics`에 **success=0** + tool/src/dst hash + timestamp. error 텍스트 없음 |

`device_status` obs의 max|Δ|=1.0은 DOWN이 아니라 **timestamp / hash 컬럼**이다. feat0는 scan 전후 전부 0.

### Diagnose head (Control)

| probe | Φ | TV(diagnose) | P(correct) | argmax |
|---|---:|---:|---|---|
| ping → down `edge_0` | 2 | 0.037 | 0.00037 → 0.00043 | 102 → **102** |
| ping healthy pair (dummy) | 0 | **0.076** | 0.00037 → 0.00051 | 102 → 95 |
| check_status `edge_0` | 2 | 0.068 | 0.00037 → 0.00044 | 102 → 95 |
| ping degraded `center→edge_0` (20.65ms) | 2 | 0.088 | **0.064 → 0.052** | 102 → 95 |

Ablation-NL도 같다: dummy TV(0.179) > evidence ping TV(0.100). Φ=2여도 P(correct device)≈0.000. Perf는 Control에서 정답 확률이 오히려 내려간다.

정책은 “도구를 썼다”에는 약간 반응한다. **GT evidence 방향로 diagnose를 옮기지는 않는다.** dummy가 evidence보다 더 흔든다.

## 갈래

**갈래 1 기각에 가깝다.** stochastic informed exact ≈ 0.7%, Device/Misconfig 0.

**갈래 2를 둘로 나눠야 한다.**

1. **Obs 손실 (환경 벡터):** DOWN과 latency가 Gym observation에 없다. 9.10의 “도구가 type/loc 증거를 노출한다”는 **시뮬레이터 툴 결과** 기준이지, MaskablePPO가 받는 `to_dict()`가 아니다. Perf 27% 복권은 latency를 볼 수 없는 정책이 scan 직후 찍는 것과 맞다.
2. **정책이 보이는 비트도 안 씀:** failed ping의 `success=0`은 obs에 들어간다. 그래도 P(correct device)는 0.0004에서 안 움직인다.

E4(diagnose-mask)만 올리면, 강제된 도구가 (a) latency를 obs에 안 넣고 (b) success=0을 diagnose가 무시하는 상태 그대로다. curriculum 실패 시 원인을 또 못 가른다.

## E4 전에

관측 래퍼(실험 코드, NetHeal 패키지 수정 아님)로라도 `DeviceStatus` 값과 latency를 벡터에 넣은 뒤, 같은 zip으로 diagnose TV / P(correct)를 다시 본다. 그때 P(correct)가 안 움직이면 정책/아키텍처, 움직이면 그다음에 E4.

산출물: `results/stochastic_evidence_audit.json`
