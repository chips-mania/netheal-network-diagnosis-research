# NetHeal Troubleshooting 학습 — 전체 결과 취합 보고서

**작성일:** 2026-09-07  
**코드:** `netheal-topology-generalization/`  
**환경(수정 금지):** `netheal-ai-agent-benchmark/`  
**Conda:** `netheal_base`  
**상세 로그:** `NetHeal_experiment_history.md` · 개별 PoC: `netheal-topology-generalization/reports/`  
**선행 요약:** `NetHeal_RL_results_summary.md` (Part A) · GNN 상세는 본 문서 Part B

이 문서는 **맨 처음 RL 감사부터** GNN expert-guided 단계까지 **한 줄기로** 취합한다.

---

## 0. 전체 한 줄 스토리

```
공식 Gym + 원본 보상 + MaskablePPO
  → Link shortcut (troubleshooting 미학습)
  → 보상·게이트·PBRS·관측 패치로도 greedy 숏컷 미해결
  → Rule ceiling 100% / BC-rule 99%  → 과제는 풀림, 병목은 데모 트리
  → pad-MLP는 경로 클론 (스케일·mix 실패)
  → GNN+pointer BC: 부분적 scale 이득 (N=8)
  → N=10 Link = commit bottleneck → Generic commit으로 회복
  → N=10 Misconfig = cue→spoke pointer binding → feature enrichment 대기
  → Mesh family 전이는 아직 실패
```

**프로젝트 가치:** “PPO를 고쳤다 / GNN이 원리를 발견했다”가 아니라,  
**인터페이스 실패 모드 → 상한 → 병목 분해 → 표현·프로토콜 개입**까지 닫은 감사.

---

## 1. 연구 질문 · 설정 · 잠금 규칙

### 1.1 질문의 진화

| 단계 | 질문 |
|---|---|
| **Part A (RL)** | 공식 RL 인터페이스 위에서, **어떤 학습 프로토콜**이 실제 troubleshooting(도구로 증거 → 타입×위치 진단)을 만드는가? |
| **Part B (GNN)** | pad-MLP 경로 암기 대신 **그래프에 grounding된 expert procedure**를 이식하면 held-out 크기·family로 더 잘 전이되는가? |

Cisco 주 대회 경로(AgentBeats/LLM)와 분리. Mesh 일반화는 Seen에서 유효 정책이 생긴 뒤로.

### 1.2 고정 실험 설정 (대부분)

| 항목 | 값 |
|---|---|
| Topology | Star 중심 (나중 Linear/Mesh ceiling·eval) |
| Faults | link / device / performance / misconfiguration |
| Algo (RL기) | MaskablePPO, `MultiInputPolicy` |
| RL timesteps | 보통 **30k** · Seed 주로 **1** |
| Eval | **100 ep** (일부 25/fault), greedy |
| Max steps | 15 |
| Cisco 패키지 | **미수정** — 래퍼/프로토콜만 |

### 1.3 공식 보상 (요지)

Exact **+16** (도구 사용 무관) · Wrong **−16** · type-only partial **없음** · Step **−0.1**.  
진단 ≈ **72-way**; post-scan 직후에도 진단 즉시 valid → 숏컷 구조적 유인.

### 1.4 관측 구멍

시뮬 툴 결과에는 증거가 있으나 Gym `to_dict()`가 DOWN/latency 등을 잘 안 노출.  
래퍼 `evidence_obs`가 **같은 키·shape**에 채움 (새 키 추가 없음).

### 1.5 표현 잠금 (뒤집어쓰지 말 것)

1. Vanilla PPO → invalid collapse.  
2. Maskable + 원본 보상 → greedy **Link 숏컷**.  
3. Link-only는 스캔만으로 100% 가능.  
4. “72-way라서 불가 / fault unlearnable” **금지** — Rule·BC-rule이 반증.  
5. 보상·마스크를 바꾼 실험은 **Cisco Control과 다른 named variant**.  
6. Best *learned* (Part A) = **Rule 궤적 BC**; RL은 발견 엔진이 아님.  
7. pad-MLP 스케일 = **경로 클론**.  
8. GNN = expert procedure **이식·전이**이지, 원리 자율 발견이 아님.  
9. N=10 Link = **commit**; Misconfig = **targeting/binding**.

---

# Part A — RL / BC 감사 (시작 ~ 경로 클론)

## A1. 결과 총표

| 단계 | Exact / 핵심 | Decision | 한 줄 |
|---|---|---|---|
| Vanilla PPO | invalid collapse | fail | 무효 진단 붕괴 |
| MaskablePPO 30k | DSR~17%, AvgSteps **2.0** | no_go | `scan→diagnose` Link 숏컷 |
| Link-only 30k | DSR **~100%** | — | 스캔만으로 링크 위치 |
| Symmetric reward | Exact↓ | fail | 숏컷 미해결 |
| Diagnosis gate (≥1 tool) | shortcut 0% | dummy_tool | 형식만 도구 (dummy ping) |
| Factorized diagnosis | Exact 0%, timeout | avoidance | 제출 회피 |
| E1 Pure PBRS | shortcut 100%, AvgSteps 2.0 | **no_go** | Control과 동일 |
| E2 Timeout=−16 | shortcut 100% | **no_go** | greedy 숏컷 유지 |
| E3 ent_coef=0.01 | greedy AvgSteps 2.0 | — | greedy 숏컷 안 죽음 |
| Ablation-NL (학습에서 Link 제거) | NL eval shortcut 100% | **no_go** | Link 경쟁 제거로도 안 풀림 |
| Obs Stage B | Device Δ≈0 | **no_go** | E4 curriculum **미실시** |
| Evidence diagnose-gate | Exact~0%, timeout **~76%** | **no_go** | 회피 |
| Repeat penalty (−2) | submit↑, Exact 낮음 | ambiguous | hard-fault 미흡 |
| BC-informed (GT 3-step) | Exact **58%** | — | Device/Perf 붕괴 |
| Rule ceiling (GT-free) | Exact **100%** | ceiling | 조건식 트리로 닫힘 |
| **BC-rule** | Exact **99%** | **go** | best learned (모방) |
| Star pad12 N=6 base | Exact **97%** | — | 고정 공간 BC |
| Zero-shot N=8 / N=10 | **76% / 64%** | — | 크기↑ 성능↓ |
| Naive FT | held-out **악화** + N=6 망각 | — | forgetting |
| Replay FT | 망각만 방지, held-out≈ZS | — | |
| Mix {4,5,6}→N=8 | **75%** ≈ N6-only | — | 일반화 이득 없음 |
| Short BC-seed + gate PPO | Exact **2%** | ambiguous | seed 미모방 |
| Full BC-rule + gate PPO | Exact **77%** | **go** | 시드가 본체; Misconfig 8% |

## A2. 초기 감사 — 왜 PPO가 troubleshooting이 아닌가

1. **Vanilla** → invalid action collapse.  
2. **MaskablePPO** → 유효 액션만, 그러나 greedy가 `scan → diagnose(link, …)`로 수렴. AvgSteps≈2.  
3. **Link-only 100%:** post-scan missing edge로 위치가 바로 분리됨 → 숏컷이 “합리적 해킹”.  
4. 툴 결과만 보면 네 fault를 구분할 수 있음. PPO는 그 트리를 **쓰지 않거나 못 맞춤**.

### 고장 미분 (왜 여러 툴이 필요한가)

| 증상 | 후보 | 가르는 툴 |
|---|---|---|
| ping 실패 | Link / Device / Misconfig | status, 반대 ping, interfaces |
| 스캔상 결손 간선 | Link / Misconfig | 비대칭 + interfaces |
| 스캔상 전부 연결 | Device / Perf | status DOWN vs ping 성공+고지연 |
| ping 성공 + 지연 10–20× | **Perf만** | |

한 툴이 “A or B”로 보이는 것이 정상. Rule/BC-rule은 **두 번째 툴**로 푼다.

## A3. 숏컷 깨기 — 보상·구조 대부분 실패

Symmetric · Gate(dummy) · Factorized(timeout) · E1 PBRS · E2 timeout · E3 entropy · Ablation-NL.

**공통:** greedy `scan→diagnose`를 Device/Perf 겹침 해소 트리로 **뒤집지 못함**.

## A4. 관측 · Evidence gate

- Stochastic audit: 도구를 써도 `P(exact∧consistent|tool)` ≈ 0.7%.  
- Obs Stage B **no_go** → E4 curriculum **올리지 않음**.  
- Evidence gate: shortcut 0% but timeout ~76% (**회피**).  
- Repeat penalty: 제출↑, Exact/hard-fault 미흡 (**ambiguous**).

## A5. 전환점 — Rule / BC

| | Rule (GT 없음) | BC-informed | BC-rule |
|---|---:|---:|---:|
| Exact | **100%** | 58% | **99%** |
| Device | 25/25 | 2/25 | 25/25 |
| Perf | 25/25 | 7/25 | 24/25 |
| Misconfig | 25/25 | 24/25 | 25/25 |
| Link | 25/25 | 25/25 | 25/25 |
| AvgSteps | 5.8 | 3.3 | 6.0 |
| 데모 | 조건식 | GT `scan→툴1→정답` | Rule 궤적 |

**해석:** 같은 MLP로도 트리를 배울 수 있다. GT 3-step은 겹침 해소를 건너뛰어 Device/Perf가 무너진다.  
→ 병목은 **“데모/프로토콜이 트리를 담느냐”**이지 MLP 용량·72-way 불가 아님.

### Gate / seed / PPO

| Stage | Exact | Shortcut | Timeout |
|---|---:|---:|---:|
| Gate-only PPO | ~0% | 0% | **76%** |
| Short seed + Gate PPO | **2%** | 0% | 50% |
| Full Rule-BC (gate OFF) | **98–99%** | 0% | 0% |
| Full BC + Gate PPO 30k | **77%** | 0% | 22% |

Full BC+gate: Device/Perf/Link **100%**, Misconfig **8%**.  
→ 게이트는 숏컷을 막지만 시드 없이는 정책이 안 생김. PPO는 BC를 넘지 못함.

## A6. Star 크기 스케일 (pad12) — 경로 클론

| N | Rule | Zero-shot (N=6 BC) | Naive FT | Replay FT |
|---:|---:|---:|---:|---|
| 6 | 100% | **97–99%** | 망각 시 ↓ | retention ~98% |
| 8 | ~97–99% | **76–79%** | **악화** | ≈ zero-shot |
| 10 | ~98–99% | **54–64%** | **악화** | ≈ zero-shot |

Mix {4,5,6} → held-out N=8 **75%** ≈ N=6-only.  
→ 다중 크기 데모 ≠ 크기 불변 방법. **경로 클론** 해석 유지. Mesh 자격 없음.

## A7. Part A 판정

| 프레임 | 판정 |
|---|---|
| 공식 보상 + MaskablePPO → troubleshooting? | **아니오** (Link 숏컷) |
| 과제 unsolvable? | **아니오** (Rule 100%, BC-rule 99%) |
| 병목 | 데모/프로토콜이 겹침 해소 트리를 담는가 |
| 크기/위상 불변 방법론? | **아니오에 가깝다** |
| RL 변형으로 발견 엔진? | **아니오** (부정 결과가 본체) |

**권고 (당시):** 추가 RL 변형 PoC 한계효용 바닥 → **GNN + Expert-guided**로 전환 (`GNN_expert_guided_spec.md`).

---

# Part B — GNN + Expert-Guided (표현 · Scale · Commit · Pointer)

## B1. 파이프라인

| Phase | 내용 | 결과 |
|---|---|---|
| 1 Rule v2 | Topology-agnostic GT-free expert | Star6/8/10·Linear·Mesh ≈ **99–100%** |
| 2 Graph interface | `graph_obs` + pointer↔flat | Smoke **go** |
| 3 B1 vs P1 | Rule demos Star{4,5,6} → MLP pad vs GNN+pointer | §B2 |
| Train audit | Tiny overfit | mem·id_rank 수정 후 P1 tiny **PASS** |

**학습 가능 조건:** 초기 P1 Seen 74%/loss≈1.2는 GNN 무용이 아니라 **(1) evidence mem 미연결 (2) Star spoke isomorphism**. 수정 후 Seen 100% → 일반화 해석 가능 단계 진입.

## B2. Phase 3 — B1 vs P1

Train: Rule v2, Star N∈{4,5,6}. Eval: Star6/8/10, Mesh6.

| Model | Star6 | Star8 | Star10 | Mesh6 |
|---|---:|---:|---:|---:|
| B1 MLP (pad12) | 89% | 76% | **57%** | 0% |
| P1 GNN+pointer | **100%** | **90%** | 42% | 9% |

P1 N=10 then: Device 68% · Perf 100% · Misconfig **0%** · Link **0%** · Timeout **58%**.

- N=6→8: P1이 B1 대비 **+14%p** → 제한적 scale 긍정.  
- N=10·Mesh: 실패/약함.  
- 표현: **`partial_scale_generalization`** (초안 게이트명 `go_scale_only`).

## B3. N=10 Failure Audit — Link는 commit

| N | Exact | Timeout | Conclude=GT | Evidence→no-commit |
|---:|---:|---:|---:|---:|
| 6 | 100% | 0% | 97% | 0% |
| 8 | 87% | 13% | 100% | 13% |
| 10 | 46% | 54% | 81% | 35% |

**Link@N10:** Exact 0% · Timeout 100% · timeout 중 conclude=GT **100%** · 궤적에 diagnose 없음.  
→ 증거는 충분한 상태에서 **제출(diagnose)을 안 함** = **commit bottleneck**.

Conclude-ready margin @N10: P(diag)~0.28, top-1 **10%**, top-2 **80%** → near-miss.  
Misconfig ready: P(diag)≈0.01 → 이후 path/targeting으로 재분류.

## B4. Commit Intervention · Ablation

### Current (×3 + w_diag=4 + Link/Misconfig ×2)

| N | Baseline | Commit | Timeout |
|---:|---:|---:|---:|
| 6 | 100% | 99% | 0% |
| 8 | 90% | **96%** | 4% |
| 10 | 42% | **67%** | 33% |

N=10 Link **0→100%**. Ready P(diag)≈0.98.  
태그: `commit_bottleneck_confirmed_and_partially_resolved`.

### Generic vs Current

| N | Baseline | Generic (×3+w4, hard=1) | Current |
|---:|---:|---:|---:|
| 8 | 90% | 86% | **96%** |
| 10 | 42% | **66%** | 67% |

N=10 Link: 0 → Generic **92%** → Current 100%.  
→ **본선은 Generic**. hard×2는 N=8 Misconfig에만 잔여 이득 (48% vs 88%).

### Factor B (weight-only)

N10 Exact: Baseline 42% → B **61%** → Generic **66%**.  
Ready-commit@N10: 9% → 53% → **96%**. Premature ≈ 0.  
판정: `weight_helps_oversample_adds`.

**잠금 문장:** 일부 scale failure는 graph 붕괴가 아니라 evidence-conditioned **diagnosis commitment**의 scale 일반화 실패이며, Generic으로 Link@N10을 fault-specific 없이 회복했다.

## B5. Misconfig — Path / Pointer

### Path buckets (Generic)

| Model | N | Exact | C commit | D incomplete |
|---|---:|---:|---:|---:|
| baseline | 10 | 0% | 56% | 44% |
| generic | 10 | **8%** | **0%** | **92%** |

asym_from_GT **100%** · iface@GT **8%** · 실패 traj = status spam.  
성공 ≈ GT=`center`.

| Fault @N10 | 주 실패 모드 |
|---|---|
| Link | **commit** → Generic으로 회복 |
| Misconfig | **cue→spoke targeting** |
| Perf | 이미 ~100% |
| Device | 잔여 scale (~68–72%) |

### First-cue spoke rank (forced-scan)

| Model | N6 | N8 | N10 top-1 / top-3 |
|---|---:|---:|---|
| Generic joint | 100% | ~44% | **0% / 31%** |
| Baseline joint | 100% | ~81% | 23% / 69% |
| Node-only | 100% | ~83% | **23% / 83%** |
| Cue curriculum | 100% | **12%** | **26% / 33%** |

Offline Rule iface: N=10 pred=GT ~100% → capacity OK.  
Curriculum: top-1 소폭↑, top-3↓, N8 붕괴 → **`curriculum_insufficient_try_features`**.

**다음:** 관측 기반 **missing-incident node features** (GT leak 없이) → first-cue 재평가.  
Curriculum과 feature를 **한 스텝에 섞지 않음**.

## B6. Fault 메커니즘 지도 (현재)

```
N=10 P1 Baseline Exact ~42%
  ├─ Link 0% ──► commit ──► Generic ~92–100%
  ├─ Misconfig 0–8% ──► spoke pointer ──► feature enrichment 대기
  ├─ Perf ~100%
  └─ Device ~68%
Mesh (Star-only train) ~9% ──► family 미달 (별도)
```

## B7. Part B 방법 스택 (권고)

| 역할 | 내용 |
|---|---|
| Primary (scale/Link) | **Generic commit** (diagnose ×3 + w_diag×4) |
| Ablation only | Current hard×2 · weight-only B |
| Misconfig next | Missing-incident / asym node features |
| 보류 | 더 큰 GNN 단독 · DAgger(early binding 전) · Mesh claim |
| 중간지표 | First-cue GT rank · Ready/Premature commit |

---

# Part C — 전체 타임라인 · 산출물 · 다음

## C1. 타임라인

```
Vanilla PPO collapse
  → MaskablePPO: Link shortcut
  → Link-only 100% / post-scan audits
  → Symmetric / Gate(dummy) / Factorized / E1–E3 / Ablation-NL
  → Obs holes → evidence_obs → Stage B no_go (E4 skip)
  → Evidence gate (timeout) / Repeat penalty
  → BC GT-3-step 58%
  → Rule ceiling 100%
  → BC-rule 99%                    ← Part A best learned
  → Star pad12 scale / replay / mix (경로 클론)
  → Short seed+gate ~2% / Full BC+gate 77%
  ────────────────────────────────
  → Rule v2 ceiling (multi-topo) 99–100%
  → Graph interface smoke go
  → P1 train audit (mem + id_rank)
  → B1 vs P1: partial_scale_generalization
  → N10 Link audit → commit fix → Generic ablation
  → Misconfig path → pointer probe → curriculum insufficient
  → [NEXT] missing-incident features
```

## C2. 재현 엔트리 (요지)

```powershell
cd netheal-topology-generalization
$env:PYTHONPATH="...\netheal-ai-agent-benchmark"
conda run -n netheal_base --no-capture-output python -m src.<module>
```

| 구간 | 모듈 예 |
|---|---|
| Rule / BC-rule | `run_rule_ceiling`, `run_bc_rule` |
| Scale / gate | `run_star_scale_poc`, `run_bc_rule_gate_ppo` |
| Rule v2 / graph | `run_rule_v2_ceiling`, `run_graph_interface_smoke` |
| B1 vs P1 | `run_graph_expert_bc` |
| Commit | `run_p1_commit_fix`, `run_p1_commit_ablation`, `run_p1_commit_factor_ablation` |
| Misconfig / pointer | `run_p1_misconfig_path_audit`, `run_p1_pointer_only_probe`, `run_p1_pointer_curriculum` |

## C3. 문서 인덱스

| 문서 | 내용 |
|---|---|
| `NetHeal_experiment_history.md` | 실험 전체 상세 로그 |
| `NetHeal_RL_results_summary.md` | Part A 전용 취합 |
| `GNN_expert_guided_spec.md` | Part B 설계 |
| `reports/step_progress.md` | 진행 로그 |
| `reports/*_poc.md` / `p1_*.md` | 수치 1차 권위 |

## C4. 다음 우선순위

1. **Missing-incident node features** → First-cue spoke @N=8/10.  
2. Feature가 top-1을 올리면 Generic(+feature) Misconfig Exact·iface@GT.  
3. Baseline vs Generic N10/Link **multi-seed**.  
4. Mesh: Star-only vs Star+Linear.  
5. DAgger는 early binding이 풀린 뒤.

## C5. 결론 (처음부터 끝까지)

1. **공식 RL 인터페이스는 troubleshooting을 스스로 발견하지 못한다** — Link 숏컷이 본체이며, 보상·게이트·PBRS 계열 부정 결과가 이를 닫는다.  
2. **과제는 풀린다** — Rule 100% / BC-rule 99%; 병목은 겹침 해소 **트리 데모**.  
3. **pad-MLP BC는 Seen 강하지만 크기 불변이 아니다** — 경로 클론.  
4. **GNN+pointer는 부분적 scale 이득**을 입증했고, N=10 Link는 **commit**으로, Misconfig는 **cue→spoke binding**으로 분해된다.  
5. Mesh·완전 크기 불변은 **아직 미달**. 다음 레버는 Misconfig feature enrichment.

---

*수치의 1차 권위는 각 `reports/*.md` / `results/*.json`이다. Cross-run 비교 시 체크포인트를 고정할 것. Named variant를 Cisco Control과 동일 과제로 부르지 말 것.*
