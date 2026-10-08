# Graph Expert-Guided Troubleshooting — 개념 정리 & 실험 설계

**작성일:** 2026-09-06  
**상태:** 설계 (미구현)  
**선행 취합:** `NetHeal_RL_results_summary.md`, `NetHeal_experiment_history.md`  
**코드 위치 (예정):** `netheal-topology-generalization/`  
**환경:** `netheal-ai-agent-benchmark` **수정 금지** (래퍼·expert·정책만 실험 레포)

---

## 0. 한 줄 요약

> **위상·크기에 묶인 정답 경로를 외우는 것이 아니라**, 그래프 위의 증거·가설·구분 툴 관계를 학습해  
> **학습에 넣지 않은 크기·topology family**에서도 troubleshooting이 되게 한다.

학습 수단의 후보는 **GNN + Expert-guided (BC → 조건부 DAgger)** 이다.  
**PPO를 살리는 것이 목표가 아니다.**

---

## 1. 배경: 지금까지가 말해 주는 것

| 사실 | 함의 |
|---|---|
| 공식 보상 + MaskablePPO → Link 숏컷 | RL이 원리를 “발견”하지 않음 |
| Rule (GT-free) Star Exact **100%** | 과제는 풀림; 상한 존재 |
| BC-rule N=6 Exact **~99%** | 같은 MLP로 **경로 모방**은 가능 |
| BC-informed (GT 3-step) **58%** | 겹침 해소 없는 데모는 Device/Perf 실패 |
| Zero-shot N=8 / N=10 **~76% / ~54–64%** | 크기만 바꿔도 붕괴 |
| Mix {4,5,6} → held-out N=8 **~75%** ≈ N=6-only | 여러 크기 암기로도 **방법론**이 안 생김 |
| Full BC + gate PPO **77%** < BC 99% | RL fine-tune은 보조·망각 위험 |

**해석 (잠금):** 현재 BC-rule은 “장애진단 방법”보다 **Star에서 Rule이 지나간 경로 복제**에 가깝다.

원하지 않는 것 (X) / 원하는 것 (O):

| X | O |
|---|---|
| Star 6대 정답 경로 암기 | “증상 → 무엇을 확인” |
| `관측 벡터 이 칸 → action 37` | “증거 → 어떤 가설을 버릴까” |
| Seen Exact만 고득점 | “후보 복수 → 어떤 툴로 구분” |
| | → **처음 보는** Star-N / Linear / Mesh 등에도 적용 |

---

## 2. 재정의된 연구 질문

### 2.1 최종 목적 (유지)

**Topology generalization:** 학습에 포함되지 않은 네트워크 크기·구조에서도 troubleshooting.

### 2.2 진짜 연구 질문 (갱신)

~~Star/Linear PPO 학습 → Mesh 테스트~~ (불충분·조기)

> **위상-불변 expert**의 troubleshooting trajectory를,  
> **그래프 객체에 grounding된** policy로 학습하면,  
> 고정 topology·고정 길이 표(MLP+pad)를 모방한 BC보다  
> **미관측 크기·topology family**에서 더 잘 일반화되는가?

### 2.3 최종 판정 기준 (단일)

> 처음 보는 그래프에서도 **evidence를 모으고**, **올바른 진단 경로**를 만들어 내는가?

부차 지표: Exact DSR, evidence acquisition, exact_consistent, tool count, shortcut rate, per-fault.

### 2.4 주장 가능한 문장 / 금지 문장

| 가능 | 금지 |
|---|---|
| Expert procedure를 graph-inductive policy로 **이식**해 전이한다 | PPO/GNN이 원리를 **스스로 발견**했다 |
| MLP path-clone 대비 held-out에서 개선 | Cisco Control과 동일 과제에서 RL로 DSR 1등 |
| 부정: 표현·액션을 바꿔도 family 전이가 안 된다 | Train에 Mesh를 넣고 “unseen Mesh 일반화” |

---

## 3. 방법론 개념

### 3.1 두 축의 역할

```
구조/크기 shift          분포 shift / 무엇을 할지
        │                         │
        ▼                         ▼
      GNN (+ pointer)         Expert-guided
   (어떻게 표현·선택)     (BC → 조건부 DAgger)
        │                         │
        └──────────┬──────────────┘
                   ▼
        Unseen graph zero-shot eval
```

| 구성요소 | 담당 | 단독으로 부족한 이유 |
|---|---|---|
| **GNN** | 가변 N·관계를 같은 계산으로 | 보상 RL만 붙이면 숏컷 그대로 |
| **Expert-guided** | 조사 순서·겹침 해소 | MLP+동일 expert면 표현 한계 잔존 |
| **DAgger** | learner 방문 상태 보정 | 구조 표현 한계는 못 고침 → **2단계** |
| **PPO/RL** | (선택) 미세조정 | 본 과제 본체 아님; forgetting 위험 |

### 3.2 원하는 학습 단위 (의미론)

암기:

```text
center 확인 → edge_2 ping → edge_2 interface → Device failure @ edge_2
⇒ “이 벡터 모양이면 action_id=37”
```

원리:

```text
ping 실패 ∧ device DOWN ⇒ 해당 **node**의 device_failure 가설 ↑
결손 간선 ∧ 양방향 비대칭 ⇒ link vs misconfig를 interfaces로 구분
고지연 ∧ ping 성공 ⇒ performance @ 해당 **edge/path**
```

이름(`edge_2` / `edge_7` / `R13`)과 무관해야 한다.

### 3.3 필수 전제 (없으면 또 클론)

#### A. Topology-agnostic Expert (1순위 작업)

현재 `src/rule_agent.py`는 `center`, `_expected_star_edges` 등 **Star 가정**이 있다.  
이 expert로 GNN을 학습하면 **메시지로 포장된 Star 절차 암기**가 된다.

**요구:** GT를 액션에 쓰지 않으면서, 다음만으로 분기하는 Rule v2.

- 관측/스캔 그래프의 결손·비대칭 엣지  
- node status (DOWN 등)  
- ping success / latency  
- interfaces 등 툴 JSON  

Star 전용 “기대 spoke 목록”에 의존하지 않을 것.  
**천장 검증:** Star / Linear / (가능하면) Mesh 소량에서 Exact 상한을 먼저 측정. Expert가 Mesh에서 무너지면 정책 일반화 실험 전에 expert를 고친다.

#### B. Graph-native 관측

노드 feature 예: status, 최근 ping/status/interfaces 요약, 발견 여부.  
엣지 feature 예: discovered UP/DOWN, latency, link-like evidence.  
글로벌: step, 스캔 여부 등.

Cisco 패키지 수정 없이, 실험 레포에서 **그래프 텐서 빌더**로 구성 (기존 `evidence_obs`와 정합).

#### C. Pointer / object-grounded 액션 (Encoder만 GNN 금지)

출력이 `Discrete(pad)` flat ID이면 pad-MLP와 같은 **ID 정렬 병목**이 남는다.

**목표 액션 인자화 (개념):**

| 카테고리 | 선택 |
|---|---|
| Topology | `scan` 등 (객체 없음) |
| Node tool | `check_status(node)`, … |
| Edge/pair tool | `ping(src, dst)`, … |
| Diagnosis | `diagnose(fault_type, node \| edge)` |

구현은 NetHeal `action_space_manager`의 valid set에 **매핑** (패키지 수정 없음).

**Ablation 필수:** GNN+flat Discrete vs GNN+pointer — flat만 올리면 “GNN 필요” 주장이 약하다.

---

## 4. 실험 설계

### 4.1 가설

**H1 (scale):** 동일 expert 데모로 학습할 때, Graph(+pointer) BC가 pad-MLP BC보다 held-out **Star N**에서 Exact·행동 지표가 낫다.

**H2 (family):** Train에 넣지 않은 **Mesh**(및 필요 시 Hier)에서 Graph 정책이 MLP보다 낫다.  
(절대치가 높아지지 않아도 **상대 개선**이면 H2 지지.)

**H3 (DAgger):** Graph BC만으로 H1/H2가 약할 때, learner 상태 DAgger가 recovery·Exact를 추가로 改善.  
H1이 이미 강하면 DAgger는 선택.

### 4.2 Train / Test 분리 (핵심 원칙)

**모든 topology family를 학습에 넣지 않는다.**

| Split | 내용 | 역할 |
|---|---|---|
| **Train** | Star N ∈ {4,5,6} (균등). 선택적으로 Linear N∈{4,5,6} **소량**만 | 절차 학습 |
| **Seen eval** | Star N=6 (train과 겹칠 수 있음) | 암기·회귀 체크 |
| **Unseen scale** | Star N=8, N=10 (**train 금지**) | H1 |
| **Unseen family** | Mesh (고정 시드 인스턴스 M1/M2/M3 등), 필요 시 Hierarchical | H2 |
| **금지** | Train에 Mesh/Hier/ N≥8 Star 포함 후 “unseen 일반화” 주장 | — |

Linear를 train에 넣을지:

- **Put-in (권장 기본):** Star-only train → Linear+Mesh를 둘 다 unseen family로.  
- **Put-out (확장):** Star+Linear train → Mesh만 held-out (더 어려운 H2).

1차 PoC는 **Star-only train**으로 단순화.

### 4.3 모델 조건

| ID | 모델 | 학습 | 비고 |
|---|---|---|---|
| **B0** | pad-MLP BC-rule | 기존과 동일 계열 | Baseline (이미 N=6≈99%, N=8/10 하락 재현) |
| **B1** | MLP + **동일 v2 expert** 데모 | fair control | Expert 업그레이드만의 효과 분리 |
| **P0** | GNN encoder + **flat** action | BC | 표현만 변경 |
| **P1** | GNN + **pointer** action | BC | Proposed 본체 |
| **P2** | P1 + DAgger | BC 후 aggregation | H1/H2 미흡 시만 |

공통:

- Expert: **Rule v2** (위상 불변), GT 액션 미사용  
- evidence_obs(또는 동등 그래프 feature) ON  
- 데모 수·epoch·seed·eval episode 수 **조건 간 고정**  
- Cisco 원본 보상으로 “학습했다”고 주장하지 않음 (본 실험은 **imitation 본체**)

### 4.4 데이터

| 항목 | 1차 PoC 값 (잠정) |
|---|---|
| Train demos | Star N=4/5/6 각 **700** 성공 ep (총 2100) 또는 동등 transition budget |
| Expert | Rule v2; 성공 trajectory만 (실패는 제외 또는 별도 로그) |
| BC epochs | 40 (기존 BC-rule과 맞춤; 수렴 시 early stop 가능) |
| DAgger (P2) | round 3–5; 매 round learner rollouts에서 expert label; 신규 전이 ≤ train demos의 50% |
| Eval | 조건당 **100 ep**, fault type 균등, deterministic |
| Seeds | policy seed 1 필수; 가능하면 2,3 |

### 4.5 지표

| 지표 | 용도 |
|---|---|
| Exact DSR | 주 점수 |
| Per-fault Exact | Device/Perf/Misconfig/Link |
| Shortcut rate | 조사 없이 진단 |
| Timeout / submit rate | 회피 |
| Avg steps / tool count | 경로 길이 |
| Evidence acquisition / exact_consistent | “찍기” vs 증거 정합 |
| (선택) Probe-type confusion | 같은 증상에서 올바른 툴 계열인지 |

### 4.6 Go / No-Go (1차 PoC)

**사전 조건 (Expert ceiling)**

- Rule v2: Star N=6 Exact ≥ **95%**  
- Rule v2: Star N=8,10 Exact ≥ **90%** (크기만으로 안 죽는지)  
- Rule v2: Mesh 소량(예: 50–100 ep) Exact ≥ **70%** 또는 “실패 모드 문서화”  
  - Mesh 천장 &lt; 50%면 **정책 실험 전에 expert 수정**. 정책 no_go로 몰지 말 것.

**H1 Go (scale)** — P1 vs B1, unseen Star N=8:

- Exact **+10%p 이상** (예: 76% → ≥86%), **또는**  
- Exact +5%p 이상이면서 Device+Perf 합산 exact_n이 유의미히 개선  

동시에 Seen Star N=6 Exact ≥ **90%** (회귀 금지).

**H2 Go (family)** — P1 vs B1, held-out Mesh:

- Exact **+10%p 이상**, 또는 MLP가 붕괴(≤40%)일 때 Graph가 ≥55% 등 **실질적 사용 가능 구간** 진입  

**H3 Go (DAgger)** — P2 vs P1:

- Unseen에서 Exact +3%p 이상 또는 timeout/recovery 명확 개선  

**No-Go 해석**

| 결과 | 해석 | 다음 |
|---|---|---|
| P0≈P1≈B1 on N=8 | 병목이 GNN encoder가 아님 (데모/액션/과제) | pointer·expert 재점검; 아키텍처 과투자 중단 |
| P1≫B1 on N=8, Mesh 동률 | scale inductive bias만; family 전이 실패 | Mesh expert·feature 보강 또는 한계로 보고 |
| P1 Seen만 높고 Unseen 동률 | 또 다른 암기 | 설계 실패 |
| Expert Mesh 천장 낮음 | 원리 expert 미완성 | **정책 학습 금지** |

### 4.7 목표 표 형태 (숫자 예시 ≠ 예측)

보고는 **상대 비교** 중심. 아래는 형식 예시일 뿐 목표 수치가 아니다.

| Model | Star 6 | Star 8 | Star 10 | Unseen Mesh |
|---|---:|---:|---:|---:|
| MLP BC (B1) | … | … | … | … |
| GNN+pointer BC (P1) | … | … | … | … |
| + DAgger (P2) | … | … | … | … |

---

## 5. 단계적 실행 계획 (한 번에 다 하지 않음)

```
Phase 0   개념 freeze (본 문서)
Phase 1   Rule v2 (위상 불변) + ceiling on Star/Linear/Mesh
Phase 2   Graph obs builder + pointer action map (유닛 테스트)
Phase 3   B1 vs P1 BC, Star-only train, eval Seen + Unseen scale
Phase 4   동일 체크포인트로 Unseen Mesh eval (H2)
Phase 5   (조건부) P2 DAgger
Phase 6   (선택) Linear train ablation / Hier eval
Phase 7   결과 → reports/graph_expert_poc.md + summary 갱신
```

### Phase 1 — Expert v2

- [x] Star 특수 분기 제거 → nominal edge skeleton vs discovery  
- [x] Link/Misconfig/Device/Perf 트리를 툴 증거 + missing edges로 재기술  
- [x] Ceiling 리포트: `reports/rule_v2_ceiling.md`  
- [x] **Go:** Star 6/8/10 = 100%/99%/99%; Linear 100%; Mesh 100% (`decision=go`)

코드: `src/rule_agent_v2.py`, `src/run_rule_v2_ceiling.py`

### Phase 2 — 인터페이스

- [x] `graph_obs.py`: node/edge tensors + mask  
- [x] `pointer_actions.py`: 카테고리×객체 → NetHeal action_id  
- [x] 유닛: N=4/6/8에서 동일 의미 액션이 **다른 id**여도 pointer resolve (`ping_ids` 5/7/9, spaces 69/151/265)  
- [x] **Go:** `reports/graph_interface_smoke.md` decision=go

코드: `src/graph_obs.py`, `src/pointer_actions.py`, `src/run_graph_interface_smoke.py`

### Phase 3–4 — 본실험

- [x] B1 (pad-MLP Rule-v2) vs P1 (GNN+pointer) 학습·평가  
- [x] 표: Seen / N=8 / N=10 / Mesh — `reports/graph_expert_poc.md`  
- [x] Go/No-Go: **no_go_seen** (P1 Star6=74% <90%; Star8 Δ=−6%p; Mesh=14%)  
- [ ] P0 GNN+flat — 미실시 (P1 Seen 미달로 후순위)

코드: `src/graph_policy.py`, `src/run_graph_expert_bc.py`  
ckpt: `models/checkpoints/bc_rule_v2_pad12_b1.zip`, `graph_pointer_p1.pt`

### Phase 5 — DAgger

- [ ] Phase 3에서 H1 **No-Go이거나** timeout/recovery 이슈가 보일 때만  
- [ ] Expert는 계속 v2; learner env와 동일 그래프 표현  

### 하지 않을 것 (본 설계 범위 밖)

- 또 다른 원본보상 MaskablePPO 숏컷 깨기 PoC  
- Stage B no_go 이후 E4 curriculum 재개  
- Train에 Mesh를 넣고 Unseen Mesh 주장  
- “GNN-PPO가 troubleshooting을 발견” 문구  
- Expert v2 없이 GNN BC만 구현  

---

## 6. 기존 결과와의 관계

| 기존 산출 | 본 설계에서의 역할 |
|---|---|
| `bc_rule_poc` / `star_scale_poc` | B0 숫자·경로 클론 근거 |
| `rule_ceiling_poc` | Star 상한; v2로 교체 대상 |
| `star_mix456_poc` | “다중 N 암기 ≠ 방법” 근거 |
| Obs/gate/E1–E3 | RL 인터페이스 한계 배경 (본선 아님) |
| Full BC+gate PPO | imitation 대비 RL fine-tune 열등 사례 |

본 실험은 **감사 다음 장**이지, PPO 시리즈의 연장전이 아니다.

---

## 7. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| Expert가 사실상 Star | Phase 1 Mesh ceiling 강제 |
| GNN+flat만 구현 | P0/P1 ablation 필수 |
| 평가 N·seed 불일치 | protocol.yaml에 split 고정 |
| DAgger를 너무 이르게 | Phase 5 조건부 |
| 패키지 수정 유혹 | 래퍼·매핑만; NetHeal 미수정 |
| 과대 수치 기대 | Go는 **상대 + 회귀 금지** |

---

## 8. 산출물 체크리스트

| 파일 (예정) | 내용 |
|---|---|
| `GNN_expert_guided_spec.md` (본 문서) | 개념·설계 |
| `reports/rule_v2_ceiling.md` | Expert 상한 |
| `reports/graph_expert_poc.md` | B1/P0/P1/(P2) 결과 |
| `results/graph_expert_poc.json` | 수치 |
| `NetHeal_RL_results_summary.md` | 완료 시 한 절 추가 |

구현 모듈 (예정 스케치):

- `src/rule_agent_v2.py` (또는 `rule_agent` 일반화)  
- `src/graph_obs.py`  
- `src/pointer_actions.py`  
- `src/run_rule_v2_ceiling.py`  
- `src/run_graph_expert_bc.py`  
- `src/run_graph_expert_dagger.py` (Phase 5)

---

## 9. 결정 실행 순서 (지금 당장)

1. **Phase 1만** 착수: Rule v2 설계·천장 측정.  
2. 천장 Go 전에는 GNN 학습 코드에 공 들이지 않는다.  
3. Phase 1 결과에 따라 Mesh 포함 H2 난이도·expert 수정을 재평가한다.

---

*이 문서는 실험 설계다. 구현·수치 확정 전 “GNN이 일반화를 해결한다”고 단정하지 말 것.*
