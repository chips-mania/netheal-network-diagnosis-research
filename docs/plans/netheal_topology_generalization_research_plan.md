# 미관측 네트워크(Mesh)로의 RL 기반 AI 장애진단 정책 Cross-Network Deployment 분석
## Cisco NetHeal AI Agent Benchmark 기반 Controlled Generalization Study

> **설계 동결 (2026-09)** — 추가 설계 논의 중단. **Step 1~8 → PPO Sanity Go/No-Go** 먼저 실행.

---

# Executive Summary — 최종 합의 · 구현 우선순위 · Go/No-Go

## 한 줄

Known network(Star/Linear)에서 학습한 RL diagnostic policy를 New network(Mesh)에 **재학습 없이 배포**했을 때, degradation이 **진단 실패(DSR↓)** 인지 **탐색 비효율(TTD-success↑, Cost↑)** 인지 분해·분석한다.

## Primary 실험 (고정)

| 항목 | 값 |
|---|---|
| Train | `topology_types=["star","linear"]`, 6 devices, Hint OFF, PPO MultiInputPolicy |
| Seen | Star / Linear 각각 별도 run |
| Unseen | **Mesh만** — M1, M2, M3 instance |
| 제외 | Hierarchical, random (Primary) |
| Step budget | Train/Eval/Baseline **동일** — 우선 `max_episode_steps=15` |
| Training seeds | **3개 필수** (N=1로 Primary 결론 금지) |
| Baseline | Random-Valid (`get_valid_actions()`) |

## 세 종류 Randomness (혼용 금지)

| 종류 | 역할 | Primary |
|---|---|---|
| **Policy / Training seed** | PPO 학습 확률 변동 | Seed 1, 2, 3 |
| **Topology instance seed** | 동일 Mesh family 내 graph 차이 | M1, M2, M3 |
| **Fault / Episode seed** | 동일 장애 eval suite | Fault Suite A/B/C per instance |

세 policy 모두 **동일 M1/M2/M3 + 동일 fault suite**에서 평가.

## Fault eval (Unseen peek 전 고정)

- 4 fault types × **20~30 ep/instance** × 3 Mesh instances ≈ **240~360 ep/policy**
- 3 training seeds → **720~1,080** unseen inference (runtime 측정 후 N 고정, **Unseen 결과 본 뒤 변경 금지**)

## Metrics (항상 함께)

1. **DSR** — `diagnosis_success_rate`
2. **Avg Steps-all** — `avg_steps`
3. **TTD-success** — success episode만 custom aggregate
4. Tool Cost, Coverage, Macro F1

※ TTD-success 단독 금지 (survivorship bias). Fault type별도 동일 주의.

## 통계 (2~3일 PoC)

- p-value 중심 ❌
- **Mean ± SD** (3 training seeds), **Effect size**, DSR Gap / Relative TTD·Cost increase ✅

## 구현 우선순위

```text
Step 1~2   Repo + 구조 파악
Step 3~4   6-device obs + Star/Linear/Mesh(M1/M2/M3) sanity
Step 5~6   Fault sanity + Runtime benchmark → protocol 고정
Step 7~8   Random-Valid baseline
Step 9     ★ PPO Sanity Go/No-Go (Star only, seed 1)
           GO → Step 10~17  |  NO-GO → PPO/환경 재검토, Generalization 금지
Step 10~11 Main train ×3 seeds + Seen eval
Step 12~17 Unseen M1/M2/M3 + aggregation + README
Step 18+   Secondary (Primary 완료 후만)
```

## 첫 Go/No-Go (Step 9)

**6-device Star, 15 steps, seed 1** — PPO가 Random-Valid 대비 실제 학습하는가?

| GO | NO-GO |
|---|---|
| Reward↑, DSR↑, Random-Valid 대비 의미 있는 성능 | Random 수준 → Generalization 실험 **진행 금지** |

## 최소 산출물

- **Table 1**: Seen vs Unseen (mean ± SD, 3 seeds)
- **Table 2**: Mesh M1 / M2 / M3 instance별
- **Table 3**: Fault breakdown (DSR + TTD-success)
- **Figure 1~4**: DSR, TTD/Cost, Fault gap, Mesh instance

## 결론 범위 (금지 vs 권장)

| 금지 | 권장 |
|---|---|
| “다른 네트워크에서도 generalize한다” | “NetHeal 동일 6-device 조건에서 Star/Linear 학습 PPO를 Mesh에 적용했을 때…” |
| Cisco 실망 검증 / 실망 구축 / NetHeal 초월 모델 | Simulation benchmark controlled experiment |

---

> **핵심 질문**  
> 비교적 단순한 네트워크 구조(Star/Linear)에서 학습한 AI 장애진단 정책을, **별도 재학습 없이** 구조가 다른 신규/별도 네트워크(Mesh, 동일 6-device)에 배포했을 때 진단 성공률(DSR)과 진단 효율(TTD-success, Tool Cost)을 어느 수준까지 유지할 수 있는가?

---

# 0. 연구 한 줄

본 연구는 Cisco의 `netheal-ai-agent-benchmark`를 단순 실행하는 데 목적이 있지 않다.

**Cisco NetHeal simulation benchmark 위에서, Known network(Star/Linear)에서 학습한 RL diagnostic policy를 New network(Mesh)에 재학습 없이 이식·배포했을 때 generalization 성능(DSR, TTD-success, Tool Cost, Coverage)을 분해·분석하는 PoC**다.

---

# 1. 연구 배경

AI 기반 네트워크 운용은 단순한 이상탐지를 넘어 장애 원인 탐색과 troubleshooting 자동화까지 확장되고 있다.

네트워크 장애진단 agent는 다음과 같은 진단 행동을 반복하면서 장애 원인을 추론한다.

```text
Topology / Device 상태 관찰
        ↓
scan / neighbor discovery
        ↓
ping / traceroute
        ↓
device/interface 상태 확인
        ↓
추가 진단
        ↓
최종 장애 원인 판단
```

그러나 실제 네트워크 운용에서는 **하나의 망이 textbook topology로 통째로 바뀌기**보다, **지역·사이트·신규 망마다 연결 구조가 다르고**, 중앙 NOC는 **한 AI 정책을 여러 망에 확장 배포**하려는 경우가 많다.

예를 들어 동일한 link failure라도:

- **Site A**(단순 chain): ping 실패 구간을 비교적 빠르게 좁힐 수 있음
- **Site B**(redundant path): ping은 성공할 수 있으나 alternate path·neighbor·interface evidence를 더 조합해야 함

즉 장애 유형은 같아도 **topology에 따라 evidence 탐색 방식**이 달라진다.

따라서 특정 망(들)에서 학습된 diagnostic policy를 **구조가 다른 별도 네트워크에 재학습 없이 적용**했을 때도 동일한 수준으로 troubleshooting이 가능한지는 별도 검증이 필요하다.

본 연구는 여기서 다음 질문을 제기한다.

> **“Known network에서 학습한 장애진단 AI를, 구조가 다른 New network에 그대로 배포할 수 있는가? — 그리고 성능이 떨어진다면 ‘못 찾는 것’인가, ‘찾지만 더 많이 탐색하는 것’인가?”**

본 Primary는 **망 구조의 점진적 evolution**(증설·이중화로 기존 망이 복잡해짐)이나 **장애/우회로 인한 active topology 변화**를 직접 모델링하지 않는다. 이는 별도 연구 축으로 둔다.

---

# 2. 연구 플랫폼

## 2.1 Cisco NetHeal AI Agent Benchmark

실험 플랫폼:

```text
cisco-ai-platform/netheal-ai-agent-benchmark
```

NetHeal은 AI agent의 네트워크 troubleshooting 능력을 **학습·평가**하기 위한 simulation benchmark다.

| | |
|---|---|
| **NetHeal 기본 목적** | AI Agent Network Troubleshooting 학습·평가 |
| **본 프로젝트 추가 질문** | Training topology vs unseen topology 분리 시 Diagnostic Policy **Generalization** |

권장 표현:

> Cisco Network Troubleshooting Simulation Benchmark인 NetHeal을 활용하여, 학습 시 경험하지 않은 topology에 RL diagnostic policy를 배포했을 때 generalization 성능을 통제 실험으로 분석하였다.

금지:

- 실제 Cisco 망에서 검증
- 실제 통신망 구축
- Cisco NetHeal보다 향상된 모델 개발
- 새로운 네트워크 AI 알고리즘 개발

본 연구에서 활용하는 주요 기능은 다음과 같다.

- 다양한 topology 생성
  - Linear
  - Star
  - Mesh
  - Hierarchical *(Primary 제외 — §7 참고)*
- 다양한 fault injection
  - Device failure
  - Link failure
  - Performance degradation
  - Misconfiguration
- Diagnostic action space
- Episode 기반 reward
- PPO 기반 RL 학습 예제
- Network troubleshooting 평가 metric

중요:

> 본 실험은 Containerlab/FRR처럼 실제 packet forwarding을 수행하는 가상 네트워크 구축 실험이 아니다.

따라서 결과를 설명할 때는 다음 표현을 사용한다.

> **“Cisco가 공개한 network troubleshooting simulation benchmark에서 controlled generalization experiment를 수행했다.”**

다음 표현은 사용하지 않는다.

> “실제 통신망을 구축했다.”

> “Cisco 실제망에서 검증했다.”

---

# 3. 연구 목적

본 연구의 목적은 크게 세 가지다.

## 목적 1. Cross-Network Deployment 성능 차이 측정

Known network(학습·seen)와 New network(unseen)에 **동일 policy를 배포**했을 때 diagnostic performance가 얼마나 달라지는지 측정한다.

## 목적 2. Generalization Failure 형태 분해

단순히 “Unseen에서 성능이 떨어지는가?”가 아니라:

- ① **장애를 아예 못 찾는가?** (DSR↓)
- ② **장애는 찾지만 더 많이 탐색·더 오래 걸리는가?** (TTD-success↑, Cost↑)

를 DSR + TTD-success + Avg Steps-all로 분리한다.

## 목적 3. Fault Type별 일반화 차이 분석

동일한 topology shift에서도:

- Link failure
- Device failure
- Misconfiguration
- Performance degradation

의 generalization 난이도가 동일하지 않을 수 있다.

Fault-type breakdown을 통해 어떤 장애가 topology 변화에 더 민감한지 확인한다.

---

# 4. 핵심 연구 질문

## RQ1. Topology Generalization

> **Known network(Star/Linear)에서 학습한 RL diagnostic policy를, 재학습 없이 New network(Mesh)에 배포했을 때 장애진단 성공률을 유지할 수 있는가?**

## RQ2. Diagnostic Efficiency

> **New network(Mesh) 배포 시 DSR 하락과 별도로, 성공 episode의 TTD-success 및 Tool Cost가 Known network(Seen) 대비 더 크게 악화되는가?**

즉 다음 두 failure mode를 구분한다.

```text
Failure Mode A
→ 아예 정답을 찾지 못함

Failure Mode B
→ 정답은 찾지만 훨씬 많이 탐색함
```

## RQ3. Fault-Type Sensitivity

> **Topology generalization 성능 저하는 fault type에 따라 다르게 나타나는가?**

## Secondary RQ (Primary 완료 후)

> **Fixed-Compute Training Strategy**: 동일 timestep budget에서 Star-only vs Star+Linear 중 어느 쪽이 New network(Mesh) 이식에 더 robust한가?

※ “순수 topology diversity의 인과효과”로 주장하지 않는다 (Model B는 Star 경험량이 적음).

---

# 5. 연구 가설

## H1. Seen > Unseen

Seen topology에서의 DSR이 Unseen topology보다 높을 것이다.

## H2. Efficiency Degradation

Unseen Mesh에서는 DSR 하락보다:

- TTD-success 증가 (성공 episode에서만 평균 step)
- Tool Cost 증가
- Topology Coverage 증가

가 더 크게 나타날 수 있다.

※ `Avg Steps-all`(성공/실패 전체 평균)은 실패 episode가 max step까지 진행할 때 DSR 하락과 함께 증가하므로, H2 해석에는 **TTD-success**를 우선 사용한다.

즉 agent가 완전히 실패하기보다 **더 많이 탐색하면서 정답을 찾는 형태**의 generalization degradation이 나타날 수 있다.

## H3. Fault-Type Difference

Link failure와 같이 path 탐색으로 비교적 명확하게 좁힐 수 있는 장애는 unseen topology에서도 generalization이 상대적으로 잘 될 수 있다.

반면 Misconfiguration 또는 Performance degradation처럼 여러 evidence를 조합해야 하는 fault는 topology 변화에 더 민감할 수 있다.

※ 이는 사전 가설이며 결과에 맞추어 해석을 변경하지 않는다.

---

# 6. Operational Persona

## 6.1 가상 페르소나

**김도윤 / 통신사 네트워크 운용 엔지니어 (중앙 NOC)**

> 특정 기업의 내부 프로세스를 재현한 인물이 아니라, 통신사 NW 인프라운용 업무를 바탕으로 구성한 가상 Operational Persona다.

### 상황

통신사는 **여러 지역·사이트 네트워크**를 중앙 NOC에서 함께 지원한다. AI 기반 장애진단 policy는 **비교적 단순한 topology**를 중심으로 학습해 일부 legacy/지역망 운영에 활용 중이다.

이후 **신규 지역망**이 투입되거나, 다른 사이트를 같은 AI로 지원하려 할 때:

- 장애 유형(link/device/performance/misconfig)은 동일할 수 있고
- ping / traceroute / status check 등 **진단 tool은 동일**하지만
- **연결 구조(redundancy, alternate path)**는 다를 수 있으며
- **망마다 AI를 처음부터 재학습**하는 것은 비용이 크다

운용자는 다음을 확인하고 싶다.

> **“Known network에서 학습한 장애진단 AI를, 구조가 다른 New network에 재학습 없이 배포해도 되는가?”**

단순 정답률뿐 아니라:

> **“정답은 맞히지만 진단 step·tool 호출·탐색 범위가 과도하게 늘어나지는 않는가?”**

도 운영 의사결정에 중요하다.

본 연구는 **cross-network diagnostic policy deployment**에 필요한 정량적 근거를 만드는 것을 목표로 한다.

## 6.2 현업 시나리오와 실험 매핑

| 현업 상황 | 예시 | Primary 실험과의 관계 |
|---|---|---|
| **A에서 쓰던 AI를 B에 적용** | 지역/사이트마다 연결 구조가 다름 | **Primary 핵심** — Star/Linear 학습 → Mesh 배포 |
| A 네트워크가 시간이 지나 B로 변경 | 증설, 이중화, 신규 링크 추가 | 관련 있으나 보통 **점진적** — Primary 범위 밖 (Future work) |
| 장애/우회로 active topology 변화 | 링크 장애로 실제 사용 경로 변경 | **별도 연구** — topology family generalization과 다름 |

NetHeal textbook topology 이름과 현업 해석:

| NetHeal | 현업 해석 |
|---|---|
| Star / Linear (Train + Seen) | Known network — 비교적 단순한 legacy/지역망에서 학습·검증 |
| Mesh (Unseen) | New network — redundancy·alternate path가 많은 **별도/신규 망** |
| 6 devices 고정 | 규모 차이가 아닌 **연결 구조 차이**만 비교 |

### 실험 흐름 (현업 관점)

```text
Known Networks (Star / Linear)
        ↓
  AI diagnostic policy 학습
        ↓
   재학습 없이 배포
        ↓
New Network (Mesh)
        ↓
  DSR / TTD-success / Tool Cost / Coverage ?
```

Primary 질문은 **“Mesh에서 떨어지냐?”**만이 아니다.

- **얼마나** degradation이 발생하는가?
- **Failure Mode A**(못 찾음) vs **Failure Mode B**(찾지만 더 많이 탐색) 중 어느 형태인가?

Secondary(Star-only vs Star+Linear → Mesh)는 같은 서사로 이어진다.

> **배포 전 multi-topology 학습이 New network 이식 성능을 실제로 개선하는가?**

## 6.3 Primary에서 주장하지 않는 것

- “Star 망이 어느 날 Mesh로 **교체**되었다”
- 실제 통신망의 **topology evolution**을 그대로 재현했다
- **Active path / 우회 routing** 변화를 모델링했다

위는 simulation benchmark 한계와 연구 범위상 **주장하지 않는다**.

---

# 7. 실험 범위

## Primary Experiment

### Train Topology

- Star
- Linear

### Seen Evaluation

- Star
- Linear

단:

- 새로운 random seed
- 새로운 fault location
- 새로운 episode

를 사용한다.

### Unseen Evaluation

- Mesh

학습 과정에는 절대 포함하지 않는다.

### Primary 설계 요약

```text
                TRAIN
        Star + Linear
          6 devices
        Hint OFF
      max_steps = 15
       PPO MultiInput
             │
             ▼
     ┌────── SEEN ──────┐
     │                   │
   Star                Linear
  6 devices            6 devices
     │                   │
     └────────┬──────────┘
              │
             VS
           UNSEEN
             Mesh
          6 devices
```

### 현업 시나리오 해석 (Cross-Network Deployment)

본 실험은 **“A topology가 B topology로 변했다”**가 아니라:

```text
Known network들(Star / Linear)에서 학습
        ↓
New network(Mesh)에 동일 diagnostic policy 배포 (재학습 없음)
```

으로 해석한다. 즉 **다른 구조를 가진 신규/별도 네트워크로 AI 진단 정책을 확장 배포**하는 상황을 모델링한다.

### Hierarchical을 Primary에서 제외한 이유

NetHeal 기본 `hierarchical` generator는 `min_devices`/`max_devices`를 따르지 않고 layer 수·layer별 device 수를 별도 랜덤 생성한다. 따라서 `min_devices=max_devices=6`이어도 실제 hierarchical network가 6대로 고정되지 않는다.

게다가 observation은 `max_devices × max_devices` 크기로 고정되고 `scan_network`도 최대 `max_devices`까지만 발견한다. hierarchical이 6대를 넘으면 topology scale confound를 넘어 observation capacity 밖 topology가 잘릴 수 있다. reset마다 action space도 rebuild되므로 PPO 조건 자체가 흔들릴 수 있다.

**Primary에서는 Hierarchical을 제외**하고, fixed-size hierarchical generator(예: `devices_per_layer=[2,2,2]`) 패치 후 평가는 **Secondary / Future work**로 둔다.

> NetHeal 기본 hierarchical generator가 min/max_devices를 따르지 않아 topology와 scale을 분리하기 어렵기 때문에 Primary에서 제외하였다. Fixed-size hierarchical generator를 구성한 평가는 후속 실험으로 둔다.

---

# 8. 가장 중요한 통제 조건

Topology generalization만 측정하기 위해 다음 조건을 Train/Test에서 동일하게 유지한다.

## 8.1 Device Count

Primary에서는 장비 수를 **정확히 하나로 고정**한다.

권장:

```text
min_devices = 6
max_devices = 6
```

이유:

```text
Star 4 devices
vs
Mesh 8 devices
```

를 비교하면 topology shift와 scale shift가 동시에 발생한다.

Primary에서는 이를 금지한다.

## 8.2 Fault Type

Train/Seen/Unseen에서 동일한 fault category set을 사용한다.

예:

```text
device failure
link failure
performance degradation
misconfiguration
```

## 8.3 Evaluation Episode Count (Fault-Type Stratified + Mesh Instances)

Uniform random fault sampling은 seed에 따라 fault type별 episode 수가 불균형해질 수 있다. Primary eval은 **fault type별 고정 episode 수** + **Mesh instance 3개(M1/M2/M3)** 를 사용한다.

### Mesh Topology Instances

같은 Mesh family라도 graph instance가 다를 수 있다. Day 1 topology sanity에서 **M1, M2, M3** seed를 고정하고 다음을 기록한다.

- edge count, connectivity, node degree, redundancy/path 구조

세 PPO policy(Training seed 1/2/3) **모두 동일 M1/M2/M3**에서 평가한다.

### Episode 수 (Unseen peek **전** runtime 측정 후 고정)

```text
각 fault type × N episodes × 3 Mesh instances
= Unseen episodes per policy

예: 4 fault × 30 ep × 3 instances = 360 ep/policy
Training seed 3개 → 1,080 unseen inference total

부담 시 N=20 → 240 ep/policy (Unseen 결과 본 뒤 변경 금지)
```

Seen(Star/Linear)도 **동일 fault-type별 N**과 protocol을 사용한다.

Train/Seen/Unseen에서 **동일 fault category set**과 **동일 fault-type별 episode 수**를 유지한다.

## 8.4 Topology Types (Run별 단일 topology, random 제외)

NetHeal 기본 `topology_types`는 `linear, star, mesh, hierarchical, random` 전부를 포함한다. topology를 명시하지 않으면 train contamination이 생긴다.

**각 run마다 topology 하나만** 지정한다. `random`은 Train/Eval 모두에서 **절대 포함하지 않는다**.

```python
# Train
topology_types=["star", "linear"]

# Seen eval 1
topology_types=["star"]

# Seen eval 2
topology_types=["linear"]

# Unseen eval
topology_types=["mesh"]

# Primary에서 random, hierarchical 절대 포함 금지
```

## 8.5 Episode Step Budget

Environment 기본값은 `max_episode_steps=100`이지만, Cisco README SB3 예제는 **15**를 사용한다.

Primary 초기값 (Train / Seen / Unseen / Baseline **전부 동일**):

```text
max_episode_steps = 15
```

Sanity training에서 PPO가 15 step budget으로 정상 학습·진단하지 못하면, **Unseen eval 전에** (test peek 없이) **25** 등으로 올리고 **Train·Eval·Baseline protocol 전체를 재고정·재학습**한다.

중요한 것은 숫자 자체보다 **모든 조건이 동일 episode budget**을 갖는 것이다.

## 8.6 Evaluation Seed (Fault / Episode Seed)

Evaluation fault·episode seed는 고정하고 artifact로 저장한다.

- Train seed(Policy seed)와 Eval seed(Fault seed)는 **분리**
- Policy seed 1/2/3 모두 **동일 fault suite** (M1+Suite A, M2+Suite B, M3+Suite C) 평가

## 8.7 세 종류 Randomness (혼용 금지)

| # | 이름 | 역할 | Primary 값 |
|---|---|---|---|
| 8.7.1 | **Policy / Training seed** | PPO 학습 확률 변동 | **3개 필수** (Optional 아님) |
| 8.7.2 | **Topology instance seed** | Mesh family 내 graph 차이 | **M1, M2, M3** |
| 8.7.3 | **Fault / Episode seed** | eval episode suite 재현 | Suite A/B/C (instance별) |

**N=1 training seed만으로 “Seen > Unseen” Primary 결론을 내리지 않는다.**

일정상 Day 1은 **Policy seed 1**으로 Go/No-Go → 이후 seed 2/3 main train.

---

# 9. User Hint 설정

Primary Experiment에서는:

```text
enable_user_hints = False
```

를 우선 사용한다.

이유:

Topology generalization 자체를 분석할 때 additional hint가 diagnostic difficulty를 완화하는 confound가 될 수 있기 때문이다.

Agent는 가능한 범위 내에서:

- observation
- discovery result
- diagnostic history
- tool response

를 바탕으로 진단한다.

시간이 남을 경우 Secondary Ablation으로:

```text
Hint OFF
vs
Hint ON
```

을 비교할 수 있으나 Primary 범위에는 포함하지 않는다.

---

# 10. Agent

## 10.1 RL Agent

Primary learning agent:

```text
PPO (Stable Baselines3)
MultiInputPolicy   # dict observation 필수
```

NetHeal observation은 dictionary 구조이며, Cisco README 공식 예제도 다음을 사용한다.

```python
from stable_baselines3 import PPO

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Evaluation
action, _ = model.predict(obs, deterministic=True)
```

평가 시 **checkpoint 로드 + `deterministic=True`**를 사용한다.

본 연구의 목적은 PPO algorithm 자체를 개선하는 것이 아니다.

따라서:

- PPO vs DQN
- PPO vs SAC
- custom deep model
- hyperparameter tournament

는 수행하지 않는다.

---

# 11. Baseline

PPO만 평가하면 unseen topology 성능 저하가 단순 RL 불안정성 때문이라는 반론이 가능하다.

따라서 Gymnasium path에서 **Random-Valid Agent** baseline을 둔다.

## Baseline — Random-Valid Agent

`examples/basic_usage.py`의 `env.action_space.sample()`은 **invalid action**을 뽑을 수 있고, environment는 invalid action에 penalty를 준다. 공정 비교를 위해 **`get_valid_actions()`**만 사용한다.

```python
valid_actions = env.get_valid_actions()
action = random.choice(valid_actions)
obs, reward, terminated, truncated, info = env.step(action)
```

- Primary는 **Gym environment + `MetricsCollectorWrapper`** path에서 직접 구현 (~20줄)
- MCP/AAA용 `netheal/aaa/dummy_agent.py`는 Primary baseline으로 사용하지 않음 (경로·protocol 상이)

역할:

> PPO가 invalid action 회피만 배워 Random보다 좋아진 것이 아니라, 실제 troubleshooting policy로 의미 있는 성능 차이를 보이는지 확인.

## Optional — Scripted Baseline

시간이 남을 경우에만 구현한다.

예:

```text
1. topology discovery
2. neighbor 확인
3. suspicious path에 ping
4. traceroute
5. interface/device status 확인
6. diagnosis
```

단, scripted baseline 개발 때문에 PPO Seen/Unseen 평가가 늦어지면 생략한다.

Primary 성공 조건은:

```text
Random-Valid baseline
vs
PPO (MultiInputPolicy)
+
Seen / Unseen controlled evaluation (Star / Linear / Mesh)
```

이다.

---

# 12. Training Configuration

초기 권장 설정:

```text
Training Topologies:
  topology_types=["star", "linear"]

Devices:
  min_devices=6, max_devices=6

Hints:
  enable_user_hints=False

Episode budget:
  max_episode_steps=15   # Train / Eval / Baseline 동일

Algorithm:
  PPO + MultiInputPolicy

Training Timesteps:
  50k~100k 후보 (sanity curve 보고 Unseen test 전 고정)

Training Seeds:
  3 (각 seed별 checkpoint 저장)
```

100k를 무조건 고정하지 않는다. 먼저 학습 curve를 확인한다.

---

# 13. PPO Go / No-Go (전체 연구보다 먼저)

PPO가 학습되지 않은 상태에서 generalization을 평가하면 연구 질문 자체가 무의미하다. **Seen에서도 학습 안 된 policy의 Unseen 비교는 금지.**

## Stage 1. Sanity Training — ★ 첫 Go/No-Go (Step 9)

```text
Star only
topology_types=["star"]
6 devices
max_episode_steps=15
Hint OFF
PPO MultiInputPolicy
Policy seed 1 (짧은 budget)
```

확인:

- reward 증가
- DSR 증가
- episode length / diagnostic behavior 개선
- **Random-Valid baseline 대비 의미 있는 성능**

| GO | NO-GO |
|---|---|
| PPO가 실제 diagnostic policy 학습 | Random 수준 → **Generalization 실험 진행 금지** |

NO-GO 시 먼저 점검: `max_episode_steps`, training timesteps, reward, observation/action 설정.

## Stage 2. Main Training (GO 이후만)

```text
Star + Linear
topology_types=["star", "linear"]
6 devices, max_episode_steps=15
PPO MultiInputPolicy
Training seeds: 1, 2, 3
50k~100k timesteps (Unseen test 전 고정)
```

각 seed별 checkpoint 저장 → Seen/Unseen eval.

---

# 14. Evaluation Protocol

각 eval run은 **topology 하나만** 포함한다 (`topology_types=[단일]`). `random` topology는 사용하지 않는다.

평가 시 동일 `max_episode_steps`, fault-type stratified suite, `enable_user_hints=False`를 유지한다.

## Seen

### Star

```text
topology_types=["star"]
Train에 사용된 topology family
BUT
new seed
new fault location
new episode (fault-type별 고정 수)
```

### Linear

```text
topology_types=["linear"]
동일 protocol
```

## Unseen — Mesh M1 / M2 / M3

```text
topology_types=["mesh"]
Train에서 Mesh 절대 포함 금지
6 devices, 동일 max_episode_steps, 동일 fault suite
Instance M1, M2, M3 각각 eval (topology instance seed 고정)
```

모든 PPO eval: `model.predict(obs, deterministic=True)` + 동일 `MetricsCollectorWrapper`.

---

# 15. Primary Metrics

NetHeal `CompetitionEvaluator` summary key와 본 연구 명칭 매핑:

| 본 연구 명칭 | NetHeal key | 설명 |
|---|---|---|
| DSR | `diagnosis_success_rate` | 전체 episode 중 진단 성공률 |
| Macro F1 | `fault_type_macro_f1` | fault class macro F1 |
| Avg Steps-all | `avg_steps` | 성공/실패 전체 평균 step |
| Normalized Steps-all | `normalized_steps` | `steps / max_episode_steps` 전체 평균 |
| Tool Cost | `tool_cost_index` | 정규화 tool cost |
| Topology Coverage | `topology_coverage` | topology 탐색 비율 |

## 15.1 DSR (Diagnosis Success Rate)

> Agent가 최종 장애진단에 성공했는가?

가장 기본적인 정확도 metric. NetHeal: `diagnosis_success_rate`.

## 15.2 TTD-success (성공 episode만)

> **진단에 성공한 episode**에서 평균 몇 step이 필요했는가?

NetHeal summary에 기본 포함되지 않으므로, `MetricsCollectorWrapper` episode trace에서 `diagnosis_success == True`인 episode만 필터해 `avg_steps`를 **custom aggregate**한다.

예: DSR 90→85, **TTD-success** 5→9 이면 “정답 능력은 비교적 유지, 성공 시 더 많이 헤맴”으로 해석 가능.

## 15.3 Avg Steps-all (전체 episode)

> 성공/실패 포함 전체 episode 평균 step.

NetHeal: `avg_steps`, `normalized_steps`.

실패 agent가 max step까지 진행하면 DSR↓와 Steps-all↑가 **동시에** 발생하므로, H2(효율 악화) 해석에는 TTD-success를 우선 사용하고 Steps-all은 보조로 본다.

## 15.4 Tool Cost

> 장애진단 과정 diagnostic action 비용.

NetHeal: `tool_cost_index`. 낮을수록 효율적.

## 15.5 Topology Coverage

> 장애진단 과정에서 topology를 얼마나 탐색했는가?

NetHeal: `topology_coverage`. Unseen Mesh에서 탐색 부담 증가 여부 확인.

## 15.6 Macro F1

Fault class별 불균형 보조 metric. NetHeal: `fault_type_macro_f1`.

## 15.7 TTD Survivorship Bias 대응

**TTD-success만 단독 사용 금지.**

Unseen에서 어려운 episode가 전부 실패하면, 성공한 **쉬운 episode만** 남아 TTD-success가 오히려 **짧아지는** survivorship bias가 발생할 수 있다.

**항상 함께 보고:**

1. DSR
2. Avg Steps-all
3. TTD-success

Fault type별 breakdown도 survivorship bias를 **완전 제거하지는 못하며 완화** 수준임을 명시한다.

---

# 16. Fault-Type Breakdown

전체 평균만 보고 끝내지 않는다.

최소:

```text
Link Failure
Device Failure
Performance Degradation
Misconfiguration
```

별로 결과를 나눈다.

예:

| Fault Type | Seen DSR | Unseen DSR | Seen TTD-success | Unseen TTD-success | ΔCost |
|---|---:|---:|---:|---:|---:|
| Link | ? | ? | ? | ? | ? |
| Device | ? | ? | ? | ? | ? |
| Performance | ? | ? | ? | ? | ? |
| Misconfiguration | ? | ? | ? | ? | ? |

---

# 17. 핵심 결과표 (최소 산출물)

## Table 1 — Seen vs Unseen

모든 값: **mean ± SD across 3 PPO training seeds**

| Condition | DSR ↑ | Avg Steps-all ↓ | TTD-success ↓ | Tool Cost ↓ | Coverage ↓ |
|---|---:|---:|---:|---:|---:|
| Seen | ? ± ? | ? ± ? | ? ± ? | ? ± ? | ? ± ? |
| Unseen | ? ± ? | ? ± ? | ? ± ? | ? ± ? | ? ± ? |

## Table 2 — Mesh Instance별

±SD = **3 training seeds 간 변동**

| Topology | Instance | DSR | TTD-success | Tool Cost |
|---|---|---:|---:|---:|
| Mesh | M1 | ? ± ? | ? | ? |
| Mesh | M2 | ? ± ? | ? | ? |
| Mesh | M3 | ? ± ? | ? | ? |

Instance 간 격차(M3만 급락 등)는 **특정 graph instance 난이도** 추가 분석. ANOVA/mixed-effects까지는 확장하지 않음.

## Table 3 — Per-Topology (보조)

| Topology | Seen/Unseen | DSR ↑ | Macro F1 ↑ | TTD-success ↓ | Avg Steps-all ↓ | Tool Cost ↓ | Coverage ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Star | Seen | ? | ? | ? | ? | ? | ? |
| Linear | Seen | ? | ? | ? | ? | ? | ? |
| Mesh | Unseen | ? | ? | ? | ? | ? | ? |

Random-Valid baseline도 동일 표 형식으로 병렬 기록.

## Figures (최소 4)

- **Figure 1**: Seen vs Unseen DSR
- **Figure 2**: Seen vs Unseen TTD-success / Tool Cost
- **Figure 3**: Fault-type Generalization Gap
- **Figure 4**: Mesh Instance별 성능 (M1/M2/M3)

---

# 18. 핵심 분석

연구 핵심은 **“차이가 있는가?”**보다 **“차이가 어떤 형태인가?”**

- p-value 중심 ❌
- **Mean ± SD + Effect Size** ✅ (Training seed 3개 PoC)

### Effect Size

```text
DSR Gap = Seen Mean DSR - Unseen Mean DSR

Relative TTD-success Increase
= (Unseen - Seen) / Seen

Relative Tool Cost Increase — 동일
```

### 결과 문장 예시

> 미관측 Mesh에서 DSR은 평균 4.8%p 감소, TTD-success 61% 증가, Tool Cost 74% 증가 → **탐색 비용 degradation**.

### Case A — Policy Generalization Failure

DSR 크게 하락 + TTD/Cost 증가 → Training topology에 **부분적 과적합**.

### Case B — Exploration Efficiency Failure

DSR 비슷 + TTD-success/Cost/Coverage 크게 증가 → **진단은 되나 탐색 비용 증가**.

> Generalization 문제는 진단 실패뿐 아니라 **탐색 비효율**로도 나타날 수 있다.

### Case C — Robust Generalization

DSR / TTD-success / Cost 모두 비슷 → 6-device NetHeal 조건에서 **비교적 robust**.

### Case D — Fault × Topology Interaction

Link vs Misconfiguration 등 fault별 격차 → topology **단독 축으로 설명 불가**. 중요하게 보고.

---

# 19. Secondary Experiment

Primary가 완전히 끝난 후에만 수행한다.

## Secondary A — Fixed-Compute Training Strategy

**“순수 topology diversity effect”라고 부르지 않는다.**

### Model A

```text
Train: Star only
100k timesteps
```

### Model B

```text
Train: Star + Linear
동일 100k timesteps
```

평가 (Unseen):

```text
Mesh M1 / M2 / M3
topology_types=["mesh"]
동일 eval suite
```

### 비교 질문

> **동일 Training Compute Budget에서 Single-Topology vs Multi-Topology 학습 중 어느 전략이 Unseen Mesh에 더 robust한가?**

주의: Model B는 Star 경험량이 적으므로 **topology diversity의 인과효과**로 주장하지 않는다.

## Secondary B — Fixed-Size Hierarchical (시간 남을 때)

Primary 완료 후, fixed `devices_per_layer=[2,2,2]` hierarchical generator 패치를 적용한 뒤 **추가 unseen eval**로 수행한다. Primary 결론과 분리 보고한다.

---

# 20. Secondary Fairness

Model A / B는 다음을 동일하게 유지한다.

- Total training timesteps
- Device count
- Fault category set
- PPO architecture
- Hyperparameters
- Eval episode count
- Eval seeds
- Unseen eval suite (Mesh, fault-type stratified)

README에 다음을 명시한다.

> **Training timestep budget은 동일하게 맞췄으며 episode 수 자체가 완전히 동일하다는 의미는 아니다.**

---

# 21. 연구에서 제외하는 것

Primary에서 수행하지 않는다.

- Fine-tuning on target topology
- Scratch retraining on target topology
- Hierarchical topology (Primary) — fixed-size generator 패치 후 Secondary/Future
- LLM multi-agent
- Containerlab
- FRRouting
- Real telemetry
- Autoencoder
- Anomaly detection
- RCA LLM
- Auto remediation
- topology scale shift
- device count generalization
- custom PPO architecture comparison
- multiple RL algorithm benchmark

---

# 22. 연구의 한계

1. NetHeal은 simulation benchmark이며 실제 packet forwarding network가 아니다.
2. 실제 통신망의 protocol timing, packet loss, control-plane convergence, hardware behavior를 완전하게 반영하지 않는다.
3. Device count를 고정하므로 topology scale generalization은 검증하지 않는다.
4. 본 실험은 NetHeal이 정의한 action/fault/reward structure에 의존한다.
5. PPO 하나를 사용하므로 algorithm generalization 비교 연구가 아니다.
6. Primary unseen topology는 **Mesh만** (M1/M2/M3). NetHeal 기본 hierarchical generator는 device count 통제 불가 → 제외.
7. **N=1 training seed**로 Primary generalization 결론을 내리지 않는다.
8. 결론은 다음 범위로 제한한다 — “다른 네트워크에서도 generalize” ❌ / “NetHeal 동일 6-device simulation에서 Star/Linear 학습 PPO를 Mesh에 적용했을 때…” ✅
9. NetHeal은 simulation이며 실제 통신망이 아님을 명시한다.
10. 결과를 실제 상용망 성능으로 일반화할 수 없다.

---

# 23. 본 연구의 기여

과장하지 않고 다음 정도로 정의한다.

> **Cisco NetHeal simulation 환경을 활용하여, Known network(Star/Linear)에서 학습한 policy를 New network(Mesh)에 재학습 없이 배포하는 cross-network deployment protocol을 구성하고, structural mismatch가 diagnostic success(DSR)와 diagnostic exploration cost(TTD-success, Tool Cost)에 미치는 영향을 정량적으로 분석한다.**

본 프로젝트의 핵심 기여는 새로운 RL 알고리즘 개발이 아니다.

```text
Benchmark 활용
+
Generalization protocol 설계
+
Confound 통제
+
Metric decomposition
+
Fault-type 분석
```

이다.

---

# 24. 포트폴리오 표현

## 권장

> Cisco NetHeal simulation benchmark에서 **Known network(Star/Linear)에서 학습한 RL 장애진단 정책**을 **재학습 없이 New network(Mesh)**에 배포했을 때 generalization을 평가했습니다. 진단 성공률(DSR)뿐 아니라 성공 episode 진단 step(TTD-success)과 Tool Cost를 분리해, **진단 실패 vs 진단 비효율** degradation 형태를 분석했습니다.

## 한 줄 (영문 키워드)

> Cross-network diagnostic policy deployment under structural mismatch (without retraining)

## 금지

> Cisco의 장애진단 AI를 개발했다.

> 실제 통신망 장애진단 시스템을 구축했다.

> Cisco NetHeal보다 개선된 모델을 만들었다.

근거가 없으면 사용하지 않는다.

---

# 25. KT NW 인프라운용 연결

본 프로젝트는 다음 업무와 연결해 설명한다.

```text
Multi-site / multi-region network support
Cross-network AI policy deployment
Network fault diagnosis
Troubleshooting under structural mismatch
Topology understanding
Operational efficiency (TTD-success, tool cost)
AI-assisted network operation
```

면접 설명 예:

> “중앙 NOC에서 여러 지역망을 지원할 때, 단순한 구조의 망에서 학습한 장애진단 AI를 redundancy가 큰 **신규/별도 망**에 **재학습 없이** 배포해도 되는지가 궁금했습니다. Cisco NetHeal benchmark로 Known(Star/Linear) vs New(Mesh)를 분리하고, DSR뿐 아니라 TTD-success와 tool cost로 **못 찾는 문제인지, 찾지만 더 많이 탐색하는 문제인지**를 구분해 봤습니다.”

---

# 26. 최종 제목

## 권장 제목

**다른 구조 네트워크로의 AI 장애진단 정책 이식 성능 분석**

## 논문형 제목 (alternate)

**네트워크 토폴로지 변화에 대한 AI 장애진단 정책의 일반화 성능 분석**

**미관측 네트워크 토폴로지에서 RL 기반 장애진단 정책의 일반화 성능 분석**

## 영문

**Generalization Analysis of RL-Based Network Fault Diagnosis Policies on Unseen Mesh Topology**

---

# Part II. 실행 Step

# Step 1. Repository 확보

- NetHeal repository clone
- Python environment 생성
- dependency 설치
- 기본 demo 실행

목표:

```text
Environment reset
Agent action
Fault injection
Episode termination
Metric output
```

까지 정상 동작 확인.

---

# Step 2. Repository 구조 파악

반드시 확인:

- Environment class
- Topology generator
- Fault injector
- Observation shape
- Action space
- Reward
- PPO example (`MultiInputPolicy`)
- Evaluator (`MetricsCollectorWrapper`, `CompetitionEvaluator`)
- Random-Valid baseline (직접 구현)
- User hint option
- `get_valid_actions()` API

산출물:

```text
reports/repo_inventory.md
```

---

# Step 3. Observation Shape 검증

확인:

```text
max_devices
observation dimensions
topology matrix shape
```

장비 수를 6으로 고정했을 때 Star/Linear/Mesh 모두 같은 observation shape인지 unit test.

Hierarchical은 Primary 범위 밖이지만, benchmark 한계 기록용으로 device count가 6으로 고정되지 않음을 `topology_sanity.md`에 문서화한다.

---

# Step 4. Topology Sanity (Star / Linear / Mesh M1·M2·M3)

각 topology 6 devices, `topology_types` 단일, 10회+ 생성.

**Mesh instance M1, M2, M3** seed 고정 후 기록:

- edge count, connectivity, node degree, redundancy/path 구조
- 서로 다른 graph instance인지 확인

Star: center+5 edge = 6 devices.

Hierarchical (기록용): min/max=6이어도 device 수 불고정 → Primary 제외 근거.

산출물: `reports/topology_sanity.md`

---

# Step 5. Fault Injection Sanity

각 fault type × topology별 소수 episode.

---

# Step 6. Runtime Benchmark + Protocol 고정 ★

**Unseen 결과 보기 전** 다음을 고정:

- fault type별 evaluation episodes **N** (20~30 후보)
- `max_episode_steps` (우선 15)
- main training timesteps budget
- Mesh M1/M2/M3 instance seeds
- fault suite seeds

산출물:

```text
config/eval_seeds.json
config/eval_suite.yaml
config/protocol.yaml
```

---

# Step 7. Protocol Artifact 저장

```text
min_devices=6, max_devices=6
max_episode_steps=15
enable_user_hints=False
topology_types (run별 단일, random/hierarchical 제외)
Policy training seeds: 1, 2, 3 (목표)
Mesh instances: M1, M2, M3
```

---

# Step 8. Random-Valid Baseline

Star / Linear / Mesh에서 **동일 eval suite** 실행.

Gym path에서 `get_valid_actions()` + `MetricsCollectorWrapper` 사용.

수집:

- DSR (`diagnosis_success_rate`)
- Macro F1
- TTD-success (custom: success episode only)
- Avg Steps-all (`avg_steps`)
- Tool Cost (`tool_cost_index`)
- Coverage (`topology_coverage`)
- Fault breakdown

산출물:

```text
results/random_valid_baseline.csv
```

---

# Step 9. PPO Sanity — ★ 첫 Go/No-Go

```text
Star only, topology_types=["star"]
6 devices, max_episode_steps=15, Hint OFF
PPO MultiInputPolicy, Policy seed 1
```

확인: Reward↑, DSR↑, **Random-Valid 대비 의미 있는 성능**

| GO | NO-GO |
|---|---|
| → Step 10 진행 | Generalization 금지, PPO/환경 재검토 |

---

# Step 10. Main PPO Training × 3 Seeds

```text
Star + Linear, topology_types=["star","linear"]
6 devices, max_episode_steps=15
Training seeds: 1, 2, 3
50k~100k timesteps (Unseen test 전 고정)
```

각 seed checkpoint 저장.

---

# Step 11. Seen Evaluation

Star / Linear 각 별도 run. 3 seeds × 동일 fault suite. `deterministic=True`.

---

# Step 12. Unseen Evaluation — Mesh M1 / M2 / M3

```text
topology_types=["mesh"]
3 policy × 3 instances × fault suite
```

Train에 Mesh 절대 미포함.

---

# Step 13. Metric Aggregation

- Table 1~3 (§17)
- TTD-success custom aggregate
- mean ± SD (3 seeds)

---

# Step 14. Effect Size

DSR Gap, Relative TTD-success/Cost increase.

---

# Step 15. Fault-Type Breakdown

DSR + TTD-success per fault (survivorship bias 주의 명시).

---

# Step 16. Case A/B/C/D 해석

`reports/primary_results.md`

---

# Step 17. Figures + README

Figure 1~4 + README (§24 구조).

---

# Step 18. Secondary (Primary 완료 + 시간 남을 때만)

Fixed-Compute: Star-only vs Star+Linear → Mesh M1/M2/M3.

---

# Step 19. Reproducibility 정리

저장:

- package versions
- policy training seeds (1, 2, 3)
- Mesh instance seeds (M1, M2, M3)
- fault / episode seeds
- topology config (run별 `topology_types`, random/hierarchical 제외)
- max_episode_steps, fault N per type
- PPO hyperparameters (MultiInputPolicy)
- model checkpoints (×3)
- TTD-success aggregation script
- raw evaluation results
- aggregation scripts

---

# Step 20. README 작성

README 구성:

1. Problem (cross-network AI policy deployment)
2. Why NetHeal
3. Research Question
4. Experimental Control
5. Train/Seen/Unseen Protocol
6. Metrics (DSR, TTD-success, Avg Steps-all, Tool Cost, Coverage, NetHeal key mapping)
7. Results
8. Fault Breakdown
9. Key Finding
10. Limitations
11. Reproduction
12. Future Work

---

# Step 21. Future Work

다음은 구현하지 않고 명시만 가능하다.

- Fixed-size hierarchical topology eval (`devices_per_layer=[2,2,2]` 패치)
- Target topology fine-tuning
- Scratch vs adaptation
- Device scale generalization
- Hint ON/OFF ablation
- Real telemetry validation
- Containerlab based physical-network emulation
- LLM agent comparison

---

# Part III. 2~3일 일정

# Day 1 — Step 1~9 (Environment + Go/No-Go)

1. Step 1~2: Repo + 구조
2. Step 3~4: 6-device obs + Star/Linear/Mesh **M1/M2/M3** sanity
3. Step 5~6: Fault sanity + **Runtime benchmark → protocol 고정**
4. Step 7~8: Protocol artifact + Random-Valid baseline
5. Step 9: **★ PPO Sanity Go/No-Go** (Star, seed 1)

**Day 1 성공 조건**

```text
PPO Sanity GO
(Random-Valid 대비 학습 확인)
+
evaluation metric pipeline 동작
```

NO-GO → Generalization 중단, PPO/환경 재검토.

---

# Day 2 — Step 10~17 (Primary)

1. Main PPO train × **3 seeds**
2. Seen Star / Linear eval
3. Unseen Mesh **M1/M2/M3** eval
4. Aggregation + Effect size + Fault breakdown
5. Table 1~3 + Figure 1~4

**Day 2 성공 조건**

```text
Table 1 (mean ± SD, 3 seeds) + 최소 3 Figures
Case A/B/C/D 해석 초안
```

---

# Day 3 — Step 18~21 (Documentation + Optional)

1. README + reproducibility (Step 19~20)
2. seed 3 미완 시 seed 2/3 보완 (가능하면)
3. Step 18 Secondary (Primary 완료 후만)
4. 포트폴리오 문장

---

# Part IV. 프로젝트 구조

```text
netheal-topology-generalization/
│
├── README.md
├── requirements.txt
│
├── config/
│   ├── train.yaml
│   ├── eval.yaml
│   ├── eval_seeds.json
│   ├── eval_suite.yaml
│   └── protocol.yaml          # M1/M2/M3, fault N, max_steps
│
├── src/
│   ├── train_ppo.py
│   ├── evaluate.py
│   ├── baselines.py          # Random-Valid agent
│   ├── aggregate_metrics.py  # TTD-success custom aggregate
│   └── plot_results.py
│
├── models/
│   └── checkpoints/
│
├── results/
│   ├── raw/
│   ├── aggregated/
│   └── figures/
│
├── reports/
│   ├── repo_inventory.md
│   ├── topology_sanity.md
│   └── primary_results.md
│
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

# Part V. 실험 무효 조건 체크리스트

다음 중 하나라도 발생하면 Seen/Unseen generalization 결론을 내리지 않는다.

- [ ] Mesh/`random`/`hierarchical`이 Train에 포함됨
- [ ] Eval run에 topology 2개 이상 섞임
- [ ] **N=1 training seed**로 Primary generalization 결론
- [ ] Train/Eval/Baseline `max_episode_steps` 불일치
- [ ] **Unseen 결과 본 뒤** episode 수·protocol 변경 (test peek)
- [ ] Policy seed 1/2/3이 **서로 다른 fault suite**로 평가됨
- [ ] Mesh M1/M2/M3 instance seed 미고정·미기록
- [ ] PPO Sanity NO-GO인데 Generalization 진행
- [ ] TTD-success만 단독 보고 (DSR/Avg Steps-all 누락)
- [ ] Random-Valid와 PPO eval protocol 불일치
- [ ] “다른 네트워크에서도 generalize” 등 범위 초과 주장
- [ ] Simulation을 실제망으로 과장

---

# Part VI. 최종 결론 문장 템플릿

## Case A — DSR 하락 + 비용 증가

> New network(Mesh) 배포 시 diagnostic success와 efficiency가 모두 저하되었다. Known network에서 형성된 diagnostic policy가 연결 구조에 부분적으로 의존했음을 시사한다.

## Case B — DSR 유지 + 비용 증가

> New network(Mesh) 배포 시에도 진단 성공률은 비교적 유지되었으나 TTD-success와 Tool Cost가 증가했다. cross-network deployment에서 degradation이 **진단 실패**보다 **topology exploration burden 증가** 형태로 나타날 수 있음을 보여준다.

## Case C — 차이 작음

> 동일 device scale과 fault 조건에서 topology family 변화만으로는 diagnostic policy 성능이 크게 저하되지 않았다. 본 benchmark 조건에서 학습된 policy가 일정 수준의 topology generalization 능력을 보였음을 시사한다.

---

# Part VII. 최종 프로젝트 한 문장

> **본 연구는 AI가 학습한 네트워크와 구조가 다른 신규 네트워크(Mesh)에 재학습 없이 배포될 때, DSR·TTD-success·Tool Cost·Coverage를 함께 측정하여 degradation이 ‘진단 실패’인지 ‘탐색 비효율’인지 구분하는 것을 목표로 한다. (NetHeal 동일 6-device simulation, 3 training seeds, mean ± SD)**
