# Obs-wrapper → (조건부) E4 — 관측 복구 후 diagnose-mask 스펙

## 0. 배경 및 질문

지금까지의 실패(E1 보상, E2 timeout, E3 ent_coef, Ablation-NL, Gate dummy-ping)는 모두 **Cisco `to_dict()` 관측을 그대로** 둔 채 학습 신호만 바꿨다.

9.10: 시뮬레이터 **도구 결과**에는 type/location을 가를 증거가 있다.  
stochastic audit (n=400): 도구를 써도 informed exact ≈ 0.7%.  
Gym 벡터는 그 증거를 버린다.

| 내부 상태 | Cisco Gym 벡터 (`observation.py` `to_dict`) |
|---|---|
| `DeviceStatus.DOWN` | `device_status[:,0] = 1` iff `status == 'active'` (문자열). enum과 불일치 → **항상 0** |
| ping/interface latency (예: 20.65ms) | `discovery_matrix`는 int8 adjacency만. CONNECTED 유지. latency는 `connection_properties`에만 있음 |
| ping fail error 문구 | `recent_diagnostics`에 success=0 + hash + timestamp. 에러 텍스트 없음 |

질문: **이 두 칸(상태, 지연)을 실험 래퍼로 벡터에 넣으면, 조사가 진단 확률을 움직이는가.**  
움직이면 그때 E4(초반 진단 잠금). 안 움직이면 E4를 올리지 않는다.

E1 / E2 / E3 / Ablation-NL / Gate / Factorized / Symmetric과 **섞지 않는다.**  
NetHeal 패키지(`netheal-ai-agent-benchmark`)는 **수정하지 않는다.**

---

## 1. 전체 순서 (한 번에 다 안 함)

```
Stage A  래퍼 + 유닛 테스트          (정책 없음)
Stage B  래퍼 위에서 Control 재학습   (보상/마스크 curriculum 없음)
Stage C  E4 diagnose-mask            (Stage B Go일 때만)
```

Stage A 실패 → 구현 버그. 학습 금지.  
Stage B No-Go → E4 금지. 다음 후보는 아키텍처/72-way 또는 LLM 경로이지, 또 다른 보상 패치가 아님.

---

## 2. Stage A — Observation wrapper

### 2.1 한 가지 개입

Gymnasium `ObservationWrapper` (실험 레포 `netheal-topology-generalization/src/`).  
`env.unwrapped`의 `StructuredObservation` / `discovery_matrix.connection_properties` / `DeviceStatus`를 읽어 **같은 dict 키·같은 shape**를 덮어쓴다.

| 키 | shape (현행) | 래퍼가 쓰는 칸 | 쓰는 값 |
|---|---|---|---|
| `device_status` | `(max_devices, 10)` float32 | `[:, 0]` | `UNKNOWN=0, UP=+1, DOWN=−1, DEGRADED=+0.5`. 문자열 `'active'` 비교 금지 |
| `device_status` | 동일 | `[:, 4]` (현행 미사용에 가깝) | 해당 행 디바이스에서 관측된 **최대 latency_ms / 100**, clip `[0, 1]` |
| `discovery_matrix` | `(max_devices, max_devices)` int8 | **값 의미만 확장** | 기본 adjacency 유지. latency ≥ 15ms인 간선은 `ConnectionStatus.DEGRADED`(=2)로 기록 (원래 CONNECTED=1과 구분) |
| `recent_diagnostics` | `(10, 6)` | `[:, 4]` | timestamp 대신 **해당 툴 결과의 latency_ms / 100**, 없으면 0. success 비트 `[:, 3]`는 유지 |

`[:, 4]`를 timestamp → latency로 바꾸는 것은 의도된 교체다. timestamp는 증거 칸이 아니었다.

**새 obs 키를 추가하지 않는다.** 키를 늘리면 MultiInputPolicy 입력이 달라져 Stage B와 이후를 갈라놓는다. shape는 Cisco와 동일하게 유지한다.

### 2.2 하지 않는 것

- NetHeal `to_dict` 원본 수정
- reward / action mask / fault 샘플러 변경
- GT를 obs에 넣기
- hash(device_type) 같은 기존 칸을 임의로 더 갈아엎기 (`[:,0]`, `[:,4]`, discovery 값, recent `[:,4]`만)

### 2.3 유닛 테스트 (학습 전 필수)

통제 Star, `inject_fault_at_location`, scan 후 해당 도구.

| # | 설정 | 통과 조건 |
|---|---|---|
| 1 | Device `edge_0` DOWN + `check_status` | 해당 디바이스 행 `device_status[i,0] == −1` |
| 2 | 동일 + ping `edge_1→edge_0` (fail) | feat0 −1 또는 recent 최신 행 success=0. feat0가 전부 0이면 실패 |
| 3 | Perf `center→edge_0` + ping (latency ≥ 15ms) | discovery 해당 간선이 2이거나 `device_status`/`recent` latency 칸 > 0.15 |
| 4 | Healthy ping `edge_1→edge_2` | feat0가 DOWN(−1)이 아님. degraded 간선이 그 쌍에 생기지 않음 |
| 5 | 래퍼 off (원본) | Device DOWN 이후에도 feat0 전부 0 (회귀: 구멍이 아직 있는지) |

5번이 깨지면 측정 코드가 원본을 잘못 읽은 것이다. 1–4가 깨지면 래퍼 구현 오류.

---

## 3. Stage B — 래퍼 위 Control 재학습

### 3.1 왜 기존 zip만으로 판정하지 않는가

E3 Control은 **feat0가 항상 0인 관측**에서 학습됐다. 래퍼로 feat0에 DOWN을 넣어도, 그 가중치는 “상수 0”에 맞춰져 있어 **P(correct)가 안 오르는 게 기본값**이다.  
“같은 zip + 새 obs”는 **음수 대조**로만 돌린다 (기대: ΔP(correct) ≈ 0). 판정에 쓰지 않는다.

### 3.2 학습 (Control-Obs)

| 항목 | 값 |
|---|---|
| 환경 | Star, 6 devices, hint OFF, **4-fault 균등**, 원본 terminal reward |
| 관측 | Stage A 래퍼 ON |
| 알고리즘 | MaskablePPO, `ent_coef=0.0`, seed 1, 30k timestep |
| 금지 | E1/E2/E3/Gate/Factorized/Symmetric/timeout/PBRS shaping/diagnose-mask |

비교 대상은 **같은 코드로 래퍼 OFF Control을 나란히 재학습하지 않는다** (또 다른 7–20% DSR 노이즈).  
행동/단락 비교는 **이 Control-Obs의 scan 직후 vs 도구 직후**와, 이미 있는 stochastic audit 숫자(래퍼 없는 E3 zip)다.

### 3.3 프로브 (필수, 재학습 후)

`run_stochastic_evidence_audit`와 동일한 통제 주입.

| probe | 보는 것 |
|---|---|
| Device `edge_0` + ping to GT | Φ≥1, feat0, **P(correct device)** scan→tool |
| Device dummy ping (healthy pair) | Φ=0, dummy TV vs evidence TV |
| Device `check_status` | feat0=−1, ΔP(correct) |
| Perf `center→edge_0` ping | latency 칸, discovery=2, **P(correct perf)** |
| Misconfig ping (참고) | 래퍼가 DOWN/block을 어떻게 쓰는지 |

지표:

- `ΔP(correct) = P(correct | after evidence tool) − P(correct | post-scan)`
- diagnose TV (evidence tool vs dummy)
- feat0 / latency 칸이 유닛 테스트와 같이 변하는지

추가로 Control-Obs를 NL-only **stochastic 200 episode** (재학습 없이 eval만):

- `P(Φ≥1 | tool-use)`
- `P(exact ∧ consistent | tool-use)`
- `P(exact ∧ consistent | Φ≥1)`

### 3.4 판정 (deterministic 프로브가 주, sto는 보조)

사전 고정.

1. **래퍼 작동 (A 통과는 전제)**  
2. **가설 지지 (관측이 진단에 연결됨):**  
   Device 또는 Perf 프로브에서 **`ΔP(correct) ≥ +0.05`**  
   그리고 dummy의 ΔP(correct)보다 evidence 쪽이 큼  
3. **Go (Stage C E4 개방):** 위 + sto `P(exact ∧ consistent | Φ≥1) ≥ 10%`  
   또는 (프로브 ΔP ≥ +0.05 이고) sto `P(exact ∧ consistent | tool-use) ≥ 5%`  
4. **No-Go:** 래퍼는 통과했는데 Device/Perf `ΔP(correct) < +0.02` 이고 sto exact-consistent | Φ 가 audit과 같이 ~3% 이하  
   → 칸은 보이는데 30k PPO가 안 씀. E4는 **열지 않음**.  
5. **애매:** ΔP는 +0.02–0.05 또는 Perf만 오름. seed/E4 안 열고 기록.

기존 audit 기준점 (래퍼 없음, E3 Control zip): Device ΔP ≈ 0.000, Perf ΔP ≈ −0.01.

음수 대조 (E3 zip + 래퍼 ON, 재학습 없음): ΔP < +0.02이면 “옛 가중치는 새 칸을 못 읽음”으로 기록만.

---

## 4. Stage C — E4 diagnose-mask curriculum (Go일 때만)

별도 구현. 여기서는 경계만 고정한다.

| 항목 | 값 |
|---|---|
| 관측 | Stage A 래퍼 ON (B에서 쓴 것) |
| 보상 | 원본 그대로 |
| 개입 | 에피소드 초반 **K스텝 동안 DIAGNOSIS 마스크 off** (scan 다음 도구를 강제) |
| K | **3으로 사전 고정** (scan + diagnostic 최소 2). 사후 그리드서치 금지 |
| 이후 | K 이후 72-way 진단 개방 (현행과 동일) |
| 금지 | E1/E2/ent_coef 동시 변경, 반복-행동 가중 감점, 초반 step penalty 인하 |

묻는 것: 관측이 진단을 움직일 때, **강제 조사가 greedy shortcut을 깨는가.**  
성공 기준은 E4 착수 전에 한 장으로 적는다 (Device/Misconfig det exact≥2 + evidence-consistent 등). **지금 적고 학습하지 않는다.**

---

## 5. 지표 표 (Stage B)

### 5.1 유닛 테스트

| # | 결과 |
|---|---|
| 1–5 | **all pass** |

### 5.2 프로브 (Control-Obs, 재학습 후)

| probe | Φ | feat0 / latency 칸 | ΔP(correct) | TV |
|---|---|---|---:|---:|
| device ping GT | 2 | feat0 DOWN=True | +0.0000 | 0.034 |
| device dummy | 0 | latency 0.062 | +0.0001 | 0.091 |
| device status | 2 | feat0 DOWN=True | +0.0000 | 0.044 |
| perf ping | 2 | latency 0.206, disc=2 | **−0.0019** | 0.088 |

래퍼 칸은 변함. Device ΔP < dummy. Perf ΔP 음수. 가설 지지 실패.

### 5.3 Stochastic NL (Control-Obs, n=200)

| Metric | E3 zip (audit, 참고) | Control-Obs |
|---|---:|---:|
| P(Φ≥1 \| tool-use) | 28.5% | 22.6% |
| P(exact ∧ cons \| tool-use) | 0.7% | **0.0%** |
| P(exact ∧ cons \| Φ≥1) | 2.4% | **0.0%** |

**Decision: no_go.** Stage C E4 미구현.

---

## 6. 이 실험이 답하는 것 / 답하지 않는 것

**답하는 것**

- DOWN/latency를 벡터에 넣으면 조사가 진단 분포를 움직이는가
- 그 전에 E4를 열어도 되는가

**답하지 않는 것**

- Cisco `to_dict`를 고치는 패치 (우리는 래퍼만)
- 4-fault 배포 정책 / Mesh 일반화
- LLM+MCP가 이 숙제를 푸는가
- 반복 감점·초반 −0.1 인하 (관측 복구 전 금지)

---

## 7. 체크리스트

### Stage A
- [x] NetHeal 패키지 미수정
- [x] shape/키 유지, `[:,0]` status, latency를 `[:,4]` + discovery degraded
- [x] 유닛 테스트 1–5 통과
- [x] E1/E2/E3/NL/Gate와 미혼합

### Stage B
- [x] Control-Obs만 30k 재학습 (래퍼 ON, 원본 reward)
- [x] E3 zip+래퍼는 음수 대조만
- [x] 프로브 ΔP(correct) + sto 조건부 기록
- [x] §3.4로 Go / No-Go / 애매 → **no_go** (Device ΔP +0.000, Perf −0.002, P(ec\|Φ)=0%)
- [x] Go가 아니면 Stage C 코드 작성·학습 안 함

### Stage C (Go 후)
- [ ] K=3 고정, 래퍼 유지, 보상 원본 — **닫힘 (B no_go)**
- [ ] E4 전용 성공 기준을 학습 전에 문서화 — **닫힘**
