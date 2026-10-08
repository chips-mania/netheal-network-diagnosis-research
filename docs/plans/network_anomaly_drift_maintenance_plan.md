# 네트워크 이상탐지 모델 Drift 대응 실험 계획서
## 동일 네트워크의 트래픽 운용 조건 변화에 따른 모델 유지·보정·적응·재학습 전략 비교

> **핵심 질문**  
> 기존 운용 조건에서 배포된 이상탐지 모델이 트래픽 특성 변화로 성능 저하를 보일 때,  
> **Threshold 재보정, Fine-tuning, Scratch Retraining 중 어느 수준의 유지보수가 필요한가?**

---

# 0. 연구 한 줄

이 프로젝트는 “더 좋은 이상탐지 모델을 만들자”가 아니다.

**동일 네트워크에서 트래픽 운용 조건이 달라졌을 때 기존 정상 기반 이상탐지 모델의 분포 변화와 오탐 증가를 정량화하고, 변화에 대응하기 위한 유지·재보정·적응·재학습 전략을 비교하는 PoC**다.

연구 질문은 더 이상 바꾸지 않는다. 이후 절과 이 절이 충돌하면 **이 절(확정된 실험 범위)을 따른다.**

---

# 확정된 실험 범위

기능을 더 얹지 않는다. 용어를 정확히 하고, Dataset 3의 짧은 정상 구간에 맞춰 실험을 줄인 뒤 Dataset inventory부터 진행한다.

## 잠긴 연구 질문

> 기존 traffic regime에서 학습한 정상 기반 이상탐지 모델이 새로운 traffic regime에서 성능 저하를 보였을 때, 모델을 다시 학습하기 전에 어느 수준의 유지보수가 필요한가?

실험 답안은 다음 순서로 **가장 싼 대응부터** 비교한다.

```text
KEEP → RECALIBRATE → ADAPT → SCRATCH → TARGET-FULL
```

Fine-tuning이 이겨야만 성공이 아니다.

- Threshold만으로 오탐을 줄이고 탐지를 유지하면, 그것이 현업적으로 더 좋은 결론이다.
- Fine-tuning이 필요하면 그 이유를 보여준다.
- Scratch가 낫다면 기존 모델을 버릴 조건에 대한 근거가 된다.

## 용어: Concept Drift가 아니라 Traffic Regime / Distribution Shift

KT 사례에서 Concept Drift라는 표현을 썼더라도, Cisco 실험에서 직접 관찰하는 것은 기본적으로 다음이다.

```text
500 Gbps Normal
        ↓
1.3 Tbps Normal

P(X)가 달라짐
```

BGP Clear라는 **이상의 의미 자체는 달라지지 않는다.** 논문·README·본문에서는 다음을 쓴다.

- distribution shift
- traffic regime shift
- input telemetry distribution shift

**Concept Drift는 본 실험의 이름으로 쓰지 않는다.**

KT 연결 문장은 다음만 사용한다.

> KT의 실제 운영 사례에서 네트워크 환경 변화에 따라 모델의 성능을 지속적으로 관리하고 재학습 여부를 판단한다는 문제의식을 확인했다. 본 실험에서는 그중 트래픽 운용 조건 변화에 따른 입력 Telemetry distribution shift를 제한적으로 다룬다.

## 데이터는 점진적 성장이 아니라 독립 run의 proxy

페르소나에서 “500G였다가 서비스가 성장해서 몇 달 뒤 1.3T가 됐다”고 말하는 것은 **시나리오**일 뿐이다. Cisco 데이터가 시간 순으로 그렇게 변한 기록은 아니다.

실제 데이터:

```text
Run A = Dataset 1, 500 Gbps
Run B = Dataset 3, 약 1.3 Tbps
```

동일 lab fabric을 **서로 다른 traffic operating point에서 수집한 독립 run**이다.

README / 한계에 반드시 명시한다.

> 본 데이터는 하나의 네트워크에서 트래픽이 시간에 따라 500 Gbps에서 1.3 Tbps로 점진적으로 증가한 기록이 아니라, 동일 lab fabric을 서로 다른 traffic operating point에서 수집한 독립 run이다. 본 연구에서는 이를 운용 조건 변화의 통제된 proxy로 사용한다.

## Primary Feature

처음부터 Family A/B × Level/Change 네 가지를 돌리지 않는다.

### Primary

**Family A + Level-based**

- traffic rate, packet rate, drops, errors 등 data-plane 절대 수준
- 첫 질문(“트래픽 운용 조건 변화가 기존 모델을 흔드는가?”)을 가장 직접 보여 줌

### Secondary (필요할 때만)

**Family B = A + control-plane**

- route count, neighbor count, BGP counters
- Family A만으로 BGP Clear 탐지가 너무 약할 때 **한 번만** 추가

**Change-based** (delta, percent change, rolling z-score)

- Primary 결과를 본 뒤 필요할 때만
- 예: Level-based Source-only FPR이 매우 높으면, 절대 traffic level에 과민한지 확인하기 위해 추가
- 처음부터 돌리지 않음

## Target 데이터는 비율 곡선이 아니라 N분

Dataset 3 정상은 첫 이벤트 이전 약 5분이므로 5% / 10% / 25% / 100% data-efficiency 곡선을 Primary로 만들지 않는다.

```text
Target-small  = clean window 시간축 앞쪽 N분
Target Test   = 이후 clean 구간
Target-full   = 확보 가능한 clean normal 전부
```

N은 지금 정하지 않고 **inventory 후** 정한다. timestamp가 충분하면 예를 들어 다음과 같이 갈 수 있다.

```text
0~2분  → Target-small (RECALIBRATE / ADAPT / SCRATCH에 동일하게 사용)
2~5분  → Clean normal test
```

timestamp가 너무 적으면 2분을 억지로 고정하지 말고, train에 쓸 분과 test에 남길 분을 inventory에서 같이 정한다.

운영 스토리에서는 비율보다 **“새로운 정상 데이터를 몇 분 확보했는가”**라고 말한다.

## 모든 결과는 FPR과 Detection을 같이 본다

Threshold를 극단적으로 높이면 FPR = 0%를 만들 수 있으나, BGP Clear를 하나도 못 잡으면 무의미하다. 모든 전략 표는 최소 다음 축을 같이 보고한다.

| Strategy | Target data | 정상 FPR ↓ | BGP Event Detection ↑ | Delay ↓ |
|---|---|---:|---:|---:|
| KEEP | 0 | | | |
| RECALIBRATE | N분 | | | |
| ADAPT | 동일 N분 | | | |
| SCRATCH | 동일 N분 | | | |
| TARGET-FULL | available full | | | |

해석 예:

```text
KEEP         FPR 43%   Detection 96%
RECALIBRATE  FPR  7%   Detection 48%
```

→ Threshold 보정으로 오탐은 줄였지만 이상까지 함께 눌렀다.

```text
RECALIBRATE  FPR  8%   Detection 92%
```

→ 굳이 모델을 재학습할 필요가 없었다.

## Observation Unit과 Device×Timestamp 집계

구현 전 inventory에서 결정한다. Device×Timestamp를 쓰면 interface telemetry를 장비 시각으로 접어야 한다.

예: `leaf1` + `10:00:00`의 여러 인터페이스 →

- `total_rx_rate`, `total_tx_rate`
- `mean_utilization` / `max_utilization`
- `total_drop`, `total_error`

집계 규칙은 feature 의미에 따른다.

| 종류 | 집계 |
|---|---|
| Traffic bytes / rate | sum |
| Error / drop count | sum |
| Utilization | mean 또는 max |
| Route / neighbor count | 이미 장비 단위면 그대로 |

Inventory에 포함할 것:

> 각 EncodingPath가 interface-level인지 device-level인지 확인하고, Device×Timestamp로 집계하는 규칙을 먼저 정의한다.

## Dataset 5 / 6은 replication이 아님

Dataset 3과 동일 조건의 독립 반복이 아니다.

- Dataset 5: 다른 부하·길이의 BGP Clear run
- Dataset 6: 앱 트래픽 0이라는 **다른 shift**

표현:

- Secondary robustness check
- Additional traffic-condition validation

“재현(replication)”이라고 부르지 않는다.

## Primary 실행 파이프라인 (이 이상 추가하지 않음)

```text
[1] Inventory
    Dataset 1: 500G Normal
    Dataset 3: 1.3T Clean Normal + BGP Clear
    EncodingPath 레벨 / 집계 규칙 / clean 길이 / N분 결정

[2] 기존 모델
    Dataset 1 Normal → AE
    Family A + Level-based

[3] 환경 변화 측정
    500G Normal vs 1.3T Clean Normal
    Wasserstein Drift

[4] KEEP
    기존 AE 그대로
    → Target Clean FPR
    → BGP Detection / Delay

[5] 동일한 Target-small (N분)
          ┌──────────────┬──────────────┐
          ↓              ↓              ↓
    RECALIBRATE        ADAPT         SCRATCH
    threshold만        fine-tune      random init

          ↓              ↓              ↓
    동일 Target Clean Test
    동일 BGP Event Test

[6] TARGET-FULL
    확보 가능한 Target clean normal 전체
    → Reference
```

끝. 이후는 Primary 완료 후에만 Secondary.

이 실험의 Confirmed Normal은 **첫 이벤트 이전 clean window**다. 이 CSV에 장비 alarm 컬럼이 없어도 pre-event clean을 정상으로 쓴다.

---

# 1. 연구 배경

네트워크 이상탐지 모델은 보통 정상 상태의 Telemetry 패턴을 학습한 뒤, 정상 분포에서 벗어난 상태를 이상으로 판단한다.

그러나 실제 네트워크는 고정된 환경이 아니다.

- 서비스 이용량 증가
- 시간대별 트래픽 변화
- 장비 증설
- 구성 변경
- 사용자 패턴 변화
- 계절성 및 주기성 변화

등으로 인해, **장애가 없어도 정상 Telemetry 분포 자체가 달라질 수 있다.**

이 경우 배포 당시에는 잘 작동하던 이상탐지 모델이 새로운 정상 상태를 이상으로 판단하면서 False Positive가 증가할 수 있다.

KT의 「AI 기반 IP코어망 장애 선감지 서비스 개발」 사례에서는 네트워크 환경 변화에 따라 모델 성능을 지속적으로 관리하고 재학습 여부를 판단하는 문제의식을 확인했다. 해당 자료는 Concept Drift라는 표현을 쓰지만, **본 Cisco 실험이 직접 다루는 것은 이상의 의미 변화가 아니라 트래픽 운용 조건 변화에 따른 입력 Telemetry distribution shift**다.

본 프로젝트는 여기서 한 단계 더 구체적인 질문을 만든다.

> **“환경 변화가 감지됐다고 항상 모델을 다시 학습해야 할까?”**

즉, 실제 운용자의 관점에서 다음 의사결정을 데이터로 비교한다.

```text
KEEP
→ 기존 모델 유지

RECALIBRATE
→ 기존 모델 유지 + Threshold 재보정

ADAPT
→ 기존 모델 + 최근 정상 데이터 Fine-tuning

RETRAIN
→ 기존 모델을 사용하지 않고 Target 데이터로 Scratch 학습
```

---

# 2. Operational Persona

## 2.1 가상 페르소나

**정우진 / 통신사 IP 네트워크 운용 엔지니어**

> ※ 특정 KT 내부 운영 절차를 그대로 재현한 인물이 아니라, 통신사 NW 인프라운용 업무를 바탕으로 구성한 가상 Operational Persona다.

### 담당 업무

- IP 네트워크 장비 상태 및 Traffic 모니터링
- Telemetry 기반 이상 알람 확인
- 장애 영향 판단
- 이상탐지 모델의 운영 상태 확인
- 불필요한 오탐 감소
- 모델 유지보수 필요 여부 판단

### 현재 상황

기존 패브릭은 약 500 Gbps 수준의 정상 트래픽에서 이상탐지 모델을 학습해 운영 중이다.

시간이 지나 서비스 이용량이 증가하면서 동일 네트워크의 정상 트래픽 수준이 약 1 Tbps 이상으로 증가했다. **이 문단은 운용 시나리오다.** 사용 데이터는 500 Gbps run과 1.3 Tbps run을 따로 수집한 것이며, 한 패브릭이 수개월에 걸쳐 성장한 시계열이 아니다.

네트워크 자체에는 장애가 없지만 기존 모델은 새로운 정상 상태를 기존 분포와 다르게 보기 시작한다.

운용자는 다음을 판단해야 한다.

> 1. 현재 변화는 장애인가, 정상 운용 조건 변화인가?
> 2. 기존 모델을 그대로 유지해도 되는가?
> 3. Threshold만 다시 맞추면 되는가?
> 4. 최근 정상 데이터로 Fine-tuning해야 하는가?
> 5. 기존 모델을 폐기하고 새 모델을 다시 학습해야 하는가?

---

# 3. 연구 목적

본 연구의 목적은 다음 세 단계를 검증하는 것이다.

### 목적 1. Drift 측정
기존 정상 분포와 새로운 트래픽 조건의 정상 분포가 얼마나 달라졌는지 정량화한다.

### 목적 2. Drift와 모델 성능 저하의 관계 확인
Input Distribution Drift가 커질 때 기존 이상탐지 모델의 Reconstruction Error 및 False Positive가 실제로 증가하는지 확인한다.

### 목적 3. 대응 전략 비교
동일한 Target 정상 데이터가 주어졌을 때 다음 전략을 비교한다.

1. 기존 모델 유지
2. Threshold 재보정
3. Fine-tuning
4. Scratch Retraining
5. 충분한 Target 데이터 기반 Full Retraining (Reference)

---

# 4. 핵심 연구 질문

## RQ1. Distribution Drift

> **트래픽 운용 조건이 달라지면 기존 정상 Telemetry 분포와 새로운 정상 분포 간 차이가 측정 가능한 Drift로 나타나는가?**

## RQ2. Model Degradation

> **Telemetry Distribution Drift가 커질수록 기존 이상탐지 모델의 정상 False Positive가 증가하는가?**

## RQ3. Maintenance Strategy

> **트래픽 조건 변화로 기존 모델의 성능이 저하됐을 때 Threshold 재보정, Fine-tuning, Scratch Retraining 중 어떤 전략이 가장 효율적인가?**

## RQ4. Pretraining Benefit

> **동일한 양의 Target 정상 데이터가 주어졌을 때, 기존 Source 정상 패턴을 유지한 Fine-tuning이 Target Scratch 학습보다 유리한가?**

---

# 5. 연구 가설

## H1. Drift 가설
500 Gbps 정상 환경과 1 Tbps 이상 정상 환경 사이에는 Traffic-related Telemetry Distribution Drift가 존재할 것이다.

## H2. FPR 가설
기존 Source 모델을 새로운 Target 정상 환경에 그대로 적용하면 정상 Reconstruction Error 및 False Positive가 증가할 것이다.

## H3. Recalibration 가설
단순 Threshold 재보정만으로도 False Positive를 줄일 수 있을 것이다. 다만 FPR만 낮아지고 Event Detection이 함께 무너지면 충분한 대응이 아니다.

## H4. Adaptation 가설
Target 정상 데이터가 제한된 조건에서는 Source-pretrained 모델의 Fine-tuning이 동일 Target 데이터로 처음부터 학습한 Scratch 모델보다 높은 데이터 효율성을 보일 수 있다.

## H5. Retraining 필요성 가설
Drift가 큰 경우 또는 기존 representation이 새로운 정상 구조를 충분히 설명하지 못하는 경우에는 Threshold 재보정이나 Fine-tuning보다 Scratch Retraining이 유리할 수 있다.

---

# 6. Cisco Dataset 개요

사용 데이터:

**Cisco Innovation Edge Network Telemetry Dataset**

Repository:

```text
https://github.com/cisco-ie/telemetry
```

## 6.1 Dataset 요약

| Dataset | Traffic | Event | Duration | 주요 용도 |
|---|---:|---|---:|---|
| 0 | 0 | 없음 | 1h | 무부하 정상 baseline |
| 1 | 500 Gbps | 없음 | 1h | 부하 있는 정상 baseline |
| 2 | 1 Tbps | BGP Clear | 1h | BGP 이상 |
| 3 | ~1.3 Tbps / 0 | BGP Clear | ~33m | 동일 이벤트의 with/without traffic |
| 4 | 1 Tbps | Port event | ~43m | 링크/포트 상태 이상 |
| 5 | 1 Tbps | BGP Clear | 2h | 앱 트래픽 있음 |
| 6 | 0 | BGP Clear | 2h | 앱 트래픽 없음 |
| 7 | 0 | VIRL BGP Clear | 72h | 시뮬레이션 |
| 8 | 0 | VIRL BGP Clear | 262h | 시뮬레이션 |
| 9 | 2.9 Tbps | Admin shutdown | 45m | 포트 이상 |
| 10 | 2 Tbps | Transceiver pull | 35m | 포트 이상 |
| 11 | 미기재 | Interface/BFD | 5h 28m | 단일 장비 MDT |
| 12 | AppMix/FlowMix | 다양한 이벤트 | 다수 run | 이벤트 일반화 |

---

# 7. Primary Source / Target 정의

## Source

**Dataset 1**

```text
500 Gbps
No anomaly
1 hour
```

역할:

> 기존 운용 조건에서 이미 배포된 정상 이상탐지 모델의 학습 데이터

## Target

**Dataset 3의 Traffic 포함 데이터**

```text
약 1.3 Tbps
BGP Clear
약 4~5초 sampling
첫 event 이전 약 5분 clean baseline
```

역할:

> 트래픽 부하가 크게 증가한 새로운 운용 조건

## Secondary robustness check (replication 아님)

**Dataset 5 / Dataset 6** — additional traffic-condition validation

- Dataset 5: 1 Tbps + BGP Clear (다른 부하·길이의 BGP run)
- Dataset 6: 0 Traffic + BGP Clear (앱 트래픽 없음이라는 다른 shift)

Dataset 3과 동일 조건의 독립 반복이 아니다. Primary 완료 후에만, 다른 traffic condition에서도 전략 비교가 흔들리지 않는지 본다.

---

# 8. 연구 범위 표현

## 사용하지 말아야 할 표현

- 서로 다른 네트워크 간 전이
- 새로운 사이트로 모델 이전
- 다른 실제망 간 Transfer
- 다양한 통신망 전체에 일반화
- 본 실험을 Concept Drift 연구라고 부르는 것
- Dataset 1→3을 시간 순 트래픽 성장 기록처럼 서술하는 것
- Dataset 5/6을 Dataset 3의 replication이라고 부르는 것

## 사용할 표현

> **동일 네트워크 패브릭에서 트래픽 운용 조건이 달라졌을 때의 정상 패턴 Drift 및 모델 적응**

또는

> **동일 네트워크의 서로 다른 Traffic Regime 간 이상탐지 모델 유지보수 전략 비교**
>
> **입력 Telemetry distribution shift / traffic regime shift**

---

# 9. 정상 구간 정의

Target에서 이벤트가 아닌 모든 구간을 정상으로 사용하지 않는다.

BGP Clear 이후에는 재수렴 영향이 남아 있을 수 있으므로 Primary 정상 데이터는 다음만 사용한다.

```text
Target Clean Normal
= 첫 Event 이전 Clean Baseline
```

### 금지

```text
Event
↓
잠깐 정상처럼 보이는 구간
↓
Train에 포함
```

### 이유

BGP 재수렴 잔여 영향이 정상 데이터에 섞일 수 있다.

---

# 10. 데이터 단위

Cisco telemetry는 long-format이며 동일 timestamp에 여러 장비/인터페이스/EncodingPath 행이 존재할 수 있다.

따라서 데이터량은 단순 row 수로 비교하지 않는다.

## Data Budget 단위

> **고유 Timestamp / Clean Window 시간**

예:

```text
Target 10%
= Target clean window 시간축 앞 10%
```

### 금지

```text
전체 row 중 random 10%
```

같은 시각의 일부 인터페이스만 더 포함하는 것은 “더 많은 운용 시간”을 의미하지 않는다.

---

# 11. Observation Unit

Inventory 이후 아래 두 방식 중 실제 schema에 맞는 방법을 선택한다.

## Option A. Device × Timestamp

```text
timestamp + node
→ 하나의 sample
```

권장 우선 검토 방식.

장점:

- 5분 clean window에서도 sample 수 확보 가능
- 장비별 정상 패턴 분석 가능

## Option B. Fabric × Timestamp

```text
timestamp
→ fabric 전체 feature 집계
```

장점:

- 전체 패브릭 상태를 하나의 시점으로 표현

단점:

- Dataset 3의 5분 clean baseline에서는 sample 수가 매우 적을 수 있음

## Device×Timestamp 집계 규칙

Option A를 쓰면 같은 `node + timestamp`의 interface-level 행을 장비 시각 하나로 접는다.

Inventory에서 각 EncodingPath가 interface-level인지 device-level인지 확인한 뒤 규칙을 고정한다.

| Feature 종류 | 집계 |
|---|---|
| Traffic bytes / data-rate / packet-rate | sum |
| Error / drop count | sum |
| Utilization / load | mean 또는 max (둘 중 inventory 후 하나 고정) |
| Route count / neighbor count | 이미 장비 단위면 그대로 |

집계 전에 ID 컬럼(`Producer`, `node-name`, `interface-name`)은 모델 입력이 아니라 정렬·집계 키로만 쓴다.

---

# 12. Feature Family

Primary는 **Family A + Level-based만** 사용한다. Family B와 Change-based는 Secondary다.

## Family A — Data-plane (Primary)

Traffic load 변화에 민감한 feature.

예:

- input/output data-rate
- packet-rate
- byte count / change
- drops
- errors
- traffic variability
- packet variability

## Family B — Data-plane + Control-plane (Secondary)

Family A만으로 BGP Clear 탐지가 너무 약할 때 한 번만 추가한다.

Family A +

- active-routes-count
- routes-counts
- established-neighbor counts
- VRF neighbor counts
- BGP 관련 continuous counters

## Shortcut Feature

Primary에서는 제외한다.

- interface UP/DOWN
- neighbor_state
- event flag
- fault code
- 직접적인 label 컬럼

---

# 13. Absolute vs Change-based Representation

Primary는 Level-based만 돌린다. Change-based는 Level-based KEEP의 FPR이 매우 높아 “절대 부하 수준에 과민한가?”가 남을 때만 Secondary로 추가한다.

## Level-based (Primary)

```text
raw/normalized absolute traffic level
```

목적:

> 실제 Traffic Regime 변화 자체가 모델에 어떤 Domain Shift를 만드는지 확인

## Change-based (Secondary)

예:

```text
delta
percent change
rolling z-score
rolling std
```

목적:

> 부하 크기 자체를 일부 제거하면 모델의 generalization이 개선되는지 확인

---

# 14. Drift Detection

## 14.1 Input Distribution Drift

Reference:

```text
Dataset 1 clean normal
```

Recent / Target:

```text
Dataset 3 pre-event clean normal
```

Feature별 Drift Score를 계산한다.

권장 지표:

- Wasserstein Distance
- Jensen-Shannon Divergence
- KS statistic

Primary에서는 구현이 단순한 **Wasserstein Distance**를 우선 검토한다.

## 14.2 Reference Normal Drift Range

Drift threshold를 임의로 정하지 않는다.

Dataset 1 정상 1시간을 여러 window로 나누고 정상끼리의 Drift를 계산한다.

예:

```text
Reference Window 1 vs Window 2
Reference Window 1 vs Window 3
...
```

이 분포를 이용해 정상적인 drift 범위를 추정한다.

예:

```text
Normal-Normal Drift
median
95 percentile
99 percentile
```

---

# 15. Model Drift

Input Drift만 보지 않는다.

Autoencoder의 정상 Reconstruction Error 분포 변화도 같이 본다.

Reference:

```text
Dataset 1 validation reconstruction error
```

Target:

```text
Dataset 3 clean normal reconstruction error
```

측정:

- 평균 Reconstruction Error
- median
- percentile
- error distribution distance
- FPR

---

# 16. Drift와 FPR 관계

Dataset 3 clean window가 약 5분이므로 **여러 Target window로 Drift vs FPR 곡선을 그리는 것은 Primary가 아니다.** Dataset 1 정상 구간 내부의 Normal–Normal Drift를 기준 범위로 쓰고, Dataset 1 vs Dataset 3은 Family A 기준 점 비교로 충분하다.

시간이 남으면 다음 관계를 참고로만 본다.

```text
Input Drift Score ↑
        ↓
Normal Reconstruction Error ↑
        ↓
False Positive Rate ↑ ?
```

산출물 예:

| Window | Drift Score | Recon Error | FPR |
|---|---:|---:|---:|
| A | ? | ? | ? |
| B | ? | ? | ? |
| C | ? | ? | ? |

이 관계가 확인되면 Drift Score를 단순 데이터 차이가 아니라 **모델 유지보수 판단 근거**로 해석할 수 있다.

---

# 17. 모델

Small MLP Autoencoder.

```text
input_dim
→ 32
→ 8
→ 32
→ input_dim
```

Autoencoder 선택 이유:

1. 정상 데이터만으로 학습 가능
2. Reconstruction Error를 anomaly score로 사용 가능
3. Source weight를 Target에서 그대로 Fine-tuning 가능
4. 모델 비교가 아니라 유지보수 전략 비교에 집중 가능

---

# 18. 비교 전략

## Strategy 0. KEEP

```text
Dataset 1
→ Source AE 학습
→ 그대로 Dataset 3에 적용
```

목적:

> 새로운 운용 조건에서 기존 모델 성능이 얼마나 저하되는지 확인

## Strategy 1. RECALIBRATE

```text
Source AE weight 유지
+
Target clean normal 일부
→ Threshold만 재설정
```

모델 weight는 변경하지 않는다.

목적:

> 가장 싼 유지보수 전략으로 충분한지 확인

## Strategy 2. ADAPT

```text
Source AE
+
동일 Target clean normal
→ Full Fine-tuning
```

권장:

```text
source lr ≈ 1e-3
fine-tuning lr ≈ 1e-4
```

## Strategy 3. SCRATCH-SMALL

```text
Random initialization
+
ADAPT와 동일 Target clean normal
→ 처음부터 학습
```

핵심 비교:

```text
ADAPT
vs
SCRATCH-SMALL
```

Target 데이터량은 반드시 동일해야 한다.

## Strategy 4. TARGET-FULL

Target에서 확보 가능한 정상 데이터를 충분히 사용하여 Scratch 학습.

Reference model로 사용한다.

---

# 19. Fair Comparison

ADAPT와 SCRATCH-SMALL은 다음이 같아야 한다.

- Target timestamp subset
- Feature set
- Architecture
- Optimizer
- Batch size
- Training budget
- Validation 방식
- Threshold calibration 방식
- Test set

차이:

```text
ADAPT
= Source pretrained initialization

SCRATCH
= Random initialization
```

---

# 20. Target Data Budget

Dataset 3 clean window는 약 5분이다. Primary에서 5% / 10% / 25% / 100% 곡선은 만들지 않는다.

```text
Target-small  = clean window 앞쪽 N분
Clean Test    = 이후 clean 구간
Target-full   = 확보 가능한 clean normal 전부
```

N은 inventory 후 결정한다. 예: timestamp가 충분하면 0~2분 small, 2~5분 test. 부족하면 train/test에 남길 분을 함께 조정한다.

운영 표현은 “Target 데이터의 X%”가 아니라 **새로운 정상 데이터 N분**이다.

ADAPT와 SCRATCH와 RECALIBRATE는 **같은 N분**을 사용한다.

---

# 21. Threshold

기본 방식:

```text
threshold
= validation normal reconstruction error의 99th percentile
```

### KEEP

Source validation threshold 사용.

### RECALIBRATE

Target budget 안의 clean normal calibration data로 threshold만 재설정.

### ADAPT / SCRATCH

동일 Target budget 내부 validation/calibration set으로 threshold 설정.

### 금지

Test 결과를 본 뒤 threshold 수정.

---

# 22. Event Metric

Target Event Test:

```text
BGP Clear ground-truth window
```

## Event Detection Rate

이벤트 window 내 anomaly alert가 최소 지속 조건을 만족하면 detected.

예:

```text
2 consecutive samples
```

단일 spike는 detection으로 보지 않는다.

## Detection Delay

```text
최초 지속조건 충족 Alert 시각
-
Event Start
```

미탐지는 NaN 처리하고 miss 개수를 별도 보고한다.

## FPR

```text
Target Clean Normal Test
```

에서만 계산한다.

Event 이후 residual 구간은 FPR에 포함하지 않는다.

---

# 23. 핵심 평가 지표

전략 비교는 FPR만 보고 끝내지 않는다. Threshold를 극단적으로 올리면 FPR은 0에 가까워질 수 있으나 Detection이 무너지면 실패다.

Primary:

1. **Input Drift Score**
2. **Reconstruction Error Drift**
3. **False Positive Rate** (Target Clean Normal Test)
4. **Event Detection Rate** (BGP Clear window)
5. **Detection Delay**

FPR과 Detection은 항상 같은 표에 나란히 둔다.

Secondary:

- Precision
- Recall
- F1
- PR-AUC / AUROC

---

# 24. 모델 Lifecycle Decision Framework

최종 결과를 다음 운영 프레임으로 해석한다.

```text
                Recent Telemetry
                       │
                       ▼
              Distribution Drift
                       │
          ┌────────────┴────────────┐
          │                         │
       Drift 작음                 Drift 큼
          │                         │
        KEEP                기존 모델 성능 확인
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                      FPR 유지               FPR 증가
                          │                     │
                        KEEP              유지보수 필요
                                                │
                           ┌────────────────────┼───────────────────┐
                           │                    │                   │
                     RECALIBRATE             ADAPT              RETRAIN
                     Threshold               Fine-tune           Scratch
```

중요:

> Drift가 감지됐다고 자동으로 Fine-tuning하지 않는다.

장애 구간을 새로운 정상으로 학습하면 안 되기 때문이다.

---

# 25. Adaptation Guardrail

새로운 데이터를 adaptation에 사용하려면 최소한 다음 조건을 만족해야 한다.

- Ground-truth event window 아님
- BGP/route/interface state 이상 없음
- 일정 시간 이상 지속되는 새로운 normal regime

이 CSV에 운용 alarm 컬럼이 없을 수 있다. 실험에서 Confirmed Normal은 **첫 이벤트 이전 clean window**로 정의한다.

즉:

> **Confirmed Normal Data Only = Target pre-event clean**

---

# 26. 결과 해석 규칙

숫자는 예시이며 목표치가 아니다. **FPR과 Detection을 항상 같이** 본다.

## Case A. RECALIBRATE로 충분

예:

```text
KEEP         FPR 43%   Detection 96%
RECALIBRATE  FPR  8%   Detection 92%
ADAPT        FPR  7%   Detection 90%
```

해석:

> 오탐을 줄이면서 탐지도 유지했다. 이 traffic regime 변화에서는 weight 변경보다 threshold 재보정이 더 단순하고 효율적이었다.

## Case A-fail. RECALIBRATE가 이상을 같이 누름

예:

```text
KEEP         FPR 43%   Detection 96%
RECALIBRATE  FPR  7%   Detection 48%
```

해석:

> Threshold 보정으로 오탐은 줄였지만 BGP 이상까지 함께 눌렀다. 유지보수 비용은 낮아도 탐지 성능이 부족하므로 ADAPT 또는 SCRATCH를 본다.

## Case B. ADAPT가 유리

예:

```text
KEEP         FPR 40%   Detection 100%
RECALIBRATE  FPR  8%   Detection  65%
SCRATCH      FPR 12%   Detection  86%
ADAPT        FPR  7%   Detection  90%
```

해석:

> 기존 정상 representation이 새 운용 조건 적응에 유용했고, 동일 N분의 Target 데이터에서 Fine-tuning이 scratch보다 FPR·탐지를 함께 잘 맞췄다.

## Case C. SCRATCH가 유리

예:

```text
ADAPT    FPR 17%   Detection 70%
SCRATCH  FPR  8%   Detection 86%
```

해석:

> Source와 Target 간 정상 분포 차이가 커 Negative Transfer가 있었거나, 기존 representation을 재사용하는 것보다 새 baseline이 적합했다.

---

# 27. 중요한 결론 원칙

본 프로젝트는 다음을 증명하려는 것이 아니다.

> “Fine-tuning이 항상 최고다.”

오히려 핵심은:

> **“환경 변화 정도와 모델 성능 저하를 측정하고, 가장 단순한 대응부터 단계적으로 비교해야 한다.”**

이다.

현업 관점에서는 가장 복잡한 AI보다 **가장 적은 비용으로 안정성을 회복하는 방법**이 더 중요하다.

---

# 28. Primary 최소 실행 세트

2일 기준 반드시 끝낼 것.

## Step 1

Dataset 1 / 3 Inventory

- 파일, row 수, unique timestamp, sampling interval
- node / interface / EncodingPath가 device-level인지 interface-level인지
- Device×Timestamp 집계 규칙
- common columns, clean window 길이, event timestamp
- Target-small N분과 clean test 구간

## Step 2

Primary feature 고정

- Family A + Level-based
- shortcut 제외
- Family B / Change-based는 이 단계에서 구현하지 않음

## Step 3

Source AE 학습

```text
Dataset 1
500 Gbps Normal
```

## Step 4

Drift 측정

```text
Dataset 1 Normal
vs
Dataset 3 Clean Normal
```

- Feature Drift
- Reconstruction Error Drift

## Step 5

KEEP 평가

- Target clean FPR
- Event Detection
- Delay

## Step 6

같은 Target Small Budget으로 비교

```text
RECALIBRATE
vs
ADAPT
vs
SCRATCH-SMALL
```

## Step 7

TARGET-FULL Reference

## Step 8

Seed 3회

모든 전략에서 FPR과 Detection을 같이 기록한다. Seed 3회는 ADAPT vs SCRATCH를 우선한다.

---

# 29. Secondary 실험

Primary가 끝난 후에만 수행. 처음부터 넣지 않는다.

1. Family B 한 세트 (Family A로 BGP 탐지가 약할 때)
2. Change-based (Level-based KEEP FPR이 매우 높을 때)
3. Dataset 5 / 6 additional traffic-condition validation (replication 아님)
4. Port event Dataset 4 / 9 / 10
5. Layer freeze
6. Dataset 12 — schema가 다르면 하지 않음

---

# 30. 하지 말 것

Primary 완료 전 다음을 하지 않는다.

- Containerlab
- FRR
- LLM Agent
- RCA
- 자동 복구
- Grafana
- Kafka
- Multi-Agent
- LSTM
- Transformer
- 여러 anomaly model 경진대회
- Dataset 11/12 schema 강제 결합

---

# 31. Leakage Checklist

다음 하나라도 위반하면 실험 무효.

- [ ] Event window가 train/adapt/scaler/threshold에 들어가지 않았다.
- [ ] Normal test를 scaler fit에 사용하지 않았다.
- [ ] ADAPT와 SCRATCH가 동일 Target timestamp subset을 사용한다.
- [ ] Budget 밖 Target data로 threshold를 만들지 않았다.
- [ ] Row random split을 하지 않았다.
- [ ] Source-only는 Target statistics를 사용하지 않았다.
- [ ] Producer / node-name / interface-name은 입력 feature가 아니다.
- [ ] ID column은 정렬/집계 key로만 사용한다.
- [ ] 대부분 상수인 컬럼을 제거했다.
- [ ] missing이 과도한 컬럼을 제거했다.
- [ ] Feature list를 artifact로 저장했다.
- [ ] Split timestamp를 저장했다.
- [ ] Seed를 기록했다.
- [ ] Hyperparameter를 저장했다.
- [ ] Test를 본 뒤 feature/threshold/epoch를 바꾸지 않았다.

---

# 32. 핵심 Figure

## Figure 1. Distribution Drift

```text
Dataset 1 Normal
vs
Dataset 3 Clean Normal
```

Feature별 Drift Score.

## Figure 2. Reconstruction Error Distribution

```text
Source Normal
vs
Target Clean Normal
```

기존 모델이 Target 정상 상태를 얼마나 낯설게 보는지 표현.

## Figure 3. Maintenance Strategy Comparison

```text
KEEP
RECALIBRATE
ADAPT
SCRATCH-SMALL
TARGET-FULL
```

FPR / Detection Rate 비교.

## Figure 4. Event Timeline

BGP event timestamp와 anomaly score를 같은 그래프에 표시.

## Figure 5. Drift vs FPR

Primary 아님. Dataset 3 clean이 짧아 점이 거의 안 나온다. 시간이 남으면 Dataset 1 내부 window만 참고.

---

# 33. 최종 결과표 예시

| Strategy | Model Weight | Target Normal Used | Threshold | FPR | Event Detection | Delay |
|---|---|---:|---|---:|---:|---:|
| KEEP | Source | 0 | Source | ? | ? | ? |
| RECALIBRATE | Source | N분 | Target | ? | ? | ? |
| ADAPT | Source→FT | 동일 N분 | Target | ? | ? | ? |
| SCRATCH | Random | 동일 N분 | Target | ? | ? | ? |
| TARGET-FULL | Random | available full | Target | ? | ? | ? |

---

# 34. 최종 README 스토리

## Problem

> 500 Gbps 환경에서 정상 데이터를 학습한 이상탐지 모델을 운영 중이라고 가정했다. 이후 동일 네트워크의 정상 트래픽이 1 Tbps 이상으로 증가했을 때 기존 모델이 새로운 정상 상태를 이상으로 오인할 수 있다. 사용 데이터는 점진적 성장 기록이 아니라, 동일 lab fabric의 서로 다른 traffic operating point에서 수집한 독립 run을 proxy로 쓴다.

## Question

> **환경 변화가 발생했을 때 기존 모델을 그대로 유지해야 할까, Threshold만 보정해야 할까, Fine-tuning해야 할까, 아니면 새로 학습해야 할까?**

## Experiment

Cisco Network Telemetry를 이용해:

```text
500 Gbps Normal
→ Source Model

1.3 Tbps Clean Normal
→ Drift / FPR 측정

BGP Clear
→ Detection 평가
```

후 다음 전략을 비교했다.

```text
KEEP
RECALIBRATE
ADAPT
SCRATCH
RETRAIN
```

---

# 35. KT 지원 프로젝트 연결 문장

> KT의 실제 운영 사례에서 네트워크 환경 변화에 따라 모델의 성능을 지속적으로 관리하고 재학습 여부를 판단한다는 문제의식을 확인했다. 본 실험에서는 그중 트래픽 운용 조건 변화에 따른 입력 Telemetry distribution shift를 제한적으로 다룬다. Cisco 공개 Telemetry의 500 Gbps 정상 run과 1.3 Tbps 정상 run을 서로 다른 traffic operating point의 proxy로 두고, KEEP·RECALIBRATE·ADAPT·SCRATCH를 False Positive와 BGP 탐지와 함께 비교했다.

---

# 36. 최종 연구 제목 후보

## 후보 1 — 가장 권장

**트래픽 운용 조건 변화에 따른 네트워크 이상탐지 모델의 Drift와 유지보수 전략 비교**

## 후보 2

**Network Telemetry Distribution Drift 기반 이상탐지 모델 적응 전략 비교**

## 후보 3

**트래픽 부하 변화 환경에서 정상 기반 이상탐지 모델의 재보정·적응·재학습 비교**

---

# 37. 최종 연구 질문

> **“기존 traffic regime에서 학습한 정상 기반 이상탐지 모델이 새로운 traffic regime에서 성능 저하를 보였을 때, 모델을 다시 학습하기 전에 어느 수준의 유지보수가 필요한가?”**

그리고 실제 실험에서 가장 중요한 하위 질문은 다음이다.

> **“동일한 Target 정상 데이터가 주어졌을 때 Threshold 재보정, Fine-tuning, Scratch Retraining 중 어떤 전략이 False Positive를 줄이면서 BGP 이상탐지 성능을 가장 잘 유지하는가?”**

---

# 38. 전체 작업 순서

1. Cisco Dataset 1 / 3 원본 파일 및 README 확인
2. Dataset inventory 작성
3. Unique timestamp / clean baseline 길이 확인
4. Observation unit 결정
5. Source / Target common schema 검증
6. Family A + Level-based 및 shortcut 제외 (Family B / Change는 Secondary)
7. EncodingPath 레벨 확인 후 Device×Timestamp 집계 규칙 고정
8. Target clean/event window 확정, N분 결정
9. Timestamp 기반 split 생성 (비율 곡선 없음)
10. Leakage checklist 자동 검증
11. Dataset 1 정상 내부 Normal-Normal Drift 계산
12. Dataset 1 → Dataset 3 Input Drift 계산
13. Source Autoencoder 학습
14. Source validation threshold 생성
15. Dataset 3 clean normal에 Source-only 적용
16. Reconstruction Error Drift 측정
17. Source-only FPR 계산
18. Target event에서 BGP Detection 평가
19. Target Small Budget 결정
20. RECALIBRATE 실행
21. ADAPT Fine-tuning 실행
22. 동일 Target subset으로 SCRATCH-SMALL 실행
23. TARGET-FULL reference 학습
24. 모든 전략을 동일 Target Test에 평가
25. Seed 3회 반복
26. 결과 mean ± std 계산
27. Distribution Drift Figure 생성
28. Reconstruction Error Figure 생성
29. Strategy Comparison Figure 생성
30. BGP Event Timeline Figure 생성
31. Drift vs FPR 다중 window는 Primary에서 생략
32. 결과가 예상과 달라도 원인 분석 (FPR과 Detection을 같이)
33. Dataset 5 / 6은 replication이 아님. Secondary robustness check 여부만 판단
34. 최종 README 작성
35. KT 기술 사례와 본 실험의 차이 명시
36. 한계 및 실무 적용 범위 작성
37. 재현용 requirements / config / artifact 저장
38. 면접용 1분 프로젝트 설명 작성

---

# 39. Cursor 구현 원칙

1. 코드를 먼저 짜지 말고 Dataset Inventory부터 완료한다.
2. 실제 컬럼을 확인하지 않은 상태에서 feature 이름을 가정하지 않는다.
3. Schema가 다르면 억지로 맞추지 않는다.
4. 모든 split과 budget은 timestamp 기준으로 관리한다.
5. Event window는 어떤 정상 학습에도 사용하지 않는다.
6. Target test는 scaler / threshold / training에 사용하지 않는다.
7. RECALIBRATE는 weight를 바꾸지 않는다.
8. ADAPT와 SCRATCH는 정확히 같은 Target normal subset을 사용한다.
9. Fine-tuning 결과가 안 좋다고 learning rate를 test 결과에 맞춰 반복 수정하지 않는다.
10. 결과가 가설과 다르면 Negative Transfer 또는 Recalibration sufficiency로 해석한다.
11. Primary 완료 전 모델 종류를 늘리지 않는다.
12. 모든 결과를 CSV/JSON artifact로 저장한다.
13. Figure는 저장된 result artifact만 읽어 생성한다.
14. README 숫자는 artifact 결과를 자동 참조하거나 최종 결과와 반드시 대조한다.
15. “새로운 네트워크 전이”, “Concept Drift 실험”, “Dataset 1→3이 시간 순 성장”이라는 표현을 사용하지 않는다.
16. 모든 전략 결과표에 FPR과 Event Detection을 같이 넣는다.
17. Primary는 Family A + Level-based와 KEEP→RECALIBRATE→ADAPT→SCRATCH→TARGET-FULL만 구현한다.

---

# 40. 연구의 최종 톤

성공하더라도 다음처럼 쓰지 않는다.

> “본 모델이 실제 통신망에 적용 가능함을 증명했다.”

대신:

> **“동일 네트워크 패브릭에서 Traffic Regime이 변화한 제한된 공개 실험 조건에서, Telemetry Drift와 기존 이상탐지 모델의 성능 저하를 확인하고, 유지·재보정·Fine-tuning·재학습 전략의 상대적 효과를 비교했다.”**

그리고 결과에 따라:

> **“환경 변화가 발생했다고 항상 새 모델이 필요한 것은 아니며, Drift 크기와 기존 모델의 성능 저하 수준에 따라 가장 낮은 비용의 대응부터 검토할 필요가 있다.”**

라는 운용 관점의 결론으로 마무리한다.
