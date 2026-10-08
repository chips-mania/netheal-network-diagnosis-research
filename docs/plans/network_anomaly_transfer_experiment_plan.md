# 네트워크 이상탐지 전이학습 실험 진행계획서
## Cisco Network Telemetry 내부 Domain 간 Normal-pattern Transfer 검증

> **핵심 질문**  
> 새로운 네트워크 조건이 추가될 때마다 이상탐지 모델을 처음부터 다시 학습해야 할까?  
> 기존 네트워크 조건에서 학습한 정상 패턴을 새로운 조건의 소량 정상 데이터로 적응시키면, 처음부터 학습하는 것보다 더 적은 데이터로 유사한 성능에 도달할 수 있을까?

---

# Part I. 논문형식 진행계획서

## 1. 연구 제목

### 국문
**서로 다른 네트워크 조건 간 정상 패턴 전이가 이상탐지 모델의 데이터 효율성에 미치는 영향**

### 영문
**Data-Efficient Adaptation of Network Anomaly Detection via Normal-Pattern Transfer Across Network Conditions**

---

## 2. 연구 배경

네트워크 이상탐지에서는 실제 장애 데이터가 희소하고, 모든 장애 유형을 사전에 정의하기 어렵기 때문에 정상 상태의 패턴을 학습한 뒤 정상 분포에서 벗어난 상태를 이상으로 판단하는 방식이 유용하다.

그러나 실제 네트워크는 트래픽 부하, 장비 상태, 토폴로지, 서비스 구성 등에 따라 정상 상태 자체가 서로 다른 분포를 가질 수 있다. 따라서 한 환경에서 학습한 이상탐지 모델을 다른 환경에 그대로 적용하면, 실제 장애가 아님에도 환경 차이를 이상으로 판단하는 Domain Shift 문제가 발생할 수 있다.

반대로 기존 환경에서 이미 학습한 정상 상태 표현을 새로운 환경에서도 재사용할 수 있다면, 새로운 네트워크 조건이 추가될 때마다 충분한 정상 데이터를 처음부터 다시 수집하고 모델을 재학습해야 하는 부담을 줄일 수 있다.

본 연구는 다음의 실무적 의문에서 출발한다.

> **“새로운 네트워크 조건이 추가될 때마다 이상탐지 모델을 처음부터 다시 학습해야 하는가?”**

---

## 3. 연구 목적

본 연구의 목적은 **Cisco Network Telemetry Dataset 내부의 서로 다른 네트워크 조건을 Source Domain과 Target Domain으로 구분하고, Source 정상 데이터로 사전학습한 이상탐지 모델이 Target의 소량 정상 데이터만으로 빠르게 적응할 수 있는지 검증하는 것**이다.

특히 다음 비교를 핵심으로 한다.

- **Target-small**: Target 정상 데이터를 소량 사용해 랜덤 초기화 상태에서 처음부터 학습
- **Adapted**: Source 정상 데이터로 사전학습한 뒤, Target-small과 동일한 양의 Target 정상 데이터로 fine-tuning

두 모델에 제공되는 Target 데이터의 양과 학습 조건을 동일하게 통제하여, 성능 차이가 Source 사전학습의 효과인지 검증한다.

---

## 4. 핵심 연구 질문

### Main Research Question

> **Source 네트워크 조건에서 정상 패턴을 사전학습한 모델은 Target 네트워크의 제한된 정상 데이터만으로, Target에서 처음부터 학습한 모델보다 효율적으로 적응할 수 있는가?**

### 세부 연구 질문

**RQ1. Domain Shift**  
Source-only 모델을 Target에 그대로 적용하면 Target 정상 상태를 이상으로 오인하는가?

**RQ2. Adaptation**  
Target의 소량 정상 데이터만으로 Source-pretrained 모델을 fine-tuning하면 Target 정상 상태에 대한 False Positive가 감소하는가?

**RQ3. Transfer Benefit**  
동일한 양의 Target 정상 데이터를 사용했을 때 Adapted 모델이 Target-small 모델보다 높은 이상탐지 성능을 보이는가?

**RQ4. Data Efficiency**  
Source pretraining을 사용하면 Target-full 모델과 유사한 성능에 도달하는 데 필요한 Target 정상 데이터량을 줄일 수 있는가?

---

## 5. 연구 가설

### H1. Domain Shift 가설
Source-only 모델은 Source와 Target의 네트워크 조건 차이로 인해 Target 정상 구간에서도 비교적 높은 Anomaly Score를 보일 것이다.

### H2. Adaptation 가설
Source-pretrained 모델에 Target 정상 데이터를 소량 제공하면 Target 정상 분포에 적응하면서 False Positive가 감소할 것이다.

### H3. Pretraining Benefit 가설
동일한 Target 데이터 예산에서 Adapted 모델은 Target-small 모델보다 높은 탐지 성능을 보일 것이다.

### H4. Diminishing Return 가설
Target 정상 데이터가 충분히 많아질수록 Source pretraining의 이점은 감소할 것이다.

---

## 6. 데이터셋

### 6.1 데이터 출처

Cisco Innovation Edge Network Telemetry Dataset

- Repository: https://github.com/cisco-ie/telemetry
- Dataset 11: https://github.com/cisco-ie/telemetry/tree/master/11

이 데이터는 네트워크 이상탐지 알고리즘의 개발 및 평가를 위해 공개된 Router Telemetry 데이터셋으로, 여러 traffic condition과 BGP/Port/BFD 관련 state-change scenario가 포함되어 있다.

---

## 7. Source / Target Domain 선정 전략

### 7.1 1순위: Cisco Dataset 내부 Domain 분리

Containerlab과 Cisco 간 전이는 장비 구조와 telemetry 체계 차이가 너무 커 Domain Gap 자체가 실험 결과를 지배할 수 있다.

따라서 본 연구에서는 먼저 **Cisco 데이터셋 내부에서 서로 다른 네트워크 조건을 Source와 Target으로 나누는 방식**을 우선한다.

### 후보 조합

- **Source 후보**: Dataset 1 — 500 Gbps traffic, anomaly 없는 정상 baseline
- **Target 후보**: Dataset 2 — 1 Tbps traffic, BGP Clear event 포함
- **보조 Target 후보**: Dataset 6 — 0 traffic, BGP Clear event 포함

> 주의: 위 조합은 후보이며, 실제 실험 시작 전에 Dataset 1/2/6의 schema, topology, sampling interval, telemetry column 호환성을 반드시 확인한다.

### 7.2 Go / No-Go 기준

다음 조건이 충족되면 Cisco 내부 Domain 실험으로 진행한다.

1. Source와 Target에서 동일한 telemetry schema 또는 충분히 매핑 가능한 공통 feature가 존재한다.
2. Target에서 event 이전 또는 별도 구간으로 정상 데이터를 확보할 수 있다.
3. Event timestamp 또는 ground truth가 존재한다.
4. Source와 Target 간 차이가 traffic load 등 해석 가능한 조건 차이로 설명 가능하다.

위 조건을 만족하지 못하면 **Fallback으로 Containerlab/FRR Source → Cisco Target 실험**으로 전환한다.

---

## 8. Feature 설계

### 8.1 기본 원칙

Source와 Target에서 의미가 동일한 feature만 사용한다.

가능하면 raw 절대값보다 **변화 기반 feature**를 우선한다.

예시:

- `rx_traffic_change`
- `tx_traffic_change`
- `rx_packet_change`
- `tx_packet_change`
- `route_count_change`
- `error_counter_change`
- `traffic_variability`
- `packet_variability`

### 8.2 변화 기반 Feature 예시

- 직전 시점 대비 차이
- percent change
- rolling mean 대비 편차
- rolling z-score
- rolling standard deviation

예:

```text
rx_traffic_change = (x_t - x_(t-1)) / (|x_(t-1)| + eps)
route_count_change = route_t - route_(t-1)
rolling_z = (x_t - rolling_mean) / rolling_std
```

### 8.3 Shortcut Feature 제외

Primary Experiment에서는 장애 정답에 가까운 explicit state feature를 제외한다.

예:

- `interface_state = DOWN`
- `neighbor_state = DOWN`

이러한 feature를 넣으면 모델이 telemetry pattern을 학습한 것이 아니라 장애 상태값을 직접 읽은 것에 가까워질 수 있다.

따라서 Primary Experiment는 continuous telemetry 중심으로 진행한다.

### 8.4 Secondary Experiment

시간이 남을 경우 explicit state feature를 추가하여 성능 차이를 비교한다.

- Primary: continuous telemetry only
- Secondary: continuous telemetry + state indicators

---

## 9. 데이터 분할

Target 정상 데이터는 시간 순서대로 분리한다.

```text
Target Normal
├── Adapt/Train Pool
├── Validation / Threshold Calibration
└── Normal Test

Target Event
└── Event Test
```

### 기본 분할 예시

- Train/Adapt Pool: 70%
- Validation: 15%
- Normal Test: 15%

Event Test는 어떤 학습, fine-tuning, normalization, threshold 결정에도 사용하지 않는다.

---

## 10. Target Data Budget

Target 데이터 효율성을 보기 위해 Train/Adapt Pool의 일부만 사용한다.

### 최소 실험 구간

- 5%
- 10%
- 25%
- 100%

가능하면 1%를 추가하되, sample 수가 지나치게 적으면 제외한다.

### 공정성 원칙

Target-small과 Adapted는 각 budget에서 **동일한 Target sample**을 사용한다.

예:

```text
Target-small-10%
Random initialization
+ Target normal 10%

Adapted-10%
Source-pretrained initialization
+ 동일한 Target normal 10%
```

Scaler fitting, normalization parameter 계산, threshold calibration에 사용되는 Target 정보도 해당 budget 안에서만 사용한다.

---

## 11. Source Data Budget

Source 정상 데이터는 기본적으로 **Target-full Train Pool과 동일한 sample 수**로 맞춘다.

예:

```text
Source Train = N samples
Target-full Train = N samples
```

이를 통해 Source가 단순히 더 많은 데이터를 보았기 때문에 유리했다는 해석을 줄인다.

---

## 12. 모델

### 12.1 모델 종류

작은 MLP Autoencoder를 사용한다.

예시 구조:

```text
Input
↓
32
↓
8
↓
32
↓
Output
```

### 12.2 Autoencoder 선택 이유

본 연구의 목적은 복잡한 모델 성능 비교가 아니라 **Source pretraining이 Target adaptation에 도움이 되는지 검증하는 것**이다.

Autoencoder는 다음 이유로 적합하다.

1. 정상 데이터만으로 학습 가능
2. Reconstruction Error를 Anomaly Score로 사용 가능
3. Source에서 학습한 weight를 Target에서 그대로 fine-tuning 가능
4. 작은 feature 수에서도 구현 및 해석이 간단함

---

## 13. 비교 모델

### Model 1. Source-only

```text
Source Normal
→ Train
→ Target Test
```

- Target 데이터 사용량: 0
- 목적: Zero-shot transfer 및 Domain Shift 확인

### Model 2. Target-small

```text
Random Initialization
+ Target Normal k%
→ Train from Scratch
→ Target Test
```

- k ∈ {5, 10, 25}
- Small-data baseline

### Model 3. Adapted

```text
Source Normal
→ Pretraining
→ Target Normal k%
→ Fine-tuning
→ Target Test
```

- Target-small과 동일한 Target sample 사용
- 본 연구의 핵심 모델

### Model 4. Target-full

```text
Random Initialization
+ Target Normal 100%
→ Train
→ Target Test
```

- Target-specific reference model

---

## 14. Fine-tuning 방식

Adapted 모델은 Source에서 학습한 전체 weight를 초기값으로 사용한다.

### 권장 방식

- Source learning rate: 약 `1e-3`
- Target fine-tuning learning rate: 약 `1e-4`
- 전체 layer fine-tuning
- 동일 architecture 유지

처음에는 layer freeze를 사용하지 않는다.

시간이 남을 경우에만 보조 실험으로 partial freeze를 비교한다.

---

## 15. 공정한 비교 조건

Target-small과 Adapted는 다음을 동일하게 유지한다.

- Architecture
- Feature set
- Target sample
- Target data budget
- Optimizer
- Batch size
- Target training epoch
- Threshold calibration 방식
- Test set

주요 차이는 단 하나다.

```text
Target-small = Random Initialization
Adapted      = Source-pretrained Initialization
```

---

## 16. Anomaly Score 및 Threshold

Autoencoder의 Reconstruction Error를 Anomaly Score로 사용한다.

예:

```text
score_t = mean((x_t - x_hat_t)^2)
```

Threshold는 학습 데이터와 분리된 정상 validation 데이터에서 결정한다.

예:

```text
threshold = 99th percentile of normal validation scores
```

Source-only는 Source validation 기준 threshold를 사용하고, Target-small / Adapted / Target-full은 각자 허용된 Target budget 범위 내의 calibration set을 사용한다.

---

## 17. 평가 지표

### 17.1 False Positive Rate (FPR)

정상 Target sample 중 Anomaly로 잘못 판단한 비율.

Domain Shift 및 adaptation 효과를 확인하는 핵심 지표다.

### 17.2 Event Detection Rate

실제 state-change event 중 모델이 이상으로 탐지한 비율.

### 17.3 Detection Delay

실제 event timestamp와 최초 anomaly alert 사이의 시간 차이.

```text
Detection Delay
= First Alert Time - Event Time
```

### 17.4 Score Separation

Normal과 Event 구간의 Anomaly Score 분포 차이를 시각적으로 비교한다.

### 17.5 보조 지표

Ground truth 형태가 허용하는 경우:

- Precision
- Recall
- F1
- PR-AUC 또는 AUROC

단, 본 연구의 중심은 FPR / Event Detection / Detection Delay로 유지한다.

---

## 18. 반복 실험

단일 random seed 결과에 의존하지 않도록 가능하면 각 조건을 3회 반복한다.

예:

```text
seed = 42, 52, 62
```

최종 결과는 다음처럼 표현한다.

```text
FPR = 7.8 ± 1.4%
Detection Rate = 89.2 ± 2.1%
```

2일 내 시간이 부족하면 10% budget 조건에서라도 최소 3회 반복한다.

---

## 19. 핵심 결과표

```text
| Model        | Target Normal Used | FPR | Event Detection | Detection Delay |
|--------------|-------------------:|----:|----------------:|----------------:|
| Source-only  | 0%                 |  ?  |       ?         |       ?         |
| Target-small | 10%                |  ?  |       ?         |       ?         |
| Adapted      | 10%                |  ?  |       ?         |       ?         |
| Target-full  | 100%               |  ?  |       ?         |       ?         |
```

핵심 비교는 **Target-small vs Adapted**이다.

---

## 20. 핵심 산출물: Data Efficiency Curve

가로축:

```text
Target Normal Data %
```

세로축:

- FPR
- Event Detection Rate
- 필요 시 F1

같은 Target data 양에서 Scratch와 Pretrained curve를 함께 표시한다.

핵심 해석은 다음과 같다.

> **Pretrained curve가 Scratch curve보다 적은 Target 데이터 구간에서 우수한가?**

그렇다면 Source pretraining이 Target 데이터 요구량을 줄였다고 해석할 수 있다.

---

## 21. 예상 결과 및 해석

### Case A. Adapted > Target-small

가장 기대하는 결과.

해석:

> Source에서 학습한 정상 패턴이 Target low-data regime에서 유용한 초기 representation으로 작용했다.

특히 Target 데이터가 적을수록 차이가 크고, Target 데이터가 증가하면서 차이가 줄어든다면 H4까지 지지한다.

### Case B. Adapted ≈ Target-small

해석:

> Source pretraining이 Target 적응에 실질적인 이점을 제공하지 못했다.

새로운 환경에서는 소량 Target 데이터로 처음부터 학습하는 것만으로 충분했을 가능성이 있다.

### Case C. Adapted < Target-small

해석:

> Source와 Target 사이 Domain Gap으로 인해 Negative Transfer가 발생했다.

이 경우 Source pretraining 자체보다 Source/Target의 구조적 유사성, feature 설계, normalization 전략이 더 중요하다는 결론을 낼 수 있다.

---

## 22. 연구의 한계

1. 본 실험은 Cisco 공개 데이터셋에 포함된 제한된 traffic/state-change scenario를 사용한다.
2. state-change event는 실제 상용망에서 자연발생한 모든 장애를 대표하지 않는다.
3. Source/Target 간 공통 feature를 만들면서 일부 telemetry 정보가 손실될 수 있다.
4. Autoencoder 하나만 사용하므로 모델 구조별 성능 차이는 연구 범위에 포함하지 않는다.
5. 본 실험의 목적은 상용 도입 가능성 증명이 아니라 **normal-pattern transfer의 데이터 효율성 가능성을 검증하는 PoC**이다.

---

## 23. 연구 범위에서 제외하는 사항

본 실험에서는 다음을 의도적으로 제외한다.

- LLM Agent
- Root Cause Analysis
- Fault Classification
- 자동 복구
- Kafka
- Grafana
- 복잡한 Multi-Agent
- 여러 Deep Learning 모델 비교
- 실제 장비 CLI 자동 실행

이번 실험의 질문은 하나다.

> **“기존 네트워크 조건에서 배운 정상 패턴이 새로운 네트워크 조건의 정상 데이터 요구량을 줄이는가?”**

---

## 24. 예상 논문 결론 문장

성공적인 경우:

> Source 네트워크의 정상 Telemetry로 사전학습한 Autoencoder는 Target 정상 데이터가 제한된 조건에서 랜덤 초기화 모델보다 낮은 False Positive와 높은 event detection 성능을 보였다. Target 데이터가 증가할수록 두 모델의 성능 차이는 감소했으며, 이는 정상 패턴 사전학습이 특히 새로운 네트워크의 초기 적응 단계에서 데이터 요구량을 줄이는 데 활용될 가능성을 시사한다.

사전학습 효과가 없는 경우:

> 서로 다른 네트워크 조건 간 정상 패턴의 직접적인 전이는 유의미한 성능 개선으로 이어지지 않았다. 이는 이상탐지 모델의 전이 가능성이 단순한 정상 상태 사전학습보다 Source와 Target 간 telemetry 분포 및 feature 구조의 유사성에 크게 의존함을 보여준다.

---

# Part II. 전체 진행 순서

## 1. Repository 및 Dataset 확보

1. Cisco telemetry repository를 clone 또는 download한다.
2. Dataset 0, 1, 2, 6을 우선 확보한다.
3. 필요 시 Dataset 11도 참고용으로 확보한다.
4. 각 Dataset의 README / header / ground truth 파일을 확인한다.

---

## 2. Dataset 구조 조사

다음 내용을 표로 정리한다.

1. Dataset 번호
2. Traffic condition
3. Event 종류
4. Sampling interval
5. 파일 수
6. Feature / column 수
7. 동일 schema 여부
8. Ground truth 존재 여부
9. 정상 구간 확보 가능 여부

산출물:

```text
reports/dataset_inventory.md
```

---

## 3. Source / Target Go-NoGo 결정

우선 후보:

```text
Source = Dataset 1
Target = Dataset 2
```

다음을 확인한다.

1. 동일 또는 매핑 가능한 telemetry 구조인가?
2. Target에 정상 adaptation 구간이 존재하는가?
3. Target event timestamp가 존재하는가?
4. Source/Target 차이를 traffic condition 등으로 명확히 설명할 수 있는가?

### GO
조건을 만족하면 Cisco 내부 Domain 실험 진행.

### NO-GO
조건을 만족하지 못하면:

```text
Containerlab/FRR Source
→ Cisco Target
```

fallback 계획으로 전환한다.

---

## 4. 공통 Feature 후보 추출

Source와 Target에 동시에 존재하는 telemetry를 조사한다.

후보를 다음 기준으로 선별한다.

1. 양쪽 환경에서 의미가 동일함
2. missing 비율이 낮음
3. 대부분 상수인 feature가 아님
4. event 전후 변화 가능성이 있음
5. explicit label leakage를 발생시키지 않음

목표 feature 수:

```text
약 5~20개
```

Dataset schema가 충분히 동일하다면 더 많은 feature를 사용할 수 있다.

---

## 5. Shortcut Feature 분리

다음과 같은 feature는 Primary 실험에서 별도 목록으로 분리한다.

- Interface UP/DOWN
- Neighbor UP/DOWN
- Event flag
- 직접적인 fault code

파일:

```text
config/shortcut_features.yaml
```

Primary 실험에서는 제외한다.

---

## 6. Time Alignment 및 전처리

1. Source/Target sampling interval 확인
2. timestamp 정렬
3. 동일 window size로 resampling
4. missing value 처리
5. constant feature 제거
6. 변화량 feature 생성
7. rolling feature 생성
8. 최종 feature 순서 고정

최종 출력:

```text
data/processed/source_normal.parquet
data/processed/target_normal.parquet
data/processed/target_event.parquet
```

---

## 7. 데이터 누수 점검

반드시 확인한다.

1. Event 구간이 training에 포함되지 않았는가?
2. Normal Test가 scaler fit에 사용되지 않았는가?
3. Target-small과 Adapted가 동일한 Target sample을 사용하는가?
4. Target budget 밖의 데이터를 threshold 설정에 사용하지 않았는가?
5. 시간 순서를 무시한 random split로 future leakage가 발생하지 않았는가?

산출물:

```text
reports/data_leakage_check.md
```

---

## 8. Target Train / Validation / Test 분리

Target Normal을 시간 순서대로:

```text
70% Train/Adapt Pool
15% Validation
15% Normal Test
```

로 분리한다.

Target Event는 별도의 Event Test로 유지한다.

Split index 또는 timestamp를 파일로 저장한다.

```text
config/splits.json
```

---

## 9. Target Budget 생성

Train/Adapt Pool에서 다음 subset을 생성한다.

```text
5%
10%
25%
100%
```

각 subset은 nested subset으로 구성한다.

예:

```text
5% ⊂ 10% ⊂ 25% ⊂ 100%
```

이렇게 해야 데이터량 증가에 따른 학습곡선을 공정하게 비교할 수 있다.

---

## 10. Source Data Budget 통제

Source 학습 sample 수를 Target-full Train Pool과 동일하게 맞춘다.

필요하면 Source 데이터를 시간 순 또는 random downsampling한다.

random downsampling을 사용할 경우 seed를 고정한다.

---

## 11. Baseline Autoencoder 구현

구현 파일 예시:

```text
src/models/autoencoder.py
```

기본 구조:

```text
input_dim
→ 32
→ 8
→ 32
→ input_dim
```

구현 사항:

1. train()
2. predict reconstruction()
3. anomaly_score()
4. save/load weights
5. reproducible seed

---

## 12. Source Pretraining

Source Normal 데이터로 Autoencoder를 학습한다.

저장:

```text
artifacts/source_model.pt
```

함께 저장:

- feature list
- scaler
- hyperparameters
- training loss
- seed

---

## 13. Source-only 평가

Source 모델을 Target Test에 바로 적용한다.

목적:

> Domain Shift가 실제로 존재하는지 확인.

확인 항목:

- Target Normal FPR
- Target Event score
- Normal/Event score distribution

이 단계에서 Source-only가 Target 정상까지 대부분 anomaly로 판단하더라도 실패가 아니다.

---

## 14. Target-small 모델 학습

각 Target budget에 대해 random initialization으로 처음부터 학습한다.

예:

```text
target_small_5
target_small_10
target_small_25
```

각 모델의 architecture와 training condition은 Adapted 모델과 동일하게 유지한다.

---

## 15. Adapted 모델 Fine-tuning

각 budget에 대해 Source pretrained weight를 불러온다.

```text
source_model
→ target 5% fine-tuning

source_model
→ target 10% fine-tuning

source_model
→ target 25% fine-tuning
```

권장:

```text
source lr ≈ 1e-3
fine-tuning lr ≈ 1e-4
```

처음에는 전체 layer를 fine-tuning한다.

---

## 16. Target-full 모델 학습

Target Normal Train Pool 100%를 사용해 random initialization부터 학습한다.

이 모델은 Target 환경에서 정상 데이터가 충분한 경우의 reference 성능으로 사용한다.

---

## 17. Threshold Calibration

각 모델별 validation data에서 anomaly threshold를 결정한다.

기본:

```text
99th percentile of validation reconstruction error
```

threshold를 Test 결과를 보고 조절하지 않는다.

---

## 18. 전체 모델 평가

모든 모델을 동일한:

- Target Normal Test
- Target Event Test

에 평가한다.

수집 지표:

1. FPR
2. Event Detection Rate
3. Detection Delay
4. Precision / Recall / F1 가능 시
5. Normal/Event anomaly score

---

## 19. Seed 반복 실험

가능하면 각 핵심 조건을 3회 반복한다.

특히:

```text
Target-small 10%
Adapted 10%
```

은 최소 3회 반복한다.

평균 ± 표준편차로 결과를 저장한다.

---

## 20. 핵심 그래프 생성

반드시 생성할 그래프:

### Figure 1. Source / Target Anomaly Score Distribution
Source-only의 Domain Shift 시각화.

### Figure 2. Scratch vs Pretrained Data Efficiency Curve
x축: Target Normal Data %  
y축: FPR 또는 Detection Metric

### Figure 3. Event Timeline
실제 event timestamp와 anomaly score 변화를 함께 표시.

### Figure 4. 모델별 FPR 비교
Source-only / Target-small / Adapted / Target-full 비교.

---

## 21. 결과 해석

결과는 다음 순서로 해석한다.

1. Domain Shift가 존재했는가?
2. Target 정상 데이터를 추가했을 때 성능이 회복됐는가?
3. 같은 Target 데이터량에서 Adapted가 Scratch보다 좋은가?
4. 어느 Target budget까지 pretraining의 이점이 유지되는가?
5. Target data가 많아질수록 차이가 줄어드는가?
6. Negative Transfer가 발생한 조건은 없는가?

---

## 22. 실패 케이스 분석

Adapted가 Target-small보다 좋지 않은 경우 반드시 분석한다.

확인:

1. Source/Target feature distribution 차이
2. Source/Target traffic load 차이
3. 특정 feature가 domain identifier처럼 작동했는지
4. normalization 문제
5. fine-tuning learning rate 문제
6. Source representation이 Target에 부적합했는지

실패 결과를 억지로 성능 개선으로 해석하지 않는다.

---

## 23. Secondary Experiment 수행 여부 결정

Primary 결과가 모두 나온 뒤 시간이 남는 경우에만 수행한다.

후보:

1. explicit state feature 추가
2. 1% Target budget 추가
3. partial layer freeze
4. Dataset 6 등 추가 Target Domain 검증

Primary 실험 완료 전에는 진행하지 않는다.

---

## 24. 최종 README 작성

README 첫 문장:

> **새로운 네트워크 조건이 추가될 때마다 이상탐지 모델을 처음부터 다시 학습해야 할까?**

README 구성:

1. Problem
2. Hypothesis
3. Dataset
4. Source / Target 정의
5. Experiment Design
6. Fairness Controls
7. Results
8. Key Findings
9. Failure / Limitations
10. Relation to Network Operations
11. Reproduction

---

## 25. 최종 산출물 체크리스트

- [ ] Dataset inventory
- [ ] Source / Target 선정 근거
- [ ] Feature definition
- [ ] Data leakage check
- [ ] Source-only model
- [ ] Target-small models
- [ ] Adapted models
- [ ] Target-full model
- [ ] FPR 결과
- [ ] Event Detection 결과
- [ ] Detection Delay 결과
- [ ] Seed 반복 결과
- [ ] Data Efficiency Curve
- [ ] Event Timeline
- [ ] 실패 사례 분석
- [ ] README
- [ ] requirements.txt
- [ ] 실행 스크립트
- [ ] 최종 결과표

---

# Part III. 권장 프로젝트 구조

```text
network-anomaly-transfer/
│
├── README.md
├── requirements.txt
│
├── config/
│   ├── splits.json
│   ├── features.yaml
│   └── shortcut_features.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   │   ├── inspect_dataset.py
│   │   ├── preprocess.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   └── autoencoder.py
│   │
│   ├── train/
│   │   ├── train_source.py
│   │   ├── train_target_scratch.py
│   │   ├── finetune_target.py
│   │   └── train_target_full.py
│   │
│   ├── evaluate/
│   │   ├── metrics.py
│   │   └── evaluate_all.py
│   │
│   └── visualize/
│       └── plots.py
│
├── artifacts/
│   ├── models/
│   ├── scalers/
│   └── results/
│
├── reports/
│   ├── dataset_inventory.md
│   ├── data_leakage_check.md
│   └── experiment_results.md
│
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

# Part IV. Cursor 작업 원칙

1. **먼저 데이터를 확인하고 모델을 만들 것.**
2. Source/Target schema가 맞지 않으면 억지로 feature를 맞추지 말 것.
3. Go/No-Go 검증 전 Containerlab 구축부터 시작하지 말 것.
4. 모든 split은 timestamp 기반으로 수행할 것.
5. Event 데이터는 학습에 절대 사용하지 말 것.
6. Target-small과 Adapted는 동일한 Target sample을 사용할 것.
7. Target Test를 보고 threshold나 hyperparameter를 조정하지 말 것.
8. Primary experiment가 끝나기 전 LLM/RCA/Grafana 등 범위를 확장하지 말 것.
9. 모든 실험은 seed, feature list, data split, hyperparameter를 기록할 것.
10. 결과가 예상과 달라도 수정해서 숨기지 말고 Negative Transfer 또는 한계로 분석할 것.

---

# 최종 핵심 문장

> **본 프로젝트는 “기존 환경에서 더 많은 데이터를 학습한 모델이 더 좋은가”를 보여주는 것이 아니라, 동일한 양의 새로운 네트워크 데이터를 사용했을 때 사전학습된 정상 패턴이 실제로 적응 속도와 데이터 효율성을 개선하는지를 검증하는 실험이다.**
