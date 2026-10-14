# NetHeal 원본 스냅샷의 Jev 평가 결과

작성일: 2026-10-11 (KST).

## 실행 상태

- 완료: Jev 유료 연결 smoke, 원본 스냅샷 100개 v2 평가, 원본 fingerprint/집계/입력 검증.
- 중단: v1 어댑터는 전체 행동 목록을 자동으로 반복 삽입해 문맥 한도를 초과했다. 원시 결과를 보존하고 v2에서 후보 생성에만 사용하도록 수정했다.
- 미실행: Sol/Claude 재호출, LAYA 8비트 optimizer/LoRA 학습, Docker 컨테이너 자체 실행, 외부 리더보드 제출.

## 결과

| 지표 | 원본 Claude Sonnet 4.5 | Jev 1.13.0 v2 |
|---|---:|---:|
| 평가 사례 | 100 | 100 |
| 원본 방식 가중 진단 성공률 | 65.04% | 25.31% |
| 단순 사례 성공률 | 66/100 (66.00%) | 34/100 (34.00%) |
| 유형 macro-F1 | 0.6373 | 0.4451 |
| 원본 방식 가중 위치 정확도 | 65.04% | 25.31% |
| 평균 단계(장비 수 가중) | 19.01 | 7.44 |

Jev 단순 유형 정확도 39.00%, 단순 위치 정확도 34.00%. 미제출도 전체 100개 분모에 포함했다.

최종 실행 상태: `{'completed': 61, 'Jev HTTP 400': 39}`. HTTP/입력 오류는 진단 오답과 구분해야 한다. 원본 evaluator에 강제 종료 trace를 넣어 전체 분모를 유지했으므로, 오류 metrics를 누락할 수 있는 원본 orchestration과는 차이가 있다. 원본에는 timeout/error가 0이었다.

API 오류 식별자: `{'max_tokens_exceeded': 39}`. 정상 응답에서 보고된 최대 입력 토큰 수는 32926이다. 입력 한도 실패를 모델의 잘못된 최종 진단으로 부르지 않는다.

최종 분류: 성공 34개, 진단 제출 후 오진 27개, 입력 오류 39개. 진단 제출 사례만 골라 계산한 점수를 전체 성능으로 사용하지 않는다.

| 장애 유형 | 사례 수 | 성공 | 오진 | 미완료/오류 |
|---|---:|---:|---:|---:|
| device_failure | 28 | 12 | 3 | 13 |
| link_failure | 21 | 9 | 2 | 10 |
| misconfiguration | 22 | 0 | 19 | 3 |
| performance_degradation | 29 | 13 | 3 | 13 |

동일 사례 paired 집계: `{'both': 31, 'claude_only': 35, 'neither': 31, 'jev_only': 3}`. both=둘 다 성공, jev_only=Jev만 성공, claude_only=Claude만 성공, neither=둘 다 실패. 입력 오류가 포함된 전체 에이전트 결과이며 모델 판단 능력만의 차이로 해석하지 않는다.

## 조건 동일성과 한계

원본 Run ID `685bad60-d554-4300-9c7e-e849301d6df7`의 100개 스냅샷과 시나리오 fingerprint가 100/100 일치했다. 고정 green 이미지의 환경/도구/채점 소스를 사용했다. 100단계, 공개된 기존 힌트, 동시성 5를 사용했다.

Windows Python 3.13.2에서 원본 FastMCP handler를 직접 호출했다. 원본 Docker Linux Python 3.11.14의 전체 orchestration과 의존성까지 재현한 것은 아니다. Jev는 도구→인자를 선택하며 생성형 설명을 쓰지 않는다. 인자 후보에 공개 list_actions를 사용하고, 실제 모델 선택 도구 결과는 모두 히스토리에 보존한다. 정답/숨겨진 장애 상태를 모델 입력에 넣지 않았다.

Jev 문맥 한도 때문에 발생한 입력 오류를 관측 절단이나 평가 중 설정 변경으로 숨기지 않았다. v1 입력 결함 수정 후의 v2 결과이며 일부 평가 사례를 이미 실행한 후이므로 완전한 최초 미관측 평가로 주장하지 않는다. 본 v2 설정은 시작 전에 해시로 고정했고 점수에 따라 변경하지 않았다.

## 해석

이 구성에서 Jev는 도구를 선택하고 일부 원본 사례를 정확히 진단할 수 있었다. 그러나 Claude 공개 구성 수준의 전체 성공률은 확보하지 못했다. 39개는 판정 이전의 문맥 한도 실패이므로 전체 격차를 추론 능력 차이로만 설명할 수 없다. 문맥 제한을 해결하면 모두 맞힐 것이라는 결론도 낼 수 없다. 원본 공개 관측 정보를 유지하면서 중복을 줄이는 입력 설계의 효과는 후속 과제이며 이번 평가 중에는 추가 조정하지 않았다. LAYA 성능이나 8비트 optimizer/LoRA 효과는 이번 결과로 판단하지 않는다.

## 시간·비용

- v2 최종 시도 평균 episode 시간: 40.09초. 동시 실행·요청 간격 최소 0.5초·Windows 직접 호출을 포함하므로 Claude 모델 속도와 직접 비교 불가.
- v2 본 평가 API 응답 대기 시간 합계(동시 요청 합산): 431.46초.
- 모델 선택 도구 호출 총 776회, 동일 도구/인자 반복 비율 0.52%. 원본 가중 redundancy_rate와 정의가 다르다.
- v1/v2 smoke·평가에서 usage가 기록된 입력 token 기준 총 추정 비용: **$0.978898**.
- 비용 근거: [TypeSafe 모델 문서](https://docs.typesafe.ai/models), 2026-10-11 확인, 입력 100만 token당 $0.042, 출력 무료. 청구서 확정액은 아니다. 400 오류 재전송 1회는 usage가 없고, 중단 시 응답 미기록 요청도 있을 수 있어 미계상 가능성을 남긴다.

상세 비용/HTTP 상태는 `verification.json`의 costs를 참조한다.

## 재현·검증

실행 명세: [v2 사전 동결 문서](../../../specs/netheal_official_jev_execution_v2_20261011.md).

```powershell
.runtime/jev-official-venv/Scripts/python.exe -X utf8 netheal-topology-generalization/scripts/jev_official_eval_v2.py evaluate
.runtime/jev-official-venv/Scripts/python.exe -X utf8 netheal-topology-generalization/scripts/report_jev_official_eval.py
```

평가 명령은 완료 episode를 재사용한다. 새 재현 실행은 별도 OUT과 사전 고정 프로토콜이 필요하다. 기존 결과를 삭제/덮어쓰지 않는다.

결과: `results/netheal_official_jev_20261011_v2/summary.json`, `verification.json`, episode별 requests/tools/helpers/result, frozen_protocol 및 실행 소스 사본. 원본 스냅샷과 provenance는 v1/upstream에 보존했다.
